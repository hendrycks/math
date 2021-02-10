"""
Tune LM on Code
"""

import io
import logging
import math
import os
import pprint
import sys
import time
import json
import argparse

import transformers

from tqdm import tqdm
from datetime import datetime

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from dataset.MATH import MATHDataset
from dataset.khan_academy import KhanAcademyMathDataset
from dataset.mathematica import MathematicaMathDataset
from dataset.mathematica_with_steps import MathematicaWithStepsMathDataset

def run_training(args, train_data):
    
    if not args.save_steps:
        # Save every epoch
        if not args.tpu_num_cores:
            save_steps = len(train_data) 
            save_steps = int(save_steps / torch.cuda.device_count())
            save_steps = int(save_steps / args.grad_acc_steps)
            save_steps = int(save_steps / args.batch_size_per_replica)
        else:
            save_steps = len(train_data) 
            save_steps = int(save_steps / 8) # 8 TPU cores is constant for now.
            save_steps = int(save_steps / args.grad_acc_steps)
            save_steps = int(save_steps / args.batch_size_per_replica)
    else:
        save_steps = args.save_steps



    print("Save Steps = ", save_steps)

    ## Checkpoint Loading ######################################################## 
    if args.load:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Loaded model from {args.load}")
    else:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)

    start_epoch = 0
    start_iteration = 0
    
    ## Dataloading ######################################################## 
    train_data.start_iteration = start_iteration

    ## Start Loop ########################################################
    print(f"Setting up Trainer")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=False,

        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.lr_warmup_steps,
        max_grad_norm=100000.0, # Essentially disable gradient clipping

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=save_steps,
        save_total_limit=10, # Only save the last epoch

        dataloader_drop_last=True,
        dataloader_num_workers=args.dataloader_num_workers,

        local_rank=args.local_rank,
        tpu_num_cores=args.tpu_num_cores,
    )

    trainer = GPT2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
    )
    trainer.remove_callback(transformers.integrations.TensorBoardCallback)
    trainer.add_callback(CustomTensorBoardCallback())

    print(f"STARTING TRAINING. save_steps={save_steps}")
    trainer.train()
    
    trainer.save_model(os.path.join(args.save_dir, "final_checkpoint"))
    print("Finished")


class GPT2Trainer(transformers.Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            print("Making AdamW Optimizer")
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

        if self.lr_scheduler is None:
            
            if self.args.warmup_steps == -1:
                print("Using constant LR")
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda steps: 1.0)
            else:
                print("Using Linear warmup LR")
                self.lr_scheduler = self.get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )

    @staticmethod
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        """
        Linear warmup from 0 to max lr, then linear decay from max_lr to 0.1*max_lr
        As done in https://arxiv.org/pdf/2010.14701.pdf
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            min_lr_multiplier = 0.1
            return max(
                min_lr_multiplier, 
                ((1 - min_lr_multiplier) * float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))) + min_lr_multiplier
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_tokenizer_gpt(args):
    """
    If args.tokenizer_merges_file is given, return a tokenizer that uses that merges_file.
    In the paper, we use this to restrict models to ingest and outuput digits. For example:
    
    >>> tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2", merges_file="merges_gpt2_single_digit_numbers.txt")
    >>> tokenizer_old = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    >>> tokenizer.encode("1")
    [16]
    >>> tokenizer_old.encode("1")
    [16]
    >>> tokenizer.encode("2")
    [17] 
    >>> tokenizer_old.encode("12")
    [1065]
    >>> tokenizer.encode("12")
    [16, 17]
    >>> tokenizer.encode("HEllo world!")
    [13909, 18798, 995, 0]
    >>> tokenizer_old.encode("HEllo world!")
    [13909, 18798, 995, 0]
    """
    if args.tokenizer_merges_file is not None:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch, merges_file=args.tokenizer_merges_file)
    else:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)
    return tokenizer

def get_dataset(args): 

    tokenizer = get_tokenizer_gpt(args)
    # print(tokenizer.tokenize("1231231234441234 blah dklkjl12490!!@ 2*x + y^k + f(x)"))  # sanity check
    
    train_data = []

    if args.mathematica_dataroot:
        for mathematica_dr in args.mathematica_dataroot:
            len_multiplier, dirname = mathematica_dr.split("@")
            len_multiplier = float(len_multiplier)

            no_steps_flist_fname = os.path.join(dirname, "no_steps_flist_relative.txt")
            with_steps_flist_fname = os.path.join(dirname, "with_steps_flist_relative.txt")

            with open(no_steps_flist_fname,"r") as f:
                no_steps_num_files = len(f.readlines())

            with open(with_steps_flist_fname,"r") as f:
                with_steps_num_files = len(f.readlines())

            if no_steps_num_files:
                train_data.append(MathematicaMathDataset(
                    dataroot=no_steps_flist_fname,
                    tokenizer=tokenizer,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode='gpt2',
                    len_multiplier=len_multiplier
                ))

            if with_steps_num_files:
                train_data.append(MathematicaWithStepsMathDataset(
                    dataroot=with_steps_flist_fname,
                    tokenizer=tokenizer,
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
                    mode='gpt2',
                    len_multiplier=len_multiplier
                ))

    if args.khan_dataroot:
        len_multiplier, dirname = args.khan_dataroot.split("@")
        len_multiplier = float(len_multiplier)
        train_data.append(KhanAcademyMathDataset(
            dataroot=dirname,
            tokenizer=tokenizer,
            max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
            mode='gpt2',
            mode_answer=args.khan_mode,
            len_multiplier=len_multiplier,
            latex_mask=args.khan_latex_mask
        ))
    
    if args.MATH_dataroot:
        train_data.append(MATHDataset(
            dataroot=args.MATH_dataroot,
            tokenizer=tokenizer,
            max_tokens=384 if args.arch == 'gpt2-xl' else 1024,
            mode='gpt2',
            mode_answer=args.MATH_mode,
            len_multiplier=1.0,
            peek_fraction=(args.MATH_peek_min, args.MATH_peek_max),
            packing=True,    # Special for fine-tuning
            randomize=True   # Special for fine-tuning
        ))


    # Print the sizes of each dataset, useful for weighting
    for dset in train_data:
        print(f"{dset.__class__.__name__}: __len__ = {len(dset)}")

    return torch.utils.data.ConcatDataset(train_data)


def main():

    ######### Arg parsing ###############################################################

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument('--tokenizer-merges-file', default=None, type=str)
    parser.add_argument('--load', default=None, type=str)

    # Dataloading
    parser.add_argument('--khan-mode', default='mixed_hints', type=str)
    parser.add_argument('--khan-dataroot', default=None, type=str)
    parser.add_argument('--khan-latex-mask', default=False, action='store_true')
    parser.add_argument('--deepmind-dataroot', default=None, type=str, action='append')
    parser.add_argument('--mathematica-dataroot', default=None, type=str, action='append')
    parser.add_argument('--mathematica-with-steps-dataroot', default=None, type=str, action='append')
    parser.add_argument('--MATH-mode', default='mixed_final_boxed_and_full', type=str, choices=['mixed_final_boxed_and_full', 'final_boxed', 'peeking', 'nopack_padding', 'mixed_full_and_peeking', 'mixed_full_and_nopack_padding'])
    parser.add_argument('--MATH-peek-min', default=0.1, type=float)
    parser.add_argument('--MATH-peek-max', default=1.0, type=float)
    parser.add_argument('--MATH-dataroot', default=None, type=str)
    parser.add_argument('--stackexchange-dataroot', default=None, type=str)
    parser.add_argument('--dataloader-num-workers', default=1, type=int)

    # Training
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--weight-decay', default=0.05, type=float)
    parser.add_argument('--lr-warmup-steps', default=-1, type=int)
    parser.add_argument('--batch-size-per-replica', default=8, type=int)
    parser.add_argument('--grad-acc-steps', default=4, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--tpu_num_cores', default=None, type=int)

    # Logging and stuff
    parser.add_argument('--save-dir', default="checkpoints/TEMP", type=str)
    parser.add_argument('--save-steps', default=0, type=int)
    parser.add_argument('--log-freq', default=5, type=int)

    args = parser.parse_args()
    args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%m-%d-%Y__%H:%M:%S"))

    ######### Start training ###############################################################

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    train_data = get_dataset(args)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "command.txt"), 'w') as f:
        f.write(pprint.pformat(argsdict))

    run_training(args, train_data)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
