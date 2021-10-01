
"""

Example:

CUDA_VISIBLE_DEVICES=6 python3 eval_math_gpt.py \
    --arch=gpt2 \
    --math-dataroot=./MATH/test/*/*.json \
    --load=/data/sauravkadavath/maths-beta__modeling__checkpoints/MATH__bbox_only_3_epochs__finetune_6_epochs__pretraining_khan_latex_loss_only__gpt117/checkpoint.pth

"""

import io
import logging
import math
import os
import pprint
import sys
import json
import time
import transformers
import numpy as np

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.MATH import MATHDataset
from dataset.khan_academy import KhanAcademyMathDataset
from dataset.util import clean_numbers, last_boxed_only, last_boxed_only_string
from math_equivalence import is_equiv

def get_level_type(fname):
    """
    Somewhat inefficient, but much easier than changing dataloader and probably fine for evaluation
    """
    with open(fname, 'r') as fp:
        try:
            problem_data = json.load(fp)
        except Exception as e:
            print(f"Error loading JSON from {fname}", e)
            raise e
    level, prob_type = problem_data['level'], problem_data['type']
    try:
        level = int(level.split("Level ")[1])
    except:
        level = None
    return level, prob_type

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def dict_to_gpu(d, device_id=None):
    new_dict = dict()
    for key, value in d.items():
        # Only move to GPU is cuda() is a function
        if 'cuda' in dir(value):
            new_dict[key] = value.cuda(device_id)
        else:
            new_dict[key] = value
    return new_dict


def get_real_sol_idxs(tokens_sol, tokenizer):
    """
    Return the start and stop indexes (inclusive) for everything inside \\boxed{...}
    """
    left_idx, right_idx = None, None
    for i in range(tokens_sol.shape[1]):
        if i < 3:
            continue

        if tokens_sol[0, i].item() and \
            tokens_sol[0, i-1].item() == 276 and \
            tokens_sol[0, i-2].item() == 3524:
            # at index i, we have the { of \\boxed{ 
            left_idx = i + 1 # Don't include the {
        
        if tokens_sol[0, i].item() == 50256:
            right_idx = i-2 # don't include the one token before the current one as well (usually the } from \boxed{})
    
    # Will error if either is not found, which we dont expect
    return left_idx, right_idx


def run_eval(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    if args.tokenizer_merges_file is not None:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch, merges_file=args.tokenizer_merges_file)
    else:
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(args.arch)

    eval_data = get_dataset(args)
    for inner_dset in eval_data.datasets:
        inner_dset.tokenizer = tokenizer

    dataloader = torch.utils.data.DataLoader(
        eval_data, 
        batch_size=1, 
        num_workers=0, 
        pin_memory=True,
    )

    """
    with torch.no_grad():
        correct = 0
        total = 0
        for i, batch in enumerate(tqdm(dataloader)):
            batch = dict_to_gpu(batch, device_id=0)
            print(batch['fnames'])
            print(batch['input_ids'])
            quit()
    """

    # Set up model
    if args.load is None:
        model = transformers.GPT2LMHeadModel.from_pretrained(args.arch)
    else:
        print(f"Loading model from {args.load}")
        model = transformers.GPT2LMHeadModel.from_pretrained(args.load)
        print(f"Successfully loaded model from {args.load}")

    model = model.eval()
    model = model.cuda()

    loss_moving_average = 0

    outputs = []
    answers = []
    types = []
    levels = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}

    with torch.no_grad():
        correct = 0
        total = 0
        skipped = 0
        mean_max_probs_correct = []
        mean_max_probs_wrong   = []
        for i, batch in enumerate(tqdm(dataloader)):

            if torch.sum(batch['input_ids']) == 0:
                skipped += 1
                print("SKIPPING", batch['fnames'][0])
                continue

            fnames = batch['fnames'][0]
            assert len(fnames) == 1
            fnames_list.append(fnames[0])
            prob_level, prob_type = get_level_type(fnames[0])
            batch = dict_to_gpu(batch, device_id=0)

            output_ids = model.generate(
                batch['input_ids'], 
                num_beams=args.num_beams, 
                early_stopping=True,
                temperature=1.0,
                max_length=384 if args.arch == 'gpt2-xl' else 1024
            )
            
            # logits = model(output_ids).logits
            # probs = F.softmax(logits, dim=2) # torch.Size([1, L, 50257])
            # max_probs, max_tokens = probs.max(2) # torch.Size([1, L]), torch.Size([1, L])

            # num_tokens_for_question = batch['input_ids'].shape[1]
            # probs_sol = max_probs[:, num_tokens_for_question-1:]
            # tokens_sol = max_tokens[:, num_tokens_for_question-1:]

            # real_sol_start_idx, real_sol_stop_idx = get_real_sol_idxs(tokens_sol, tokenizer)
            # if real_sol_start_idx is None or real_sol_stop_idx is None:
            #     skipped += 1
            #     print("BAD ANSWER, SKIPPING", batch['fnames'][0])
            #     continue
            # probs_sol = probs_sol[:, real_sol_start_idx:real_sol_stop_idx + 1]
            # mean_probs_sol = torch.mean(probs_sol).item()
            mean_probs_sol = 0

            output_tokens = get_model_output(batch['input_ids'][0], output_ids[0], tokenizer)

            # Print this iteration
            output_str = tokenizer.decode(output_tokens)
            output_full = output_str
            output_str = last_boxed_only_string(output_str)

            if args.math_mode == "eval_peeking":
                answer_str = last_boxed_only_string(tokenizer.decode(batch['labels'][0]))
            else:
                answer_str = tokenizer.decode(batch['labels'][0])

            output, answer = remove_boxed(output_str), remove_boxed(answer_str)

            print("Problem String:")
            print(tokenizer.decode(batch['input_ids'][0]) + "\n")
            print("Model output:")
            print(output_full)
            print(output)
            print("Correct answer:")
            print(answer)
            print("fname")
            print(fnames)
            print("--------------------------------------------")

            # scratchwork_fname = "___".join(fnames[0].split("/")[-2:])
            # with open(f"scratchwork_Temp2e-1_{args.arch}/{scratchwork_fname}.txt", 'w') as f:
            #     f.write("Problem String:" + "\n")
            #     f.write(tokenizer.decode(batch['input_ids'][0]) + "\n")
            #     f.write("Model output:" + "\n")
            #     f.write(output_full + "\n")
            #     f.write(str(output) + "\n")
            #     f.write("Correct answer:" + "\n")
            #     f.write(answer + "\n")
            #     f.write("--------------------------------------------" + "\n")

            outputs.append(output)
            answers.append(answer)
            types.append(prob_type)
            levels.append(prob_level)

            equiv = is_equiv(output, answer)
            if (prob_level, prob_type) in cors:
                cors[(prob_level, prob_type)].append(equiv)
            else:
                cors[(prob_level, prob_type)] = [equiv]
            
            if prob_level in level_cors:
                level_cors[prob_level].append(equiv)
            else:
                if prob_level is not None:
                    level_cors[prob_level] = [equiv]
            
            if prob_type in subject_cors:
                subject_cors[prob_type].append(equiv)
            else:
                if prob_type is not None:
                    subject_cors[prob_type] = [equiv]
            
            if equiv:
                correct += 1
                mean_max_probs_correct.append(mean_probs_sol)
            else:
                mean_max_probs_wrong.append(mean_probs_sol)

            # print("CORRECT", mean_max_probs_correct)
            # print("WRONG", mean_max_probs_wrong)
            
            total += 1

    subjects = ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']

    print(f"Average of mean_max_probs_correct = {sum(mean_max_probs_correct)}/{len(mean_max_probs_correct)} = ", sum(mean_max_probs_correct)/len(mean_max_probs_correct))
    print(f"Average of mean_max_probs_wrong   = {sum(mean_max_probs_wrong)}/{len(mean_max_probs_wrong)} = ", sum(mean_max_probs_wrong)/len(mean_max_probs_wrong))

    # now save outputs and answers
    with open(f"outputs_answers_Temp2e-1_{args.arch}.txt", "w+") as f:
        for k, (output, answer, prob_type, prob_level, fname) in enumerate(zip(outputs, answers, types, levels, fnames_list)):
            f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | FNAME: {}\n".format(k, prob_type, prob_level, output, answer, fname))

        # print(cors)
        for prob_type in subjects:
            for prob_level in [1, 2, 3, 4, 5]:
                if (prob_level, prob_type) in cors:
                    cors_list = cors[(prob_level, prob_type)]
                    print("{} Level {} Accuracy = {}/{} = {:.3f}".format(prob_type, prob_level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                    f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(prob_type, prob_level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))

        print("#####################")
        f.write("#####################\n")
        # also get accuracies for each 
        for level in sorted(level_cors):
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")

        for subject in subjects:
            # for subject in sorted(subject_cors):
            if subject in subject_cors:
                cors_list = subject_cors[subject]
                print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
        print("#####################")
        f.write("#####################\n")
        
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total))
        print("Skipped = {}".format(skipped))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total))
        f.write("Skipped = {}".format(skipped))
    
    print()
    
def get_model_output(context, full_output, tokenizer):
    """
    Given the context and the full model output (context + generated),
    extract just the generated tokens.
    Remove the last token if it is <|endoftext|>
    """
    ret = full_output[len(context):]
    if ret[-1] == tokenizer.eos_token_id:
        ret = ret[:-1]
    return ret

def get_dataset(args):
    all_datasets = []

    if args.math_dataroot is not None:
        if args.math_mode == 'gpt2-eval':
            all_datasets.append(
                MATHDataset(
                    dataroot=args.math_dataroot, 
                    tokenizer=None, # Set in run_training(), not in dataset creation 
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024, 
                    mode='gpt2-eval', 
                )
            )
        else:
            all_datasets.append(
                MATHDataset(
                    dataroot=args.math_dataroot, 
                    tokenizer=None, # Set in run_training(), not in dataset creation 
                    max_tokens=384 if args.arch == 'gpt2-xl' else 1024, 
                    mode='gpt2-eval',
                    mode_answer=args.math_mode,
                    peek_fraction=args.peek_fraction
                )
            )

    
    train_data = torch.utils.data.ConcatDataset(all_datasets)
    return train_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language Modelling on Code")
    parser.add_argument('--arch', default='gpt2', choices=transformers.GPT2_PRETRAINED_MODEL_ARCHIVE_LIST)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--num-beams', default=20, type=int)
    parser.add_argument('--tokenizer-merges-file', default=None, type=str)

    # Dataloading
    parser.add_argument('--math-dataroot', default=None, type=str)
    parser.add_argument('--math-mode', default='gpt2-eval', type=str)
    parser.add_argument('--peek-fraction', type=float, default=1.0)
    
    # Others
    parser.add_argument('--workers', default=4, type=int)

    args = parser.parse_args()

    run_eval(args)
