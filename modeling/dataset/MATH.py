"""
Load MATH Data for training.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import json
import glob
import logging
import io
import random
import numpy as np

from dataset.util import last_boxed_only, _clean_numbers, last_boxed_only_string, only_until_first_boxed_from_tokens

from multiprocessing import Manager

from torch.multiprocessing import Pool
from dataset.base_math_dataset import BaseMathDataset

class MATHDataset(BaseMathDataset):
    """Configurable Math Dataset.
    """

    def __len__(self):
        return int(len(self.samples) * self.len_multiplier)

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """

        all_filenames = glob.glob(self.dataroot)
        samples_raw = []
        for fname in all_filenames:
            with open(fname, 'r') as fp:
                try:
                    problem_data = json.load(fp)
                except Exception as e:
                    print(f"Error loading JSON from {fname}", e)
                    raise e
            curr_sample_raw = (problem_data['problem'], problem_data['solution'], fname)
            for e in curr_sample_raw:
                assert e
            samples_raw.append(curr_sample_raw)
        
        manager = Manager()
        samples_raw = manager.list(samples_raw)
        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def clean_filter_sample_gpt(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        if self.mode_answer == 'peeking_only':
            return self.clean_filter_sample_peeking_gpt(sample)
        if self.mode_answer == 'mixed_full_and_peeking':
            if random.random() < 0.5:
                return self.clean_filter_sample_peeking_gpt(sample)
            else:
                _mode_answer = 'full'
        elif self.mode_answer == 'mixed_full_and_nopack_padding':
            if random.random() < 0.5:
                return self.clean_filter_sample_nopackpadding_gpt(sample)
            else:
                _mode_answer = 'full'
        elif self.mode_answer == 'mixed_final_boxed_and_full':
            if random.random() < 0.5:
                _mode_answer = 'full'
            else:
                _mode_answer = 'final_boxed'
        elif self.mode_answer == 'full':
            _mode_answer = 'full'
        elif self.mode_answer == 'final_boxed':
            _mode_answer = 'final_boxed'
        else:
            raise NotImplementedError(f"self.mode_answer = {self.mode_answer} not recognized.")


        if _mode_answer == 'full':
            question, answer = sample

            if self.clean_numbers:
                question = _clean_numbers(question)
                answer   = _clean_numbers(answer)

            answer_final = last_boxed_only_string(answer)

            question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
            
            sep_ids_2        = torch.LongTensor(self.tokenizer.encode("\nFULL SOLUTION:\n", verbose=False))
            answer_ids       = self.tokenizer.encode(answer, verbose=False)
            answer_ids.append(self.tokenizer.eos_token_id)
            answer_ids       = torch.LongTensor(answer_ids)
            
            input_ids = torch.cat([
                question_ids, 
                sep_ids_2,
                answer_ids
            ], dim=0)

            # Only answer_ids contribute to the loss
            label_ids = torch.cat([
                torch.ones_like(question_ids) * -100, 
                torch.ones_like(sep_ids_2) * -100, 
                answer_ids.clone()
            ], dim=0)
        
        elif _mode_answer == 'final_boxed':
            question, answer = sample

            if self.clean_numbers:
                question = _clean_numbers(question)
                answer   = _clean_numbers(answer)
            answer_final = last_boxed_only_string(answer)
            if not answer_final:
                print("ERROR FROM", question, answer)
                return None

            question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
            
            sep_ids_1        = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
            answer_final_ids = self.tokenizer.encode(answer_final, verbose=False)
            answer_final_ids.append(self.tokenizer.eos_token_id)
            answer_final_ids = torch.LongTensor(answer_final_ids)

            input_ids = torch.cat([
                question_ids, 
                sep_ids_1, 
                answer_final_ids,
            ], dim=0)

            # Only answer_ids contribute to the loss
            label_ids = torch.cat([
                torch.ones_like(question_ids) * -100, 
                torch.ones_like(sep_ids_1) * -100, 
                answer_final_ids.clone(),
            ], dim=0)
        
        else:
            raise NotImplementedError()
        
        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_nopackpadding_gpt(self, sample):

        if sample == None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer   = _clean_numbers(answer)

        answer_final = last_boxed_only_string(answer)

        question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids          = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
        final_answer_ids = torch.LongTensor(self.tokenizer.encode(answer_final, verbose=False))

        # Stop early if this Q,A pair is too long
        num_to_pad = 32
        padding_tensor = torch.ones((num_to_pad)) * 220 # 220 is the token for space in the case of GPT2 models
        
        input_ids = torch.cat([
            question_ids, 
            padding_tensor,
            sep_ids,
            final_answer_ids
        ], dim=0)

        # Only answer_ids contribute to the loss
        label_ids = torch.cat([
            torch.ones_like(question_ids) * -100, 
            torch.ones_like(padding_tensor) * -100,
            torch.ones_like(sep_ids) * -100,
            final_answer_ids.clone()
        ], dim=0)
        
        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_nopackpadding_gpt_eval(self, sample):

        if sample == None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer   = _clean_numbers(answer)

        answer_final = last_boxed_only_string(answer)

        question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids          = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
        final_answer_ids = torch.LongTensor(self.tokenizer.encode(answer_final, verbose=False))

        num_to_pad = 32
        padding_tensor = torch.ones((num_to_pad)) * 220 # 220 is the token for space in the case of GPT2 models
        
        input_ids = torch.cat([
            question_ids, 
            padding_tensor,
            sep_ids,
        ], dim=0)

        # Only answer_ids contribute to the loss
        label_ids = torch.cat([
            final_answer_ids.clone()
        ], dim=0)

        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] + label_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None
        
        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_peeking_gpt(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer   = _clean_numbers(answer)

        answer_final = last_boxed_only_string(answer)

        question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question + "\nFULL SOLUTION:\n", verbose=False))
        answer_ids       = self.tokenizer.tokenize(answer)
        answer_ids       = only_until_first_boxed_from_tokens(answer, answer_ids)
        answer_ids       = torch.LongTensor(self.tokenizer.encode(answer_ids, verbose=False))

        # Take a fraction
        if isinstance(self.peek_fraction, tuple):
            final_idx = int(len(answer_ids) * random.uniform(*self.peek_fraction))
        else:
            final_idx = int(len(answer_ids) * self.peek_fraction)

        # # Override peeking fraction
        # final_idx = int(len(answer_ids) * np.random.choice([0.25, 0.5, 0.75, 1.0], p=[1/6, 1/6, 1/3, 1/3]))

        answer_ids = answer_ids[:final_idx]

        sep_ids          = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
        final_answer_ids = torch.LongTensor(self.tokenizer.encode(answer_ids[final_idx:]))
        
        input_ids = torch.cat([
            question_ids, 
            answer_ids,
            sep_ids,
            final_answer_ids
        ], dim=0)

        # Only answer_ids contribute to the loss
        label_ids = torch.cat([
            torch.ones_like(question_ids) * -100, 
            torch.ones_like(answer_ids) * -100,
            torch.ones_like(sep_ids) * -100,
            final_answer_ids.clone()
        ], dim=0)
        
        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_peeking_gpt_eval(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer   = _clean_numbers(answer)

        answer_final = last_boxed_only_string(answer)

        question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question + "\nFULL SOLUTION:\n", verbose=False))
        answer_ids       = self.tokenizer.tokenize(answer)
        answer_ids_full = torch.LongTensor(self.tokenizer.encode(answer))
        answer_ids       = only_until_first_boxed_from_tokens(answer, answer_ids)
        if len(answer_ids) == 0:
            return None
        answer_ids       = torch.LongTensor(self.tokenizer.encode(answer_ids, verbose=False))

        # Take a fraction
        if isinstance(self.peek_fraction, tuple):
            final_idx = int(len(answer_ids) * random.uniform(*self.peek_fraction))
        else:
            final_idx = int(len(answer_ids) * self.peek_fraction)
        
        answer_ids = answer_ids[:final_idx]

        # sep_ids          = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER\n", verbose=False))
        final_answer_ids = answer_ids_full[final_idx:]
        print(final_answer_ids)
        
        input_ids = torch.cat([
            question_ids, 
            answer_ids,
            # sep_ids,
        ], dim=0)

        # Only answer_ids contribute to the loss
        label_ids = torch.cat([
            final_answer_ids.clone()
        ], dim=0)
        
        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] + label_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_gpt_eval(self, sample):
        """
        Does tokenization for final model evaluation. This should return
        input_ids as the context and labels as the true answer.
        """

        if sample == None:
            return None

        if self.mode_answer == 'eval_peeking':
            return self.clean_filter_sample_peeking_gpt_eval(sample)
        elif self.mode_answer == 'eval_nopack_padding':
            return self.clean_filter_sample_nopackpadding_gpt_eval(sample)

        question, answer = sample

        if self.clean_numbers:
            question = _clean_numbers(question)
            answer   = _clean_numbers(answer)
        answer_final = last_boxed_only_string(answer)

        assert not answer.isspace()

        question_ids = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
        sep_ids      = torch.LongTensor(self.tokenizer.encode("\FULL SOLUTION:\n", verbose=False))
        answer_final_ids   = torch.LongTensor(self.tokenizer.encode(answer_final, verbose=False)) # Loss only counted on these tokens.

        input_ids = torch.cat([
            question_ids, 
            sep_ids, 
        ], dim=0)

        label_ids = torch.cat([
            answer_final_ids.clone()
        ], dim=0)
        
        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] + label_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None
        
        return {
            'input_ids_list' : input_ids.tolist(),
            'label_ids_list' : label_ids.tolist()
        }

