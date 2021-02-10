"""
Load Khan Data for Math training.
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F
import json
import glob
import logging
import random
import io
from tqdm import tqdm
import os

from dataset.util import last_boxed_only, _clean_numbers, last_boxed_only_string

from multiprocessing import Manager

from torch.multiprocessing import Pool
from dataset.base_math_dataset import BaseMathDataset

class MathematicaWithStepsMathDataset(BaseMathDataset):
    """Configurable Math Dataset.
    """

    def __len__(self):
        return int(len(self.samples) * self.len_multiplier)

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """

        with open(self.dataroot, 'r') as fp:
            all_filenames = fp.readlines()
 
        print(f"{self.__class__.__name__}: Loading samples from {len(all_filenames)} files.")
        samples_raw = []
        for fname in tqdm(all_filenames):
            fname = fname.rstrip()
            fname = os.path.join(os.path.dirname(os.path.dirname(self.dataroot)), fname[2:])
            with open(fname, 'r') as fp:
                question = ""
                answers  = []
                reading_question = True
                curr_section = ""
                for line in fp:
                    if line == "Problem:\n":
                        reading_question = True
                    elif line == "Answer:\n":
                        if reading_question:
                            # curr_section contains Q
                            question = curr_section
                        else:
                            # curr_section contains an A
                            answers.append(curr_section)
                        curr_section = ""
                        reading_question = False
                    else:
                        curr_section += line
                
                # The last answer needs to be recorded.
                answers.append(curr_section)
            
            for a in answers:
                samples_raw.append((question, a, fname))

        # manager = Manager()
        # samples_raw = manager.list(samples_raw)
        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")

    def clean_filter_sample_gpt(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        question, answer = sample
        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = _clean_numbers(answer)

        if self.mode_answer == 'default':
            question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))

            sep_ids          = self.tokenizer.encode("\nFULL SOLUTION:\n", verbose=False)
            sep_ids.append(self.tokenizer.eos_token_id)
            sep_ids          = torch.LongTensor(sep_ids)

            answer_ids       = self.tokenizer.encode(answer, verbose=False)
            answer_ids.append(self.tokenizer.eos_token_id)
            answer_ids       = torch.LongTensor(answer_ids)
            
            # Use full solution
            input_ids = torch.cat([
                question_ids, 
                sep_ids,
                answer_ids
            ], dim=0)

            label_ids = torch.cat([
                torch.ones_like(question_ids) * -100, 
                torch.ones_like(sep_ids) * -100, 
                answer_ids.clone()
            ], dim=0)
        else:
            raise NotImplementedError()
        
        # Stop early if this Q,A pair is too long
        if question_ids.shape[0] + sep_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"{self.__class__.__name__} Skipping due to input_ids being too big. question_ids.shape[0] + sep_ids.shape[0] = {question_ids.shape[0] + sep_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

    def clean_filter_sample_t5(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        question, answer = sample
        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = _clean_numbers(answer)

        if self.mode_answer == 'default':
            question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question + "\nFULL SOLUTION:\n", verbose=False))
            answer_ids       = torch.LongTensor(self.tokenizer.encode(answer, verbose=False))

            input_ids = torch.cat([
                question_ids, 
            ], dim=0)

            label_ids = torch.cat([
                answer_ids
            ], dim=0)
        else:
            raise NotImplementedError()
        
        # Stop early if this Q,A pair is too long
        if input_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"{self.__class__.__name__} Skipping due to input_ids being too big. input_ids.shape[0] = {input_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }
        
