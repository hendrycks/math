"""
Base Dataset for Math training
"""

import torch
import torch.nn as nn 
import torch.nn.functional as F

import pprint
import random
import glob
import logging
import io
import os
import time

from torch.multiprocessing import Pool

from dataset.util import last_boxed_only, clean_numbers, last_boxed_only_string

class BaseMathDataset(torch.utils.data.Dataset):
    """Configurable APPS Dataset.
    """

    def __init__(self, dataroot, tokenizer, max_tokens, mode, mode_answer='default', len_multiplier=1.0, packing=None, randomize=None, pack_end=None, clean_numbers=False, latex_mask=False, peek_fraction=(0.1, 1.0)):
        self.dataroot = dataroot
        self.tokenizer = tokenizer # Set in run_training(), not in dataset creation
        self.max_tokens = max_tokens
        self.mode = mode
        self.mode_answer = mode_answer # Used in subclass
        self.len_multiplier = len_multiplier
        self.clean_numbers = clean_numbers
        self.latex_mask = latex_mask
        self.peek_fraction = peek_fraction

        if self.mode in {'gpt2'}:
            self.clean_sample = self.clean_filter_sample_gpt
            self.packing = True
            self.randomize = True
            self.include_fnames = False
            self.pack_end = True
        elif self.mode in {'gpt2-eval'}:
            self.clean_sample = self.clean_filter_sample_gpt_eval
            self.packing = False
            self.randomize = False
            self.include_fnames = True
            self.pack_end = True
        else:
            raise NotImplementedError()

        if packing != None:
            print("Overriding packing to be", packing)
            self.packing = packing
        if randomize != None:
            print("Overriding randomize to be", randomize)
            self.randomize = randomize
        if pack_end != None:
            print("Overriding pack_end to be", pack_end)
            self.pack_end = pack_end
        
        self.initialize()

        self.bad_fnames = set()
        self.i = 0
    
    
    def initialize(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):

        # Each worker needs a different seed....
        random.seed(os.getpid() + time.time() + random.random())

        # Sampling with replacement.
        # We need to pack random elements to get close to self.max_tokens
        curr_input_ids = []
        curr_label_ids = []
        curr_fnames = []
        num_samples = 0
        while len(curr_input_ids) + 1 <= self.max_tokens and len(curr_label_ids) + 1 <= self.max_tokens:
            curr_sample, fname = self.get_random_sample()
            if curr_sample is None:
                # This only happens in eval modes
                return {
                    "input_ids" : torch.zeros([self.max_tokens]),
                    "labels" : torch.zeros([self.max_tokens]),
                    "fnames" : [fname]
                }

            if not self.pack_end and (
                (len(curr_input_ids) + 1 + len(curr_sample['input_ids_list']) > self.max_tokens) or 
                (len(curr_label_ids) + 1 + len(curr_sample['label_ids_list']) > self.max_tokens)
            ):
                # Do not include curr_sample if either the input_ids or the label_ids will run off the end.
                break

            # Add curr_sample to the current inputs and labels
            curr_input_ids.extend(curr_sample['input_ids_list'])
            curr_label_ids.extend(curr_sample['label_ids_list'])
            curr_fnames.append(fname)
            
            num_samples += 1

            # Break on the first iteration if we don't want to do packing.
            if not self.packing:
                break
    
        input_ids = torch.LongTensor(curr_input_ids)
        label_ids = torch.LongTensor(curr_label_ids)

        # Sanity check
        if 'eval' not in self.mode:
            assert len(curr_input_ids) == len(curr_label_ids)

        input_ids = input_ids[:self.max_tokens]
        label_ids = label_ids[:self.max_tokens]

        if len(curr_input_ids) < self.max_tokens and 'eval' not in self.mode:
            # Pad
            num_to_pad = self.max_tokens - len(curr_input_ids)
            input_ids = F.pad(input_ids, [0, num_to_pad], mode='constant', value=self.tokenizer.pad_token_id)

        if len(curr_label_ids) < self.max_tokens and 'eval' not in self.mode:
            num_to_pad = self.max_tokens - len(curr_label_ids)
            label_ids = F.pad(label_ids, [0, num_to_pad], mode='constant', value=-100)
        
        # Sanity check
        if 'eval' not in self.mode:
            assert input_ids.shape[0] == label_ids.shape[0] == self.max_tokens, f"{input_ids.shape[0]}, {label_ids.shape[0]}, {self.max_tokens}"
        
        if self.include_fnames:
            return {
                "input_ids" : input_ids,
                "labels" : label_ids,
                "fnames" : curr_fnames
            }
        else:
            # This is the format required by our GPT2Trainer class
            return {
                "input_ids" : input_ids,
                "labels" : label_ids
            }

    def get_random_sample(self):
        """
        Get a full on random sample (used for training)
        """
        random_sample = None
        while random_sample is None:
            if self.randomize:
                q, a, fname = random.choice(self.samples)
            else:
                q, a, fname = self.samples[self.i]
                self.i = (self.i + 1) % len(self.samples)
            
            random_sample = self.clean_sample((q, a))

            if not self.randomize:
                break
        
        return random_sample, fname

    def clean_filter_sample_gpt(self, sample):
        raise NotImplementedError()

    def clean_filter_sample_gpt_eval(self, sample):
        raise NotImplementedError()

    def clean_filter_sample_t5(self, sample):
        raise NotImplementedError()

    def clean_filter_sample_t5_eval(self, sample):
        raise NotImplementedError()

