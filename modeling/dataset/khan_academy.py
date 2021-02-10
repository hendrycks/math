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

from dataset.util import last_boxed_only, _clean_numbers, last_boxed_only_string

from multiprocessing import Manager

from torch.multiprocessing import Pool
from dataset.base_math_dataset import BaseMathDataset

class KhanAcademyMathDataset(BaseMathDataset):
    """Configurable Math Dataset.
    """

    def __len__(self):
        return int(len(self.samples) * self.len_multiplier)

    def initialize(self):
        """
        Set up self.samples by loading from the dataroot
        """

        all_filenames = glob.glob(self.dataroot)
        print(f"{self.__class__.__name__}: Loading samples from {len(all_filenames)} files.")
        samples_raw = []
        for fname in tqdm(all_filenames):
            # Each fname is a json file with the following structure:
            # {
            #     "problem": "How many positive three-digit ...?",
            #     "level": "Level 24",
            #     "type": "Counting & Probability",
            #     "solution": "<Blah>",
            #     "discuss": ""
            # }
            with open(fname, 'r') as fp:
                problem_data = json.load(fp)
            
            q = problem_data.get('question', None) or problem_data['problem']

            if 'hints' in problem_data:
                a = problem_data['hints']
            elif 'solution' in problem_data:
                print(f"Falling back to 'solution': {fname}")
                a = problem_data['solution']
            else:
                print("Malformed file")
                print(fname)
                print(problem_data)


            assert q is not None and a is not None

            curr_sample_raw = (q, a, fname)
            samples_raw.append(curr_sample_raw)
        
        manager = Manager()
        samples_raw = manager.list(samples_raw)
        self.samples = samples_raw
        del samples_raw

        print(f"{self.__class__.__name__}: Loaded {len(self.samples)} samples.")


    def tokenize_latex_mask_full_answer(self, answer_full):
        """
        Tokenize the full answer string.
        If needed, mask the tokenized version to only include latex tokens.
        """

        answer_full_ids       = self.tokenizer.encode(answer_full, verbose=False)

        tokenized = self.tokenizer.tokenize(answer_full)
        mask = [None for _ in tokenized]

        prev_char = None
        in_latex = False
        for i, token in enumerate(tokenized):
            for char in token:
                if char == '$' and prev_char != '\\':
                    # Found a dollar sign that represents latex start/end.
                    mask[i] = 1
                    in_latex = not in_latex
                else:
                    mask[i] = int(in_latex)
        
        assert len(mask) == len(answer_full_ids)
        for i in range(len(answer_full_ids)):
            if mask[i] == 0:
                answer_full_ids[i] = -100

        answer_full_ids.append(self.tokenizer.eos_token_id)
        answer_full_ids       = torch.LongTensor(answer_full_ids)
        return answer_full_ids


    def clean_filter_sample_gpt(self, sample):
        """
        Does the actual tokenization. Should be parallelized because it can be a bit slow.
        """

        if sample == None:
            return None

        question, answer = sample
        if self.clean_numbers:
            question = _clean_numbers(question)
            answer = list(map(_clean_numbers, answer))

        if self.mode_answer == 'mixed_hints':
            answer_full = "".join(answer)
            answer_final = answer[-1]

            question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question, verbose=False))
            
            if random.random() < 0.5:
                # Use full solution
                sep_ids_2             = torch.LongTensor(self.tokenizer.encode("\nFULL SOLUTION:\n", verbose=False))
                answer_full_ids       = self.tokenizer.encode(answer_full, verbose=False)
                answer_full_ids.append(self.tokenizer.eos_token_id)
                answer_full_ids       = torch.LongTensor(answer_full_ids)
                if self.latex_mask:
                    answer_full_ids_label = self.tokenize_latex_mask_full_answer(answer_full)
                else:
                    answer_full_ids_label = answer_full_ids.clone()

                input_ids = torch.cat([
                    question_ids, 
                    sep_ids_2,
                    answer_full_ids
                ], dim=0)

                label_ids = torch.cat([
                    torch.ones_like(question_ids) * -100, 
                    torch.ones_like(sep_ids_2) * -100, 
                    answer_full_ids_label
                ], dim=0)
            else:
                # Use only final answer
                sep_ids_1        = torch.LongTensor(self.tokenizer.encode("\nFINAL ANSWER:\n", verbose=False))
                answer_final_ids = self.tokenizer.encode(answer_final, verbose=False)
                answer_final_ids.append(self.tokenizer.eos_token_id)
                answer_final_ids = torch.LongTensor(answer_final_ids)

                input_ids = torch.cat([
                    question_ids, 
                    sep_ids_1, 
                    answer_final_ids,
                ], dim=0)

                label_ids = torch.cat([
                    torch.ones_like(question_ids) * -100, 
                    torch.ones_like(sep_ids_1) * -100, 
                    answer_final_ids.clone(),
                ], dim=0)
        else:
            raise NotImplementedError()
        
        # Stop early if this Q,A pair is too long
        if question_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"{self.__class__.__name__} Skipping due to input_ids being too big. question_ids.shape[0] = {question_ids.shape[0]}.")
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
            answer = list(map(_clean_numbers, answer))

        if self.mode_answer == 'mixed_hints':
            answer_full = "".join(answer)
            answer_final = answer[-1]
            
            if random.random() < 0.5:
                # Use full solution
                question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question + "\nFULL SOLUTION:\n", verbose=False))
                answer_ids       = torch.LongTensor(self.tokenizer.encode(answer_full, verbose=False))
            else:
                # Use only final answer
                question_ids     = torch.LongTensor(self.tokenizer.encode("\nQUESTION:\n" + question + "\nFINAL ANSWER:\n", verbose=False))
                answer_ids       = torch.LongTensor(self.tokenizer.encode(answer_final, verbose=False))
        else:
            raise NotImplementedError()
        
        input_ids = torch.cat([
            question_ids, 
        ], dim=0)

        label_ids = torch.cat([
            answer_ids
        ], dim=0)

       # Stop early if this Q,A pair is too long
        if question_ids.shape[0] > self.max_tokens:
            # Print reason for skipping
            # print(f"{self.__class__.__name__} Skipping due to input_ids being too big. question_ids.shape[0] = {question_ids.shape[0]}.")
            return None

        input_ids = input_ids.tolist()
        label_ids = label_ids.tolist()

        return {
            'input_ids_list' : input_ids,
            'label_ids_list' : label_ids
        }

