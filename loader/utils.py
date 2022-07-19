#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/loader/utils.py
# Project: /data/zhangruochi/projects/pseudoESM/loader
# Created Date: Saturday, July 16th 2022, 11:33:05 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jul 19 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 Silexon Ltd
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import os
import torch
from torch.utils.data import DataLoader
from .dataset import FastaDataset
from torch.utils.data import ChainDataset
from pathlib import Path
import numpy as np



class DataCollector(object):
    def __init__(self, tokenizer,
                 mlm: bool = True,
                 mlm_probability: float = 0.15):

        self.tokenizer = tokenizer
        self.mlm = mlm
        self.mlm_probability = mlm_probability

    def __call__(self, batch):

        batch_converter = self.tokenizer.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        labels = batch_tokens.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        special_tokens_mask = np.array([
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ])
        special_tokens_mask = torch.tensor(special_tokens_mask,
                                            dtype=torch.bool)


        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices
        batch_tokens[indices_replaced] = self.tokenizer.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(
            labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        batch_tokens[indices_random] = random_words[indices_random]

        return batch_tokens, labels


def make_loaders(collate_fn,
                 train_dir,
                 valid_dir,
                 test_dir,
                 batch_size=32,
                 num_workers=1):
    
    train_loader = None
    if train_dir and os.path.exists(train_dir):

        train_loader = DataLoader(ChainDataset([FastaDataset(file_path) for file_path in Path(train_dir).glob("*.fasta")]) ,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    valid_loader = None
    if valid_dir and os.path.exists(valid_dir):
        valid_loader = DataLoader(ChainDataset([
            FastaDataset(file_path)
            for file_path in Path(valid_dir).glob("*.fasta")]),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_loader = DataLoader(ChainDataset(
            FastaDataset(file_path)
            for file_path in Path(test_dir).glob("*.fasta")),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)

    dataset_loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    return dataset_loader
