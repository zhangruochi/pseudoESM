#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/loader/utils.py
# Project: /data/zhangruochi/projects/pseudoESM/loader
# Created Date: Saturday, July 16th 2022, 11:33:05 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Jul 17 2022
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
import json
import torch
from torch.utils.data import DataLoader
from .dataset import FastaDataset
from torch.utils.data import ChainDataset
from pathlib import Path


def make_loaders(tokenizer,
                 collate_fn,
                 train_dir='',
                 valid_dir='',
                 test_dir='',
                 batch_size=32,
                 num_workers=4):
    train_loader = None
    if train_dir and os.path.exists(train_dir):

        train_loader = DataLoader(ChainDataset(FastaDataset(tokenizer, file_path) for file_path in Path(train_dir).glob("*.fasta")) ,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)

    valid_loader = None
    if valid_dir and os.path.exists(valid_dir):
        valid_loader = DataLoader(ChainDataset(
            FastaDataset(tokenizer, file_path)
            for file_path in Path(valid_dir).glob("*.fasta")),
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    test_loader = None
    if test_dir and os.path.exists(test_dir):
        test_loader = DataLoader(ChainDataset(
            FastaDataset(tokenizer, file_path)
            for file_path in Path(test_dir).glob("*.fasta")),
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader
