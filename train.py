#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/train.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Saturday, July 16th 2022, 7:14:39 pm
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
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


import hydra
import torch
import esm
from omegaconf import DictConfig
from pseudoESM.loader.utils import make_loaders, DataCollector
from pseudoESM.esm.data import Alphabet
from pseudoESM.esm.model import ProteinBertModel

from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertTokenizer


@hydra.main(config_name="train_conf.yaml")
def main(cfg: DictConfig):

    orig_cwd = hydra.utils.get_original_cwd()
    tokenizer = Alphabet.from_architecture("protein_bert_base")
    collate_fn = DataCollector(tokenizer)

    # load dataset
    train_loader, valid_loader, test_loader = make_loaders(
        collate_fn,
        train_dir=os.path.join(orig_cwd, cfg.data.train_dir),
        valid_dir=os.path.join(orig_cwd, cfg.data.valid_dir),
        test_dir=os.path.join(orig_cwd, cfg.data.test_dir),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers)

    class Args():
        def __init__(self):
            self.arch = 'protein_bert_base'
            self.embed_dim = 768
            self.layers = 6
            self.ffn_embed_dim = 3072
            self.attention_heads = 12
            self.final_bias = True

    args = Args()

    model = ProteinBertModel(args, tokenizer)

if __name__ == "__main__":
    main()