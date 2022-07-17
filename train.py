#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/train.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Saturday, July 16th 2022, 7:14:39 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Mon Jul 18 2022
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
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hydra
import torch
import esm
import random
import numpy as np
from torch import nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from omegaconf import DictConfig
from pseudoESM.loader.utils import make_loaders, DataCollector
from pseudoESM.esm.data import Alphabet
from pseudoESM.esm.model import ProteinBertModel
from pseudoESM.std_logger import Logger
from pseudoESM.utils.utils import get_device
from pseudoESM.trainer import Trainer
from pseudoESM.loss import LossFunc




@hydra.main(config_name="train_conf.yaml")
def main(cfg: DictConfig):

    # fix random seed
    random.seed(cfg.train.random_seed)
    torch.manual_seed(cfg.train.random_seed)
    np.random.seed(cfg.train.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.random_seed)

    orig_cwd = hydra.utils.get_original_cwd()
    device = get_device(cfg)

    Logger.info("using master device: {}".format(device))


    tokenizer = Alphabet.from_architecture("protein_bert_base")
    collate_fn = DataCollector(tokenizer)

    # load dataset
    dataloaders = make_loaders(
        collate_fn,
        train_dir=os.path.join(orig_cwd, cfg.data.train_dir),
        valid_dir=os.path.join(orig_cwd, cfg.data.valid_dir),
        test_dir=os.path.join(orig_cwd, cfg.data.test_dir),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers)

    model = ProteinBertModel(cfg.model.protein_bert_base, tokenizer)
    model.to(device)
    Logger.info("model arch:{}".format(model))

    warmup_steps = cfg.train.warmup_steps

    Logger.info("total warmup steps: {}".format(warmup_steps))

    optimizer = AdamW(model.parameters(),
                      lr=cfg.train.learning_rate,
                      eps=cfg.train.adam_epsilon)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=warmup_steps * 100,
        last_epoch=-1)


    criterion = LossFunc()

    trainer = Trainer(model,
                         criterion,
                         dataloaders,
                         optimizer,
                         scheduler,
                         device,
                         cfg)

    Logger.info("start training......")
    trainer.run()



if __name__ == "__main__":
    main()