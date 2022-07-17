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
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import hydra
import torch
import esm
import tqdm
from torch import nn as nn
from torch.optim import AdamW,Adam
from pytorch_transformers import WarmupLinearSchedule
from pseudoESM.utils.utils import get_device
from pseudoESM.mlflow_logger import MLflowLogger
from pseudoESM.utils.optim_schedule import ScheduledOptim
from omegaconf import DictConfig
from pseudoESM.loader.utils import make_loaders, DataCollector
from pseudoESM.esm.data import Alphabet
from pseudoESM.esm.model import ProteinBertModel
from callbacks import MyPrintingCallback
from trainer import Trainer

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

    device = get_device(cfg)

    model = ProteinBertModel(args, tokenizer)
    model.to(device=torch.device("cuda:{}".format(cfg.train.device_ids[0]
                                                ) if torch.cuda.is_available()
                               and len(cfg.train.device_ids) > 0 else "cpu"))    
    dataset_loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader
    }

    # for _, data in enumerate(dataset_loader["valid"]):
    #         batch = tuple(t.to(device) for t in data)
    #         input_ids, input_lables = batch
    #         print(input_ids.device,'*****')
    #         model=model.to(device)
    #         # forward
    #         pred_logits = model(input_ids)['logits']
    #         break
    # exit()

    no_decay = ['bias', 'LayerNorm.weight']
    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        cfg.train.weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    
    # optimizer = Adam(model.parameters(), 
    #                 lr=cfg.train.learning_rate, 
    #                 betas=(0.9, 0.999), 
    #                 weight_decay=cfg.train.weight_decay)
    # optim_schedule = ScheduledOptim(optimizer, 768, n_warmup_steps=10000)

    
    num_train_optimization_steps = max(int(
        len(dataset_loader["train"].dataset) / cfg.train.batch_size /
        cfg.train.gradient_accumulation_steps) * cfg.train.num_epoch, 1)

    warmup_steps = int(cfg.train.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=cfg.train.learning_rate,
                      eps=cfg.train.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=warmup_steps,
                                     t_total=num_train_optimization_steps)

    criterion = nn.NLLLoss(ignore_index=0)
    my_trainer = Trainer(model,
                         criterion,
                         dataset_loader,
                         optimizer,
                         scheduler,
                         device,
                         cfg,
                         callbacks=[MyPrintingCallback()])
    logger = MLflowLogger(cfg, my_trainer)
    logging.info("training {} start......".format("esm model"))
    my_trainer.train_model(logger)

if __name__ == "__main__":
    main()