#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/train.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Saturday, July 16th 2022, 7:14:39 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Thu Jul 21 2022
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
import random
import numpy as np
from torch import nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from pseudoESM.loader.utils import make_loaders, DataCollector
from pseudoESM.esm.data import Alphabet
from pseudoESM.esm.model import ProteinBertModel
from pseudoESM.std_logger import Logger
from pseudoESM.utils.utils import get_device, load_weights
from pseudoESM.trainer import Trainer
from pseudoESM.loss import LossFunc
from pseudoESM.evaluator import Evaluator
import mlflow
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from omegaconf import OmegaConf

#
parser = argparse.ArgumentParser(description='Train a recognizer')
parser.add_argument('--local_rank', type=int, default=0)
args = parser.parse_args()


def main():
    orig_cwd = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(orig_cwd,"train_conf.yaml")
    cfg = OmegaConf.load(cfg_path)
    # fix random seed
    random.seed(cfg.train.random_seed)
    torch.manual_seed(cfg.train.random_seed)
    np.random.seed(cfg.train.random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.random_seed)

    #set up distributed device
    local_rank = int(args.local_rank)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    tokenizer = Alphabet.from_architecture("protein_bert_base")
    collate_fn = DataCollector(tokenizer,
                               mlm=cfg.model.task.mlm,
                               mlm_probability=cfg.model.task.mlm_probability,
                               max_length=cfg.model.task.max_length)

    #-------------------- load dataset --------------------
    dataloaders = make_loaders(collate_fn,
                               train_dir=os.path.join(orig_cwd,
                                                      cfg.data.train_dir),
                               valid_dir=os.path.join(orig_cwd,
                                                      cfg.data.valid_dir),
                               test_dir=os.path.join(orig_cwd,
                                                     cfg.data.test_dir),
                               batch_size=cfg.train.batch_size,
                               num_workers=cfg.train.num_workers)

    # -------------------- load model --------------------
    model = ProteinBertModel(cfg.model.protein_bert_base, tokenizer)

    # DistributedDataParallel
    model.to(device)
    Logger.info("model arch:{}".format(model))
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

    num_training_steps = cfg.train.num_epoch * cfg.data.total_train_num // cfg.train.batch_size // cfg.train.gradient_accumulation_steps
    warmup_steps = int(cfg.train.warmup_steps_ratio * num_training_steps)

    Logger.info("total warmup steps: {}".format(warmup_steps))

    optimizer = AdamW(model.parameters(),
                      lr=cfg.train.learning_rate,
                      eps=cfg.train.adam_epsilon)

    criterion = LossFunc()

    if cfg.other.debug:
        cfg.train.num_epoch = 10
        cfg.train.gradient_accumulation_steps = 1
        cfg.logger.log_per_steps = 1
        cfg.train.eval_per_steps = 5
        cfg.other.debug_step = 5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    1,
                                                    gamma=0.99,
                                                    last_epoch=-1,
                                                    verbose=False)
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            last_epoch=-1)

    trainer = Trainer(model, criterion, dataloaders, optimizer, scheduler,
                      device, cfg)

    if cfg.logger.log:
        # log hyper-parameters
        for p, v in cfg.data.items():
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model.protein_bert_base.items():
            mlflow.log_param(p, v)

    Logger.info("start training......")
    trainer.run()

    Logger.info("finished training......")

    Logger.info("loading best weights......")
    load_weights(model, trainer.best_model_path, device)

    Logger.info("start evaluating......")
    evaluetor = Evaluator(model, dataloaders["test"], criterion, device, cfg)

    test_metrics = evaluetor.run()

    Logger.info("test | loss: {:.4f} | acc: {:.4f} | f1: {:.4f}".format(
        test_metrics["test_loss"], test_metrics["test_acc"],
        test_metrics["test_f1"]))

    for metric_name, metric_v in test_metrics.items():
        mlflow.log_metric("test/{}".format(metric_name), metric_v, step=1)

    Logger.info("finished evaluating......")


if __name__ == "__main__":
    main()