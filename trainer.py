#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/trainer.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Wednesday, September 29th 2021, 3:41:32 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jul 29 2022
# Modified By: Qiong Zhou
# -----
# Copyright (c) 2021 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2021 Silexon Ltd
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

from .std_logger import Logger
import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import numpy as np
from pathlib import Path
import random
import mlflow
from tqdm import tqdm
from .loss import compute_metrics
from .utils.utils import is_parallel
import shutil
import os
import gc


class Trainer(object):
    def __init__(self, net, criterion, dataloaders, optimizer, scheduler,
                 device, global_rank, world_size, cfg):

        self.net = net
        self.device = device
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = cfg.train.num_epoch
        self.batch_size = cfg.train.batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.cfg = cfg

        self.global_train_step = 0
        self.global_valid_step = 0

        ## save checkpoint
        self.best_f1 = 0
        self.best_model_path = Path(".")

        self.root_level_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))

        ## amp
        self.scaler = None



    def evaluate(self):
        losses = []
        accs = []
        f1s = []

        total_eval_step = self.cfg.data.total_valid_num // (
            self.cfg.train.batch_size)

        if self.world_size != 0:
            total_eval_step = total_eval_step // self.world_size

        self.net.eval()
        with torch.no_grad():
            loss = acc = f1 = 0

            for step, data in tqdm(
                    enumerate(self.dataloaders["valid"]),
                    total=total_eval_step,
                    desc="evaluating | loss: {}, acc: {} | f1: {}".format(
                        loss, acc, f1)):

                batch = tuple(t.to(self.device) for t in data)
                batch_ids, true_labels = batch
                # forward
                pred_logits = self.net(batch_ids,
                                       need_head_weights=False,
                                       return_contacts=False)['logits']
                loss = self.criterion(pred_logits, true_labels).item()
                metrics = compute_metrics(pred_logits, true_labels)
                acc = metrics["acc"]
                f1 = metrics["f1"]
                losses.append(loss)
                accs.append(acc)
                f1s.append(f1)

                if self.cfg.other.debug and step >= self.cfg.other.debug_step:
                    break

        valid_loss = np.mean(losses)
        valid_acc = np.mean(accs)
        valid_f1 = np.mean(f1)

        return {
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_f1": valid_f1
        }

    def run(self):

        if self.cfg.train.amp:
            self.scaler = GradScaler(init_scale=2**16,
                                     growth_factor=2,
                                     backoff_factor=0.5,
                                     growth_interval=2000,
                                     enabled=True)

        for epoch in range(self.num_epoch):

            self.net.train()

            for _, data in enumerate(self.dataloaders["train"]):

                batch = tuple(t.to(self.device) for t in data)
                batch_ids, true_labels = batch

                # forward
                if self.cfg.train.amp:
                    with autocast():
                        pred_logits = self.net(batch_ids, need_head_weights=False, return_contacts=False)['logits']
                        train_loss = self.criterion(pred_logits, true_labels)
                else:
                    pred_logits = self.net(batch_ids, need_head_weights=False, return_contacts=False)['logits']
                    train_loss = self.criterion(pred_logits, true_labels)

                # backward
                if self.cfg.train.gradient_accumulation_steps > 1:
                    train_loss /= self.cfg.train.gradient_accumulation_steps


                if (self.global_train_step +
                        1) % self.cfg.train.gradient_accumulation_steps == 0:

                    self.optimizer.zero_grad()

                    if self.cfg.train.amp:
                        self.scaler.scale(train_loss).backward()
                    else:
                        train_loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.train.max_grad_norm)

                    if self.cfg.train.amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()


                ## logging

                if (self.global_train_step + 1
                ) % self.cfg.logger.log_per_steps == 0:

                    cur_lr = self.scheduler.optimizer.state_dict()['param_groups'][0]['lr']

                    if self.global_rank == 0:
                        mlflow.log_metric("train/loss",
                                        train_loss.item(),
                                        step=self.global_train_step)
                        mlflow.log_metric("lr",
                                        cur_lr,
                                        step=self.global_train_step)
                        Logger.info(
                                "lr: {:.8f}".format(cur_lr))

                        Logger.info("train | epoch: {:d}  step: {:d} | loss: {:.4f}".format(epoch,
                            self.global_train_step, train_loss.item()))


                ### evaluating

                if (self.global_train_step +
                        1) % self.cfg.train.eval_per_steps == 0:

                    valid_metrics = self.evaluate()

                    if self.global_rank == 0:

                        Logger.info(
                            "valid | epoch: {:d} | step: {:d} | loss: {:.4f} | acc: {:.4f} | f1: {:.4f}"
                            .format(epoch,
                                    self.global_valid_step,
                                    valid_metrics["valid_loss"],
                                    valid_metrics["valid_acc"],
                                    valid_metrics["valid_f1"]))

                        if self.cfg.logger.log:
                            for metric_name, metric_v in valid_metrics.items():
                                mlflow.log_metric("valid/{}".format(metric_name),
                                                metric_v,
                                                step=self.global_valid_step)

                    self.global_valid_step += 1
                    self.net.train()

                    if valid_metrics["valid_f1"] >= self.best_f1:
                        self.best_f1 = valid_metrics["valid_f1"]

                        self.best_model_path = Path(
                            "model_step_{}_f1_{}".format(
                                self.global_valid_step,
                                round(valid_metrics["valid_f1"], 3)))
                        if self.global_rank == 0:
                            if self.best_model_path.exists():
                                shutil.rmtree(self.best_model_path)

                            if self.cfg.logger.log:
                                mlflow.pytorch.save_model(
                                    (self.net.module
                                    if is_parallel(self.net) else self.net),
                                    self.best_model_path,
                                    code_paths=[
                                        os.path.join(self.root_level_dir, "esm")
                                    ])

                self.global_train_step += 1

                if self.cfg.other.debug and self.global_train_step >= self.cfg.other.debug_step:
                    break
