#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/trainer.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Wednesday, September 29th 2021, 3:41:32 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Mon Jul 18 2022
# Modified By: Ruochi Zhang
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
import numpy as np
from pathlib import Path
import random
import mlflow
from tqdm import tqdm
from .loss import compute_metrics


class Trainer(object):
    def __init__(self, net, criterion, dataloaders, optimizer, scheduler,
                 device, cfg):

        self.net = net
        self.device = device
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = cfg.train.num_epoch
        self.batch_size = cfg.train.batch_size
        self.cfg = cfg

        self.global_train_step = 0
        self.global_eval_step = 0

    def evaluate(self):
        losses = []
        accs = []
        f1s = []
        self.net.eval()
        with torch.no_grad():
            for _, data in enumerate(self.dataloaders["valid"]):
                batch = tuple(t.to(self.device) for t in data)

                batch_ids, true_labels = batch

                # forward
                pred_logits = self.net(batch_ids)['logits']
                loss = self.criterion(pred_logits, true_labels).item()
                metrics = compute_metrics(pred_logits, true_labels)

                losses.append(loss)
                accs.append(metrics["acc"])
                f1s.append(metrics["f1"])

        valid_loss = np.mean(losses)
        valid_acc = np.mean(accs)
        valid_f1 = np.mean(f1s)

        return {
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "valid_f1": valid_f1
        }

    def run(self):

        for epoch in range(self.num_epoch):

            self.net.train()

            for _, data in enumerate(self.dataloaders["train"]):

                batch = tuple(t.to(self.device) for t in data)
                batch_ids, true_labels = batch

                # forward
                pred_logits = self.net(batch_ids)['logits']
                train_loss = self.criterion(pred_logits, true_labels)

                # backward

                if self.cfg.train.gradient_accumulation_steps > 1:
                    train_loss = train_loss / self.cfg.train.gradient_accumulation_steps

                train_loss.backward()

                if (self.global_train_step + 1) % self.cfg.train.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.net.parameters(), self.cfg.train.max_grad_norm)

                    self.optimizer.step()
                    self.net.zero_grad()
                    self.scheduler.step()
                    Logger.info(
                        "lr: {}".format(self.scheduler.optimizer.state_dict()
                                        ['param_groups'][0]['lr']))

                if self.cfg.logger.log:
                    mlflow.log_metric("train/loss",
                                    train_loss.item(),
                                    step=self.global_train_step)

                Logger.info("train | step: {} | loss: {}".format(
                    self.global_train_step, train_loss.item()))


                if (self.global_train_step + 1) % self.cfg.train.eval_steps == 0:
                    eval_metrics = self.evaluate()

                    Logger.info(
                        "valid | step: {} | loss: {} | acc: {} | f1: {}".
                        format(self.global_eval_step,
                               eval_metrics["valid_loss"], eval_metrics["valid_acc"],
                               eval_metrics["valid_f1"]))

                    for metric_name, metric_v in eval_metrics.items():
                        mlflow.log_metric("eval/{}".format(metric_name),
                                          metric_v,
                                          step=self.global_eval_step)

                    self.global_eval_step += 1
                    self.net.train()

                self.global_train_step += 1

            epoch += 1