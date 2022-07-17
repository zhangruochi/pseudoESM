#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/trainer.py
# Project: /data/zhangruochi/projects/autopatent3/iupac_ner
# Created Date: Wednesday, September 29th 2021, 3:41:32 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sun Apr 10 2022
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
import logging

from zmq import device
from logger.std import logger as std_logger

import torch
import numpy as np
from pathlib import Path
import random


class Trainer(object):
    def __init__(self, net, criterion, dataloaders, optimizer, scheduler,
                 device, cfg, callbacks):

        self.net = net
        self.device = device
        self.criterion = criterion
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epoch = cfg.train.num_epoch
        self.batch_size = cfg.train.batch_size
        self.random_seed = cfg.train.random_seed

        # for logger
        self.train_evaluation_batch = cfg.train.train_evaluation_batch
        self.valid_evaluation_batch = cfg.train.valid_evaluation_batch

        self.cfg = cfg

        self.train_loss = None
        self.train_batch_idx = 0

        self.evaluation_train_batch_idx = 0
        self.evaluation_train_batch_loss = None
        self.evaluetion_train_epoch_loss = None

        self.valid_batch_idx = 0
        self.valid_batch_loss = None
        self.valid_epoch_loss = None

        self.epoch = 0

        self.total_train_batch = len(self.dataloaders["train"])
        self.total_evaluation_valid_batch = self.train_evaluation_batch
        self.total_evaluation_valid_batch = self.valid_evaluation_batch
        self.callbacks = callbacks

        # fix random seed
        self.random_fix()

    def random_fix(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def train_epoch(self):
        # set training mode
        self.net.train()

        for _, data in enumerate(self.dataloaders["train"]):

            batch = tuple(t.to(self.device) for t in data)
            input_ids, input_lables = batch

            # forward
            pred_logits = self.net(input_ids)['logits']
            self.train_loss = self.criterion(pred_logits.transpose(1, 2),input_lables )

            # backward
            self.optimizer.zero_grad()
            self.train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),
                                           self.cfg.train.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            logging.info("lr: {}".format(self.scheduler.optimizer.state_dict()['param_groups'][0]['lr']))
            
            self.train_batch_idx += 1

            # logging
            if self.logger.log:
                self.logger.on_train_batch_end()

            ## callbacks
            for callback in self.callbacks:
                callback.on_train_batch_end(self)

    def evaluate_train_epoch(self):
        losses = []
        self.evaluation_train_batch_idx = 0
        self.net.eval()
        
        with torch.no_grad():
            for _, data in enumerate(self.dataloaders["train"]):

                batch = tuple(t.to(self.device) for t in data)

                input_ids, input_lables = batch
                # forward
                pred_logits = self.net(input_ids)['logits']
                self.evaluation_train_batch_loss = self.criterion(pred_logits.transpose(1, 2),input_lables).cpu().item()

                losses.append(self.evaluation_train_batch_loss)
                self.evaluation_train_batch_idx += 1

                if self.evaluation_train_batch_idx == self.train_evaluation_batch:
                    break

                # callbacks
                for callback in self.callbacks:
                    callback.on_evaluate_train_batch_end(self)

        self.evaluetion_train_epoch_loss = np.mean(losses)


        if self.logger.log:
            self.logger.log_metric("evaluation_train_epoch_loss",
                                   self.evaluetion_train_epoch_loss,
                                   self.epoch)
            self.logger.log_evidence("train")

    def evaluate_valid_epoch(self):
        losses = []
        self.valid_batch_idx = 0

        self.net.eval()
        with torch.no_grad():
            for _, data in enumerate(self.dataloaders["valid"]):
                batch = tuple(t.to(self.device) for t in data)
                input_ids, input_lables = batch
                # forward
                pred_logits = self.net(input_ids)['logits']
                print('+'*30)
                print(pred_logits.shape,input_lables.shape)
                print(np.unique(input_lables.cpu().numpy()))
                print('+'*30)
                self.valid_batch_loss = self.criterion(pred_logits.transpose(1, 2),input_lables).cpu().item()
                
                losses.append(self.valid_batch_loss)
                self.valid_batch_idx += 1

                if self.valid_batch_idx == self.valid_evaluation_batch:
                    break

                # callbacks
                for callback in self.callbacks:
                    callback.on_evaluate_valid_batch_end(self)

        self.valid_epoch_loss = np.mean(losses)
        

        if self.logger.log:
            self.logger.log_metric("valid_epoch_loss", self.valid_epoch_loss,
                                   self.epoch)
            self.logger.log_evidence("valid")

    def train_model(self, logger):

        for callback in self.callbacks:
            callback.on_init_end(self)

        self.logger = logger

        for self.epoch in range(self.num_epoch):

            self.epoch_dir = Path("epoch_{}".format(self.epoch))

            if not self.epoch_dir.exists():
                self.epoch_dir.mkdir(parents=True, exist_ok=True)

            self.train_epoch()
            self.evaluate_train_epoch()
            self.evaluate_valid_epoch()

            for callback in self.callbacks:
                callback.on_epoch_end(self)

            self.epoch += 1

        if self.logger.log:
            self.logger.on_end()