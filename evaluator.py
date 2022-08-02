#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/evaluator.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Tuesday, July 19th 2022, 2:47:37 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Aug 02 2022
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
import torch
from .loss import compute_metrics
import numpy as np
from tqdm import tqdm
import os

from torch.distributed import ReduceOp


class Evaluator():
    def __init__(self, model, test_loader, criterion, device, cfg):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader
        self.cfg = cfg

    def run(self):
        loss_list = []

        logits_list = []
        true_list = []
        self.model.eval()
        with torch.no_grad():
            loss = acc = f1 = 0
            for step, data in tqdm(
                    enumerate(self.test_loader),
                    desc="evaluating | loss: {}, acc: {} | f1: {}".format(
                        loss, acc, f1)):
                batch = tuple(t.to(self.device) for t in data)

                batch_ids, true_labels = batch
                # forward
                pred_logits = self.model(batch_ids)['logits']
                loss = self.criterion(pred_logits, true_labels).item()

                logits_list.append(pred_logits)
                true_list.append(true_labels)
                loss_list.append(loss)

                if step > 100:
                    break


        pred_logits = torch.concat(logits_list, dim=1)
        true_labels = torch.concat(true_list, dim=1)

        metrics = compute_metrics(pred_logits, true_labels)

        test_loss = torch.tensor(np.mean(loss_list))
        test_ece = torch.exp(test_loss)
        test_acc = torch.tensor(metrics["acc"])
        test_f1 = torch.tensor(metrics["f1"])


        # print("RANK: {}: loss {}".format(int(os.environ['RANK']), test_loss))
        # print("RANK: {}: ece {}".format(int(os.environ['RANK']), test_ece))
        # print("RANK: {}: acc {}".format(int(os.environ['RANK']), test_acc))
        # print("RANK: {}: f1 {}".format(int(os.environ['RANK']), test_f1))


    
        if self.cfg.mode.gpu:
            torch.distributed.barrier()
            torch.distributed.all_reduce(test_loss, op=ReduceOp.SUM)
            torch.distributed.all_reduce(test_ece, op=ReduceOp.SUM)
            torch.distributed.all_reduce(test_acc, op=ReduceOp.SUM)
            torch.distributed.all_reduce(test_f1, op=ReduceOp.SUM)
        
            test_loss /= torch.distributed.get_world_size()
            test_ece /= torch.distributed.get_world_size()
            test_acc /= torch.distributed.get_world_size()
            test_f1 /= torch.distributed.get_world_size()
            

        # print("RANK: {}: loss {}".format(int(os.environ['RANK']), test_loss))
        # print("RANK: {}: ece {}".format(int(os.environ['RANK']), test_ece))
        # print("RANK: {}: acc {}".format(int(os.environ['RANK']), test_acc))
        # print("RANK: {}: f1 {}".format(int(os.environ['RANK']), test_f1))

        return {
            "test_loss": test_loss.item(),
            "test_acc": test_acc.item(),
            "test_f1": test_f1.item(),
            "test_ece": test_ece.item(),
        }
