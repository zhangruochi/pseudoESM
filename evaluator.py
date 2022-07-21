#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/evaluator.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Tuesday, July 19th 2022, 2:47:37 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed Jul 20 2022
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

class Evaluator():
    def __init__(self, model, test_loader, criterion, device, cfg):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.test_loader = test_loader
        self.cfg = cfg

    def run(self):
        losses = []
        accs = []
        f1s = []
        self.model.eval()
        with torch.no_grad():
            loss = acc = f1 = 0
            for step, data in tqdm(
                    enumerate(self.test_loader),
                    total=self.cfg.data.total_test_num //
                    self.cfg.train.batch_size,
                    desc="evaluating | loss: {}, acc: {} | f1: {}".format(
                        loss, acc, f1)):
                batch = tuple(t.to(self.device) for t in data)

                batch_ids, true_labels = batch

                # forward
                pred_logits = self.model(batch_ids)['logits']
                loss = self.criterion(pred_logits, true_labels).item()
                metrics = compute_metrics(pred_logits, true_labels)

                acc = metrics["acc"]
                f1 = metrics["f1"]

                losses.append(loss)
                accs.append(acc)
                f1s.append(f1)

                if self.cfg.other.debug and step >= self.cfg.other.debug_step:
                    break

        test_loss = np.mean(losses)
        test_acc = np.mean(accs)
        test_f1 = np.mean(f1s)

        return {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1
        }
