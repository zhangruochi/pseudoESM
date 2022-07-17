#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/loss.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Sunday, July 17th 2022, 11:48:02 pm
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
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


class LossFunc(object):
    def __init__(self):
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(
            ignore_index=-100, reduction='mean', label_smoothing=0.0)

    def __call__(self, pred, true):

        pred = pred.permute((0,2,1))
        loss = self.cross_entropy_loss(pred, true)

        return loss


def compute_metrics(pred, true):
    # Get the predictions

    pred = torch.argmax(pred, dim=-1)
    masked_tokens = true != -100

    pred = pred[masked_tokens].cpu().numpy()
    true = true[masked_tokens].cpu().numpy()

    acc = accuracy_score(pred, true)
    f1 = f1_score(pred, true, average='weighted')

    return {"acc": acc, "f1": f1}
