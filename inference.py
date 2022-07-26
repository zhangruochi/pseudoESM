#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/inference.py
# Project: /data/zhangruochi/projects/pseudoESM
# Created Date: Tuesday, July 19th 2022, 10:17:31 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jul 26 2022
# Modified By: Qiong Zhou
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

import mlflow

import os
import sys
import torch.nn.functional as F
import torch
from omegaconf import OmegaConf
from pathlib import Path
from pseudoESM.esm.data import Alphabet
from pseudoESM.utils.utils import get_device


root_dir = os.path.dirname(os.path.abspath(__file__))

from pseudoESM.loader.utils import make_loaders,  DataCollector
from pseudoESM.evaluator import Evaluator
from pseudoESM.loss import LossFunc
import numpy as np


def load_model(cfg):
    model_path = os.path.join(root_dir, cfg.inference.model_path)
    print("loading model from : {}".format(model_path))
    model = mlflow.pytorch.load_model(model_path, map_location="cpu")
    model.eval()
    return model



if __name__ == "__main__":
    cfg = OmegaConf.load(os.path.join(root_dir, "configs/inference.yaml"))
    orig_cwd = os.path.dirname(__file__)

    device = torch.device("cuda:{}".format(cfg.inference.device_ids[0]
                                        ) if torch.cuda.is_available()
                        and len(cfg.inference.device_ids) > 0 else "cpu")
    
    model = load_model(cfg)
    tokenizer = Alphabet.from_architecture("protein_bert_base")
    collate_fn = DataCollector(tokenizer)
    model.to(device)
    dataloader = make_loaders(collate_fn,
                        test_dir=os.path.join(orig_cwd, cfg.inference.test_dir),
                        batch_size=cfg.inference.batch_size,
                        num_workers=cfg.inference.num_workers)["test"]

    evaluetor = Evaluator(model, dataloader, LossFunc(), device, cfg)
    test_metrics = evaluetor.run()

    print(test_metrics)

    # ECE: 15.462533462714758