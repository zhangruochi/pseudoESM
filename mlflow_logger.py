#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/logger/mlflow_logger.py
# Project: /data/zhangruochi/projects/autopatent3/iupac_ner
# Created Date: Tuesday, October 19th 2021, 3:52:40 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Apr 05 2022
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
import os
import shutil
import mlflow
import traceback
import numpy as np
import cv2
from .utils.utils import save_cm, is_parallel
from .utils.plotting import print_prediction
from pathlib import Path

from abc import ABC


class Logger(ABC):
    def on_train_batch_end(self):
        pass

    def on_evaluate_train_batch_end(self):
        pass

    def on_evaluate_valid_batch_end(self):
        pass

    def on_evaluate_train_epoch_end(self):
        pass

    def on_evaluate_valid_epoch_end(self):
        pass

    def on_train_batch_end(self):
        pass

    def on_end(self):
        pass


class MLflowLogger(Logger):
    def __init__(self, cfg, trainer):

        self.log = cfg.logger.log

        if cfg.logger.logger_type == "mlflow":
            self.log_params = mlflow.log_params
            self.log_param = mlflow.log_param
            self.log_metric = mlflow.log_metric
            self.log_image = mlflow.log_image
            self.log_text = mlflow.log_text
            self.log_to_python_func = mlflow.pyfunc.log_model
            self.save_to_python_func = mlflow.pyfunc.save_model
            self.save_pytorch_model = mlflow.pytorch.save_model
        self.trainer = trainer
        self.cfg = cfg

        self.best_model_path = Path("epoch_0") / "model"
        self.best_metric = 0

        self.root_level_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))

        if os.path.exists(self.cfg.logger.log_dir):
            shutil.rmtree(self.cfg.logger.log_dir)

    def on_train_batch_end(self):
        if self.trainer.train_batch_idx % self.cfg.logger.per_n_steps == 0:
            try:
                self.log_metric(
                    "lr",
                    float(self.trainer.scheduler.optimizer.state_dict()
                          ['param_groups'][0]['lr']),
                    self.trainer.train_batch_idx)
                self.log_metric("train_loss",
                                float(self.trainer.train_loss.item()),
                                self.trainer.train_batch_idx)
            except Exception as e:
                traceback.print_exc()

    def on_evaluate_train_epoch_end(self, res_res):
        try:
            for k, v in res_res.items():
                save_path = self.trainer.epoch_dir / "train_{}.png".format(k)
                if isinstance(v, np.ndarray):
                    save_cm(v, str(save_path))
                    self.log_image(
                        cv2.imread(str(save_path))[:, :, ::-1],
                        "epoch_{}/train_{}.png".format(self.trainer.epoch, k))
                elif isinstance(v, str):
                    save_path = self.trainer.epoch_dir / "train_{}.txt".format(
                        k)
                    self.log_text(v, save_path)
                elif isinstance(v, float):
                    self.log_metric("train_" + k, float(v), self.trainer.epoch)

        except Exception as e:
            traceback.print_exc()

    def on_evaluate_valid_epoch_end(self, res_res):
        try:
            for k, v in res_res.items():
                save_path = self.trainer.epoch_dir / "valid_{}.png".format(k)
                if isinstance(v, np.ndarray):
                    save_cm(v, str(save_path))
                    self.log_image(
                        cv2.imread(str(save_path))[:, :, ::-1],
                        "epoch_{}/valid_{}.png".format(self.trainer.epoch, k))
                elif isinstance(v, str):
                    save_path = self.trainer.epoch_dir / "valid_{}.txt".format(
                        k)
                    self.log_text(v, save_path)
                elif isinstance(v, float):
                    self.log_metric("valid_" + k, float(v), self.trainer.epoch)
        except Exception as e:
            traceback.print_exc()

    def on_end(self):
        if os.path.exists(self.cfg.logger.final_artifact_path):
            shutil.rmtree(self.cfg.logger.final_artifact_path)

        shutil.copytree(self.best_model_path.parent,
                        self.cfg.logger.final_artifact_path)

    def log_evidence(self, split):
        for i in range(self.cfg.logger.num_evidence):
            text = print_prediction(
                self.trainer.net, self.trainer.dataloaders[split].dataset,
                self.trainer.device)

            try:
                self.log_text(
                    text, self.trainer.epoch_dir /
                    "{}_pred_vs_true_{}_result.txt".format(split, i + 1))

            except Exception as e:
                traceback.print_exc()
                with open(
                        self.trainer.epoch_dir /
                        "{}_pred_vs_true_{}_result.txt".format(split, i + 1),
                        "w") as f:
                    f.write(text)

    def save_checkpoint(self, eval_res):
        if eval_res[self.cfg.logger.comparison_matric] >= self.best_metric:
            self.best_metric = eval_res[self.cfg.logger.comparison_matric]
            self.best_model_path = self.trainer.epoch_dir / "model"
            self.log_metric("best_epoch", float(self.trainer.epoch))
            self.log_metric(
                "best_{}".format(self.cfg.logger.comparison_matric),
                self.best_metric, self.trainer.epoch)

            if self.best_model_path.exists():
                shutil.rmtree(self.best_model_path)

            self.save_pytorch_model(
                (self.trainer.net.module
                 if is_parallel(self.trainer.net) else self.trainer.net),
                self.best_model_path,
                code_paths=[os.path.join(self.root_level_dir, "models")])
