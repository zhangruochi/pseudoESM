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
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import torch
import esm
from omegaconf import DictConfig, OmegaConf
from pseudoESM.loader.dataset import FastaDataset
from pseudoESM.loader.utils import make_loaders
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments, BertTokenizer
# from esm.model import ProteinBertModel


import datasets
from datasets import load_dataset, load_metric
from transformers import BertForMaskedLM, BertConfig



@hydra.main(config_name="train_conf.yaml")
def main(cfg: DictConfig):

    orig_cwd = hydra.utils.get_original_cwd()

    tokenizer = BertTokenizer(vocab_file=os.path.join(orig_cwd, "./vocab.txt"),
                              do_lower_case=False,
                              do_basic_tokenize=True,
                              never_split=None,
                              unk_token='[UNK]',
                              pad_token='[PAD]',
                              cls_token='[CLS]',
                              mask_token='[MASK]')

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                    mlm=True,
                                                    mlm_probability=0.15)

    # load dataset
    train_loader, valid_loader, test_loader = make_loaders(
        tokenizer,
        data_collator,
        train_dir=os.path.join(orig_cwd, cfg.data.train_dir),
        valid_dir=os.path.join(orig_cwd, cfg.data.valid_dir),
        test_dir=os.path.join(orig_cwd, cfg.data.test_dir),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers)

    print(next(iter(train_loader))["input_ids"].shape)
    print(next(iter(valid_loader))["input_ids"].shape)
    print(next(iter(test_loader))["input_ids"].shape)


    # training_args = TrainingArguments(output_dir=cfg.log.output_dir,
    #                                   overwrite_output_dir=True,
    #                                   num_train_epochs=1,
    #                                   per_gpu_train_batch_size=4,
    #                                   save_steps=10_000,
    #                                   save_total_limit=2,
    #                                   weight_decay=0.1,
    #                                   warmup_steps=1000,
    #                                   lr_scheduler_type="cosine",
    #                                   learning_rate=5e-4,
    #                                   evaluation_strategy="steps")

    # metric = load_metric("accuracy")

    # def compute_metrics(eval_preds):
    #     preds, labels = eval_preds
    #     # preds have the same shape as the labels, after the argmax(-1) has been calculated
    #     # by preprocess_logits_for_metrics
    #     labels = labels.reshape(-1)
    #     preds = preds.reshape(-1)
    #     mask = labels != -100
    #     labels = labels[mask]
    #     preds = preds[mask]
    #     return metric.compute(predictions=preds, references=labels)


    # configuration = BertConfig(vocab_size = 32)
    # model = BertForMaskedLM(configuration)

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=compute_metrics
    # )


    # trainer.train()
    # trainer.save_model("./EsperBERTo")

if __name__ == "__main__":
    main()