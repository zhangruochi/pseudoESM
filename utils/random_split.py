#!/usr/bin/env python
# -*- coding:utf-8 -*-
###
# Filename: /data/zhouqiong/github_space/pseudoESM/unit_test/random_split.py
# Path: /data/zhouqiong/github_space/pseudoESM/unit_test
# Created Date: Wednesday, July 27th 2022, 5:34:38 pm
# Author: Qiong Zhou
# 
# Copyright (c) 2022 Silexon Inc.
###

import re
import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
import multiprocessing
from functools import partial
import time


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(
            f, keep_gaps=keep_gaps, keep_insertions=keep_insertions, to_upper=to_upper
        ):
            yield result


def read_alignment_lines(
    lines,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    seq = desc = None

    def parse(s):
        if not keep_gaps:
            s = re.sub("-", "", s)
        if not keep_insertions:
            s = re.sub("[a-z]", "", s)
        return s.upper() if to_upper else s

    for line in lines:
        # Line may be empty if seq % file_line_width == 0
        if len(line) > 0 and line[0] == ">":
            if seq is not None:
                yield desc, parse(seq)
            desc = line.strip()
            seq = ""
        else:
            assert isinstance(seq, str)
            seq += line.strip()
    assert isinstance(seq, str) and isinstance(desc, str)
    yield desc, parse(seq)

def get_generator_len(file_path):
    data_generator = read_fasta(file_path)
    print('***genarator length is {}'.format(len(list(data_generator))))

def multi_process(sub_path):
    data_generator = read_fasta(os.path.join(path,sub_path))
    data_generator = list(data_generator)
    data_len = len(data_generator)
    # eval_index = np.linspace(np.random.randint(10,50),data_len,int(data_len*0.01),dtype=np.int32)
    # test_index = eval_index-np.random.randint(1,9)
    # print(test_index,eval_index)
    eval_index = np.random.choice(data_len,12000,replace=False)
    test_index = np.random.choice(data_len,12000,replace=False)
    if_same = np.where(eval_index == test_index)
    eval_index = np.delete(eval_index,if_same)
    test_index = np.delete(test_index,if_same)
    
    for index,val in enumerate(data_generator):
        if index in eval_index:
            with open(eval_data_file,'a') as f:
                f.write(val[0]+'\n')
                f.write(val[1]+'\n')
        elif index in test_index:
            with open(test_data_file,'a') as f:
                f.write(val[0]+'\n')
                f.write(val[1]+'\n')
        else:
            with open(train_data_file,'a') as f:
                f.write(val[0]+'\n')
                f.write(val[1]+'\n')
    print('{} finish'.format(sub_path))

if __name__ == '__main__':
    path = '/data/zhouqiong/github_space/pseudoESM/datatt/ssssss'
    train_data_file = '/data/zhouqiong/github_space/pseudoESM/datatt/train/train.fasta'
    eval_data_file = '/data/zhouqiong/github_space/pseudoESM/datatt/eval/eval.fasta'
    test_data_file = '/data/zhouqiong/github_space/pseudoESM/datatt/test/test.fasta'
    if os.path.exists(train_data_file):
        os.remove(train_data_file)
    if os.path.exists(eval_data_file):
        os.remove(eval_data_file)
    if os.path.exists(test_data_file):
        os.remove(test_data_file)

    sub_paths = os.listdir(path)
    # max_workers = 40
    # with multiprocessing.Pool(max_workers) as p:
    #     sdf_list = p.map(partial(multi_process), sub_paths)
    start_time = time.time()
    for sub_path in sub_paths:
        data_generator = read_fasta(os.path.join(path,sub_path))
        data_generator = list(data_generator)
        data_len = len(data_generator)

        eval_index = np.random.choice(data_len,12000,replace=False)
        test_index = np.random.choice(data_len,12000,replace=False)
        
        if_same = np.where(eval_index == test_index)
        eval_index = np.delete(eval_index,if_same)
        test_index = np.delete(test_index,if_same)

        for index,val in enumerate(data_generator):
            if index in eval_index:
                with open(eval_data_file,'a') as f:
                    f.write(val[0]+'\n')
                    f.write(val[1]+'\n')
            elif index in test_index:
                with open(test_data_file,'a') as f:
                    f.write(val[0]+'\n')
                    f.write(val[1]+'\n')
            else:
                with open(train_data_file,'a') as f:
                    f.write(val[0]+'\n')
                    f.write(val[1]+'\n')
            
        print('{} finish'.format(sub_path))
    end = time.time()
    print('*****{}'.format(end-start_time))

    