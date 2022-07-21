#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/pseudoESM/unit_test/test_alphabet.py
# Project: /data/zhangruochi/projects/pseudoESM/unit_test
# Created Date: Wednesday, July 20th 2022, 11:01:10 pm
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


def _test_esm1b(alphabet):
    import torch

    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", "MKTVRQG"),
        ("protein2 with mask", "KALTA<mask>ISQP"),
        ("protein3", "K A <mask> I S Q"),
    ]
    _, _, batch_tokens = batch_converter(data)
    expected_tokens = torch.tensor([
        [0, 20, 15, 11, 7, 10, 16, 6, 2, 1, 1, 1],
        [0, 15, 5, 4, 11, 5, 32, 12, 8, 16, 14, 2],
        [0, 15, 5, 32, 12, 8, 16, 2, 1, 1, 1, 1],
    ])
    assert torch.allclose(batch_tokens, expected_tokens)


def test_esm1b_alphabet():
    import esm

    _, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    _test_esm1b(alphabet)


def test_esm1v_alphabet():
    import esm

    _, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    _test_esm1b(alphabet)


def test_esm1_msa1b_alphabet():
    import torch
    import esm

    # Load ESM-1b model
    _, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    batch_converter = alphabet.get_batch_converter()

    data = [
        ("protein1", "MKTVRQG"),
        ("protein2", "KALTRAI"),
        ("protein3", "KAAISQQ"),
    ]
    _, _, batch_tokens = batch_converter(data)
    expected_tokens = torch.tensor([[
        [0, 20, 15, 11, 7, 10, 16, 6],
        [0, 15, 5, 4, 11, 10, 5, 12],
        [0, 15, 5, 5, 12, 8, 16, 16],
    ]])
    assert torch.allclose(batch_tokens, expected_tokens)