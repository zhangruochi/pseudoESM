import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from typing import Dict
import re
from pathlib import Path


def read_fasta(
    path,
    keep_gaps=True,
    keep_insertions=True,
    to_upper=False,
):
    with open(path, "r") as f:
        for result in read_alignment_lines(f,
                                           keep_gaps=keep_gaps,
                                           keep_insertions=keep_insertions,
                                           to_upper=to_upper):
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


class FastaDataset(IterableDataset):

    def __init__(self, fasta_file_path):

        super(FastaDataset).__init__()
        self.fasta_file_path = fasta_file_path
        self.data_generator = read_fasta(str(self.fasta_file_path))

    def __iter__(self):

        for line in self.data_generator:
            yield line
