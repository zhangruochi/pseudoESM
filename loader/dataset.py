import os
import json
import torch
from torch.utils.data import IterableDataset
from typing import Dict
import re
from pathlib import Path
import math


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

    def __init__(self, fasta_file_path, global_rank, world_size, train = True):
        super(FastaDataset).__init__()
        self.fasta_file_path = fasta_file_path
        self.global_rank = global_rank
        self.world_size = world_size
        self.start, self.end = self._get_file_info(fasta_file_path)

        self.data_generator = read_fasta(str(self.fasta_file_path))
        self.train = train

        # if not self.train:
        #     self.global_rank = 0
        #     self.world_size = 0

    def __len__(self):
        return self.end - self.start


    def _get_file_info(self, file_path):
        start = 0
        end = sum(1 for _ in open(file_path)) // 2

        return start, end

    def _sample_generator(self, start, end):
        for i, line in enumerate(self.data_generator):
            if i < start:
                continue
            if i >= end:
                return StopIteration()
            yield line

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single worker
            iter_start = self.start
            iter_end   = self.end
        elif self.global_rank == 0 and self.world_size == 0:
            workers = worker_info.num_workers
            per_worker = int(
                math.ceil((self.end - self.start) / float(workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        else:  # multiple workers
            workers = self.world_size * worker_info.num_workers

            per_worker = int(
                math.ceil((self.end - self.start) / float(workers)))
            per_rank = int(math.ceil((self.end - self.start) / float(self.world_size)))

            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker + self.global_rank * per_rank
            iter_end = min(iter_start + per_worker, self.end)

        # workers
        sample_iterator = self._sample_generator(iter_start, iter_end)

        return sample_iterator