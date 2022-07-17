import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(orig_cwd))

from pathlib import Path
from pseudoESM.loader.dataset import FastaDataset

def test_tokenizer():

    train_dataset = FastaDataset(
        "../data/train/1.fasta"
    )

    for _ in train_dataset:
        print(_)
        break


if __name__ == "__main__":
    test_tokenizer()
