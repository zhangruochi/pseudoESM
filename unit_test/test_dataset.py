import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(orig_cwd))


from pathlib import Path
from pseudoESM.loader.dataset import FastaDataset
from transformers import BertTokenizer


def test_dataset():

    train_dataset = FastaDataset("../data/train/1.fasta"
    )

    for i, _ in enumerate(train_dataset):
        print(_)
        if i > 10:
            break

if __name__ == "__main__":
    test_dataset()
