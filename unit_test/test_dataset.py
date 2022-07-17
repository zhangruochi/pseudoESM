import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(orig_cwd))


from pathlib import Path
from pseudoESM.loader.dataset import FastaDataset
from transformers import BertTokenizer


def test_tokenizer():
    tokenizer = BertTokenizer(vocab_file = "../vocab.txt",
                            do_lower_case = False,
                            do_basic_tokenize = True,
                            never_split = None,
                            unk_token = '[UNK]',
                            pad_token = '[PAD]',
                            cls_token = '[CLS]',
                            mask_token = '[MASK]')

    train_dataset = FastaDataset(
        tokenizer, "../data/train/1.fasta"
    )

    for i, _ in enumerate(train_dataset):
        for k, v in _.items():

            print(k, v.shape)
            

        if i > 10:
            break

if __name__ == "__main__":
    test_tokenizer()
