import torch
import sys
import os

orig_cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, orig_cwd)
sys.path.insert(0, os.path.dirname(orig_cwd))

from omegaconf import DictConfig, OmegaConf
from pseudoESM.loader.utils import make_loaders, DataCollector
from pseudoESM.esm.data import Alphabet
import hydra

from omegaconf import OmegaConf



def test_loader():

    cfg = OmegaConf.load("../train_conf.yaml")
    tokenizer = Alphabet.from_architecture("protein_bert_base")

    collate_fn = DataCollector(tokenizer)

    # load dataset
    dataloaders = make_loaders(
        collate_fn,
        train_dir=os.path.join(orig_cwd, cfg.data.train_dir),
        valid_dir=os.path.join(orig_cwd, cfg.data.valid_dir),
        test_dir=os.path.join(orig_cwd, cfg.data.test_dir),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers)

    tokens, labels = next(iter(dataloaders["train"]))

    print(tokens[3])
    print(labels[3])


if __name__ == "__main__":
    test_loader()
