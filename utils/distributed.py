import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os


def single_machine_multi_gpu_setup():
    dist.init_process_group(backend='nccl')

def cleanup():
    dist.destroy_process_group()