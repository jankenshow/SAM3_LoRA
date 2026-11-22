"""torchrun process lifecycle and scalar aggregation, independent of the trainer."""

import os
from contextlib import contextmanager

import torch
from torch import distributed as dist


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


@contextmanager
def distributed_runtime():
    launched = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if launched:
        if not torch.cuda.is_available():
            raise ValueError("Distributed training requires CUDA/NCCL")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    try:
        yield launched
    finally:
        if launched and dist.is_initialized():
            dist.destroy_process_group()


def aggregate_mean(total, count, device):
    values = torch.tensor([total, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(values)
    if values[1].item() == 0:
        raise ValueError("DataLoader produced no batches")
    return (values[0] / values[1]).item()
