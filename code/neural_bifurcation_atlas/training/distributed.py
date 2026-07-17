from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as distributed
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool


def initialize_distributed(backend: str | None = None) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not distributed.is_initialized():
        selected = backend or ("nccl" if torch.cuda.is_available() else "gloo")
        distributed.init_process_group(backend=selected, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    return DistributedContext(rank, local_rank, world_size, enabled)


def wrap_distributed(model: nn.Module, context: DistributedContext) -> nn.Module:
    if not context.distributed:
        return model
    if torch.cuda.is_available():
        return DistributedDataParallel(model, device_ids=[context.local_rank])
    return DistributedDataParallel(model)


def reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    if not context.distributed:
        return value
    result = value.detach().clone()
    distributed.all_reduce(result, op=distributed.ReduceOp.SUM)
    return result / context.world_size


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        distributed.barrier()


def shutdown_distributed(context: DistributedContext) -> None:
    if context.distributed and distributed.is_initialized():
        distributed.destroy_process_group()
