from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from neural_bifurcation_atlas.types import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            parameters, lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
        )
    raise ValueError(f"unknown optimizer {config.optimizer}")


def cosine_warmup(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: Optimizer, config: TrainingConfig, total_steps: int) -> LambdaLR:
    if config.scheduler.lower() == "cosine":
        return LambdaLR(
            optimizer, lambda step: cosine_warmup(step, config.warmup_steps, total_steps)
        )
    if config.scheduler.lower() == "constant":
        return LambdaLR(optimizer, lambda step: 1.0)
    raise ValueError(f"unknown scheduler {config.scheduler}")
