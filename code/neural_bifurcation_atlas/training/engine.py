from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from neural_bifurcation_atlas.losses import manifold_objective
from neural_bifurcation_atlas.models.manifold import InvertibleSlowManifold
from neural_bifurcation_atlas.training.checkpoint import save_checkpoint
from neural_bifurcation_atlas.types import ExperimentConfig


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    loss: float
    reconstruction: float
    normal_form: float
    learning_rate: float
    steps: int


class RunningAverage:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.total += value * count
        self.count += count

    @property
    def mean(self) -> float:
        return self.total / max(self.count, 1)


class ManifoldTrainer:
    def __init__(
        self,
        model: InvertibleSlowManifold,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        config: ExperimentConfig,
        output: Path,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.output = output
        self.device = torch.device(
            config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.training.precision == "fp16" and self.device.type == "cuda"
        )
        self.logger = logging.getLogger(__name__)
        self.global_step = 0

    def _autocast(self) -> torch.amp.autocast:
        enabled = self.config.training.precision in {"fp16", "bf16"} and self.device.type == "cuda"
        dtype = torch.float16 if self.config.training.precision == "fp16" else torch.bfloat16
        return torch.amp.autocast(self.device.type, enabled=enabled, dtype=dtype)

    def train_batch(self, embedding: torch.Tensor, accumulation_index: int = 0) -> dict[str, float]:
        self.model.train()
        embedding = embedding.to(self.device)
        with self._autocast():
            coordinate, reconstruction = self.model(embedding)
            loss, terms = manifold_objective(
                embedding, reconstruction, coordinate, self.config.science.normal_form_weight
            )
            scaled_loss = loss / self.config.training.gradient_accumulation
        self.scaler.scale(scaled_loss).backward()
        boundary = (accumulation_index + 1) % self.config.training.gradient_accumulation == 0
        if boundary:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_step += 1
        return {name: float(value) for name, value in terms.items()}

    def train_epoch(self, loader: DataLoader[Any], epoch: int) -> EpochSummary:
        loss_average = RunningAverage()
        reconstruction_average = RunningAverage()
        normal_average = RunningAverage()
        self.optimizer.zero_grad(set_to_none=True)
        for batch_index, batch in enumerate(loader):
            embedding = batch["embedding"] if isinstance(batch, dict) else batch
            terms = self.train_batch(embedding, batch_index)
            count = int(embedding.shape[0])
            loss_average.update(terms["total"], count)
            reconstruction_average.update(terms["reconstruction"], count)
            normal_average.update(terms["normal_form"], count)
        learning_rate = float(self.optimizer.param_groups[0]["lr"])
        summary = EpochSummary(
            epoch,
            loss_average.mean,
            reconstruction_average.mean,
            normal_average.mean,
            learning_rate,
            self.global_step,
        )
        self.logger.info(
            "epoch=%d loss=%.6f reconstruction=%.6f normal_form=%.6f",
            epoch,
            summary.loss,
            summary.reconstruction,
            summary.normal_form,
        )
        return summary

    def fit(self, loader: DataLoader[Any]) -> list[EpochSummary]:
        summaries: list[EpochSummary] = []
        for epoch in range(self.config.training.epochs):
            summary = self.train_epoch(loader, epoch)
            summaries.append(summary)
            save_checkpoint(
                self.output / "latest.pt",
                self.model,
                self.optimizer,
                epoch,
                self.global_step,
                self.config.seed,
                {"loss": summary.loss},
            )
        return summaries
