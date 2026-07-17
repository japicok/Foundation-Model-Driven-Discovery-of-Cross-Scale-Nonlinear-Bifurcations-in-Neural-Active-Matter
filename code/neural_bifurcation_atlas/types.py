from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import torch


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    root: Path
    sample_rate: int
    window_seconds: int
    stride_seconds: int
    channels: int


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int
    manifold_dim: int
    hidden_dim: int
    invertible_blocks: int
    atlas_hidden: tuple[int, ...]
    atlas_classes: int
    backbones: tuple[str, ...]


@dataclass(frozen=True)
class ScienceConfig:
    normal_form_weight: float
    coherence_threshold: float
    normal_coordinate_tolerance: float
    disagreement_sigma: float
    coherence_window: int
    exponent_window_seconds: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    gradient_accumulation: int
    epochs: int
    learning_rate: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_steps: int
    precision: str
    gradient_clip: float
    seeds: int
    world_size: int


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_resamples: int
    calibration_bins: int
    alpha: float


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    device: str
    data: DataConfig
    model: ModelConfig
    science: ScienceConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


@dataclass(frozen=True)
class WindowRecord:
    subject_id: str
    signal: torch.Tensor
    label: int
    timestamp: float
    drug_class: int
    infusion_rate: float
    susceptibility: float


class AlignmentResult(NamedTuple):
    aligned: tuple[torch.Tensor, ...]
    rotations: tuple[torch.Tensor, ...]
    residual: torch.Tensor


class CoherenceResult(NamedTuple):
    score: torch.Tensor
    transition: torch.Tensor
    neuronal_population: torch.Tensor
    population_brain: torch.Tensor


class DisagreementResult(NamedTuple):
    score: torch.Tensor
    candidate: torch.Tensor
    baseline_mean: torch.Tensor
    baseline_std: torch.Tensor


class ExponentResult(NamedTuple):
    beta: float
    intercept: float
    lower: float
    upper: float
    samples: int


BifurcationLabel = Literal["fold", "hopf", "pitchfork"]
