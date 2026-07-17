from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from neural_bifurcation_atlas.types import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    ScienceConfig,
    TrainingConfig,
)


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("configuration section must be a mapping")
    return value


def load_config(path: Path, overrides: tuple[str, ...] = ()) -> ExperimentConfig:
    base = OmegaConf.load(path)
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(list(overrides)))
    raw = OmegaConf.to_container(merged, resolve=True)
    root = _mapping(raw)
    data = _mapping(root["data"])
    model = _mapping(root["model"])
    science = _mapping(root["science"])
    training = _mapping(root["training"])
    evaluation = _mapping(root["evaluation"])
    return ExperimentConfig(
        seed=int(root["seed"]),
        device=str(root["device"]),
        data=DataConfig(
            dataset=str(data["dataset"]),
            root=Path(str(data.get("root", "data"))),
            sample_rate=int(data["sample_rate"]),
            window_seconds=int(data["window_seconds"]),
            stride_seconds=int(data["stride_seconds"]),
            channels=int(data["channels"]),
        ),
        model=ModelConfig(
            embedding_dim=int(model["embedding_dim"]),
            manifold_dim=int(model["manifold_dim"]),
            hidden_dim=int(model["hidden_dim"]),
            invertible_blocks=int(model["invertible_blocks"]),
            atlas_hidden=tuple(int(x) for x in model["atlas_hidden"]),
            atlas_classes=int(model["atlas_classes"]),
            backbones=tuple(str(x) for x in model["backbones"]),
        ),
        science=ScienceConfig(**{k: v for k, v in science.items()}),
        training=TrainingConfig(**{k: v for k, v in training.items() if k != "status"}),
        evaluation=EvaluationConfig(**{k: v for k, v in evaluation.items()}),
    )
