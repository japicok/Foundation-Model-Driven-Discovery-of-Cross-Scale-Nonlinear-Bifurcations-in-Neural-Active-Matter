from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from neural_bifurcation_atlas.config import load_config
from neural_bifurcation_atlas.models.manifold import InvertibleSlowManifold
from neural_bifurcation_atlas.training.engine import ManifoldTrainer
from neural_bifurcation_atlas.training.optim import build_optimizer, build_scheduler
from neural_bifurcation_atlas.training.seed import set_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="nam-train")
    value.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    value.add_argument("--embeddings", type=Path, required=True)
    value.add_argument("--output", type=Path, default=Path("outputs/main"))
    value.add_argument("override", nargs="*")
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config(arguments.config, tuple(arguments.override))
    set_seed(config.seed)
    embeddings = torch.load(arguments.embeddings, map_location="cpu", weights_only=True)
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("embedding file must contain a tensor")
    dataset = TensorDataset(embeddings)
    loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=True)
    model = InvertibleSlowManifold(
        config.model.embedding_dim,
        config.model.manifold_dim,
        config.model.hidden_dim,
        config.model.invertible_blocks,
    )
    optimizer = build_optimizer(model, config.training)
    steps = config.training.epochs * max(len(loader), 1)
    scheduler = build_scheduler(optimizer, config.training, steps)
    trainer = ManifoldTrainer(model, optimizer, scheduler, config, arguments.output)
    trainer.fit(loader)


if __name__ == "__main__":
    main()
