from __future__ import annotations

import numpy as np
import torch

from neural_bifurcation_atlas.types import ExponentResult


def embedding_velocity(embedding: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    velocity = torch.zeros_like(embedding)
    velocity[:-1] = (embedding[1:] - embedding[:-1]) / delta
    if embedding.shape[0] > 1:
        velocity[-1] = velocity[-2]
    return velocity


def order_parameter(channel_embeddings: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    if channel_embeddings.ndim != 3:
        raise ValueError("expected time, channel, embedding dimensions")
    velocity = embedding_velocity(channel_embeddings, delta)
    return velocity.mean(dim=1).norm(dim=-1)


def log_log_regression(
    order: torch.Tensor, control: torch.Tensor, critical: float
) -> tuple[float, float]:
    distance = (control - critical).abs()
    mask = torch.isfinite(order) & torch.isfinite(distance) & (order > 0) & (distance > 0)
    x = torch.log(distance[mask])
    y = torch.log(order[mask])
    if x.numel() < 3:
        raise ValueError("at least three valid samples are required")
    centered = x - x.mean()
    beta = (centered * (y - y.mean())).sum() / centered.square().sum().clamp_min(1e-12)
    intercept = y.mean() - beta * x.mean()
    return float(beta), float(intercept)


def fit_critical_exponent(
    order: torch.Tensor,
    control: torch.Tensor,
    critical: float,
    resamples: int = 1000,
    seed: int = 1729,
) -> ExponentResult:
    beta, intercept = log_log_regression(order, control, critical)
    distance = (control - critical).abs()
    mask = torch.isfinite(order) & torch.isfinite(distance) & (order > 0) & (distance > 0)
    valid_order = order[mask]
    valid_control = control[mask]
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        indices = torch.from_numpy(
            rng.integers(0, valid_order.shape[0], valid_order.shape[0])
        ).long()
        try:
            estimates[index] = log_log_regression(
                valid_order[indices], valid_control[indices], critical
            )[0]
        except ValueError:
            estimates[index] = beta
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return ExponentResult(beta, intercept, float(lower), float(upper), int(valid_order.shape[0]))


def pseudo_time(embedding: torch.Tensor, neighbors: int = 8) -> torch.Tensor:
    distances = torch.cdist(embedding, embedding)
    adjacency = torch.topk(
        distances, k=min(neighbors + 1, embedding.shape[0]), largest=False
    ).indices[:, 1:]
    increments = torch.zeros(embedding.shape[0], device=embedding.device, dtype=embedding.dtype)
    for index in range(1, embedding.shape[0]):
        previous_neighbors = adjacency[index - 1]
        local = distances[index, previous_neighbors].mean()
        increments[index] = increments[index - 1] + local
    span = increments[-1].clamp_min(1e-12)
    return increments / span
