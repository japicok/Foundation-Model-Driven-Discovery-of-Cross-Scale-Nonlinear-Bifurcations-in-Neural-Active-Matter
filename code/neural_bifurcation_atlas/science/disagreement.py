from __future__ import annotations

import itertools

import torch

from neural_bifurcation_atlas.types import DisagreementResult


def covariance_inverse(
    embeddings: tuple[torch.Tensor, ...], regularization: float = 1e-5
) -> torch.Tensor:
    stacked = torch.cat(embeddings, dim=0)
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / max(stacked.shape[0] - 1, 1)
    identity = torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
    return torch.linalg.pinv(covariance + regularization * identity)


def squared_mahalanobis(
    first: torch.Tensor, second: torch.Tensor, precision: torch.Tensor
) -> torch.Tensor:
    difference = first - second
    return torch.einsum("...d,de,...e->...", difference, precision, difference)


def pairwise_distances(
    embeddings: tuple[torch.Tensor, ...], precision: torch.Tensor
) -> torch.Tensor:
    distances = [
        squared_mahalanobis(embeddings[i], embeddings[j], precision)
        for i, j in itertools.combinations(range(len(embeddings)), 2)
    ]
    return torch.stack(distances, dim=-1)


def rolling_statistics(score: torch.Tensor, window: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.zeros_like(score)
    standard = torch.zeros_like(score)
    for index in range(score.shape[0]):
        start = max(0, index - window + 1)
        values = score[start : index + 1]
        mean[index] = values.mean()
        standard[index] = values.std(unbiased=False)
    return mean, standard


def local_maxima(score: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(score, dtype=torch.bool)
    if score.shape[0] >= 3:
        result[1:-1] = (score[1:-1] > score[:-2]) & (score[1:-1] >= score[2:])
    return result


def disagreement_score(
    embeddings: tuple[torch.Tensor, ...],
    window: int,
    sigma: float = 3.0,
    precision: torch.Tensor | None = None,
) -> DisagreementResult:
    if len(embeddings) < 2:
        raise ValueError("at least two aligned embeddings are required")
    selected_precision = precision if precision is not None else covariance_inverse(embeddings)
    score = pairwise_distances(embeddings, selected_precision).sum(dim=-1)
    baseline_mean, baseline_std = rolling_statistics(score, window)
    candidate = (score > baseline_mean + sigma * baseline_std) & local_maxima(score)
    return DisagreementResult(score, candidate, baseline_mean, baseline_std)
