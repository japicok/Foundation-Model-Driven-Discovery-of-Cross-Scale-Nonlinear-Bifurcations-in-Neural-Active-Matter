from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float


@dataclass(frozen=True)
class PairedComparison:
    difference: float
    lower: float
    upper: float
    p_value: float


def _stratified_indices(target: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    groups = []
    for label in np.unique(target):
        candidates = np.flatnonzero(target == label)
        groups.append(rng.choice(candidates, size=candidates.size, replace=True))
    indices = np.concatenate(groups)
    rng.shuffle(indices)
    return indices


def bootstrap_auc(
    target: np.ndarray, probability: np.ndarray, resamples: int = 10000, seed: int = 1729
) -> BootstrapInterval:
    if np.unique(target).size != 2:
        raise ValueError("binary targets are required")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for index in range(resamples):
        sampled = _stratified_indices(target, rng)
        values[index] = roc_auc_score(target[sampled], probability[sampled])
    lower, upper = np.quantile(values, [0.025, 0.975])
    return BootstrapInterval(
        float(roc_auc_score(target, probability)),
        float(lower),
        float(upper),
        float(values.std(ddof=1)),
    )


def paired_bootstrap_auc(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    resamples: int = 10000,
    seed: int = 1729,
) -> PairedComparison:
    rng = np.random.default_rng(seed)
    differences = np.empty(resamples)
    for index in range(resamples):
        sampled = _stratified_indices(target, rng)
        differences[index] = roc_auc_score(target[sampled], first[sampled]) - roc_auc_score(
            target[sampled], second[sampled]
        )
    estimate = roc_auc_score(target, first) - roc_auc_score(target, second)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    p_value = (np.count_nonzero(differences <= 0) + 1) / (resamples + 1)
    return PairedComparison(float(estimate), float(lower), float(upper), float(p_value))
