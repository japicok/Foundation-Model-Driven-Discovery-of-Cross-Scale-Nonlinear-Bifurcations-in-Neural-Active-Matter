from __future__ import annotations

import numpy as np


def holm_bonferroni(p_values: np.ndarray, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    if p_values.ndim != 1:
        raise ValueError("p-values must be one-dimensional")
    count = p_values.size
    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate((count - np.arange(count)) * p_values[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    rejected = adjusted <= alpha
    return adjusted, rejected


def one_sided_effect_size(first: np.ndarray, second: np.ndarray) -> float:
    difference = first - second
    return float(difference.mean() / difference.std(ddof=1))
