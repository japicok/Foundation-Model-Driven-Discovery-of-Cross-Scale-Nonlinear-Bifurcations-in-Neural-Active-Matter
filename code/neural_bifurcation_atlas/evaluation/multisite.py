from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2


@dataclass(frozen=True)
class HeterogeneityResult:
    pooled: float
    q: float
    degrees_of_freedom: int
    p_value: float
    i_squared: float


def heterogeneity(estimates: np.ndarray, standard_errors: np.ndarray) -> HeterogeneityResult:
    if estimates.shape != standard_errors.shape:
        raise ValueError("estimate and error shapes must match")
    weights = 1.0 / np.square(standard_errors)
    pooled = float(np.sum(weights * estimates) / np.sum(weights))
    q = float(np.sum(weights * np.square(estimates - pooled)))
    degrees = max(estimates.size - 1, 0)
    p_value = float(chi2.sf(q, degrees)) if degrees > 0 else 1.0
    i_squared = max(0.0, (q - degrees) / q * 100.0) if q > 0 else 0.0
    return HeterogeneityResult(pooled, q, degrees, p_value, i_squared)


def fixed_effect_pool(estimates: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    weights = 1.0 / variances
    estimate = float(np.sum(weights * estimates) / np.sum(weights))
    error = float(np.sqrt(1.0 / np.sum(weights)))
    return estimate, error
