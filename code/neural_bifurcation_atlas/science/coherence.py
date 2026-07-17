from __future__ import annotations

import torch

from neural_bifurcation_atlas.data.signal import analytic_signal
from neural_bifurcation_atlas.types import CoherenceResult


def instantaneous_phase(coordinate: torch.Tensor) -> torch.Tensor:
    return torch.angle(analytic_signal(coordinate))


def rolling_complex_mean(value: torch.Tensor, window: int) -> torch.Tensor:
    result = torch.zeros_like(value)
    cumulative = torch.cumsum(value, dim=0)
    for index in range(value.shape[0]):
        start = index - window
        total = cumulative[index] if start < 0 else cumulative[index] - cumulative[start]
        result[index] = total / min(index + 1, window)
    return result


def phase_locking_value(first: torch.Tensor, second: torch.Tensor, window: int) -> torch.Tensor:
    if first.shape != second.shape:
        raise ValueError("phase shapes must match")
    difference = torch.exp(1j * (first - second))
    return rolling_complex_mean(difference, window).abs()


def three_scale_coherence(
    neuronal: torch.Tensor,
    population: torch.Tensor,
    brain: torch.Tensor,
    window: int,
    threshold: float = 0.8,
    coordinate_tolerance: float = 0.1,
) -> CoherenceResult:
    if neuronal.shape != population.shape or population.shape != brain.shape:
        raise ValueError("scale coordinates must share shape")
    neuronal_phase = instantaneous_phase(neuronal)
    population_phase = instantaneous_phase(population)
    brain_phase = instantaneous_phase(brain)
    neuronal_population = phase_locking_value(neuronal_phase, population_phase, window)
    population_brain = phase_locking_value(population_phase, brain_phase, window)
    score = neuronal_population * population_brain
    transition = (score >= threshold) & (brain.abs() < coordinate_tolerance)
    return CoherenceResult(score, transition, neuronal_population, population_brain)


def multi_scale_coherence(coordinates: tuple[torch.Tensor, ...], window: int) -> torch.Tensor:
    if len(coordinates) < 2:
        raise ValueError("at least two scales are required")
    phases = tuple(instantaneous_phase(value) for value in coordinates)
    values = [
        phase_locking_value(first, second, window)
        for first, second in zip(phases[:-1], phases[1:], strict=True)
    ]
    return torch.stack(values).prod(dim=0)


def coherence_optimal_scale_count(
    mean_log_coherence: float,
    gain_constant: float,
    sample_count: int,
    variance_constant: float,
    mean_features: float,
) -> int:
    value = abs(mean_log_coherence) * gain_constant * sample_count
    denominator = variance_constant * mean_features
    return max(1, int((value / denominator) ** 0.5))
