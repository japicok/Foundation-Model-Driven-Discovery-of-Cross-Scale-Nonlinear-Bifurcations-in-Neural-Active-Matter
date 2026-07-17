from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    auroc: float
    accuracy: float
    sensitivity: float
    specificity: float
    brier: float
    ece: float


def binary_confusion(
    probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> tuple[int, int, int, int]:
    prediction = probability >= threshold
    positive = target.bool()
    true_positive = int((prediction & positive).sum())
    true_negative = int((~prediction & ~positive).sum())
    false_positive = int((prediction & ~positive).sum())
    false_negative = int((~prediction & positive).sum())
    return true_positive, true_negative, false_positive, false_negative


def expected_calibration_error(
    probability: torch.Tensor, target: torch.Tensor, bins: int = 20
) -> float:
    edges = torch.linspace(0.0, 1.0, bins + 1, device=probability.device)
    error = torch.zeros((), device=probability.device)
    for index in range(bins):
        lower = edges[index]
        upper = edges[index + 1]
        mask = (probability >= lower) & (
            probability < upper if index < bins - 1 else probability <= upper
        )
        if mask.any():
            confidence = probability[mask].mean()
            accuracy = target[mask].float().mean()
            error += mask.float().mean() * (confidence - accuracy).abs()
    return float(error)


def brier_score(probability: torch.Tensor, target: torch.Tensor) -> float:
    return float((probability - target.float()).square().mean())


def binary_metrics(
    probability: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, bins: int = 20
) -> BinaryMetrics:
    true_positive, true_negative, false_positive, false_negative = binary_confusion(
        probability, target, threshold
    )
    total = true_positive + true_negative + false_positive + false_negative
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    accuracy = (true_positive + true_negative) / max(total, 1)
    auroc = float(roc_auc_score(target.detach().cpu().numpy(), probability.detach().cpu().numpy()))
    return BinaryMetrics(
        auroc,
        accuracy,
        sensitivity,
        specificity,
        brier_score(probability, target),
        expected_calibration_error(probability, target, bins),
    )


def cohens_kappa(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.shape != second.shape:
        raise ValueError("label shapes must match")
    observed = float((first == second).float().mean())
    labels = torch.unique(torch.cat([first, second]))
    expected = 0.0
    for label in labels:
        expected += float((first == label).float().mean() * (second == label).float().mean())
    return (observed - expected) / max(1.0 - expected, 1e-12)


def clinical_impact_per_thousand(
    specificity: float, baseline_specificity: float, prevalence: float
) -> float:
    return 1000.0 * (specificity - baseline_specificity) * (1.0 - prevalence)


def number_needed(value_per_thousand: float) -> int:
    if value_per_thousand <= 0:
        raise ValueError("clinical impact must be positive")
    return int(np.ceil(1000.0 / value_per_thousand))


def decision_curve(
    probability: torch.Tensor, target: torch.Tensor, thresholds: torch.Tensor
) -> torch.Tensor:
    count = target.numel()
    benefits = torch.empty_like(thresholds)
    for index, threshold in enumerate(thresholds):
        true_positive, _, false_positive, _ = binary_confusion(
            probability, target, float(threshold)
        )
        odds = threshold / (1.0 - threshold).clamp_min(1e-12)
        benefits[index] = true_positive / count - false_positive / count * odds
    return benefits


def lead_time(
    predicted: torch.Tensor, events: torch.Tensor, sample_period: float = 1.0
) -> torch.Tensor:
    predicted_indices = torch.nonzero(predicted, as_tuple=False).flatten()
    event_indices = torch.nonzero(events, as_tuple=False).flatten()
    values: list[torch.Tensor] = []
    for event in event_indices:
        candidates = predicted_indices[predicted_indices <= event]
        if candidates.numel() > 0:
            values.append((event - candidates[-1]).float() * sample_period)
    return torch.stack(values) if values else torch.empty(0)
