import math

import pytest
import torch
from neural_bifurcation_atlas.science.alignment import generalized_procrustes, orthogonal_procrustes
from neural_bifurcation_atlas.science.coherence import (
    coherence_optimal_scale_count,
    multi_scale_coherence,
    phase_locking_value,
    three_scale_coherence,
)
from neural_bifurcation_atlas.science.disagreement import (
    covariance_inverse,
    disagreement_score,
    local_maxima,
    pairwise_distances,
    rolling_statistics,
    squared_mahalanobis,
)
from neural_bifurcation_atlas.science.exponents import (
    embedding_velocity,
    fit_critical_exponent,
    log_log_regression,
    order_parameter,
    pseudo_time,
)
from neural_bifurcation_atlas.science.normal_form import (
    estimate_control,
    lowpass,
    normal_form_loss,
    normal_form_residual,
    slowing_indicators,
    temporal_derivative,
)


def test_procrustes_recovers_rotation() -> None:
    generator = torch.Generator().manual_seed(4)
    source = torch.randn(64, 3, generator=generator)
    matrix = torch.randn(3, 3, generator=generator)
    rotation, _ = torch.linalg.qr(matrix)
    target = source @ rotation
    fitted = orthogonal_procrustes(source, target)
    assert torch.allclose(source @ fitted, target, atol=1e-5)


def test_generalized_alignment_reduces_residual() -> None:
    generator = torch.Generator().manual_seed(5)
    reference = torch.randn(100, 4, generator=generator)
    values = []
    for _ in range(4):
        rotation, _ = torch.linalg.qr(torch.randn(4, 4, generator=generator))
        values.append(reference @ rotation)
    result = generalized_procrustes(tuple(values))
    assert result.residual < 1e-4
    assert len(result.rotations) == 4


def test_phase_locking_identical_phases() -> None:
    phase = torch.linspace(0, 8 * math.pi, 100)
    score = phase_locking_value(phase, phase, 20)
    assert torch.allclose(score, torch.ones_like(score), atol=1e-6)


def test_three_scale_transition_gate() -> None:
    time = torch.linspace(0, 10 * math.pi, 200)
    coordinate = torch.sin(time) * 0.05
    result = three_scale_coherence(coordinate, coordinate, coordinate, 20, 0.8, 0.1)
    assert result.transition.any()
    assert torch.all(result.score > 0.99)


def test_multi_scale_product() -> None:
    time = torch.sin(torch.linspace(0, 5 * math.pi, 128))
    score = multi_scale_coherence((time, time, time, time), 12)
    assert score.min() > 0.99


@pytest.mark.parametrize(
    "samples,features,expected", [(1000, 100, 3), (4000, 400, 3), (900, 100, 2)]
)
def test_scale_count(samples: int, features: int, expected: int) -> None:
    result = coherence_optimal_scale_count(-0.9, 1.0, samples, 1.0, features)
    assert result == expected


def test_mahalanobis_zero_identity() -> None:
    value = torch.randn(20, 5)
    precision = torch.eye(5)
    assert torch.equal(squared_mahalanobis(value, value, precision), torch.zeros(20))


def test_pairwise_distance_count() -> None:
    values = tuple(torch.randn(30, 4) for _ in range(5))
    distances = pairwise_distances(values, torch.eye(4))
    assert distances.shape == (30, 10)


def test_covariance_precision_is_symmetric() -> None:
    values = tuple(torch.randn(40, 6) for _ in range(3))
    precision = covariance_inverse(values)
    assert torch.allclose(precision, precision.T, atol=1e-5)


def test_rolling_statistics_constant() -> None:
    mean, standard = rolling_statistics(torch.ones(20), 5)
    assert torch.equal(mean, torch.ones(20))
    assert torch.equal(standard, torch.zeros(20))


def test_local_maxima() -> None:
    score = torch.tensor([0.0, 2.0, 1.0, 3.0, 0.0])
    assert torch.equal(local_maxima(score), torch.tensor([False, True, False, True, False]))


def test_disagreement_spike() -> None:
    first = torch.zeros(30, 3)
    second = torch.zeros(30, 3)
    second[20] = 10
    result = disagreement_score((first, second), 10, 2.0, torch.eye(3))
    assert result.candidate[20]


def test_temporal_derivative_linear() -> None:
    coordinate = torch.arange(20).float()
    assert torch.allclose(temporal_derivative(coordinate), torch.ones(20))


def test_lowpass_constant() -> None:
    coordinate = torch.ones(30)
    assert torch.allclose(lowpass(coordinate, 7), coordinate)


def test_estimated_control_satisfies_smoothed_relation() -> None:
    coordinate = torch.linspace(-1, 1, 100)
    control = estimate_control(coordinate, window=1)
    residual = normal_form_residual(coordinate, control)
    assert residual.square().mean() < 1e-10


def test_normal_form_loss_nonnegative() -> None:
    assert normal_form_loss(torch.randn(100)) >= 0


def test_slowing_indicator_shapes() -> None:
    indicators = slowing_indicators(torch.randn(100), 12)
    assert set(indicators) == {"variance", "autocorrelation", "skewness"}
    assert all(value.shape == (100,) for value in indicators.values())


def test_embedding_velocity_constant_slope() -> None:
    embedding = torch.arange(20).float().unsqueeze(-1).repeat(1, 3)
    velocity = embedding_velocity(embedding)
    assert torch.allclose(velocity, torch.ones_like(velocity))


def test_order_parameter_aligned_channels() -> None:
    time = torch.arange(20).float()
    embedding = time[:, None, None].repeat(1, 4, 3)
    order = order_parameter(embedding)
    assert torch.allclose(order, torch.full_like(order, math.sqrt(3)))


def test_log_log_regression_recovers_half() -> None:
    control = torch.linspace(0.01, 1.0, 100)
    order = control.sqrt()
    beta, intercept = log_log_regression(order, control, 0.0)
    assert beta == pytest.approx(0.5, abs=1e-5)
    assert intercept == pytest.approx(0.0, abs=1e-5)


def test_bootstrap_exponent_contains_truth() -> None:
    control = torch.linspace(0.01, 1.0, 100)
    order = 2.0 * control.sqrt()
    result = fit_critical_exponent(order, control, 0.0, 100, 9)
    assert result.lower <= 0.5 <= result.upper


def test_pseudo_time_monotonic() -> None:
    embedding = torch.arange(30).float().unsqueeze(-1)
    value = pseudo_time(embedding, 3)
    assert torch.all(value[1:] >= value[:-1])
    assert value[-1] == pytest.approx(1.0)
