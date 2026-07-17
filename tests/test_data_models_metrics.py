from pathlib import Path

import numpy as np
import pytest
import torch
from neural_bifurcation_atlas.config import load_config
from neural_bifurcation_atlas.data.datasets import (
    ArrayWindowDataset,
    SyntheticTransitionDataset,
    collate_windows,
)
from neural_bifurcation_atlas.data.signal import (
    analytic_signal,
    band_power,
    common_average_reference,
    fft_bandpass,
    high_gamma_envelope,
    linear_detrend,
    preprocess_eeg,
    resample_linear,
    robust_scale,
    spectral_slope,
    window_signal,
)
from neural_bifurcation_atlas.data.splits import assert_disjoint, subject_disjoint_split
from neural_bifurcation_atlas.evaluation.bootstrap import bootstrap_auc, paired_bootstrap_auc
from neural_bifurcation_atlas.evaluation.multiple_testing import holm_bonferroni
from neural_bifurcation_atlas.evaluation.multisite import fixed_effect_pool, heterogeneity
from neural_bifurcation_atlas.losses import (
    atlas_loss,
    focal_loss,
    manifold_objective,
    reconstruction_loss,
)
from neural_bifurcation_atlas.metrics import (
    binary_confusion,
    binary_metrics,
    brier_score,
    clinical_impact_per_thousand,
    cohens_kappa,
    decision_curve,
    expected_calibration_error,
    lead_time,
    number_needed,
)
from neural_bifurcation_atlas.models.atlas import BifurcationAtlas, TransitionHead
from neural_bifurcation_atlas.models.backbones import (
    ConvTokenBackbone,
    SpectralTokenBackbone,
    TemporalPatchBackbone,
    build_local_ensemble,
)
from neural_bifurcation_atlas.models.manifold import (
    InvertibleSlowManifold,
    ResidualTransform,
    ScaleManifolds,
)


def test_array_dataset_validation() -> None:
    with pytest.raises(ValueError):
        ArrayWindowDataset(torch.randn(3, 10), torch.zeros(3), ["a"] * 3)


def test_array_dataset_record() -> None:
    dataset = ArrayWindowDataset(
        torch.randn(4, 2, 20), torch.tensor([0, 1, 0, 1]), ["a", "a", "b", "b"]
    )
    record = dataset[1]
    assert record.subject_id == "a"
    assert record.label == 1
    assert record.signal.shape == (2, 20)


def test_synthetic_dataset_is_deterministic() -> None:
    first = SyntheticTransitionDataset(seed=2)
    second = SyntheticTransitionDataset(seed=2)
    assert torch.equal(first.signals, second.signals)
    assert torch.equal(first.labels, second.labels)


def test_synthetic_dataset_has_two_classes() -> None:
    dataset = SyntheticTransitionDataset(subjects=2, windows_per_subject=8)
    assert set(dataset.labels.tolist()) == {0, 1}


def test_collation() -> None:
    dataset = SyntheticTransitionDataset(subjects=1, windows_per_subject=4)
    batch = collate_windows([dataset[0], dataset[1]])
    assert batch["signal"].shape[0] == 2
    assert batch["label"].shape == (2,)


def test_subject_split_is_disjoint() -> None:
    subjects = [f"s{i // 4}" for i in range(40)]
    split = subject_disjoint_split(subjects, seed=3)
    assert_disjoint(split, subjects)
    assert len(split.train) + len(split.validation) + len(split.test) == 40


def test_invalid_split_fractions() -> None:
    with pytest.raises(ValueError):
        subject_disjoint_split(["a", "b"], 0.9, 0.2)


def test_linear_detrend_removes_ramp() -> None:
    ramp = torch.arange(100).float().repeat(3, 1)
    assert linear_detrend(ramp).abs().max() < 1e-4


def test_common_average_reference() -> None:
    signal = torch.randn(4, 100)
    referenced = common_average_reference(signal)
    assert referenced.mean(dim=-2).abs().max() < 1e-6


def test_robust_scale_median() -> None:
    signal = torch.randn(5, 101)
    scaled = robust_scale(signal)
    assert scaled.median(dim=-1).values.abs().max() < 1e-6


def test_resample_shape() -> None:
    signal = torch.randn(2, 3, 100)
    assert resample_linear(signal, 64).shape == (2, 3, 64)


def test_fft_bandpass_rejects_out_of_band() -> None:
    time = torch.arange(1000) / 100.0
    signal = torch.sin(2 * torch.pi * 5 * time) + torch.sin(2 * torch.pi * 30 * time)
    filtered = fft_bandpass(signal, 100, 4, 6)
    reference = torch.sin(2 * torch.pi * 5 * time)
    assert torch.mean((filtered - reference).square()) < 1e-4


def test_invalid_bandpass() -> None:
    with pytest.raises(ValueError):
        fft_bandpass(torch.randn(100), 100, 40, 60)


def test_preprocess_is_finite() -> None:
    value = preprocess_eeg(torch.randn(4, 1000), 100)
    assert torch.isfinite(value).all()


def test_window_signal_shape() -> None:
    windows = window_signal(torch.randn(3, 100), 20, 5)
    assert windows.shape == (3, 17, 20)


def test_short_window_rejected() -> None:
    with pytest.raises(ValueError):
        window_signal(torch.randn(10), 20, 2)


def test_band_power_prefers_carrier_band() -> None:
    time = torch.arange(1000) / 100.0
    signal = torch.sin(2 * torch.pi * 10 * time)
    alpha = band_power(signal, 100, 8, 12)
    delta = band_power(signal, 100, 1, 4)
    assert alpha > delta * 100


def test_high_gamma_envelope_shape() -> None:
    signal = torch.randn(4, 1000)
    assert high_gamma_envelope(signal, 100).shape == (1000,)


def test_analytic_signal_magnitude() -> None:
    time = torch.linspace(0, 20 * torch.pi, 1000)
    magnitude = analytic_signal(torch.sin(time)).abs()
    assert magnitude[10:-10].mean() == pytest.approx(1.0, abs=0.02)


def test_spectral_slope_shape() -> None:
    assert spectral_slope(torch.randn(3, 1000), 100).shape == (3,)


@pytest.mark.parametrize("batch,channels,samples,dimension", [(2, 4, 256, 16), (3, 8, 512, 32)])
def test_conv_backbone(batch: int, channels: int, samples: int, dimension: int) -> None:
    model = ConvTokenBackbone(channels, dimension)
    output = model(torch.randn(batch, channels, samples))
    assert output.shape[0] == batch
    assert output.shape[-1] == dimension


def test_spectral_backbone() -> None:
    model = SpectralTokenBackbone(4, 16, 32)
    output = model(torch.randn(2, 4, 256))
    assert output.shape == (2, 32, 16)


def test_temporal_backbone() -> None:
    model = TemporalPatchBackbone(4, 16, 32)
    output = model(torch.randn(2, 4, 256))
    assert output.shape == (2, 8, 16)


def test_local_ensemble() -> None:
    model = build_local_ensemble(4, 16)
    outputs = model(torch.randn(2, 4, 256))
    assert set(outputs) == {"spectral", "temporal", "convolutional"}
    assert all(value.shape == (2, 16) for value in outputs.values())


def test_residual_transform_inverse() -> None:
    model = ResidualTransform(8, 16)
    value = torch.randn(20, 8)
    recovered = model.inverse(model(value), 100)
    assert torch.allclose(value, recovered, atol=1e-5)


def test_manifold_shapes() -> None:
    model = InvertibleSlowManifold(16, 3, 32, 2)
    value = torch.randn(20, 16)
    coordinate, reconstruction = model(value)
    assert coordinate.shape == (20, 3)
    assert reconstruction.shape == value.shape


def test_manifold_dimension_validation() -> None:
    with pytest.raises(ValueError):
        InvertibleSlowManifold(2, 3)


def test_scale_manifolds() -> None:
    model = ScaleManifolds(16, 32, 2)
    value = torch.randn(10, 16)
    outputs = model(value, value, value)
    assert all(output.shape == (10, 3) for output in outputs)


def test_atlas_output() -> None:
    model = BifurcationAtlas((16, 8), 3)
    logits = model(torch.randn(10), torch.randint(0, 3, (10,)), torch.randn(10))
    assert logits.shape == (10, 3)
    assert model.predict(torch.randn(10), torch.zeros(10), torch.randn(10)).shape == (10,)


def test_transition_head() -> None:
    assert TransitionHead(16)(torch.randn(10, 16)).shape == (10,)


def test_losses() -> None:
    value = torch.randn(20, 8)
    reconstruction = value + 0.1
    coordinate = torch.randn(20, 3)
    total, terms = manifold_objective(value, reconstruction, coordinate)
    assert total > 0
    assert set(terms) == {"total", "reconstruction", "normal_form"}
    assert reconstruction_loss(value, value) == 0


def test_classification_losses() -> None:
    logits = torch.randn(20, 3)
    targets = torch.randint(0, 3, (20,))
    assert atlas_loss(logits, targets) > 0
    assert focal_loss(logits, targets) > 0


def test_binary_confusion() -> None:
    probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
    target = torch.tensor([1, 0, 1, 0])
    assert binary_confusion(probability, target) == (1, 1, 1, 1)


def test_perfect_binary_metrics() -> None:
    probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
    target = torch.tensor([1, 1, 0, 0])
    result = binary_metrics(probability, target)
    assert result.auroc == 1.0
    assert result.accuracy == 1.0


def test_calibration_metrics_bounds() -> None:
    probability = torch.rand(100)
    target = torch.randint(0, 2, (100,))
    assert 0 <= expected_calibration_error(probability, target) <= 1
    assert 0 <= brier_score(probability, target) <= 1


def test_kappa_identity() -> None:
    labels = torch.tensor([0, 1, 2, 1, 0])
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)


def test_clinical_translation() -> None:
    value = clinical_impact_per_thousand(0.95, 0.80, 0.1)
    assert value == pytest.approx(135.0)
    assert number_needed(value) == 8


def test_decision_curve_shape() -> None:
    output = decision_curve(
        torch.rand(100), torch.randint(0, 2, (100,)), torch.linspace(0.05, 0.5, 10)
    )
    assert output.shape == (10,)


def test_lead_time() -> None:
    predicted = torch.tensor([False, True, False, False, True, False])
    event = torch.tensor([False, False, False, True, False, True])
    assert torch.equal(lead_time(predicted, event), torch.tensor([2.0, 1.0]))


def test_bootstrap_auc_interval() -> None:
    target = np.array([0] * 50 + [1] * 50)
    probability = np.linspace(0, 1, 100)
    result = bootstrap_auc(target, probability, 100, 4)
    assert result.estimate == 1.0
    assert result.lower == 1.0


def test_paired_bootstrap() -> None:
    target = np.array([0] * 50 + [1] * 50)
    strong = np.linspace(0, 1, 100)
    weak = np.random.default_rng(4).random(100)
    result = paired_bootstrap_auc(target, strong, weak, 100, 4)
    assert result.difference > 0


def test_holm_adjustment() -> None:
    adjusted, rejected = holm_bonferroni(np.array([0.001, 0.01, 0.04]))
    assert np.all(adjusted >= np.array([0.001, 0.01, 0.04]))
    assert rejected[0]


def test_heterogeneity_identical() -> None:
    result = heterogeneity(np.array([0.9, 0.9, 0.9]), np.array([0.01, 0.01, 0.01]))
    assert result.q == pytest.approx(0.0)
    assert result.i_squared == 0.0


def test_fixed_effect_pool() -> None:
    estimate, error = fixed_effect_pool(np.array([0.8, 0.9]), np.array([0.01, 0.01]))
    assert estimate == pytest.approx(0.85)
    assert error > 0


def test_config_loading() -> None:
    path = Path(__file__).parents[1] / "configs" / "test.yaml"
    config = load_config(path, ("training.epochs=2",))
    assert config.training.epochs == 2
    assert config.model.manifold_dim == 3
