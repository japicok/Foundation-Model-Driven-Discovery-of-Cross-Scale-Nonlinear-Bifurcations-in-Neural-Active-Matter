from __future__ import annotations

import torch
import torch.nn.functional as functional


def temporal_derivative(coordinate: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    if coordinate.shape[0] < 2:
        return torch.zeros_like(coordinate)
    derivative = torch.empty_like(coordinate)
    derivative[0] = (coordinate[1] - coordinate[0]) / delta
    derivative[-1] = (coordinate[-1] - coordinate[-2]) / delta
    if coordinate.shape[0] > 2:
        derivative[1:-1] = (coordinate[2:] - coordinate[:-2]) / (2.0 * delta)
    return derivative


def lowpass(value: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return value
    padding = window // 2
    flat = value.transpose(0, -1).reshape(-1, 1, value.shape[0])
    padded = functional.pad(flat, (padding, padding), mode="replicate")
    kernel = torch.ones(1, 1, window, device=value.device, dtype=value.dtype) / window
    filtered = functional.conv1d(padded, kernel)
    filtered = filtered[..., : value.shape[0]]
    return filtered.reshape(*value.transpose(0, -1).shape).transpose(0, -1)


def estimate_control(
    normal_coordinate: torch.Tensor, delta: float = 1.0, window: int = 15
) -> torch.Tensor:
    derivative = temporal_derivative(normal_coordinate, delta)
    return lowpass(normal_coordinate.square() + derivative, window)


def normal_form_residual(
    normal_coordinate: torch.Tensor, control: torch.Tensor, delta: float = 1.0
) -> torch.Tensor:
    derivative = temporal_derivative(normal_coordinate, delta)
    return derivative - (control - normal_coordinate.square())


def normal_form_loss(
    normal_coordinate: torch.Tensor, control: torch.Tensor | None = None, delta: float = 1.0
) -> torch.Tensor:
    selected = control if control is not None else estimate_control(normal_coordinate, delta)
    return normal_form_residual(normal_coordinate, selected, delta).square().mean()


def slowing_indicators(coordinate: torch.Tensor, window: int) -> dict[str, torch.Tensor]:
    count = coordinate.shape[0]
    variance = torch.zeros_like(coordinate)
    autocorrelation = torch.zeros_like(coordinate)
    skewness = torch.zeros_like(coordinate)
    for index in range(count):
        start = max(0, index - window + 1)
        values = coordinate[start : index + 1]
        centered = values - values.mean()
        variance[index] = centered.square().mean()
        scale = variance[index].sqrt().clamp_min(1e-8)
        skewness[index] = (centered / scale).pow(3).mean()
        if values.shape[0] > 1:
            numerator = (centered[:-1] * centered[1:]).sum()
            denominator = centered.square().sum().clamp_min(1e-8)
            autocorrelation[index] = numerator / denominator
    return {"variance": variance, "autocorrelation": autocorrelation, "skewness": skewness}
