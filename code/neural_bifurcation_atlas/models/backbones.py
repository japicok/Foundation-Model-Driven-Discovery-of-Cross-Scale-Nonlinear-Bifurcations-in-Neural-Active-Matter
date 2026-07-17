from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn


class EEGBackbone(nn.Module, ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def token_embeddings(self, signal: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.token_embeddings(signal)


class ConvTokenBackbone(EEGBackbone):
    def __init__(self, channels: int, dimension: int, kernel: int = 15, stride: int = 8) -> None:
        super().__init__()
        self.dimension = dimension
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, dimension // 2, kernel, stride=stride, padding=kernel // 2),
            nn.GELU(),
            nn.Conv1d(dimension // 2, dimension, 5, stride=2, padding=2),
            nn.GroupNorm(1, dimension),
            nn.GELU(),
        )

    @property
    def embedding_dim(self) -> int:
        return self.dimension

    def token_embeddings(self, signal: torch.Tensor) -> torch.Tensor:
        return self.encoder(signal).transpose(1, 2)


class SpectralTokenBackbone(EEGBackbone):
    def __init__(self, channels: int, dimension: int, bins: int = 64) -> None:
        super().__init__()
        self.dimension = dimension
        self.bins = bins
        self.projection = nn.Sequential(
            nn.Linear(channels * 2, dimension),
            nn.LayerNorm(dimension),
            nn.GELU(),
            nn.Linear(dimension, dimension),
        )

    @property
    def embedding_dim(self) -> int:
        return self.dimension

    def token_embeddings(self, signal: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(signal, dim=-1).abs()[..., : self.bins]
        spectrum = spectrum.transpose(1, 2)
        phase = torch.angle(torch.fft.rfft(signal, dim=-1))[..., : self.bins].transpose(1, 2)
        return self.projection(torch.cat([torch.log1p(spectrum), phase], dim=-1))


class TemporalPatchBackbone(EEGBackbone):
    def __init__(self, channels: int, dimension: int, patch: int = 32) -> None:
        super().__init__()
        self.dimension = dimension
        self.patch = patch
        self.projection = nn.Linear(channels * patch, dimension)
        layer = nn.TransformerEncoderLayer(
            dimension, 4, dimension * 2, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, 2)

    @property
    def embedding_dim(self) -> int:
        return self.dimension

    def token_embeddings(self, signal: torch.Tensor) -> torch.Tensor:
        usable = signal.shape[-1] // self.patch * self.patch
        patches = signal[..., :usable].unfold(-1, self.patch, self.patch)
        patches = patches.permute(0, 2, 1, 3).flatten(2)
        return self.transformer(self.projection(patches))


class BackboneEnsemble(nn.Module):
    def __init__(self, backbones: Mapping[str, EEGBackbone], shared_dimension: int) -> None:
        super().__init__()
        if len(backbones) < 2:
            raise ValueError("ensemble needs at least two backbones")
        self.names = tuple(backbones)
        self.backbones = nn.ModuleDict(backbones)
        self.adapters = nn.ModuleDict(
            {
                name: nn.Linear(backbone.embedding_dim, shared_dimension)
                for name, backbone in backbones.items()
            }
        )

    def forward(self, signal: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        for name in self.names:
            tokens = self.backbones[name].token_embeddings(signal)
            outputs[name] = self.adapters[name](tokens).mean(dim=1)
        return outputs

    def stacked(self, signal: torch.Tensor) -> torch.Tensor:
        outputs = self(signal)
        return torch.stack([outputs[name] for name in self.names], dim=0)


def build_local_ensemble(channels: int, dimension: int) -> BackboneEnsemble:
    backbones: dict[str, EEGBackbone] = {
        "spectral": SpectralTokenBackbone(channels, dimension),
        "temporal": TemporalPatchBackbone(channels, dimension),
        "convolutional": ConvTokenBackbone(channels, dimension),
    }
    return BackboneEnsemble(backbones, dimension)
