from __future__ import annotations

import torch
from torch import nn


class BifurcationAtlas(nn.Module):
    labels = ("fold", "hopf", "pitchfork")

    def __init__(self, hidden: tuple[int, ...] = (128, 64), classes: int = 3) -> None:
        super().__init__()
        dimensions = (3, *hidden, classes)
        layers: list[nn.Module] = []
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend([nn.Linear(input_dim, output_dim), nn.GELU()])
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

    def forward(
        self, control: torch.Tensor, receptor: torch.Tensor, susceptibility: torch.Tensor
    ) -> torch.Tensor:
        features = torch.stack([control, receptor.float(), susceptibility], dim=-1)
        return self.network(features)

    def predict(
        self, control: torch.Tensor, receptor: torch.Tensor, susceptibility: torch.Tensor
    ) -> torch.Tensor:
        return self(control, receptor, susceptibility).argmax(dim=-1)


class TransitionHead(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, dimension // 2),
            nn.GELU(),
            nn.Linear(dimension // 2, 1),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.network(embedding).squeeze(-1)
