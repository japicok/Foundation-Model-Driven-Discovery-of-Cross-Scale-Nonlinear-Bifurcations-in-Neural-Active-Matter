from __future__ import annotations

import torch
from torch import nn


class ResidualTransform(nn.Module):
    def __init__(self, dimension: int, hidden: int, contraction: float = 0.8) -> None:
        super().__init__()
        self.contraction = contraction
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dimension),
        )
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.2)
                nn.init.zeros_(module.bias)

    def residual(self, value: torch.Tensor) -> torch.Tensor:
        return self.contraction * self.network(value)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.residual(value)

    def inverse(self, value: torch.Tensor, iterations: int = 30) -> torch.Tensor:
        estimate = value
        for _ in range(iterations):
            estimate = value - self.residual(estimate)
        return estimate


class InvertibleSlowManifold(nn.Module):
    def __init__(
        self, embedding_dim: int, manifold_dim: int = 3, hidden_dim: int = 512, blocks: int = 2
    ) -> None:
        super().__init__()
        if manifold_dim > embedding_dim:
            raise ValueError("manifold dimension exceeds embedding dimension")
        self.embedding_dim = embedding_dim
        self.manifold_dim = manifold_dim
        self.transforms = nn.ModuleList(
            [ResidualTransform(embedding_dim, hidden_dim) for _ in range(blocks)]
        )

    def transform(self, embedding: torch.Tensor) -> torch.Tensor:
        value = embedding
        for transform in self.transforms:
            value = transform(value)
        return value

    def inverse_transform(self, transformed: torch.Tensor) -> torch.Tensor:
        value = transformed
        for transform in reversed(self.transforms):
            value = transform.inverse(value)
        return value

    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.transform(embedding)[..., : self.manifold_dim]

    def decode(
        self, coordinate: torch.Tensor, transverse: torch.Tensor | None = None
    ) -> torch.Tensor:
        if transverse is None:
            shape = (*coordinate.shape[:-1], self.embedding_dim - self.manifold_dim)
            transverse = torch.zeros(shape, device=coordinate.device, dtype=coordinate.dtype)
        return self.inverse_transform(torch.cat([coordinate, transverse], dim=-1))

    def reconstruct(self, embedding: torch.Tensor) -> torch.Tensor:
        transformed = self.transform(embedding)
        coordinate = transformed[..., : self.manifold_dim]
        zeros = torch.zeros_like(transformed[..., self.manifold_dim :])
        return self.inverse_transform(torch.cat([coordinate, zeros], dim=-1))

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coordinate = self.encode(embedding)
        reconstruction = self.reconstruct(embedding)
        return coordinate, reconstruction


class ScaleManifolds(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, blocks: int) -> None:
        super().__init__()
        self.neuronal = InvertibleSlowManifold(embedding_dim, 3, hidden_dim, blocks)
        self.population = InvertibleSlowManifold(embedding_dim, 3, hidden_dim, blocks)
        self.brain = InvertibleSlowManifold(embedding_dim, 3, hidden_dim, blocks)

    def forward(
        self,
        neuronal: torch.Tensor,
        population: torch.Tensor,
        brain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.neuronal.encode(neuronal),
            self.population.encode(population),
            self.brain.encode(brain),
        )
