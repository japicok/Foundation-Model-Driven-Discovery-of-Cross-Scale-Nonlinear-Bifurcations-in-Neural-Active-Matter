from __future__ import annotations

import torch

from neural_bifurcation_atlas.types import AlignmentResult


def center_embedding(embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = embedding.mean(dim=0, keepdim=True)
    return embedding - mean, mean


def orthogonal_procrustes(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if source.shape != target.shape:
        raise ValueError("source and target shapes must match")
    covariance = source.transpose(-2, -1) @ target
    left, _, right = torch.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right
    if torch.linalg.det(rotation) < 0:
        left = left.clone()
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def generalized_procrustes(
    embeddings: tuple[torch.Tensor, ...],
    iterations: int = 20,
    tolerance: float = 1e-7,
) -> AlignmentResult:
    if len(embeddings) < 2:
        raise ValueError("at least two embeddings are required")
    shape = embeddings[0].shape
    if any(value.shape != shape for value in embeddings):
        raise ValueError("all embeddings must have the same shape")
    centered = tuple(center_embedding(value)[0] for value in embeddings)
    reference = centered[0]
    rotations: tuple[torch.Tensor, ...] = tuple(
        torch.eye(shape[-1], device=reference.device, dtype=reference.dtype) for _ in embeddings
    )
    aligned = centered
    for _ in range(iterations):
        rotations = tuple(orthogonal_procrustes(value, reference) for value in centered)
        aligned = tuple(
            value @ rotation for value, rotation in zip(centered, rotations, strict=True)
        )
        updated = torch.stack(aligned).mean(dim=0)
        updated = updated / updated.norm().clamp_min(1e-12) * reference.norm().clamp_min(1e-12)
        change = (updated - reference).norm() / reference.norm().clamp_min(1e-12)
        reference = updated
        if float(change) <= tolerance:
            break
    residual = torch.stack([(value - reference).square().mean() for value in aligned]).mean().sqrt()
    return AlignmentResult(aligned, rotations, residual)


def apply_rotations(
    embeddings: tuple[torch.Tensor, ...], rotations: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    if len(embeddings) != len(rotations):
        raise ValueError("embedding and rotation counts differ")
    return tuple(
        center_embedding(value)[0] @ rotation
        for value, rotation in zip(embeddings, rotations, strict=True)
    )
