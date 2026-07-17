from __future__ import annotations

import torch
import torch.nn.functional as functional

from neural_bifurcation_atlas.science.normal_form import normal_form_loss


def reconstruction_loss(embedding: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
    return functional.mse_loss(reconstruction, embedding)


def manifold_objective(
    embedding: torch.Tensor,
    reconstruction: torch.Tensor,
    coordinate: torch.Tensor,
    normal_form_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    reconstruction_term = reconstruction_loss(embedding, reconstruction)
    normal_term = normal_form_loss(coordinate[..., 0])
    total = reconstruction_term + normal_form_weight * normal_term
    return total, {
        "total": total.detach(),
        "reconstruction": reconstruction_term.detach(),
        "normal_form": normal_term.detach(),
    }


def atlas_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return functional.cross_entropy(logits, targets)


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    cross_entropy = functional.cross_entropy(logits, targets, reduction="none")
    probability = torch.exp(-cross_entropy)
    return ((1.0 - probability).pow(gamma) * cross_entropy).mean()
