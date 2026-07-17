from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SiameseProjection(nn.Module):
    """One shared head for both Qwen text and image embeddings."""

    def __init__(self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(embeddings), dim=-1)


def distance_regression_loss(
    query: torch.Tensor,
    image: torch.Tensor,
    target_similarity: torch.Tensor,
    negative: torch.Tensor | None = None,
    margin: float = 0.2,
    variance_floor: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Plan loss: direct L2 target regression plus ranking and anti-collapse terms."""
    distance = torch.linalg.vector_norm(query - image, dim=-1)
    target_distance = 2.0 * (1.0 - target_similarity.clamp(0, 1))
    dist_loss = F.mse_loss(distance, target_distance)
    rank_loss = torch.zeros((), device=query.device)
    if negative is not None:
        negative_distance = torch.linalg.vector_norm(query - negative, dim=-1)
        rank_loss = F.relu(margin + distance - negative_distance).mean()
    joined = torch.cat([query, image], dim=0)
    var_loss = F.relu(variance_floor - joined.std(dim=0, unbiased=False)).square().mean()
    total = dist_loss + 0.25 * rank_loss + 0.05 * var_loss
    return total, {"distance": dist_loss, "ranking": rank_loss, "variance": var_loss}

