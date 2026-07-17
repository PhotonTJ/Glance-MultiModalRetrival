from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualQuantizer(nn.Module):
    """Small RQ-VAE-style residual quantizer for hierarchical semantic IDs."""

    def __init__(self, input_dim: int = 256, latent_dim: int = 64, levels: int = 3, codebook_size: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.GELU(), nn.Linear(128, input_dim))
        self.codebooks = nn.Parameter(torch.randn(levels, codebook_size, latent_dim) * 0.02)

    def quantize(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = latent
        quantized = torch.zeros_like(latent)
        codes, usage_losses = [], []
        for codebook in self.codebooks:
            distances = torch.cdist(residual, codebook)
            indices = distances.argmin(dim=1)
            selected = F.embedding(indices, codebook)
            # A differentiable soft assignment keeps codebook use broad.  Without
            # it, a few code tuples dominate even when nominal capacity is large.
            probabilities = F.softmax(-distances / 0.10, dim=1)
            usage = probabilities.mean(dim=0).clamp_min(1e-8)
            usage_losses.append((usage * (usage * codebook.shape[0]).log()).sum())
            quantized = quantized + selected
            residual = residual - selected
            codes.append(indices)
        return quantized, torch.stack(codes, dim=1), torch.stack(usage_losses).mean()

    def forward(self, vectors: torch.Tensor, beta: float = 0.25, usage_weight: float = 0.02) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        latent = self.encoder(vectors)
        quantized, codes, usage_loss = self.quantize(latent)
        reconstruction = self.decoder(latent + (quantized - latent).detach())
        reconstruction_loss = F.mse_loss(reconstruction, vectors)
        codebook_loss = F.mse_loss(quantized, latent.detach())
        commitment_loss = F.mse_loss(latent, quantized.detach())
        loss = reconstruction_loss + codebook_loss + beta * commitment_loss + usage_weight * usage_loss
        return reconstruction, codes, {"total": loss, "reconstruction": reconstruction_loss, "codebook": codebook_loss, "commitment": commitment_loss, "usage": usage_loss}

    def soft_quantized_prefixes(self, vectors: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
        """Differentiable cumulative code prefixes used by attribute supervision."""
        residual = self.encoder(vectors)
        cumulative = torch.zeros_like(residual)
        prefixes = []
        for codebook in self.codebooks:
            probabilities = F.softmax(-torch.cdist(residual, codebook) / temperature, dim=1)
            selected = probabilities @ codebook
            cumulative = cumulative + selected
            prefixes.append(cumulative)
            residual = residual - selected
        return torch.stack(prefixes, dim=1)

    @torch.no_grad()
    def semantic_ids(self, vectors: torch.Tensor) -> torch.Tensor:
        return self.quantize(self.encoder(vectors))[1]
