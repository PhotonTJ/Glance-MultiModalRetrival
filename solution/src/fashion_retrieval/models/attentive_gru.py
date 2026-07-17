from __future__ import annotations

import torch
from torch import nn


class AttentionGRUComposer(nn.Module):
    """Compose an ordered sequence of frozen text-component embeddings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_sizes: dict[str, int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        representation_dim = hidden_dim * 2
        self.attentions = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False),
            )
            for name in output_sizes
        })
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            name: nn.Linear(representation_dim, size)
            for name, size in output_sizes.items()
        })

    def forward(
        self,
        components: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        states, _ = self.gru(components)
        logits = {}; attentions = {}
        for name, head in self.heads.items():
            attention_logits = self.attentions[name](states).squeeze(-1)
            attention_logits = attention_logits.masked_fill(~mask, torch.finfo(states.dtype).min)
            attention = torch.softmax(attention_logits, dim=1)
            pooled = torch.sum(states * attention.unsqueeze(-1), dim=1)
            logits[name] = head(self.dropout(pooled)); attentions[name] = attention
        return logits, attentions
