"""View fusion network: images -> feats -> self-attn -> head."""

from __future__ import annotations

import torch
from torch import nn


class ViewFusionHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        attn_dim: int = 256,
        num_views: int = 2,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_views = num_views
        self.proj = nn.Linear(in_dim, attn_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attn_dim,
            nhead=num_heads,
            dim_feedforward=attn_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(attn_dim)
        self.head = nn.Linear(attn_dim, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.ndim != 3:
            raise ValueError("Expected feats shape [batch, views, dim]")
        x = self.proj(feats)
        x = self.attn(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
