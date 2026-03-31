"""Backbone protocol and helpers."""

from __future__ import annotations

from typing import Protocol

import torch


class Backbone(Protocol):
    out_dim: int

    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
