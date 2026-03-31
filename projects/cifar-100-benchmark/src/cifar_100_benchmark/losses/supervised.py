"""Supervised losses."""

from __future__ import annotations

from torch import nn


def build_supervised_loss(name: str = "cross_entropy") -> nn.Module:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported supervised loss: {name}")
