"""Common components for SSL pretraining methods."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def update_momentum(student: nn.Module, teacher: nn.Module, m: float) -> None:
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters(), strict=True):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


def make_ema_copy(module: nn.Module) -> nn.Module:
    teacher = deepcopy(module)
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher
