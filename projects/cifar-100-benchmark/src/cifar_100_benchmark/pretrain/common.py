"""Common components for SSL pretraining methods."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import inf

import torch
from torch import nn
from omegaconf import DictConfig


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


@dataclass(slots=True)
class EarlyStopper:
    enabled: bool
    min_epochs: int
    patience: int
    min_delta: float
    best_loss: float = inf
    best_epoch: int = 0
    bad_epochs: int = 0

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "EarlyStopper":
        return cls(
            enabled=bool(cfg.pretrain.get("early_stop_enabled", False)),
            min_epochs=int(cfg.pretrain.get("early_stop_min_epochs", 10)),
            patience=int(cfg.pretrain.get("early_stop_patience", 5)),
            min_delta=float(cfg.pretrain.get("early_stop_min_delta", 0.005)),
        )

    def step(self, epoch: int, loss_value: float) -> bool:
        if loss_value < self.best_loss - self.min_delta:
            self.best_loss = loss_value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return (
            self.enabled
            and epoch >= self.min_epochs
            and self.bad_epochs >= self.patience
        )
