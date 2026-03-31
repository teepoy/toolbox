"""Validation loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.eval.metrics import topk_accuracy


@dataclass(slots=True)
class ValMetrics:
    loss: float
    top1: float
    top5: float


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device
) -> ValMetrics:
    model.eval()
    loss_sum = 0.0
    n = 0
    t1_sum = 0.0
    t5_sum = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        acc = topk_accuracy(logits, y, ks=(1, 5))
        bs = x.size(0)
        n += bs
        loss_sum += float(loss.item()) * bs
        t1_sum += acc[1] * bs
        t5_sum += acc[5] * bs
    if n == 0:
        return ValMetrics(loss=0.0, top1=0.0, top5=0.0)
    return ValMetrics(loss=loss_sum / n, top1=t1_sum / n, top5=t5_sum / n)
