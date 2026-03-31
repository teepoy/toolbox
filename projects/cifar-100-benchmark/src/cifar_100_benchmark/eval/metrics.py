"""Evaluation metrics."""

from __future__ import annotations

import torch


def topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)
) -> dict[int, float]:
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out: dict[int, float] = {}
    batch = targets.size(0)
    for k in ks:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out[k] = float(correct_k.mul_(100.0 / batch).item())
    return out
