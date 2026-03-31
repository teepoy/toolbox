"""Contrastive and distillation losses for SSL methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    q: torch.Tensor, k: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.t() / temperature
    labels = torch.arange(q.shape[0], device=q.device)
    return F.cross_entropy(logits, labels)


def supcon_loss(
    features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1
) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    logits = features @ features.t() / temperature
    mask = labels.unsqueeze(1) == labels.unsqueeze(0)
    logits_mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_mask = mask & logits_mask
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
    return -pos_log_prob.mean()


def dino_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temp_student: float,
    temp_teacher: float,
) -> torch.Tensor:
    s = F.log_softmax(student / temp_student, dim=-1)
    t = F.softmax(teacher.detach() / temp_teacher, dim=-1)
    return -(t * s).sum(dim=-1).mean()
