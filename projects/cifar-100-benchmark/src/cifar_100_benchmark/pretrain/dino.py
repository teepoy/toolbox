"""DINO-style pretraining trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.contrastive import dino_loss
from cifar_100_benchmark.models.builders import build_backbone
from cifar_100_benchmark.pretrain.common import MLP, make_ema_copy, update_momentum
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


class DINOModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        out_dim = int(backbone.out_dim)
        self.head = MLP(out_dim, out_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone.forward_features(x))


def run_dino(
    cfg: DictConfig, loader: DataLoader, device: torch.device, out_dir: Path
) -> Path:
    student = DINOModel(
        build_backbone(cfg.model.backbone).to(device),
        proj_dim=int(cfg.pretrain.proj_dim),
    ).to(device)
    teacher_backbone = make_ema_copy(student.backbone).to(device)
    teacher_head = make_ema_copy(student.head).to(device)

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    logger = JsonlLogger(out_dir / "metrics.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg.pretrain.epochs) + 1):
        loss_sum = 0.0
        n = 0
        for x1, x2, _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            s1 = student(x1)
            s2 = student(x2)
            with torch.no_grad():
                t1 = teacher_head(teacher_backbone.forward_features(x1))
                t2 = teacher_head(teacher_backbone.forward_features(x2))
            loss = 0.5 * (
                dino_loss(
                    s1,
                    t2,
                    float(cfg.pretrain.temp_student),
                    float(cfg.pretrain.temp_teacher),
                )
                + dino_loss(
                    s2,
                    t1,
                    float(cfg.pretrain.temp_student),
                    float(cfg.pretrain.temp_teacher),
                )
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            update_momentum(
                student.backbone, teacher_backbone, float(cfg.pretrain.momentum)
            )
            update_momentum(student.head, teacher_head, float(cfg.pretrain.momentum))

            bs = x1.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs

        row = {
            "epoch": epoch,
            "ssl_loss": round(loss_sum / max(1, n), 4),
            "method": "dino",
        }
        logger.log(row)
        print_metrics_table("SSL DINO", [row])

    ckpt = out_dir / "backbone.pt"
    torch.save(
        {"method": "dino", "backbone_state_dict": student.backbone.state_dict()}, ckpt
    )
    return ckpt
