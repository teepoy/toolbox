"""MoCo v3 style pretraining trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.contrastive import info_nce_loss
from cifar_100_benchmark.models.builders import build_backbone
from cifar_100_benchmark.pretrain.common import (
    EarlyStopper,
    MLP,
    make_ema_copy,
    update_momentum,
)
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


class MoCoV3Model(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        out_dim = int(cast(int, getattr(backbone, "out_dim")))
        self.projector = MLP(out_dim, out_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = cast(Any, self.backbone).forward_features(x)
        return cast(MLP, self.projector).forward(feats)


def run_mocov3(
    cfg: DictConfig, loader: DataLoader, device: torch.device, out_dir: Path
) -> Path:
    model = MoCoV3Model(
        backbone=build_backbone(cfg.model.backbone).to(device),
        proj_dim=int(cfg.pretrain.proj_dim),
    ).to(device)
    target_backbone = make_ema_copy(model.backbone).to(device)
    target_projector = make_ema_copy(model.projector).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
    )
    logger = JsonlLogger(out_dir / "metrics.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    early_stopper = EarlyStopper.from_cfg(cfg)

    for epoch in range(1, int(cfg.pretrain.epochs) + 1):
        loss_sum = 0.0
        n = 0
        for x1, x2, _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            q1 = model.encode(x1)
            q2 = model.encode(x2)
            with torch.no_grad():
                f1 = cast(Any, target_backbone).forward_features(x1)
                f2 = cast(Any, target_backbone).forward_features(x2)
                k1 = cast(MLP, target_projector).forward(f1)
                k2 = cast(MLP, target_projector).forward(f2)
            loss = 0.5 * (
                info_nce_loss(q1, k2, temperature=float(cfg.pretrain.temperature))
                + info_nce_loss(q2, k1, temperature=float(cfg.pretrain.temperature))
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            update_momentum(
                model.backbone, target_backbone, float(cfg.pretrain.momentum)
            )
            update_momentum(
                model.projector, target_projector, float(cfg.pretrain.momentum)
            )

            bs = x1.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs

        epoch_loss = loss_sum / max(1, n)
        row = {
            "epoch": epoch,
            "ssl_loss": round(epoch_loss, 4),
            "method": "mocov3",
        }
        logger.log(row)
        print_metrics_table("SSL MoCoV3", [row])
        if early_stopper.step(epoch, epoch_loss):
            stop_row = {
                "event": "early_stop",
                "method": "mocov3",
                "epoch": epoch,
                "best_epoch": early_stopper.best_epoch,
                "best_ssl_loss": round(early_stopper.best_loss, 4),
            }
            logger.log(stop_row)
            print_metrics_table("SSL MoCoV3", [stop_row])
            break

    ckpt = out_dir / "backbone.pt"
    torch.save(
        {"method": "mocov3", "backbone_state_dict": model.backbone.state_dict()}, ckpt
    )
    return ckpt
