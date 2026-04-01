"""SupCon pretraining trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.contrastive import supcon_loss
from cifar_100_benchmark.models.builders import build_backbone
from cifar_100_benchmark.pretrain.common import EarlyStopper, MLP
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


class SupConModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        out_dim = int(cast(int, getattr(backbone, "out_dim")))
        self.projector = MLP(out_dim, out_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        feats = cast(Any, self.backbone).forward_features(x)
        return cast(MLP, self.projector).forward(feats)


def run_supcon(
    cfg: DictConfig, loader: DataLoader, device: torch.device, out_dir: Path
) -> Path:
    model = SupConModel(
        backbone=build_backbone(cfg.model.backbone).to(device),
        proj_dim=int(cfg.pretrain.proj_dim),
    ).to(device)
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
        for x1, x2, y in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z1 = model.encode(x1)
            z2 = model.encode(x2)
            feats = torch.cat([z1, z2], dim=0)
            labels = torch.cat([y, y], dim=0)
            loss = supcon_loss(
                feats, labels, temperature=float(cfg.pretrain.temperature)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = x1.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs

        epoch_loss = loss_sum / max(1, n)
        row = {
            "epoch": epoch,
            "ssl_loss": round(epoch_loss, 4),
            "method": "supcon",
        }
        logger.log(row)
        print_metrics_table("SSL SupCon", [row])
        if early_stopper.step(epoch, epoch_loss):
            stop_row = {
                "event": "early_stop",
                "method": "supcon",
                "epoch": epoch,
                "best_epoch": early_stopper.best_epoch,
                "best_ssl_loss": round(early_stopper.best_loss, 4),
            }
            logger.log(stop_row)
            print_metrics_table("SSL SupCon", [stop_row])
            break

    ckpt = out_dir / "backbone.pt"
    torch.save(
        {"method": "supcon", "backbone_state_dict": model.backbone.state_dict()}, ckpt
    )
    return ckpt
