"""BYOL pretraining trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.byol import byol_loss
from cifar_100_benchmark.models.builders import build_backbone
from cifar_100_benchmark.pretrain.common import MLP, make_ema_copy, update_momentum
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


class BYOLModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int, pred_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        out_dim = int(backbone.out_dim)
        self.projector = MLP(out_dim, out_dim, proj_dim)
        self.predictor = MLP(proj_dim, out_dim, pred_dim)

    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.backbone.forward_features(x))
        p = self.predictor(z)
        return p

    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone.forward_features(x))


def run_byol(
    cfg: DictConfig,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> Path:
    backbone = build_backbone(cfg.model.backbone).to(device)
    model = BYOLModel(
        backbone=backbone,
        proj_dim=int(cfg.pretrain.proj_dim),
        pred_dim=int(cfg.pretrain.pred_dim),
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

    momentum = float(cfg.pretrain.momentum)
    epochs = int(cfg.pretrain.epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for x1, x2, _ in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            p1 = model.forward_online(x1)
            p2 = model.forward_online(x2)
            with torch.no_grad():
                z1 = target_projector(target_backbone.forward_features(x1))
                z2 = target_projector(target_backbone.forward_features(x2))
            loss = 0.5 * (byol_loss(p1, z2) + byol_loss(p2, z1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            update_momentum(model.backbone, target_backbone, momentum)
            update_momentum(model.projector, target_projector, momentum)

            bs = x1.size(0)
            n += bs
            loss_sum += float(loss.item()) * bs

        row = {
            "epoch": epoch,
            "ssl_loss": round(loss_sum / max(1, n), 4),
            "method": "byol",
        }
        logger.log(row)
        print_metrics_table("SSL BYOL", [row])

    ckpt = out_dir / "backbone.pt"
    torch.save(
        {
            "method": "byol",
            "backbone_state_dict": model.backbone.state_dict(),
        },
        ckpt,
    )
    return ckpt
