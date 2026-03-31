"""SupCon pretraining trainer."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.contrastive import supcon_loss
from cifar_100_benchmark.models.builders import build_backbone
from cifar_100_benchmark.pretrain.common import MLP
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


class SupConModel(nn.Module):
    def __init__(self, backbone: nn.Module, proj_dim: int) -> None:
        super().__init__()
        self.backbone = backbone
        out_dim = int(backbone.out_dim)
        self.projector = MLP(out_dim, out_dim, proj_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone.forward_features(x))


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

        row = {
            "epoch": epoch,
            "ssl_loss": round(loss_sum / max(1, n), 4),
            "method": "supcon",
        }
        logger.log(row)
        print_metrics_table("SSL SupCon", [row])

    ckpt = out_dir / "backbone.pt"
    torch.save(
        {"method": "supcon", "backbone_state_dict": model.backbone.state_dict()}, ckpt
    )
    return ckpt
