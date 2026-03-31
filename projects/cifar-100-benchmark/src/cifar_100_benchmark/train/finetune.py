"""Supervised fine-tuning loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from cifar_100_benchmark.losses.supervised import build_supervised_loss
from cifar_100_benchmark.models.builders import build_classifier
from cifar_100_benchmark.train.validate import validate
from cifar_100_benchmark.utils.logging import JsonlLogger, print_metrics_table


@dataclass(slots=True)
class FinetuneResult:
    best_val_top1: float
    ckpt_path: Path


def _build_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optim.name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
        )
    if cfg.optim.name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.optim.lr),
            momentum=float(cfg.optim.momentum),
            weight_decay=float(cfg.optim.weight_decay),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.name}")


def _build_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer):
    if cfg.scheduler.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(cfg.train.epochs)
        )
    if cfg.scheduler.name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(x) for x in cfg.scheduler.milestones],
            gamma=float(cfg.scheduler.gamma),
        )
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")


def finetune(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    init_ckpt: str | None = None,
) -> FinetuneResult:
    model = build_classifier(cfg.model).to(device)
    if init_ckpt:
        payload = torch.load(init_ckpt, map_location="cpu")
        state = payload.get("backbone_state_dict", payload)
        model.backbone.load_state_dict(state, strict=False)
    loss_fn = build_supervised_loss(str(cfg.loss.name))
    optimizer = _build_optimizer(cfg, model)
    scheduler = _build_scheduler(cfg, optimizer)
    logger = JsonlLogger(out_dir / "metrics.jsonl")

    best_top1 = -1.0
    best_path = out_dir / "best.pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(cfg.train.epochs) + 1):
        model.train()
        run_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            n += bs
            run_loss += float(loss.item()) * bs
        scheduler.step()

        val_m = validate(model, val_loader, loss_fn, device)
        train_loss = run_loss / max(1, n)
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_m.loss, 4),
            "val_top1": round(val_m.top1, 2),
            "val_top5": round(val_m.top5, 2),
        }
        logger.log(row)
        print_metrics_table(title="Finetune Epoch", rows=[row])

        if val_m.top1 > best_top1:
            best_top1 = val_m.top1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone_state_dict": model.backbone.state_dict(),
                    "cfg": {"model": str(cfg.model.backbone.name)},
                    "best_val_top1": best_top1,
                },
                best_path,
            )

    return FinetuneResult(best_val_top1=best_top1, ckpt_path=best_path)
