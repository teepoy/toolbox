"""Dispatcher for SSL pretraining methods."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from cifar_100_benchmark.pretrain.byol import run_byol
from cifar_100_benchmark.pretrain.dino import run_dino
from cifar_100_benchmark.pretrain.mocov3 import run_mocov3
from cifar_100_benchmark.pretrain.supcon import run_supcon


def run_pretrain(
    cfg: DictConfig, loader: DataLoader, device: torch.device, out_dir: Path
) -> Path:
    method = str(cfg.pretrain.name)
    if method == "byol":
        return run_byol(cfg, loader, device, out_dir)
    if method == "mocov3":
        return run_mocov3(cfg, loader, device, out_dir)
    if method == "supcon":
        return run_supcon(cfg, loader, device, out_dir)
    if method == "dino":
        return run_dino(cfg, loader, device, out_dir)
    raise ValueError(f"Unsupported pretrain method: {method}")
