from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from .config import _resolve_paths, load_config


def load_config_v2(
    base_path: str = "configs/base.yaml", overlay_path: str = "configs/v2.yaml"
) -> DictConfig:
    base_cfg = load_config(base_path)
    overlay_cfg = OmegaConf.load(overlay_path)
    cfg = OmegaConf.merge(base_cfg, overlay_cfg)
    _resolve_paths(cfg)
    return cfg
