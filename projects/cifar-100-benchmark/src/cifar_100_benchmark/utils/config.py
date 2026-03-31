"""OmegaConf config loading/composition helpers."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def _merge_group_config(
    base: DictConfig, config_dir: Path, key: str, subdir: str
) -> DictConfig:
    group_name = base.get(key)
    if not group_name:
        return base
    group_path = config_dir / subdir / f"{group_name}.yaml"
    if not group_path.exists():
        raise FileNotFoundError(f"Missing config group file: {group_path}")
    group_cfg = OmegaConf.load(group_path)
    return OmegaConf.merge(base, group_cfg)


def load_config(
    config_dir: Path, experiment_name: str, overrides: list[str] | None = None
) -> DictConfig:
    base = OmegaConf.load(config_dir / "default.yaml")
    base = _merge_group_config(base, config_dir, "data_config", "data")
    base = _merge_group_config(base, config_dir, "model_config", "model")
    base = _merge_group_config(base, config_dir, "loss_config", "loss")
    base = _merge_group_config(base, config_dir, "optim_config", "optim")
    base = _merge_group_config(base, config_dir, "scheduler_config", "scheduler")
    base = _merge_group_config(base, config_dir, "pretrain_config", "pretrain")
    base = _merge_group_config(base, config_dir, "runtime_config", "runtime")
    exp_cfg = OmegaConf.load(config_dir / "experiment" / f"{experiment_name}.yaml")
    cfg = OmegaConf.merge(base, exp_cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
