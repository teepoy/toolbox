from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str = "configs/base.yaml") -> DictConfig:
    cfg = OmegaConf.load(path)
    _resolve_paths(cfg)
    return cfg


def _resolve_paths(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    cache_path = Path(cfg.paths.cache_parquet)
    plots_dir = Path(cfg.paths.plots_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
