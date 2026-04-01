from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf


def _default_config() -> DictConfig:
    return OmegaConf.create(
        {
            "experiment": {
                "name": "smoke",
                "output_dir": "outputs/default",
                "seed": 42,
            },
            "dataset": {
                "name": "nkirschi/oxford-flowers",
                "train_split": "train",
                "val_split": "validation",
                "test_split": "test",
                "image_key": "image",
                "label_key": "label",
                "train_subset": None,
                "val_subset": None,
                "test_subset": None,
                "val_fraction": 0.1,
                "test_fraction": 0.1,
                "num_workers": 2,
                "class_split": {
                    "enabled": False,
                    "mode": "first_half",
                    "train_fraction": 0.5,
                    "seed": 42,
                },
            },
            "model": {
                "backbone_name": "convnextv2_atto",
                "pretrained": True,
                "allow_random_init_fallback": False,
                "projection_dim": 256,
                "projection_hidden_dim": 512,
                "normalize_embeddings": True,
            },
            "train": {
                "image_size": 224,
                "batch_size": 16,
                "epochs": 1,
                "lr": 5e-5,
                "weight_decay": 1e-4,
                "temperature": 0.07,
                "log_every_steps": 10,
                "save_every_epochs": 1,
                "use_amp": True,
                "self_supervised": {
                    "enabled": False,
                    "epochs": 20,
                    "lr": 1e-4,
                    "weight_decay": 1e-4,
                    "temperature": 0.07,
                },
            },
            "benchmark": {
                "batch_size": 32,
                "knn_k_values": [1, 5],
                "reference_split": "train",
                "eval_split": "validation",
                "linear_probe": {
                    "epochs": 25,
                    "lr": 1e-2,
                    "weight_decay": 1e-4,
                    "batch_size": 64,
                },
            },
            "runtime": {"device": "auto"},
        }
    )


def load_config(
    config_path: str | Path, overrides: list[str] | None = None
) -> DictConfig:
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    base = _default_config()
    loaded = OmegaConf.load(config_file)
    override_cfg = OmegaConf.from_dotlist(overrides or [])
    config = cast(DictConfig, OmegaConf.merge(base, loaded, override_cfg))
    OmegaConf.set_struct(config, True)
    return config


def resolve_output_dir(config: DictConfig) -> Path:
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
