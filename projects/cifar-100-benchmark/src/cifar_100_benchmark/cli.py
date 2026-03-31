"""CLI entrypoint for benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from cifar_100_benchmark.runner.experiment import run_experiment
from cifar_100_benchmark.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR-100 few-shot benchmark runner")
    parser.add_argument(
        "--experiment",
        default="smoke",
        help="Experiment config name under configs/experiment",
    )
    parser.add_argument("--config-dir", default="configs", help="Config directory")
    parser.add_argument(
        "--override", action="append", default=[], help="OmegaConf dotlist override"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config_dir), args.experiment, args.override)
    OmegaConf.set_struct(cfg, False)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
