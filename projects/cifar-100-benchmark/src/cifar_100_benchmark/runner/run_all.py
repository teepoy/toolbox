"""Run smoke first, then selected full experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from cifar_100_benchmark.runner.experiment import run_experiment
from cifar_100_benchmark.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke then full experiment")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--full-experiment", default="full_reduced")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)

    if not args.skip_smoke:
        smoke_name = "smoke_128" if args.full_experiment.endswith("2res") else "smoke"
        smoke_cfg = load_config(config_dir, smoke_name, args.override)
        run_experiment(smoke_cfg)

    full_cfg = load_config(config_dir, args.full_experiment, args.override)
    run_experiment(full_cfg)


if __name__ == "__main__":
    main()
