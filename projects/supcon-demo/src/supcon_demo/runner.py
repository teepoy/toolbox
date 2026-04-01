from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from supcon_demo.benchmark import run_benchmark
from supcon_demo.config import load_config, resolve_output_dir
from supcon_demo.data import load_data
from supcon_demo.model import SupConModel
from supcon_demo.training import train_self_supervised, train_supcon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SupCon finetuning and benchmark flow"
    )
    parser.add_argument(
        "--config", required=True, help="Path to an OmegaConf YAML config"
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in OmegaConf dotlist format, e.g. train.epochs=2",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _summarize_metrics(baseline_metrics: dict, finetuned_metrics: dict) -> dict:
    summary = {"baseline": baseline_metrics, "finetuned": finetuned_metrics}
    for section in ("knn", "linear_probe"):
        summary[f"delta_{section}"] = {}
        for key, value in finetuned_metrics[section].items():
            base_value = baseline_metrics[section].get(key)
            if base_value is not None:
                summary[f"delta_{section}"][key] = value - base_value
    return summary


def _benchmark_and_log(
    *,
    model,
    reference_loader,
    eval_loader,
    config,
    device: torch.device,
    output_path,
    stage_name: str,
) -> dict:
    metrics = run_benchmark(
        model=model,
        reference_loader=reference_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        output_path=output_path,
        stage_name=stage_name,
    )
    print(f"[{stage_name}] knn={metrics['knn']} linear_probe={metrics['linear_probe']}")
    return metrics


def _select_eval_loader(data_bundle: dict, split_name: str):
    split_to_loader = {
        "train": data_bundle["eval_train_loader"],
        "validation": data_bundle["val_loader"],
        "val": data_bundle["val_loader"],
        "test": data_bundle["test_loader"],
    }
    key = split_name.lower()
    if key not in split_to_loader:
        raise ValueError(
            f"Unsupported benchmark split {split_name!r}. Choose from train|validation|test."
        )
    return split_to_loader[key]


def main() -> None:
    args = parse_args()
    config = load_config(args.config, overrides=args.override)
    output_dir = resolve_output_dir(config)
    device = resolve_device(str(config.runtime.device))
    set_seed(int(config.experiment.seed))

    print(
        f"[setup] experiment={config.experiment.name} device={device.type} output_dir={output_dir}"
    )

    model = SupConModel(
        backbone_name=str(config.model.backbone_name),
        pretrained=bool(config.model.pretrained),
        projection_dim=int(config.model.projection_dim),
        projection_hidden_dim=int(config.model.projection_hidden_dim),
        normalize_embeddings=bool(config.model.normalize_embeddings),
        allow_random_init_fallback=bool(config.model.allow_random_init_fallback),
    ).to(device)

    data_bundle = load_data(config, model)
    print(
        "[data] "
        + " ".join(
            f"{key}={value}" for key, value in data_bundle["split_sizes"].items()
        )
        + f" num_classes={data_bundle['num_classes']}"
    )

    class_split_info = data_bundle.get("class_split")
    if class_split_info and bool(class_split_info.get("enabled")):
        print(f"[data] class_split={class_split_info}")

    reference_split = str(config.benchmark.reference_split)
    eval_split = str(config.benchmark.eval_split)
    reference_loader = _select_eval_loader(data_bundle, reference_split)
    eval_loader = _select_eval_loader(data_bundle, eval_split)
    print(f"[benchmark] reference_split={reference_split} eval_split={eval_split}")

    (output_dir / "resolved_config.yaml").write_text(
        OmegaConf.to_yaml(config), encoding="utf-8"
    )

    baseline_metrics = _benchmark_and_log(
        model=model,
        reference_loader=reference_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        output_path=output_dir / "baseline_metrics.json",
        stage_name="baseline",
    )

    self_supervised_metrics = None
    post_ssl_metrics = None
    if bool(config.train.self_supervised.enabled):
        self_supervised_metrics = train_self_supervised(
            model=model,
            train_loader=data_bundle["train_loader"],
            device=device,
            config=config,
            output_dir=output_dir,
        )
        post_ssl_metrics = _benchmark_and_log(
            model=model,
            reference_loader=reference_loader,
            eval_loader=eval_loader,
            config=config,
            device=device,
            output_path=output_dir / "post_self_supervised_metrics.json",
            stage_name="post_self_supervised",
        )

    training_metrics = train_supcon(
        model=model,
        train_loader=data_bundle["train_loader"],
        device=device,
        config=config,
        output_dir=output_dir,
    )

    finetuned_metrics = _benchmark_and_log(
        model=model,
        reference_loader=reference_loader,
        eval_loader=eval_loader,
        config=config,
        device=device,
        output_path=output_dir / "finetuned_metrics.json",
        stage_name="finetuned",
    )

    summary = {
        "experiment": str(config.experiment.name),
        "device": device.type,
        "split_sizes": data_bundle["split_sizes"],
        "used_pretrained_weights": bool(model.used_pretrained_weights),
        "self_supervised": self_supervised_metrics,
        "post_self_supervised": post_ssl_metrics,
        "training": training_metrics,
        **_summarize_metrics(baseline_metrics, finetuned_metrics),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"[done] wrote summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
