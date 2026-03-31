"""Run one experiment unit (shot, seed, family, imgsz)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf

from cifar_100_benchmark.data.cifar100 import (
    HFDatasetWrapper,
    HFPairViewDataset,
    load_cifar100,
    make_loader,
    make_ssl_pair_transforms,
    make_transforms,
    select_indices,
)
from cifar_100_benchmark.data.splits import (
    SeedSplit,
    ShotSplit,
    build_seed_split,
    build_shot_split,
    load_seed_split,
    load_shot_split,
    save_seed_split,
    save_shot_split,
)
from cifar_100_benchmark.eval.report import read_summary, write_leaderboard, write_summary
from cifar_100_benchmark.losses.supervised import build_supervised_loss
from cifar_100_benchmark.models.builders import build_backbone, build_classifier
from cifar_100_benchmark.pretrain.run import run_pretrain
from cifar_100_benchmark.train.finetune import finetune
from cifar_100_benchmark.train.svm import run_svm
from cifar_100_benchmark.train.validate import validate
from cifar_100_benchmark.utils.logging import console
from cifar_100_benchmark.utils.seed import set_seed


def _resolve_device(cfg: DictConfig) -> torch.device:
    want = str(cfg.runtime.device)
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _materialize_seed_split(cfg: DictConfig, train_ds: Dataset, seed: int, splits_dir: Path) -> SeedSplit:
    seed_dir = splits_dir / "seed_splits"
    split_path = seed_dir / f"seed_{seed}.json"
    if split_path.exists():
        return load_seed_split(split_path)
    split = build_seed_split(
        train_ds,
        seed=seed,
        ssl_pool_per_class=int(cfg.data.ssl_pool_per_class),
        label_key=str(cfg.data.label_key),
    )
    save_seed_split(split, seed_dir)
    return split


def _materialize_shot_split(
    cfg: DictConfig,
    train_ds: Dataset,
    seed_split: SeedSplit,
    shot: int,
    seed: int,
    splits_dir: Path,
) -> ShotSplit:
    shot_dir = splits_dir / "shot_splits"
    split_path = shot_dir / f"shot_{shot}_seed_{seed}.json"
    if split_path.exists():
        return load_shot_split(split_path)
    split = build_shot_split(
        train_ds,
        seed_split=seed_split,
        shot=shot,
        seed=seed,
        label_key=str(cfg.data.label_key),
        val_per_class=int(cfg.data.val_per_class),
    )
    save_shot_split(split, shot_dir)
    return split


def _make_supervised_loaders(cfg: DictConfig, split: ShotSplit, train_ds: Dataset, test_ds: Dataset):
    tfs = make_transforms(int(cfg.data.image_size))
    fewshot_ds = select_indices(train_ds, split.fewshot_train)
    val_ds = select_indices(train_ds, split.val)
    train_torch = HFDatasetWrapper(fewshot_ds, transform=tfs["train"], label_key=str(cfg.data.label_key))
    val_torch = HFDatasetWrapper(val_ds, transform=tfs["eval"], label_key=str(cfg.data.label_key))
    test_torch = HFDatasetWrapper(test_ds, transform=tfs["eval"], label_key=str(cfg.data.label_key))
    train_loader = make_loader(train_torch, int(cfg.data.batch_size), True, int(cfg.data.num_workers))
    val_loader = make_loader(val_torch, int(cfg.data.eval_batch_size), False, int(cfg.data.num_workers))
    test_loader = make_loader(test_torch, int(cfg.data.eval_batch_size), False, int(cfg.data.num_workers))
    return train_loader, val_loader, test_loader


def _make_ssl_loader(cfg: DictConfig, seed_split: SeedSplit, train_ds: Dataset):
    ssl_ds = select_indices(train_ds, seed_split.ssl_pool_fixed)
    t1, t2 = make_ssl_pair_transforms(int(cfg.data.image_size))
    ssl_torch = HFPairViewDataset(ssl_ds, t1, t2, label_key=str(cfg.data.label_key))
    return make_loader(ssl_torch, int(cfg.data.batch_size), True, int(cfg.data.num_workers))


def _row_exists(results: list[dict[str, object]], family: str, imgsz: int, shot: int, seed: int) -> bool:
    def to_int(value: object, default: int = -1) -> int:
        if value is None:
            return default
        text = str(value).strip()
        if not text or text.lower() == "none":
            return default
        return int(text)

    for r in results:
        if (
            str(r.get("family")) == family
            and to_int(r.get("imgsz", -1)) == imgsz
            and to_int(r.get("shot", -1)) == shot
            and to_int(r.get("seed", -1)) == seed
        ):
            return True
    return False


def run_experiment(cfg: DictConfig) -> None:
    set_seed(int(cfg.runtime.seed))
    device = _resolve_device(cfg)
    run_root = Path(str(cfg.runtime.output_dir))
    summary_path = run_root / "summary.csv"
    splits_dir = run_root / "splits"
    results = read_summary(summary_path)

    bundle = load_cifar100()
    img_sizes = [int(x) for x in cfg.experiment.get("img_sizes", [int(cfg.data.image_size)])]

    for imgsz in img_sizes:
        for seed in cfg.experiment.seeds:
            local_seed_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
            local_seed_cfg.runtime.seed = int(seed)
            local_seed_cfg.data.image_size = int(imgsz)
            seed_split = _materialize_seed_split(local_seed_cfg, bundle.train, int(seed), splits_dir)

            ssl_ckpts: dict[str, str] = {}
            for family in cfg.experiment.families:
                family = str(family)
                if family not in {"byol", "mocov3", "supcon", "dino"}:
                    continue
                pretrain_out = run_root / "pretrain" / f"{family}_img{imgsz}_seed{seed}"
                ckpt_path = pretrain_out / "backbone.pt"
                if ckpt_path.exists():
                    ssl_ckpts[family] = str(ckpt_path)
                    console.print(f"[cyan]Reuse SSL ckpt[/cyan] {family}_img{imgsz}_seed{seed}")
                    continue
                ssl_loader = _make_ssl_loader(local_seed_cfg, seed_split, bundle.train)
                local_seed_cfg.pretrain.name = family
                ssl_ckpts[family] = str(run_pretrain(local_seed_cfg, ssl_loader, device, pretrain_out))

            for shot in cfg.experiment.shots:
                local_cfg = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
                local_cfg.runtime.seed = int(seed)
                local_cfg.data.image_size = int(imgsz)

                shot_split = _materialize_shot_split(
                    local_cfg, bundle.train, seed_split, int(shot), int(seed), splits_dir
                )
                train_loader, val_loader, test_loader = _make_supervised_loaders(
                    local_cfg, shot_split, bundle.train, bundle.test
                )

                for family in cfg.experiment.families:
                    family = str(family)
                    run_id = f"{family}_img{imgsz}_shot{shot}_seed{seed}"
                    if _row_exists(results, family, int(imgsz), int(shot), int(seed)):
                        console.print(f"[cyan]Skip existing[/cyan] {run_id}")
                        continue

                    out_dir = run_root / "runs" / run_id
                    out_dir.mkdir(parents=True, exist_ok=True)

                    init_ckpt = ssl_ckpts.get(family)

                    if family == "svm":
                        backbone = build_backbone(local_cfg.model.backbone).to(device)
                        svm_result = run_svm(backbone, train_loader, test_loader, device)
                        results.append(
                            {
                                "family": family,
                                "imgsz": int(imgsz),
                                "shot": int(shot),
                                "seed": int(seed),
                                "val_top1": -1.0,
                                "test_top1": round(svm_result.top1, 4),
                                "ssl_pool_mode": "fixed",
                            }
                        )
                        write_summary(results, summary_path)
                        write_leaderboard(results, run_root / "leaderboard.csv")
                        console.print(f"[green]Finished[/green] {run_id}: test_top1={svm_result.top1:.2f}")
                        continue

                    if family == "yolo26n":
                        local_cfg.model.backbone.name = "yolo26n"
                        local_cfg.model.backbone.pretrained = False
                    elif family == "convnext32":
                        local_cfg.model.backbone.name = "convnext32_atto"
                        local_cfg.model.backbone.pretrained = False
                    elif family == "official":
                        local_cfg.model.backbone.name = "convnextv2_atto"
                        local_cfg.model.backbone.pretrained = True
                    elif family == "random":
                        local_cfg.model.backbone.name = "convnextv2_atto"
                        local_cfg.model.backbone.pretrained = False
                    else:
                        local_cfg.model.backbone.name = "convnextv2_atto"
                        local_cfg.model.backbone.pretrained = False

                    ft_result = finetune(local_cfg, train_loader, val_loader, device, out_dir, init_ckpt=init_ckpt)
                    model = build_classifier(local_cfg.model).to(device)
                    payload = torch.load(ft_result.ckpt_path, map_location="cpu")
                    model.load_state_dict(payload["model_state_dict"], strict=False)
                    test_metrics = validate(model, test_loader, build_supervised_loss("cross_entropy"), device)
                    results.append(
                        {
                            "family": family,
                            "imgsz": int(imgsz),
                            "shot": int(shot),
                            "seed": int(seed),
                            "val_top1": round(ft_result.best_val_top1, 4),
                            "test_top1": round(test_metrics.top1, 4),
                            "ssl_pool_mode": "fixed",
                        }
                    )
                    write_summary(results, summary_path)
                    write_leaderboard(results, run_root / "leaderboard.csv")
                    console.print(f"[green]Finished[/green] {run_id}: test_top1={test_metrics.top1:.2f}")

    write_summary(results, summary_path)
    write_leaderboard(results, run_root / "leaderboard.csv")
