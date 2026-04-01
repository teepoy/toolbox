"""Few-shot split generation with shot-independent SSL pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset


@dataclass(slots=True)
class SeedSplit:
    seed: int
    ssl_pool_fixed: list[int]
    supervised_pool: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "ssl_pool_fixed": self.ssl_pool_fixed,
            "supervised_pool": self.supervised_pool,
        }


@dataclass(slots=True)
class ShotSplit:
    shot: int
    seed: int
    fewshot_train: list[int]
    val: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "shot": self.shot,
            "seed": self.seed,
            "fewshot_train": self.fewshot_train,
            "val": self.val,
        }


def _group_indices_by_label(dataset: Dataset, label_key: str) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    labels = dataset[label_key]
    for idx, y in enumerate(labels):
        grouped.setdefault(int(y), []).append(int(idx))
    return grouped


def build_seed_split(
    train_ds: Dataset,
    seed: int,
    ssl_pool_per_class: int,
    label_key: str = "fine_label",
) -> SeedSplit:
    rng = np.random.default_rng(seed)
    grouped = _group_indices_by_label(train_ds, label_key)

    ssl_pool_fixed: list[int] = []
    supervised_pool: list[int] = []

    for cls, cls_indices in sorted(grouped.items()):
        cls_arr = np.array(cls_indices, dtype=np.int64)
        rng.shuffle(cls_arr)
        if ssl_pool_per_class >= len(cls_arr):
            raise ValueError(
                f"ssl_pool_per_class={ssl_pool_per_class} exceeds class size for class {cls}"
            )
        cls_ssl = cls_arr[:ssl_pool_per_class].tolist()
        cls_supervised = cls_arr[ssl_pool_per_class:].tolist()
        ssl_pool_fixed.extend(cls_ssl)
        supervised_pool.extend(cls_supervised)

    return SeedSplit(
        seed=seed,
        ssl_pool_fixed=sorted(ssl_pool_fixed),
        supervised_pool=sorted(supervised_pool),
    )


def build_shot_split(
    train_ds: Dataset,
    seed_split: SeedSplit,
    shot: int,
    seed: int,
    label_key: str = "fine_label",
    val_per_class: int = 50,
) -> ShotSplit:
    rng = np.random.default_rng(seed + shot * 1000)
    supervised_ds = train_ds.select(seed_split.supervised_pool)
    grouped_local = _group_indices_by_label(supervised_ds, label_key)
    local_to_global = seed_split.supervised_pool

    fewshot_train: list[int] = []
    val: list[int] = []

    for cls, local_indices in sorted(grouped_local.items()):
        local_arr = np.array(local_indices, dtype=np.int64)
        rng.shuffle(local_arr)
        if shot + val_per_class >= len(local_arr):
            raise ValueError(
                f"shot={shot} and val_per_class={val_per_class} exceed supervised pool size for class {cls}"
            )
        fewshot_local = local_arr[:shot].tolist()
        val_local = local_arr[shot : shot + val_per_class].tolist()
        fewshot_train.extend([int(local_to_global[i]) for i in fewshot_local])
        val.extend([int(local_to_global[i]) for i in val_local])

    return ShotSplit(shot=shot, seed=seed, fewshot_train=sorted(fewshot_train), val=sorted(val))


def save_seed_split(split: SeedSplit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed_{split.seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_seed_split(path: Path) -> SeedSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SeedSplit(
        seed=int(payload["seed"]),
        ssl_pool_fixed=[int(x) for x in payload["ssl_pool_fixed"]],
        supervised_pool=[int(x) for x in payload["supervised_pool"]],
    )


def save_shot_split(split: ShotSplit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shot_{split.shot}_seed_{split.seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_shot_split(path: Path) -> ShotSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ShotSplit(
        shot=int(payload["shot"]),
        seed=int(payload["seed"]),
        fewshot_train=[int(x) for x in payload["fewshot_train"]],
        val=[int(x) for x in payload["val"]],
    )
