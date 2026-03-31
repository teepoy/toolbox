"""Few-shot split generation and persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset


@dataclass(slots=True)
class SplitIndices:
    shot: int
    seed: int
    fewshot_train: list[int]
    val: list[int]
    ssl_pool: list[int]

    def to_dict(self) -> dict[str, object]:
        return {
            "shot": self.shot,
            "seed": self.seed,
            "fewshot_train": self.fewshot_train,
            "val": self.val,
            "ssl_pool": self.ssl_pool,
        }


def _group_indices_by_label(dataset: Dataset, label_key: str) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {}
    labels = dataset[label_key]
    for idx, y in enumerate(labels):
        grouped.setdefault(int(y), []).append(int(idx))
    return grouped


def build_split_indices(
    train_ds: Dataset,
    shot: int,
    seed: int,
    label_key: str = "fine_label",
    val_per_class: int = 50,
) -> SplitIndices:
    rng = np.random.default_rng(seed)
    grouped = _group_indices_by_label(train_ds, label_key)

    fewshot_train: list[int] = []
    val: list[int] = []
    ssl_pool: list[int] = []

    for cls, cls_indices in sorted(grouped.items()):
        cls_arr = np.array(cls_indices, dtype=np.int64)
        rng.shuffle(cls_arr)
        if shot + val_per_class >= len(cls_arr):
            raise ValueError(
                f"shot={shot} and val_per_class={val_per_class} exceed class size for class {cls}"
            )
        fewshot = cls_arr[:shot].tolist()
        cls_val = cls_arr[shot : shot + val_per_class].tolist()
        cls_ssl = cls_arr[shot + val_per_class :].tolist()
        fewshot_train.extend(fewshot)
        val.extend(cls_val)
        ssl_pool.extend(cls_ssl)

    return SplitIndices(
        shot=shot,
        seed=seed,
        fewshot_train=sorted(fewshot_train),
        val=sorted(val),
        ssl_pool=sorted(ssl_pool),
    )


def save_split(split: SplitIndices, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"shot_{split.shot}_seed_{split.seed}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(split.to_dict(), f, indent=2, sort_keys=True)
    return path


def load_split(path: Path) -> SplitIndices:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitIndices(
        shot=int(payload["shot"]),
        seed=int(payload["seed"]),
        fewshot_train=[int(x) for x in payload["fewshot_train"]],
        val=[int(x) for x in payload["val"]],
        ssl_pool=[int(x) for x in payload["ssl_pool"]],
    )
