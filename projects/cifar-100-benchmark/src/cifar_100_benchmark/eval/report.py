"""Result aggregation and reporting."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def _to_int(value: object, default: int) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "none":
        return default
    return int(text)


def read_summary(in_csv: Path) -> list[dict[str, object]]:
    if not in_csv.exists():
        return []
    rows: list[dict[str, object]] = []
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    for row in rows:
        if not str(row.get("imgsz", "")).strip():
            row["imgsz"] = "64"
    return rows


def write_summary(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    preferred = ["family", "imgsz", "shot", "seed", "val_top1", "test_top1"]
    keyset = {k for row in rows for k in row.keys()}
    keys = [k for k in preferred if k in keyset] + sorted([k for k in keyset if k not in preferred])
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})


def write_leaderboard(rows: list[dict[str, object]], out_csv: Path) -> None:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (
            str(r.get("family")),
            _to_int(r.get("shot"), default=-1),
            _to_int(r.get("imgsz"), default=64),
        )
        grouped[key].append(float(str(r.get("test_top1"))))

    out: list[dict[str, object]] = []
    for (family, shot, imgsz), vals in sorted(grouped.items()):
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append(
            {
                "family": family,
                "shot": shot,
                "imgsz": imgsz,
                "mean_top1": round(mean, 4),
                "std_top1": round(std, 4),
                "runs": len(vals),
            }
        )
    write_summary(out, out_csv)
