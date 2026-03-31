"""Result aggregation and reporting."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


def write_summary(rows: list[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_leaderboard(rows: list[dict[str, object]], out_csv: Path) -> None:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (str(r["family"]), int(r["shot"]))
        grouped[key].append(float(r["test_top1"]))

    out: list[dict[str, object]] = []
    for (family, shot), vals in sorted(grouped.items()):
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        out.append(
            {
                "family": family,
                "shot": shot,
                "mean_top1": round(mean, 4),
                "std_top1": round(std, 4),
                "runs": len(vals),
            }
        )
    write_summary(out, out_csv)
