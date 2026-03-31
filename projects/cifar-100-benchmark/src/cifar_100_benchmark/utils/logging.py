"""Logging helpers built on rich."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


console = Console()


@dataclass(slots=True)
class JsonlLogger:
    path: Path

    def log(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


def print_metrics_table(title: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title=title)
    for key in rows[0].keys():
        table.add_column(str(key))
    for row in rows:
        table.add_row(*[str(row[k]) for k in row.keys()])
    console.print(table)
