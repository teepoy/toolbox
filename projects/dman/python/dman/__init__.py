from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

try:
    from dman.dman import (
        DmanDataset,
        DmanDatasetBuilder,
        DmanDatasetUpdater,
        create_dataset,
        load_dataset,
        update_dataset,
    )
except ImportError:
    pass


def _candidate_binaries() -> list[Path]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent.parent
    return [
        project_root / "target" / "release" / "dman-cli",
        project_root / "target" / "debug" / "dman-cli",
    ]


def _resolve_cli() -> str:
    installed = shutil.which("dman-cli")
    if installed:
        return installed

    for candidate in _candidate_binaries():
        if candidate.exists():
            return str(candidate)

    raise RuntimeError(
        "dman-cli is not available. Build it first with `cargo build --release -p dman-cli`."
    )


def main() -> int:
    cli = _resolve_cli()
    completed = subprocess.run([cli, *sys.argv[1:]], check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
