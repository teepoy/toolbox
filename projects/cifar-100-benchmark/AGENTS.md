# AGENTS.md

This file is for coding agents working in `cifar-100-benchmark`.
It documents practical commands and repo-specific coding conventions.

## 1) Project Overview

- Language: Python (`>=3.13`)
- Packaging: `pyproject.toml` + `uv_build`
- Main code root: `src/cifar_100_benchmark/`
- Config system: OmegaConf YAMLs under `configs/`
- Primary runtime: PyTorch + timm + ultralytics + datasets
- Current workflows are experiment-driven (smoke/full benchmark runs)

## 2) Repository Layout (Important Paths)

- `src/cifar_100_benchmark/cli.py`: single-run entrypoint
- `src/cifar_100_benchmark/runner/run_all.py`: smoke/full orchestrator
- `src/cifar_100_benchmark/runner/experiment.py`: experiment loop logic
- `src/cifar_100_benchmark/pretrain/`: SSL methods (BYOL/MoCo/SupCon/DINO)
- `src/cifar_100_benchmark/train/`: finetune/validate/SVM
- `src/cifar_100_benchmark/data/`: CIFAR loading + split generation
- `src/cifar_100_benchmark/eval/`: metric aggregation/report writing
- `configs/`: default + grouped experiment/model/pretrain/runtime configs
- `artifacts*/`: generated outputs (checkpoints, logs, CSVs)

## 3) Environment and Setup Commands

Run from repo root: `/home/jin/Desktop/toolbox/projects/cifar-100-benchmark`.

- Create/update env and install deps:
  - `uv sync`
- Use local venv Python directly when in doubt:
  - `.venv/bin/python --version`
- Run module commands with source path:
  - `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.cli --experiment smoke`

## 4) Build / Lint / Test Commands

### Build

- Package build:
  - `uv build`

### Lint / Static Checks

There is no dedicated linter config committed yet (no `ruff.toml`, `mypy.ini`, etc.).
Use lightweight checks that work with current repo state:

- Syntax/import sanity over source tree:
  - `PYTHONPATH=src .venv/bin/python -m compileall src`
- Optional quick import smoke:
  - `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.cli --help`

If you add Ruff/Mypy later, document and prefer:

- `ruff check src`
- `ruff format src`
- `mypy src`

### Test

Current state: there is no `tests/` directory and `pytest` is not installed by default.

- If/when pytest is added, full test run:
  - `PYTHONPATH=src .venv/bin/python -m pytest`
- If/when pytest is added, single test file:
  - `PYTHONPATH=src .venv/bin/python -m pytest tests/test_x.py`
- If/when pytest is added, single test function:
  - `PYTHONPATH=src .venv/bin/python -m pytest tests/test_x.py::test_name`
- If using stdlib unittest single test:
  - `PYTHONPATH=src .venv/bin/python -m unittest tests.test_x.TestClass.test_method`

## 5) Common Experiment Commands

- Smoke benchmark:
  - `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.cli --experiment smoke`
- Full reduced benchmark:
  - `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.runner.run_all --full-experiment full_reduced`
- Use custom output dir (recommended for isolated reruns):
  - `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.runner.run_all --full-experiment full_reduced --override runtime.output_dir=artifacts_run_x`

## 6) Code Style Guidelines

### Imports

- Keep import groups ordered:
  1) standard library
  2) third-party
  3) local package (`cifar_100_benchmark.*`)
- Prefer explicit imports; avoid `from x import *`.
- Keep one logical import per line for readability.

### Formatting

- Follow existing style already used in repo:
  - docstring at module top
  - moderate line length (Black-compatible style)
  - trailing commas in multiline literals/calls when useful
- Preserve readability over compactness.

### Types

- Use type hints on all public functions/methods.
- Prefer concrete types for collections (`list[int]`, `dict[str, object]`).
- Use `@dataclass(slots=True)` for simple structured records.
- Keep return types explicit for orchestration and training helpers.

### Naming

- `snake_case`: functions, variables, module filenames
- `PascalCase`: classes and dataclasses
- `UPPER_SNAKE_CASE`: constants
- Config keys should remain stable and descriptive (`ssl_pool_per_class`, `early_stop_patience`, etc.).

### Error Handling

- Fail fast on invalid config/state with explicit exceptions.
- Prefer specific exceptions with actionable messages:
  - `ValueError` for bad config values
  - `RuntimeError` for runtime/model initialization failures
- Avoid broad `except Exception` unless wrapping external boundary calls.

### Logging and Metrics

- Use structured JSONL logging for metrics (`JsonlLogger`).
- Keep terminal output concise and readable (`rich` tables).
- Include enough run metadata in outputs (`family`, `imgsz`, `shot`, `seed`, split mode).

### Config and Reproducibility

- All tunable runtime behavior should be config-driven (OmegaConf).
- Do not hardcode experiment-specific constants in training loops.
- Respect seed-setting and deterministic split generation.
- Keep split protocol changes backward-aware (new files/keys rather than silent behavior swaps).

### Model/Training Conventions

- Keep backbone/head/loss/trainer separation intact.
- Reuse shared helpers instead of duplicating logic across SSL methods.
- For new SSL methods, integrate through `pretrain/run.py` dispatcher.
- If adding new families, update both runner family handling and config presets.

### Artifacts and File Hygiene

- Do not commit generated artifacts (`artifacts*`, checkpoints, metrics logs).
- Keep report CSV schemas stable when possible.
- If schema changes are necessary, update readers/writers together.

## 7) Rule Files (Cursor/Copilot)

Checked locations:

- `.cursorrules`
- `.cursor/rules/`
- `.github/copilot-instructions.md`

At time of writing, none are present in this repository.

If any are added later, treat them as higher-priority local guidance and update this file.

## 8) Agent Working Norms

- Make minimal, focused changes.
- Prefer config changes over code changes for experiment tuning.
- Validate with at least a smoke run before large reruns.
- When rerunning heavy experiments, use a new `runtime.output_dir` to avoid mixed-result ambiguity.
- Summarize behavioral diffs and exact commands used in final handoff.
