# AGENTS — online-anomaly-demo

Multimodal streaming OOD (out-of-distribution) anomaly detection PoC.
Four-stage pipeline: data prep, stream simulation, detection + clustering, visualization.

---

## Quick reference

| Action | Command |
|---|---|
| Install / sync deps | `uv sync --python 3.12` |
| Full pipeline | `uv run python scripts/run_pipeline.py` |
| Stage 1 only (embeddings) | `uv run python scripts/run_stage1.py` |
| Stages 2-4 demo | `uv run python scripts/run_stage2_4_demo.py` |
| Entry point (alias) | `uv run python main.py` |
| Type-check | `uv run mypy src/ scripts/ main.py` |
| Lint (if ruff added) | `uv run ruff check src/ scripts/` |
| Format (if ruff added) | `uv run ruff format src/ scripts/` |

**There are no tests yet.** If you add tests, place them in `tests/` and run via:
```bash
uv run pytest tests/               # all tests
uv run pytest tests/test_foo.py     # single file
uv run pytest tests/test_foo.py::test_bar  # single test
```

---

## Environment

- **Python**: 3.12 (pinned in `.python-version`; required for `faiss-cpu` + `hdbscan`)
- **Package manager**: `uv` (no pip, no poetry, no conda)
- **Config**: OmegaConf, loaded from `configs/base.yaml`
- **No linter/formatter config exists yet** — follow the style already in the code

---

## Project layout

```
configs/base.yaml               # all runtime knobs (OmegaConf)
main.py                         # thin entry point → scripts/run_pipeline.py
scripts/
  run_pipeline.py               # end-to-end orchestrator
  run_stage1.py                 # embedding cache only
  run_stage2_4_demo.py          # stream + detect + viz demo
src/online_anomaly_demo/
  __init__.py
  config.py                     # load_config(), path resolution
  class_names.py                # FLOWERS102_CLASS_NAMES constant
  stage1_prepare.py             # CLIP embedding extraction
  stage2_stream.py              # StreamSimulator, StreamBatch dataclass
  stage3_vector_store.py        # FAISS-backed VectorStore
  stage3_detector.py            # OODDetector (cross-modal + memory)
  stage3_cluster.py             # ClusterEngine (HDBSCAN)
  stage4_viz.py                 # matplotlib timeline + scatter plots
artifacts/                      # generated outputs (git-ignored)
```

---

## Code style

### Imports

1. `from __future__ import annotations` at the top of every module (already consistent).
2. Order: stdlib → third-party → local (relative with leading dot).
3. Local imports use **explicit relative imports**: `from .stage2_stream import StreamBatch`.
4. Scripts use `sys.path.append` to add `src/` — keep this pattern for scripts.

### Formatting

- 4-space indentation, no tabs.
- Max line length ~88 (black-style). Break long expressions with parentheses, not backslashes.
- Trailing commas in multi-line collections and function calls.
- Single blank line between functions within a class; two blank lines between top-level definitions.

### Type annotations

- All function signatures are typed (parameters and return types).
- Use `from __future__ import annotations` for PEP 604 union syntax (`X | Y`).
- Generic containers: `list[...]`, `dict[...]`, `tuple[...]` (lowercase, not `typing.List`).
- Use `np.ndarray` for numpy arrays (not parameterized).
- Config parameters are `DictConfig` from OmegaConf; cast explicitly at point of use:
  `int(cfg.stream.batch_size)`, `float(cfg.detection.text_image_threshold)`, `str(cfg.data.hf_dataset)`.

### Naming

- **Classes**: `PascalCase` — `OODDetector`, `VectorStore`, `StreamSimulator`, `ClusterEngine`.
- **Functions / methods**: `snake_case` — `build_or_load_embedding_cache`, `process_batch`.
- **Private helpers**: leading underscore — `_resolve_device`, `_l2_normalize`, `_reduce_to_2d`.
- **Constants**: `UPPER_SNAKE_CASE` — `FLOWERS102_CLASS_NAMES`.
- **Config keys**: `snake_case` in YAML, accessed as attributes (`cfg.stream.batch_size`).

### Error handling

- Minimal try/except; let exceptions propagate unless there is a specific recovery strategy.
- `TypeError` for unexpected types in helper functions (see `_extract_feature_tensor`).
- No bare `except:` or `except Exception:` catch-alls.

### Data patterns

- `@dataclass` for structured data (`StreamBatch`).
- NumPy arrays for batch numeric data; Python lists for variable-length metadata.
- Pandas DataFrames for tabular data and parquet I/O.
- FAISS `IndexFlatIP` for cosine-similarity vector search (vectors are L2-normalized before insert/query).

### Config access

- All config goes through `OmegaConf.load()` → `DictConfig`.
- Always cast DictConfig values to Python types before use — never pass `DictConfig` scalars to NumPy/torch.
- Paths resolved via `pathlib.Path`; directories created with `mkdir(parents=True, exist_ok=True)`.

---

## Key dependencies

| Package | Purpose |
|---|---|
| `transformers` + `torch` + `timm` | CLIP model for image/text embeddings |
| `faiss-cpu` | Approximate nearest-neighbor search |
| `hdbscan` | Density-based clustering |
| `omegaconf` | Typed YAML config |
| `datasets` | HuggingFace dataset loading |
| `pandas` + `pyarrow` | Tabular data + parquet I/O |
| `matplotlib` | Timeline + scatter plots |
| `umap-learn` / `scikit-learn` | Dimensionality reduction (UMAP / t-SNE) |
| `plotly` | (available, not yet used in core pipeline) |
| `gradio` | (available, not yet used in core pipeline) |
| `dagster` / `langgraph` | (available, not yet used in core pipeline) |

---

## Common tasks for agents

### Adding a new pipeline stage
1. Create `src/online_anomaly_demo/stageN_name.py`.
2. Add any new config knobs to `configs/base.yaml`.
3. Wire it into `scripts/run_pipeline.py` (follow the existing linear stage pattern).
4. Use `DictConfig` parameter for all configurable values — cast at point of use.

### Adding tests
1. Create `tests/` directory at project root.
2. Add `pytest` to `[project.optional-dependencies]` or `[dependency-groups]` in `pyproject.toml`.
3. One test file per module: `tests/test_stage3_detector.py`, etc.

### Modifying config
- Edit `configs/base.yaml` — all config is centralized there.
- Access new keys via dot notation: `cfg.section.key`.
- Cast to Python types: `int(...)`, `float(...)`, `str(...)`, `bool(...)`.

---

## Do NOT

- Add `as any`, `@ts-ignore` equivalents (`# type: ignore` without specific error code).
- Suppress linter warnings globally — fix the code instead.
- Import from `typing` for containers available as builtins (`list`, `dict`, `tuple`).
- Mix `pip install` with `uv` — this project is `uv`-only.
- Commit `artifacts/` or `.venv/` contents.
- Use bare `except:` blocks.
