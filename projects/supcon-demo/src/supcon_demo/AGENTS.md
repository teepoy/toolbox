# SUPCON PACKAGE GUIDE

## OVERVIEW
`src/supcon_demo` is the full runtime package: config merge, dataset assembly, model init, SupCon training, and benchmark evaluation all live here.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| CLI flow | `runner.py` | `main()` parses `--config` and `--override` |
| Config defaults | `config.py` | `_default_config()` defines valid override surface |
| Dataset and loaders | `data.py` | HF load fallback, split derivation, transforms |
| Backbone / projection head | `model.py` | `SupConModel.encode`, `.project`, `.forward` |
| Training loop | `training.py` | AdamW, AMP, checkpoint, `training_metrics.json` |
| Retrieval + linear probe | `benchmark.py` | `run_benchmark()` writes metrics JSON |
| SupCon loss math | `losses.py` | Expects `(batch, views, dim)` projections |

## FLOW CONTRACT
`runner.main()` does this in order: load config -> resolve output dir -> choose device -> seed RNGs -> build `SupConModel` -> `load_data()` -> baseline `run_benchmark()` -> `train_supcon()` -> finetuned `run_benchmark()` -> write `summary.json`.

## MODULE CONVENTIONS
- `config.py`: merges `_default_config()`, YAML file, then dotlist overrides. Struct mode is enabled after merge.
- `config.py`: keep new config keys in `_default_config()` so the package schema stays discoverable in code and in `resolved_config.yaml`.
- `data.py`: training dataset returns `(view_one, view_two, label)`; eval datasets return `(image, label)`.
- `data.py`: if `datasets.load_dataset()` fails, code downloads parquet shards into `outputs/.dataset_cache/<dataset>` and loads them locally.
- `data.py`: supported `dataset.class_split.mode` values are only `first_half` and `random`.
- `model.py`: `encode()` is the benchmark interface; `forward()` returns `(features, projections)` for training.
- `model.py`: pretrained fallback only happens when `allow_random_init_fallback` is true; `used_pretrained_weights` tracks the outcome.
- `training.py`: two views are concatenated before the forward pass, then reshaped back to `(batch, 2, dim)` before loss.
- `training.py`: AMP and `GradScaler` are CUDA-only even if config says `use_amp: true`.
- `benchmark.py`: linear probe remaps train labels to contiguous IDs; unseen eval labels are ignored in probe accuracy.

## ANTI-PATTERNS
- Do not change `model.forward()` output shape without updating `training.py` reshape logic.
- Do not feed single-view batches into `train_supcon()`; the loss path assumes two views per sample.
- Do not add config fields in ad-hoc call sites; route them through `_default_config()` and `load_config()` so resolved configs stay reproducible.
- Do not remove `used_pretrained_weights` plumbing unless you also update checkpoint and summary writers.
- Do not add hidden network side effects outside `data.py`; dataset download behavior is centralized there.

## VALIDATION PATH
- Fast end-to-end check: `uv run supcon-demo-flow --config configs/smoke.yaml`
- Primary success artifacts: `resolved_config.yaml`, `baseline_metrics.json`, `training_metrics.json`, `checkpoint.pt`, `finetuned_metrics.json`, `summary.json`
- No unit tests exist yet; if adding code, prefer smoke validation plus targeted synthetic tests around shapes and config handling.

## NOTES
- `py.typed` is present; keep typing-friendly signatures when extending the package.
- Logging is plain `print(...)`; there is no structured logger to preserve.
- `save_every_epochs` exists in config but current training code writes one final checkpoint path and metrics artifact.
