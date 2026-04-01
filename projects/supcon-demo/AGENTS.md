# PROJECT KNOWLEDGE BASE

## OVERVIEW
Small Python 3.13 demo for supervised-contrastive finetuning and embedding evaluation on `nkirschi/oxford-flowers`. One CLI runs baseline benchmark -> finetune -> post-finetune benchmark and writes artifacts under the configured output directory.

## STRUCTURE
```text
supcon-demo/
├── configs/          # Run profiles: smoke, full, class-disjoint
├── outputs/          # Generated run artifacts; also holds dataset parquet cache
├── src/supcon_demo/  # Runtime package: CLI, config, data, model, training, benchmark
├── pyproject.toml    # uv_build package metadata + console script
├── README.md         # Run instructions + expected smoke outputs
├── run_full_flow.sh  # Convenience wrapper for full.yaml
└── uv.lock           # Locked uv environment
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Run the project | `src/supcon_demo/runner.py` | CLI entry used by `supcon-demo-flow` |
| Understand package internals | `src/supcon_demo/AGENTS.md` | Runtime flow, shapes, training/benchmark contracts |
| Adjust config profiles | `configs/AGENTS.md` | Profile intent, safe overrides, config pitfalls |
| Re-run long experiment | `run_full_flow.sh` | Wraps `configs/full.yaml` |
| Check current usage docs | `README.md` | Run commands and expected smoke outputs |

## CONVENTIONS
- Use `uv`, not ad-hoc `pip install`; docs and scripts assume `uv sync` then `uv run ...`.
- Runtime configs are OmegaConf YAML merged onto `_default_config()` in `config.py`.
- CLI overrides use repeated OmegaConf dotlist entries: `--override train.epochs=2`.
- `OmegaConf.set_struct(config, True)` is enabled after merge; keep new keys in `_default_config()` so the schema stays explicit and documented.
- `runtime.device: auto` means CUDA when available, otherwise CPU.
- Output directories are created eagerly from `experiment.output_dir`.

## ANTI-PATTERNS (THIS PROJECT)
- Do not run `configs/full.yaml` or `configs/class_disjoint.yaml` if pretrained weights are unavailable; those profiles keep `allow_random_init_fallback: false`.
- Do not assume `datasets.load_dataset()` is enough; `data.py` may fall back to downloading parquet shards into `outputs/.dataset_cache`.
- Do not add new config keys only at the CLI; add them to `_default_config()` so the project schema and resolved configs stay consistent.
- Do not change benchmark split names away from `train`, `validation`/`val`, or `test`; `runner.py` rejects others.
- Do not expect AMP on CPU even with `train.use_amp: true`; mixed precision only activates on CUDA.

## UNIQUE STYLES
- Training uses two augmented views per sample for SupCon, but benchmarking uses single-view embeddings.
- Benchmark results are stored as JSON artifacts, not printed summaries alone.
- Dataset access is resilient by design: normal HF path first, direct parquet fallback second.
- `used_pretrained_weights` is tracked through model init, training metrics, checkpoint, and final summary.

## COMMANDS
```bash
uv sync
uv run supcon-demo-flow --config configs/smoke.yaml
uv run supcon-demo-flow --config configs/full.yaml --override train.epochs=12
uv run supcon-demo-flow --config configs/class_disjoint.yaml
./run_full_flow.sh
```

## NOTES
- There is no `tests/` tree or CI workflow in this repo; the smoke profile is the current fast validation path.
- `outputs/` is disposable generated state, but deleting it also removes the parquet dataset cache.
- The repo currently contains `.venv/` and `.mypy_cache/`; treat both as local artifacts, not source.
- Python requirement is `>=3.13`; avoid assuming older interpreters will work.
