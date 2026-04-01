# CONFIG PROFILES GUIDE

## OVERVIEW
`configs/` holds the runnable OmegaConf profiles for this project. They do not replace defaults in `config.py`; they override that base schema.

## WHERE TO LOOK
| Profile | Use | Notes |
|---------|-----|-------|
| `smoke.yaml` | Fast validation | Small subsets, 1 epoch, allows random-init fallback |
| `full.yaml` | Longer real run | Full dataset, 100 epochs, requires pretrained weights |
| `class_disjoint.yaml` | Generalization experiment | Enables `dataset.class_split`, evaluates on held-out classes |

## CONVENTIONS
- Keep profile keys aligned with `_default_config()` in `src/supcon_demo/config.py`.
- Keep new schema keys defined in `_default_config()` first, then reference them from YAML profiles.
- Prefer CLI dotlist overrides for one-off tweaks: `--override train.epochs=2`.
- `experiment.output_dir` is part of the profile contract; every profile should write to its own folder.
- `runtime.device: auto` is the default unless a profile needs explicit device pinning.
- `benchmark.reference_split` and `benchmark.eval_split` must stay within `train`, `validation`/`val`, or `test`.

## PROFILE DIFFERENCES
- `smoke.yaml`: subset sizes `192/96/96`, 1 training epoch, `allow_random_init_fallback: true`. Safe default for repo validation.
- `full.yaml`: no subsets, larger batch sizes, 100 epochs, `allow_random_init_fallback: false`. Assumes pretrained weights are available.
- `class_disjoint.yaml`: same long-run posture as `full.yaml`, but enables `dataset.class_split` and benchmarks `validation -> test` instead of `train -> validation`.

## ANTI-PATTERNS
- Do not invent new keys only in YAML; add them to `_default_config()` first.
- Do not point multiple profiles at the same `experiment.output_dir` unless overwriting artifacts is intentional.
- Do not use `class_split.mode` values other than `first_half` or `random`.
- Do not switch long-run profiles to `allow_random_init_fallback: true` casually; that changes experiment meaning.
- Do not use `full.yaml` as the default validation command; use `smoke.yaml` for quick checks.

## SAFE OVERRIDES
- Common short-run overrides: `train.epochs`, `train.batch_size`, `benchmark.linear_probe.epochs`, `dataset.train_subset`, `dataset.val_subset`, `dataset.test_subset`.
- High-impact overrides: `dataset.name`, `model.backbone_name`, `model.pretrained`, `dataset.class_split.*`, `experiment.output_dir`.

## NOTES
- The runner writes the merged config to `resolved_config.yaml`; inspect that file when debugging override behavior.
- Profiles are human-readable experiment presets, not the full schema authority.
