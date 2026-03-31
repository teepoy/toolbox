## CIFAR-100 Few-Shot Benchmark

This repository now includes a modular benchmark pipeline for CIFAR-100 few-shot experiments with:

- YOLO26n classification baseline
- ConvNeXtV2-Atto official pretrained baseline
- SSL pretrained families: BYOL, MoCo v3, SupCon-pretrain, DINO-style
- SVM baseline on frozen features
- View fusion network module (`images -> feats -> self-attn -> head`) for later use

### Structure

- `src/cifar_100_benchmark/data`: dataset loading and deterministic split generation
- `src/cifar_100_benchmark/models`: backbones and heads
- `src/cifar_100_benchmark/losses`: supervised and SSL losses
- `src/cifar_100_benchmark/pretrain`: SSL methods
- `src/cifar_100_benchmark/train`: finetune, validate, svm
- `src/cifar_100_benchmark/eval`: metrics and aggregation
- `configs`: OmegaConf-based experiment and component configs

### Run

Smoke run:

```bash
.venv/bin/python -m cifar_100_benchmark.cli --experiment smoke
```

Reduced full run (shots 10,20,50 with seed 0):

```bash
.venv/bin/python -m cifar_100_benchmark.runner.run_all --full-experiment full_reduced
```

Full saved configuration for later (shots 1,5,10,20,50 and seeds 0,1,2):

```bash
.venv/bin/python -m cifar_100_benchmark.runner.run_all --full-experiment full
```

Artifacts are saved under `artifacts/` (split files, checkpoints, metrics, summary, leaderboard).
