## SupCon Finetune Demo

This repository contains a small end-to-end example for improving embedding quality with supervised contrastive learning on the Hugging Face dataset `nkirschi/oxford-flowers` using `ConvNeXt V2 Atto` as the backbone.

The flow does three things in sequence:

1. Benchmarks the base backbone embeddings.
2. Finetunes the model with a SupCon objective.
3. Benchmarks the finetuned embeddings with both k-NN retrieval accuracy and a linear probe.

## What Is Included

- A smoke profile for fast end-to-end validation: [configs/smoke.yaml](/home/jin/Desktop/toolbox/projects/supcon-demo/configs/smoke.yaml)
- A fuller profile intended for later longer runs: [configs/full.yaml](/home/jin/Desktop/toolbox/projects/supcon-demo/configs/full.yaml)
- A reusable CLI entry point: [src/supcon_demo/runner.py](/home/jin/Desktop/toolbox/projects/supcon-demo/src/supcon_demo/runner.py)
- A convenience shell script for later full runs: [run_full_flow.sh](/home/jin/Desktop/toolbox/projects/supcon-demo/run_full_flow.sh)

## Run

Sync the environment:

```bash
uv sync
```

Run the validated smoke profile:

```bash
uv run supcon-demo-flow --config configs/smoke.yaml
```

Run the fuller profile later:

```bash
./run_full_flow.sh
```

You can override config values from the CLI:

```bash
uv run supcon-demo-flow --config configs/full.yaml --override train.epochs=12 --override train.batch_size=32
```

## Outputs

Each run writes artifacts under the configured output directory.

For the smoke profile this is [outputs/smoke](/home/jin/Desktop/toolbox/projects/supcon-demo/outputs/smoke).

Expected files:

- `resolved_config.yaml`
- `baseline_metrics.json`
- `training_metrics.json`
- `checkpoint.pt`
- `finetuned_metrics.json`
- `summary.json`

## Verified Smoke Run

The smoke run was executed successfully on March 30, 2026 with:

- Dataset: `nkirschi/oxford-flowers`
- Device: CUDA
- Train/validation/test subset sizes: `192 / 96 / 96`

Observed benchmark results:

- Baseline k-NN: `top_1=0.3958`, `top_5=0.5833`
- Finetuned k-NN: `top_1=0.4375`, `top_5=0.5938`
- Baseline linear probe: `0.3542`
- Finetuned linear probe: `0.3646`

## Notes

- The dataset loader first tries `datasets.load_dataset`. If the local `huggingface_hub` path fails, it falls back to downloading the dataset parquet shards directly from the dataset repo and loads them locally.
- The smoke profile allows a fallback to randomly initialized ConvNeXt weights if pretrained weight download is unavailable. The full profile keeps pretrained weights mandatory.
