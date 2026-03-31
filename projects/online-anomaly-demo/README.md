# Online Anomaly Demo

Multimodal streaming OOD anomaly detection PoC with four stages:

1. Data preparation + multimodal fabrication (image/text prompts)
2. Time-machine stream simulator (`T0`, `T1`, `T2`)
3. Core modules (`FAISS`, dual-path OOD detector, `HDBSCAN` clustering)
4. Offline visualization (OOD timeline + clustered OOD scatter)

## Environment

Use Python 3.12 for compatibility with `faiss-cpu` and `hdbscan`.

```bash
uv sync --python 3.12
```

## Run

End-to-end pipeline:

```bash
uv run python scripts/run_pipeline.py
```

Stage 1 only (cache embeddings parquet):

```bash
uv run python scripts/run_stage1.py
```

Stage 2-4 demo:

```bash
uv run python scripts/run_stage2_4_demo.py
```

## Config

Main config file: `configs/base.yaml`

Important knobs:

- `stream.batch_size`
- `stream.t0_steps`, `stream.t1_steps`, `stream.t2_steps`
- `stream.t2_new_class_ratio`, `stream.t2_mismatch_ratio`
- `detection.text_image_threshold`, `detection.memory_threshold`
- `clustering.trigger_size`

## Outputs

- Embedding cache parquet: `artifacts/cache/embeddings.parquet`
- Timeline plot: `artifacts/plots/ood_timeline.png`
- Cluster plot: `artifacts/plots/ood_clusters.png`
- Stats parquet: `artifacts/run_stats.parquet`
- Run report json: `artifacts/run_report.json`
