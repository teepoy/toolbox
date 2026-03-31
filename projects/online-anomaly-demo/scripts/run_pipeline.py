from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from online_anomaly_demo.config import load_config
from online_anomaly_demo.stage1_prepare import build_or_load_embedding_cache
from online_anomaly_demo.stage2_stream import StreamSimulator
from online_anomaly_demo.stage3_cluster import ClusterEngine
from online_anomaly_demo.stage3_detector import OODDetector
from online_anomaly_demo.stage3_vector_store import VectorStore
from online_anomaly_demo.stage4_viz import plot_ood_clusters, plot_ood_timeline


def main() -> None:
    cfg = load_config()

    embeddings_df = build_or_load_embedding_cache(cfg)
    sim = StreamSimulator(embeddings_df, cfg)

    bootstrap_emb, bootstrap_meta = sim.bootstrap_memory_data()
    store = VectorStore(dim=bootstrap_emb.shape[1])
    store.add_signatures(bootstrap_emb, bootstrap_meta)

    detector = OODDetector(cfg, store)
    cluster_engine = ClusterEngine(cfg)

    stats: list[dict[str, float | int | str]] = []
    for batch in sim.stream():
        stat = detector.process_batch(batch)
        stats.append(stat)

    stats_df = pd.DataFrame(stats)
    timeline_path = plot_ood_timeline(stats_df, cfg)

    unknown_vectors, unknown_meta = detector.get_unknown_pool()
    cluster_result = {"labels": pd.Series(dtype=int).to_numpy(), "clusters": []}
    representatives: list[int] = []
    if (
        cluster_engine.should_trigger(len(unknown_meta))
        and unknown_vectors.shape[0] > 0
    ):
        cluster_result = cluster_engine.cluster(unknown_vectors)
        for cluster in cluster_result["clusters"]:
            representatives.extend(cluster["representative_indices"])

    labels = cluster_result["labels"]
    labels_arr = (
        labels
        if labels.shape[0] == unknown_vectors.shape[0]
        else pd.Series([-1] * unknown_vectors.shape[0]).to_numpy()
    )

    clusters_path = plot_ood_clusters(unknown_vectors, labels_arr, representatives, cfg)

    stats_path = Path(cfg.paths.artifacts_dir) / "run_stats.parquet"
    stats_df.to_parquet(stats_path, index=False)

    report = {
        "timeline_plot": str(timeline_path),
        "clusters_plot": str(clusters_path),
        "stats_parquet": str(stats_path),
        "unknown_pool_size": len(unknown_meta),
        "clusters_found": len(cluster_result["clusters"]),
    }
    report_path = Path(cfg.paths.artifacts_dir) / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
