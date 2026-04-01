from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from online_anomaly_demo.config_v2 import load_config_v2
from online_anomaly_demo.stage1_prepare import build_or_load_embedding_cache
from online_anomaly_demo.stage2_stream import StreamSimulator
from online_anomaly_demo.stage3_cluster import ClusterEngine
from online_anomaly_demo.stage3_detector_v2 import OODDetectorV2
from online_anomaly_demo.stage3_vector_store_v2 import VectorStoreV2
from online_anomaly_demo.stage4_viz import plot_ood_clusters, plot_ood_timeline


def main() -> None:
    cfg = load_config_v2()
    df = build_or_load_embedding_cache(cfg)

    sim = StreamSimulator(df, cfg)
    memory_emb, memory_meta = sim.bootstrap_memory_data()
    store = VectorStoreV2(dim=memory_emb.shape[1])
    store.add_signatures(memory_emb, memory_meta)

    detector = OODDetectorV2(cfg, store)
    cluster_engine = ClusterEngine(cfg)

    stats: list[dict[str, float | int | str]] = []
    for batch in sim.stream():
        stat = detector.process_batch(batch)
        stats.append(stat)
        print(
            f"batch={stat['batch_id']:03d} phase={stat['phase']} "
            f"ood={stat['ood_rate']:.3f} cross={stat['cross_modal_rate']:.3f} "
            f"unknown={stat['unknown_rate']:.3f} consensus={stat['consensus_rate']:.3f}"
        )

    stats_df = pd.DataFrame(stats)
    timeline_path = plot_ood_timeline(stats_df, cfg)

    unknown_vectors, unknown_meta = detector.get_unknown_pool()
    cluster_result = (
        cluster_engine.cluster(unknown_vectors)
        if cluster_engine.should_trigger(len(unknown_meta))
        else {"labels": pd.Series(dtype=int).to_numpy(), "clusters": []}
    )
    representatives: list[int] = []
    for cluster in cluster_result["clusters"]:
        representatives.extend(cluster["representative_indices"])

    labels = cluster_result["labels"]
    labels_arr = (
        labels
        if labels.shape[0] == unknown_vectors.shape[0]
        else pd.Series([-1] * unknown_vectors.shape[0]).to_numpy()
    )
    cluster_path = plot_ood_clusters(unknown_vectors, labels_arr, representatives, cfg)

    scores_df = pd.DataFrame(detector.get_score_rows())
    if bool(cfg.logging_v2.save_scores):
        score_path = Path(cfg.logging_v2.scores_parquet)
        score_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_parquet(score_path, index=False)
        print(f"score parquet: {score_path}")

    print(f"timeline plot: {timeline_path}")
    print(f"cluster plot: {cluster_path}")
    print(f"unknown pool size: {len(unknown_meta)}")
    print(f"clusters found: {len(cluster_result['clusters'])}")


if __name__ == "__main__":
    main()
