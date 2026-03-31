from __future__ import annotations

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
    df = build_or_load_embedding_cache(cfg)

    sim = StreamSimulator(df, cfg)
    memory_emb, memory_meta = sim.bootstrap_memory_data()
    store = VectorStore(dim=memory_emb.shape[1])
    store.add_signatures(memory_emb, memory_meta)

    detector = OODDetector(cfg, store)
    cluster_engine = ClusterEngine(cfg)

    stats = []
    for batch in sim.stream():
        stat = detector.process_batch(batch)
        stats.append(stat)
        print(
            f"batch={stat['batch_id']:03d} phase={stat['phase']} "
            f"ood={stat['ood_rate']:.3f} cross={stat['cross_modal_rate']:.3f} "
            f"unknown={stat['unknown_rate']:.3f}"
        )

    stats_df = pd.DataFrame(stats)
    timeline_path = plot_ood_timeline(stats_df, cfg)

    unknown_vectors, unknown_meta = detector.get_unknown_pool()
    cluster_result = (
        cluster_engine.cluster(unknown_vectors)
        if cluster_engine.should_trigger(len(unknown_meta))
        else {"labels": pd.Series(dtype=int).to_numpy(), "clusters": []}
    )
    rep = []
    for cluster in cluster_result["clusters"]:
        rep.extend(cluster["representative_indices"])

    labels = cluster_result["labels"]
    labels_arr = (
        labels
        if labels.shape[0] == unknown_vectors.shape[0]
        else pd.Series([-1] * unknown_vectors.shape[0]).to_numpy()
    )
    cluster_path = plot_ood_clusters(unknown_vectors, labels_arr, rep, cfg)

    print(f"timeline plot: {timeline_path}")
    print(f"cluster plot: {cluster_path}")
    print(f"unknown pool size: {len(unknown_meta)}")
    print(f"clusters found: {len(cluster_result['clusters'])}")


if __name__ == "__main__":
    main()
