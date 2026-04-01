from __future__ import annotations

import json
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

    embeddings_df = build_or_load_embedding_cache(cfg)
    sim = StreamSimulator(embeddings_df, cfg)

    bootstrap_emb, bootstrap_meta = sim.bootstrap_memory_data()
    store = VectorStoreV2(dim=bootstrap_emb.shape[1])
    store.add_signatures(bootstrap_emb, bootstrap_meta)

    detector = OODDetectorV2(cfg, store)
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

    scores_df = pd.DataFrame(detector.get_score_rows())
    score_path = Path(cfg.logging_v2.scores_parquet)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if bool(cfg.logging_v2.save_scores):
        scores_df.to_parquet(score_path, index=False)

    report = {
        "timeline_plot": str(timeline_path),
        "clusters_plot": str(clusters_path),
        "stats_parquet": str(stats_path),
        "score_parquet": str(score_path) if bool(cfg.logging_v2.save_scores) else None,
        "unknown_pool_size": len(unknown_meta),
        "clusters_found": len(cluster_result["clusters"]),
        "metrics": _compute_metrics(scores_df),
    }
    report_path = Path(cfg.paths.artifacts_dir) / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


def _compute_metrics(scores_df: pd.DataFrame) -> dict[str, object]:
    if scores_df.empty:
        return {}

    target = scores_df["is_new_type"].astype(bool) | scores_df["is_mismatch"].astype(bool)
    prediction = scores_df["final_ood"].astype(bool)
    metrics: dict[str, object] = {
        "overall": _binary_metrics(target=target, prediction=prediction),
        "by_phase": {},
    }

    by_phase: dict[str, object] = {}
    for phase, phase_df in scores_df.groupby("phase"):
        phase_target = phase_df["is_new_type"].astype(bool) | phase_df[
            "is_mismatch"
        ].astype(bool)
        phase_prediction = phase_df["final_ood"].astype(bool)
        by_phase[str(phase)] = _binary_metrics(
            target=phase_target,
            prediction=phase_prediction,
        )
    metrics["by_phase"] = by_phase
    return metrics


def _binary_metrics(target: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    tp = int((prediction & target).sum())
    tn = int((~prediction & ~target).sum())
    fp = int((prediction & ~target).sum())
    fn = int((~prediction & target).sum())

    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    accuracy = (tp + tn) / total if total else 0.0

    return {
        "samples": int(total),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "accuracy": float(accuracy),
    }


if __name__ == "__main__":
    main()
