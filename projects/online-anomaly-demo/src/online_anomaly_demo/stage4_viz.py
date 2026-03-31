from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from sklearn.manifold import TSNE
from umap import UMAP


def plot_ood_timeline(stats_df: pd.DataFrame, cfg: DictConfig) -> Path:
    out_path = Path(cfg.paths.plots_dir) / "ood_timeline.png"
    plt.figure(figsize=(10, 4), dpi=int(cfg.visualization.figure_dpi))
    plt.plot(stats_df["batch_id"], stats_df["ood_rate"], label="OOD rate", linewidth=2)
    plt.plot(
        stats_df["batch_id"],
        stats_df["cross_modal_rate"],
        label="Cross-modal",
        alpha=0.7,
    )
    plt.plot(
        stats_df["batch_id"], stats_df["unknown_rate"], label="Unknown class", alpha=0.7
    )

    t0_end = int(cfg.stream.t0_steps) - 1
    t1_end = int(cfg.stream.t0_steps) + int(cfg.stream.t1_steps) - 1
    plt.axvline(t0_end, linestyle="--", linewidth=1, color="gray")
    plt.axvline(t1_end, linestyle="--", linewidth=1, color="gray")
    plt.text(0, 1.02, "T0", transform=plt.gca().transAxes)
    plt.text(0.35, 1.02, "T1", transform=plt.gca().transAxes)
    plt.text(0.7, 1.02, "T2", transform=plt.gca().transAxes)

    plt.xlabel("Batch Number")
    plt.ylabel("Rate")
    plt.title("OOD Detection Rate Over Time")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_ood_clusters(
    ood_vectors: np.ndarray,
    labels: np.ndarray,
    representative_indices: list[int],
    cfg: DictConfig,
) -> Path:
    out_path = Path(cfg.paths.plots_dir) / "ood_clusters.png"
    if ood_vectors.shape[0] == 0:
        plt.figure(figsize=(8, 6), dpi=int(cfg.visualization.figure_dpi))
        plt.text(0.5, 0.5, "No OOD vectors available", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_path)
        plt.close()
        return out_path

    reduced = _reduce_to_2d(ood_vectors, cfg)
    plt.figure(figsize=(8, 6), dpi=int(cfg.visualization.figure_dpi))
    unique_labels = sorted(set(labels.tolist()))
    for label in unique_labels:
        mask = labels == label
        name = "noise" if label == -1 else f"cluster {label}"
        plt.scatter(reduced[mask, 0], reduced[mask, 1], s=18, alpha=0.8, label=name)

    if representative_indices:
        rep = reduced[representative_indices]
        plt.scatter(
            rep[:, 0], rep[:, 1], marker="*", s=220, c="black", label="representatives"
        )

    plt.title("T2 OOD Pool Clusters")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(alpha=0.2)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def _reduce_to_2d(vectors: np.ndarray, cfg: DictConfig) -> np.ndarray:
    reducer_name = str(cfg.visualization.reducer).lower()
    if reducer_name == "umap":
        reducer = UMAP(
            n_components=2,
            n_neighbors=int(cfg.visualization.umap_n_neighbors),
            min_dist=float(cfg.visualization.umap_min_dist),
            random_state=int(cfg.seed),
        )
        return reducer.fit_transform(vectors)

    tsne = TSNE(
        n_components=2,
        perplexity=float(cfg.visualization.tsne_perplexity),
        random_state=int(cfg.seed),
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(vectors)
