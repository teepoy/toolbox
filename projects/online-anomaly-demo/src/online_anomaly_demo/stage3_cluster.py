from __future__ import annotations

import hdbscan
import numpy as np
from omegaconf import DictConfig


class ClusterEngine:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def should_trigger(self, unknown_pool_size: int) -> bool:
        return unknown_pool_size >= int(self.cfg.clustering.trigger_size)

    def cluster(self, vectors: np.ndarray) -> dict[str, object]:
        if vectors.shape[0] == 0:
            return {"labels": np.array([]), "clusters": []}

        model = hdbscan.HDBSCAN(
            min_cluster_size=int(self.cfg.clustering.min_cluster_size),
            min_samples=int(self.cfg.clustering.min_samples),
        )
        labels = model.fit_predict(vectors)

        clusters = []
        for label in sorted(set(labels)):
            if label == -1:
                continue
            idx = np.where(labels == label)[0]
            cluster_vectors = vectors[idx]
            center = cluster_vectors.mean(axis=0)
            dists = np.linalg.norm(cluster_vectors - center, axis=1)
            rep_n = int(self.cfg.clustering.representatives_per_cluster)
            local_rep = np.argsort(dists)[:rep_n]
            global_rep = idx[local_rep]
            clusters.append(
                {
                    "label": int(label),
                    "size": int(len(idx)),
                    "center": center,
                    "representative_indices": global_rep.tolist(),
                }
            )

        return {"labels": labels, "clusters": clusters}
