from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict[str, Any]] = []

    def add_signatures(
        self, embeddings: np.ndarray, metadata: list[dict[str, Any]]
    ) -> None:
        emb = self._normalize(embeddings.astype(np.float32))
        self.index.add(emb)
        self.metadata.extend(metadata)

    def search(
        self, query_embedding: np.ndarray, top_k: int, threshold: float
    ) -> list[dict[str, Any]]:
        query = self._normalize(query_embedding.astype(np.float32))
        sims, idx = self.index.search(query, top_k)
        results: list[dict[str, Any]] = []
        for row in range(query.shape[0]):
            best_sim = float(sims[row, 0]) if idx[row, 0] >= 0 else -1.0
            best_idx = int(idx[row, 0]) if idx[row, 0] >= 0 else -1
            hit_meta = self.metadata[best_idx] if best_idx >= 0 else None
            results.append(
                {
                    "best_similarity": best_sim,
                    "is_known": best_sim >= threshold,
                    "best_index": best_idx,
                    "best_metadata": hit_meta,
                }
            )
        return results

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)
