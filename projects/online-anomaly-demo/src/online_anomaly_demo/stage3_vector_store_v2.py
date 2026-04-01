from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class VectorStoreV2:
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
        self, query_embedding: np.ndarray, top_k: int, aggregate: str
    ) -> list[dict[str, Any]]:
        query = self._normalize(query_embedding.astype(np.float32))
        if self.index.ntotal == 0:
            return [self._empty_result() for _ in range(query.shape[0])]

        effective_top_k = max(1, min(int(top_k), int(self.index.ntotal)))
        sims, idx = self.index.search(query, effective_top_k)
        results: list[dict[str, Any]] = []
        for row in range(query.shape[0]):
            valid_mask = idx[row] >= 0
            valid_idx = idx[row][valid_mask]
            valid_sims = sims[row][valid_mask].astype(np.float32)

            if valid_sims.size == 0:
                results.append(self._empty_result())
                continue

            best_idx = int(valid_idx[0])
            neighbor_metadata = [self.metadata[int(i)] for i in valid_idx.tolist()]
            results.append(
                {
                    "best_similarity": float(valid_sims[0]),
                    "aggregate_similarity": self._aggregate_similarity(
                        valid_sims, aggregate
                    ),
                    "neighbor_similarities": valid_sims.tolist(),
                    "best_index": best_idx,
                    "best_metadata": self.metadata[best_idx],
                    "neighbor_metadata": neighbor_metadata,
                }
            )
        return results

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)

    def _aggregate_similarity(self, sims: np.ndarray, aggregate: str) -> float:
        mode = aggregate.lower()
        if mode == "max":
            return float(np.max(sims))
        if mode == "median":
            return float(np.median(sims))
        if mode == "min":
            return float(np.min(sims))
        return float(np.mean(sims))

    def _empty_result(self) -> dict[str, Any]:
        return {
            "best_similarity": -1.0,
            "aggregate_similarity": -1.0,
            "neighbor_similarities": [],
            "best_index": -1,
            "best_metadata": None,
            "neighbor_metadata": [],
        }
