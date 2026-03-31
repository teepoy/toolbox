from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from .stage2_stream import StreamBatch
from .stage3_vector_store import VectorStore


class OODDetector:
    def __init__(self, cfg: DictConfig, vector_store: VectorStore):
        self.cfg = cfg
        self.vector_store = vector_store
        self.unknown_pool_vectors: list[np.ndarray] = []
        self.unknown_pool_meta: list[dict[str, int | str]] = []
        self.cross_modal_pool_meta: list[dict[str, int | str]] = []

    def process_batch(self, batch: StreamBatch) -> dict[str, float | int]:
        text_image_sim = np.sum(batch.image_emb * batch.text_emb, axis=1)
        cross_modal_mask = text_image_sim < float(
            self.cfg.detection.text_image_threshold
        )

        search_results = self.vector_store.search(
            query_embedding=batch.image_emb,
            top_k=int(self.cfg.detection.top_k),
            threshold=float(self.cfg.detection.memory_threshold),
        )
        unknown_mask = np.array(
            [not item["is_known"] for item in search_results], dtype=bool
        )

        for idx in np.where(cross_modal_mask)[0]:
            self.cross_modal_pool_meta.append(
                {
                    "batch_id": int(batch.batch_id),
                    "sample_id": int(batch.sample_id[idx]),
                    "class_id": int(batch.class_id[idx]),
                    "class_name": str(batch.class_name[idx]),
                }
            )

        for idx in np.where(unknown_mask)[0]:
            self.unknown_pool_vectors.append(batch.image_emb[idx].copy())
            self.unknown_pool_meta.append(
                {
                    "batch_id": int(batch.batch_id),
                    "sample_id": int(batch.sample_id[idx]),
                    "class_id": int(batch.class_id[idx]),
                    "class_name": str(batch.class_name[idx]),
                }
            )

        ood_mask = cross_modal_mask | unknown_mask
        return {
            "batch_id": int(batch.batch_id),
            "phase": batch.phase,
            "ood_rate": float(np.mean(ood_mask)),
            "cross_modal_rate": float(np.mean(cross_modal_mask)),
            "unknown_rate": float(np.mean(unknown_mask)),
            "ood_count": int(np.sum(ood_mask)),
            "cross_modal_count": int(np.sum(cross_modal_mask)),
            "unknown_count": int(np.sum(unknown_mask)),
        }

    def get_unknown_pool(self) -> tuple[np.ndarray, list[dict[str, int | str]]]:
        if not self.unknown_pool_vectors:
            return np.empty((0, 0), dtype=np.float32), []
        return np.stack(self.unknown_pool_vectors).astype(np.float32), list(
            self.unknown_pool_meta
        )
