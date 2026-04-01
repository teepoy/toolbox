from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import DictConfig

from .stage2_stream import StreamBatch
from .stage3_vector_store_v2 import VectorStoreV2


class OODDetectorV2:
    def __init__(self, cfg: DictConfig, vector_store: VectorStoreV2):
        self.cfg = cfg
        self.vector_store = vector_store
        self.unknown_pool_vectors: list[np.ndarray] = []
        self.unknown_pool_meta: list[dict[str, Any]] = []
        self.score_rows: list[dict[str, Any]] = []

    def process_batch(self, batch: StreamBatch) -> dict[str, float | int | str]:
        detection_cfg = self.cfg.detection_v2
        text_image_sim = np.sum(batch.image_emb * batch.text_emb, axis=1)
        low_text_mask = text_image_sim < float(detection_cfg.text_image_threshold)

        search_results = self.vector_store.search(
            query_embedding=batch.image_emb,
            top_k=int(detection_cfg.top_k),
            aggregate=str(detection_cfg.memory_aggregate),
        )
        best_memory_sim = np.array(
            [float(item["best_similarity"]) for item in search_results],
            dtype=np.float32,
        )
        aggregate_memory_sim = np.array(
            [float(item["aggregate_similarity"]) for item in search_results],
            dtype=np.float32,
        )

        strict_unknown_mask = aggregate_memory_sim < float(
            detection_cfg.memory_threshold
        )
        memory_guard_mask = aggregate_memory_sim < float(
            detection_cfg.memory_consistency_threshold
        )
        consensus_unknown_mask = memory_guard_mask & low_text_mask
        ood_mask = strict_unknown_mask | consensus_unknown_mask

        fused_score = self._compute_fused_score(
            aggregate_memory_sim=aggregate_memory_sim,
            text_image_sim=text_image_sim,
        )
        calibrated_score = np.clip(
            fused_score * float(detection_cfg.calibration_scale)
            + float(detection_cfg.calibration_bias),
            0.0,
            1.0,
        )

        for idx in np.where(strict_unknown_mask)[0]:
            self.unknown_pool_vectors.append(batch.image_emb[idx].copy())
            self.unknown_pool_meta.append(
                {
                    "batch_id": int(batch.batch_id),
                    "sample_id": int(batch.sample_id[idx]),
                    "class_id": int(batch.class_id[idx]),
                    "class_name": str(batch.class_name[idx]),
                    "memory_similarity": float(aggregate_memory_sim[idx]),
                    "best_memory_similarity": float(best_memory_sim[idx]),
                    "text_image_similarity": float(text_image_sim[idx]),
                    "final_ood": bool(ood_mask[idx]),
                }
            )

        for idx, result in enumerate(search_results):
            best_meta = result["best_metadata"]
            self.score_rows.append(
                {
                    "batch_id": int(batch.batch_id),
                    "phase": batch.phase,
                    "sample_id": int(batch.sample_id[idx]),
                    "class_id": int(batch.class_id[idx]),
                    "class_name": str(batch.class_name[idx]),
                    "is_known_class": bool(batch.is_known_class[idx]),
                    "is_new_type": bool(batch.is_new_type[idx]),
                    "is_mismatch": bool(batch.is_mismatch[idx]),
                    "noise_applied": bool(batch.noise_applied[idx]),
                    "text_image_similarity": float(text_image_sim[idx]),
                    "low_text_similarity": bool(low_text_mask[idx]),
                    "best_memory_similarity": float(best_memory_sim[idx]),
                    "memory_similarity": float(aggregate_memory_sim[idx]),
                    "memory_top_k": int(len(result["neighbor_similarities"])),
                    "memory_neighbor_similarities": ",".join(
                        f"{sim:.6f}" for sim in result["neighbor_similarities"]
                    ),
                    "memory_strict_unknown": bool(strict_unknown_mask[idx]),
                    "memory_guard_trigger": bool(memory_guard_mask[idx]),
                    "consensus_unknown": bool(consensus_unknown_mask[idx]),
                    "fused_score": float(fused_score[idx]),
                    "calibrated_score": float(calibrated_score[idx]),
                    "final_ood": bool(ood_mask[idx]),
                    "decision_reason": self._decision_reason(
                        strict_unknown=bool(strict_unknown_mask[idx]),
                        consensus_unknown=bool(consensus_unknown_mask[idx]),
                    ),
                    "neighbor_class_id": self._meta_int(best_meta, "class_id"),
                    "neighbor_class_name": self._meta_str(best_meta, "class_name"),
                    "neighbor_sample_id": self._meta_int(best_meta, "sample_id"),
                }
            )

        return {
            "batch_id": int(batch.batch_id),
            "phase": batch.phase,
            "ood_rate": float(np.mean(ood_mask)),
            "cross_modal_rate": float(np.mean(low_text_mask)),
            "unknown_rate": float(np.mean(strict_unknown_mask)),
            "ood_count": int(np.sum(ood_mask)),
            "cross_modal_count": int(np.sum(low_text_mask)),
            "unknown_count": int(np.sum(strict_unknown_mask)),
            "consensus_rate": float(np.mean(consensus_unknown_mask)),
            "mean_memory_similarity": float(np.mean(aggregate_memory_sim)),
            "mean_text_image_similarity": float(np.mean(text_image_sim)),
            "mean_fused_score": float(np.mean(fused_score)),
            "mean_calibrated_score": float(np.mean(calibrated_score)),
        }

    def get_unknown_pool(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if not self.unknown_pool_vectors:
            return np.empty((0, self.vector_store.dim), dtype=np.float32), []
        return np.stack(self.unknown_pool_vectors).astype(np.float32), list(
            self.unknown_pool_meta
        )

    def get_score_rows(self) -> list[dict[str, Any]]:
        return list(self.score_rows)

    def _compute_fused_score(
        self, aggregate_memory_sim: np.ndarray, text_image_sim: np.ndarray
    ) -> np.ndarray:
        detection_cfg = self.cfg.detection_v2
        memory_component = np.clip(
            float(detection_cfg.memory_consistency_threshold) - aggregate_memory_sim,
            0.0,
            1.0,
        )
        text_component = np.clip(
            float(detection_cfg.text_image_threshold) - text_image_sim,
            0.0,
            1.0,
        )
        return (
            float(detection_cfg.memory_weight) * memory_component
            + float(detection_cfg.text_weight) * text_component
        ).astype(np.float32)

    def _decision_reason(self, strict_unknown: bool, consensus_unknown: bool) -> str:
        if strict_unknown:
            return "strict_memory"
        if consensus_unknown:
            return "memory_text_consensus"
        return "in_distribution"

    def _meta_int(self, metadata: dict[str, Any] | None, key: str) -> int | None:
        if metadata is None or key not in metadata:
            return None
        return int(metadata[key])

    def _meta_str(self, metadata: dict[str, Any] | None, key: str) -> str | None:
        if metadata is None or key not in metadata:
            return None
        return str(metadata[key])
