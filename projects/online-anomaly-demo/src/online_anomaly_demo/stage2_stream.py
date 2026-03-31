from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig


@dataclass
class StreamBatch:
    phase: str
    batch_id: int
    image_emb: np.ndarray
    text_emb: np.ndarray
    class_id: np.ndarray
    class_name: list[str]
    sample_id: np.ndarray
    is_known_class: np.ndarray
    is_new_type: np.ndarray
    is_mismatch: np.ndarray
    noise_applied: np.ndarray


class StreamSimulator:
    def __init__(self, embeddings_df: pd.DataFrame, cfg: DictConfig):
        self.cfg = cfg
        self.batch_size = int(cfg.stream.batch_size)
        self.rng = np.random.default_rng(int(cfg.seed))
        self.batch_counter = 0
        self.known_ids = set(int(x) for x in cfg.data.known_class_ids)
        self.new_ids = set(int(x) for x in cfg.data.new_class_ids)

        self.df = embeddings_df.copy()
        self.df["image_emb"] = self.df["image_emb"].apply(
            lambda x: np.asarray(x, dtype=np.float32)
        )
        self.df["text_emb"] = self.df["text_emb"].apply(
            lambda x: np.asarray(x, dtype=np.float32)
        )

        self.known_df = self.df[self.df["class_id"].isin(self.known_ids)].reset_index(
            drop=True
        )
        self.new_df = self.df[self.df["class_id"].isin(self.new_ids)].reset_index(
            drop=True
        )

    def bootstrap_memory_data(self) -> tuple[np.ndarray, list[dict[str, int | str]]]:
        image_emb = np.stack(self.known_df["image_emb"].to_list()).astype(np.float32)
        metadata = self.known_df[["sample_id", "class_id", "class_name"]].to_dict(
            orient="records"
        )
        return image_emb, metadata

    def stream(self):
        for _ in range(int(self.cfg.stream.t0_steps)):
            yield self._sample_t0()
        for _ in range(int(self.cfg.stream.t1_steps)):
            yield self._sample_t1()
        for _ in range(int(self.cfg.stream.t2_steps)):
            yield self._sample_t2()

    def _sample_t0(self) -> StreamBatch:
        sampled = self.known_df.sample(
            n=self.batch_size, replace=True, random_state=self._rs()
        ).reset_index(drop=True)
        batch = self._to_batch(sampled, phase="T0")
        return batch

    def _sample_t1(self) -> StreamBatch:
        major_id = int(self.cfg.stream.t1_major_class_id)
        major_ratio = float(self.cfg.stream.t1_major_class_ratio)
        major_n = int(round(self.batch_size * major_ratio))
        minor_n = self.batch_size - major_n

        major_df = self.known_df[self.known_df["class_id"] == major_id]
        minor_df = self.known_df[self.known_df["class_id"] != major_id]
        sampled = (
            pd.concat(
                [
                    major_df.sample(n=major_n, replace=True, random_state=self._rs()),
                    minor_df.sample(n=minor_n, replace=True, random_state=self._rs()),
                ],
                ignore_index=True,
            )
            .sample(frac=1.0, random_state=self._rs())
            .reset_index(drop=True)
        )

        batch = self._to_batch(sampled, phase="T1")
        noise_std = float(self.cfg.stream.t1_noise_std)
        batch.image_emb = self._normalize(
            batch.image_emb
            + self.rng.normal(0.0, noise_std, batch.image_emb.shape).astype(np.float32)
        )
        batch.text_emb = self._normalize(
            batch.text_emb
            + self.rng.normal(0.0, noise_std, batch.text_emb.shape).astype(np.float32)
        )
        batch.noise_applied[:] = True
        return batch

    def _sample_t2(self) -> StreamBatch:
        new_ratio = float(self.cfg.stream.t2_new_class_ratio)
        mismatch_ratio = float(self.cfg.stream.t2_mismatch_ratio)
        new_n = int(round(self.batch_size * new_ratio))
        known_n = self.batch_size - new_n

        sampled = (
            pd.concat(
                [
                    self.known_df.sample(
                        n=known_n, replace=True, random_state=self._rs()
                    ),
                    self.new_df.sample(n=new_n, replace=True, random_state=self._rs()),
                ],
                ignore_index=True,
            )
            .sample(frac=1.0, random_state=self._rs())
            .reset_index(drop=True)
        )

        batch = self._to_batch(sampled, phase="T2")
        mismatch_n = int(round(self.batch_size * mismatch_ratio))
        mismatch_idx = self.rng.choice(self.batch_size, size=mismatch_n, replace=False)
        self._apply_text_mismatch(batch, mismatch_idx)
        return batch

    def _apply_text_mismatch(
        self, batch: StreamBatch, mismatch_idx: np.ndarray
    ) -> None:
        for idx in mismatch_idx:
            target_class = batch.class_id[idx]
            candidates = np.where(batch.class_id != target_class)[0]
            if len(candidates) == 0:
                continue
            swap_idx = int(self.rng.choice(candidates))
            batch.text_emb[idx] = batch.text_emb[swap_idx]
            batch.is_mismatch[idx] = True

    def _to_batch(self, sampled: pd.DataFrame, phase: str) -> StreamBatch:
        image_emb = np.stack(sampled["image_emb"].to_list()).astype(np.float32)
        text_emb = np.stack(sampled["text_emb"].to_list()).astype(np.float32)
        class_id = sampled["class_id"].to_numpy(dtype=np.int64)
        is_new_type = np.array([cid in self.new_ids for cid in class_id], dtype=bool)
        is_known = ~is_new_type

        batch = StreamBatch(
            phase=phase,
            batch_id=self.batch_counter,
            image_emb=image_emb,
            text_emb=text_emb,
            class_id=class_id,
            class_name=sampled["class_name"].tolist(),
            sample_id=sampled["sample_id"].to_numpy(dtype=np.int64),
            is_known_class=is_known,
            is_new_type=is_new_type,
            is_mismatch=np.zeros(len(sampled), dtype=bool),
            noise_applied=np.zeros(len(sampled), dtype=bool),
        )
        self.batch_counter += 1
        return batch

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, 1e-12, None)

    def _rs(self) -> int:
        return int(self.rng.integers(0, 1_000_000_000))
