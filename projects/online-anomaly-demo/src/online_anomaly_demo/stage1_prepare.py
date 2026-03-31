from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .class_names import FLOWERS102_CLASS_NAMES


def build_or_load_embedding_cache(cfg: DictConfig) -> pd.DataFrame:
    cache_path = Path(cfg.paths.cache_parquet)
    if cache_path.exists() and not bool(cfg.embedding.force_recompute):
        return pd.read_parquet(cache_path)

    df = _load_hf_dataframe(cfg)
    image_emb, text_emb = _extract_clip_embeddings(df, cfg)
    out_df = df.drop(columns=["image"]).copy()
    out_df["image_emb"] = [vec.tolist() for vec in image_emb]
    out_df["text_emb"] = [vec.tolist() for vec in text_emb]
    out_df.to_parquet(cache_path, index=False)
    return out_df


def _load_hf_dataframe(cfg: DictConfig) -> pd.DataFrame:
    ds_dict = load_dataset(cfg.data.hf_dataset)
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for split_name, split_ds in ds_dict.items():
        for sample in split_ds:
            class_id = int(sample["label"])
            class_name = FLOWERS102_CLASS_NAMES[class_id]
            prompt = str(cfg.data.prompt_template).format(class_name=class_name)
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split_name,
                    "class_id": class_id,
                    "class_name": class_name,
                    "prompt": prompt,
                    "image": sample["image"],
                }
            )
            sample_id += 1
    return pd.DataFrame(rows)


def _extract_clip_embeddings(
    df: pd.DataFrame, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    model_name = str(cfg.embedding.model_name)
    device = _resolve_device(str(cfg.embedding.device))
    batch_size = int(cfg.embedding.batch_size)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    image_all: list[np.ndarray] = []
    text_all: list[np.ndarray] = []

    with torch.no_grad():
        for start in tqdm(range(0, len(df), batch_size), desc="CLIP embedding"):
            end = min(start + batch_size, len(df))
            batch = df.iloc[start:end]
            images = batch["image"].tolist()
            prompts = batch["prompt"].tolist()

            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_features = _extract_feature_tensor(model.get_image_features(**image_inputs))
            image_features = (
                _l2_normalize(image_features).cpu().numpy().astype(np.float32)
            )

            text_inputs = processor(
                text=prompts, return_tensors="pt", padding=True, truncation=True
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_features = _extract_feature_tensor(model.get_text_features(**text_inputs))
            text_features = (
                _l2_normalize(text_features).cpu().numpy().astype(np.float32)
            )

            image_all.append(image_features)
            text_all.append(text_features)

    return np.concatenate(image_all, axis=0), np.concatenate(text_all, axis=0)


def _resolve_device(device_cfg: str) -> str:
    if device_cfg != "auto":
        return device_cfg
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _extract_feature_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]
    raise TypeError(f"Unsupported feature output type: {type(output)}")
