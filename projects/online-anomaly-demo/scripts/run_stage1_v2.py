from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from online_anomaly_demo.config_v2 import load_config_v2
from online_anomaly_demo.stage1_prepare import build_or_load_embedding_cache


def main() -> None:
    cfg = load_config_v2()
    df = build_or_load_embedding_cache(cfg)
    print(f"cache rows: {len(df)}")
    print(f"classes: {df['class_id'].nunique()}")
    emb_dim = len(df.iloc[0]["image_emb"]) if len(df) else 0
    print(f"embedding dim: {emb_dim}")
    print(f"cache path: {cfg.paths.cache_parquet}")
    print(f"artifacts dir: {cfg.paths.artifacts_dir}")


if __name__ == "__main__":
    main()
