# issues.md — convnext32

## Known Pre-existing Issues (do NOT fix in this plan)

- `builders.py` lines 29, 66: Pyright type errors on `DictConfig` attribute access (false positives)
- `yolo26.py` line 11: `YOLO` import from `ultralytics` — ultralytics API changed, pre-existing
- Default image_size is 64 in configs/default.yaml — smoke experiment likely uses imgsz=64;
  ConvNeXt32 handles any size ≥8 due to adaptive pooling, so this is fine

## Potential Gotchas

- `drop_path_rate` must be accessed via `cfg.get("drop_path_rate", 0.1)` since OmegaConf
  DictConfig doesn't support `.get()` — use `getattr(cfg, "drop_path_rate", 0.1)` or
  add it as a required field in the YAML config
- GRN requires computing spatial mean: `x.mean(dim=(1,2), keepdim=True)` — shape must be
  (B, H, W, C) in channels-last layout during block computation
- `channels_last` LayerNorm: use `normalized_shape=(C,)` with `data_format="channels_last"`
  as a wrapper, OR convert to channels-first before LN and back — be consistent

## 2026-04-01 — Verification
- No new issues found during checklist verification or runtime sanity check.
