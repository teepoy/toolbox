# learnings.md — convnext32

## Architecture Conventions (from codebase)

- Backbone wrappers expose `self.out_dim: int` and `forward_features(x) -> Tensor(B, D)`
- `forward(x)` delegates to `forward_features(x)` for pure backbone mode
- `timm.create_model(..., num_classes=0)` is used for feature extractor mode in convnextv2.py
  — our ConvNeXt32 implements this natively via `num_classes` param
- `load_backbone_weights` helper pattern exists in convnextv2.py — replicate if needed
- `nn.Module` subclass, `slots=True` dataclasses used elsewhere
- Logging: use `from cifar_100_benchmark.utils.logging import console` for rich output
- Type hints required on all public methods
- Import order: stdlib → third-party → local (cifar_100_benchmark.*)

## Config Conventions

- Model configs live in `configs/model/<name>.yaml`
- Must have: `model.backbone.name`, `model.backbone.pretrained`, `model.num_classes`
- Head config always included: `model.head.name: linear` (or view_fusion)
- `backbone.model_name` is timm-specific — not needed for custom backbones

## Builder Pattern

- `build_backbone(cfg: DictConfig)` dispatches on `cfg.name`
- Must return an `nn.Module` with `.out_dim` attribute
- `build_classifier(cfg)` calls `build_backbone` then `build_head` — no changes needed there

## Runner Family Pattern

- `experiment.py` dispatches families in a loop: `for family in cfg.experiment.families`
- SSL families handled separately (`byol`, `mocov3`, `supcon`, `dino`)
- Non-SSL families: `official`, `random`, `yolo26n`, and now `convnext32`
- Family sets `local_cfg.model.backbone.name` and `local_cfg.model.backbone.pretrained`

## Pre-existing LSP Errors (do NOT fix)

- `builders.py` lines 29, 66: pyright false positives on DictConfig attribute access
- `yolo26.py` line 11: YOLO import from ultralytics (ultralytics API change, pre-existing)

## 2026-04-01 — Verification
- Confirmed convnext32.py matches the required 40→80→160→320 channel path, 2/2/6/2 depth split, and 3×3 stride-1 stem.
- The model uses 12 total ConvNeXt blocks with a linear drop-path schedule from 0.0 to drop_path_rate.
- forward_features returns a 2D (B, 320) tensor after GAP → Flatten → LayerNorm.
