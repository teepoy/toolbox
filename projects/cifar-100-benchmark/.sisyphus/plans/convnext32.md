# Plan: ConvNeXt-32 — ConvNeXt Adapted for 32×32 Images

## Overview

Build a custom ConvNeXt variant designed natively for 32×32 CIFAR inputs.
The key change: replace the aggressive 4×4 s4 stem (which crushes 32→8) with a
3×3 s1 "channel-projection" stem so spatial resolution is preserved through Stage 1.
Downsampling is deferred to 3 inter-stage layers (÷2 each), yielding a 4×4 final
feature map before GAP — proportionally equivalent to the original 7×7 for 224×224.

Architecture summary:
- Stem: Conv2d(3→40, 3×3, s1, p1) + LN  [no spatial reduction]
- Stage 1: 2 blocks, C=40,  32×32
- ↓ DS1:   LN → Conv2d(40→80, 2×2, s2)   → 16×16
- Stage 2: 2 blocks, C=80,  16×16
- ↓ DS2:   LN → Conv2d(80→160, 2×2, s2)  → 8×8
- Stage 3: 6 blocks, C=160, 8×8
- ↓ DS3:   LN → Conv2d(160→320, 2×2, s2) → 4×4
- Stage 4: 2 blocks, C=320, 4×4
- Head: GAP → LN → Linear(320→num_classes)
- out_dim = 320, params ≈ 3.5M

Block internals (ConvNeXt V2 style):
  DWConv(7×7) → LN → PW(C→4C) → GELU → GRN → PW(4C→C) → DropPath → residual

## TODOs

- [x] Implement `ConvNeXt32Backbone` in `src/cifar_100_benchmark/models/backbone/convnext32.py`
  - Pure PyTorch — no timm dependency
  - Stem: Conv2d(3→40, 3×3, s1, p1) → LayerNorm(40, eps=1e-6, data_format="channels_last")
  - ConvNeXtBlock with: DWConv(7×7, pad=3, groups=C), LN, PW(C→4C), GELU, GRN, PW(4C→C), DropPath
  - DownsampleLayer: LN(C, channels_last) → Conv2d(C→2C, 2×2, s2)
  - Stages: depths=[2,2,6,2], channels=[40,80,160,320]
  - Head: AdaptiveAvgPool2d(1) → flatten → LN(320) → Linear(320→num_classes)
  - Expose `self.out_dim = 320` and `forward_features(x)` returning (B, 320) tensor
  - Accept `num_classes: int = 0` (0 = feature extractor mode, >0 = full classifier mode)
  - Accept `drop_path_rate: float = 0.1` with linear schedule across all blocks

- [x] Register `convnext32_atto` in `src/cifar_100_benchmark/models/builders.py`
  - Import `ConvNeXt32Backbone` from `cifar_100_benchmark.models.backbone.convnext32`
  - Add `if name == "convnext32_atto":` branch in `build_backbone()` that constructs `ConvNeXt32Backbone(num_classes=0, drop_path_rate=float(cfg.get("drop_path_rate", 0.1)))`
  - `out_dim` must be 320

- [x] Create config preset `configs/model/convnext32_atto.yaml`
  - Mirror structure of `configs/model/convnextv2_atto.yaml`
  - `backbone.name: convnext32_atto`
  - `backbone.pretrained: false`
  - `backbone.drop_path_rate: 0.1`
  - No `backbone.model_name` needed (not timm-based)

- [x] Add `convnext32` family mapping in `src/cifar_100_benchmark/runner/experiment.py`
  - In the finetune family dispatch block, add: `elif family == "convnext32": local_cfg.model.backbone.name = "convnext32_atto"; local_cfg.model.backbone.pretrained = False`
  - In the SSL family check block, skip `convnext32` (it is a finetune-only baseline, not an SSL method)

- [x] Smoke-test verification
  - Run: `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.cli --experiment smoke`
  - Verify no errors and metrics are logged

## Final Verification Wave

- [x] F1 — Architecture correctness review: verify spatial dims 32→16→8→4, out_dim=320, block structure (DWConv+LN+GRN+DropPath), param count ~3.5M
- [x] F2 — Integration review: verify builders.py dispatch, config YAML completeness, runner family mapping
- [x] F3 — Code quality review: Python type hints, import ordering (stdlib→third-party→local), no hardcoded magic numbers, no broad except blocks (pre-existing ClassifierModel type-hint absence waived — not introduced by this plan)
- [x] F4 — Smoke run passes: `PYTHONPATH=src .venv/bin/python -m cifar_100_benchmark.cli --experiment smoke` exits 0 with convnext32 family included
