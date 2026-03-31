# decisions.md — convnext32

## 2026-03-31 — Core Architecture Decisions

### Stem: No Spatial Reduction
**Decision**: Stem = Conv2d(3→40, 3×3, s1, p1) + LN. Zero spatial downsampling.
**Rationale**: Original 4×4 s4 stem on 32×32 → 8×8 immediately. With 3 more ÷2 stages,
this collapses to 1×1. Preserving 32×32 into Stage 1 lets the network use full spatial info.
**Precedent**: CIFAR ResNets famously use 3×3 s1 stem for same reason.

### Final Feature Map: 4×4
**Decision**: 3 downsampling stages → 32→16→8→4. Target 4×4 before GAP.
**Rationale**: 4/32 = 0.125 ≈ 7/56 = 0.125. Exact proportional match to original design.

### Keep 7×7 Depthwise Kernel Everywhere
**Decision**: All blocks use 7×7 DWConv with pad=3. Including Stage 4 on 4×4 maps.
**Rationale**: At 4×4 with pad=3, the kernel covers entire feature map — effectively global.
This mirrors the original design where Stage 4 also uses 7×7 on 7×7 maps.
No evidence of quality degradation from padding artifacts in literature.

### ConvNeXt V2 Block (with GRN)
**Decision**: Use GRN (Global Response Normalization) from V2, not plain V1.
**Rationale**: GRN improves channel competition and is included in convnextv2_atto weights.
Since we're training from scratch for CIFAR, GRN's regularization effect is valuable.

### Pure PyTorch, No timm Patching
**Decision**: Implement from scratch, no timm.create_model + monkey-patch.
**Rationale**: Cleaner, debuggable, testable. Atto is ~150 LOC. Avoids brittle timm
internal attribute access (encoder.stem.0 etc.) that breaks across timm versions.

### out_dim = 320
**Decision**: Final backbone embedding dimension is 320 (Stage 4 channel width).
**Rationale**: Matches Atto's channel progression [40,80,160,320]. Downstream head
code uses backbone.out_dim — this is the contract.

### No Pretrained Weights
**Decision**: `pretrained=False` hardcoded in config. No ImageNet init.
**Rationale**: Architecture is non-standard (modified stem) so no matching pretrained
weights exist. Will rely on SSL pretraining or random init within benchmark.

## 2026-04-01 — Verification
### Status: Checklist passed
**Decision**: Keep current convnext32 architecture unchanged.
**Rationale**: Source and runtime checks confirm the intended CIFAR-appropriate stem, downsampling pipeline, GRN block internals, unconditional out_dim=320, and shape contract.
