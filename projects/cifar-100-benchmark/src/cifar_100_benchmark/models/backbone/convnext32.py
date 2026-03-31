"""ConvNeXt backbone natively designed for 32×32 CIFAR images.

Architecture overview:
  Stem (s1, 32×32) → Stage1 (2 blocks) → Down (16×16) → Stage2 (2 blocks)
  → Down (8×8) → Stage3 (6 blocks) → Down (4×4) → Stage4 (2 blocks)
  → GAP → LayerNorm → (B, 320)

Channel progression: 40 → 80 → 160 → 320 (mirrors ConvNeXt-Atto).
Uses ConvNeXt V2 blocks with Global Response Normalization (GRN).
No timm dependency — pure PyTorch implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Stochastic depth (DropPath) — native implementation, no timm required
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Per-sample stochastic depth (Drop Path) regularization."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * noise.div_(keep_prob)


# ---------------------------------------------------------------------------
# Utility: channels-first LayerNorm wrapper
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first (B, C, H, W) tensors.

    Permutes to channels-last, applies nn.LayerNorm on the last dim,
    then permutes back to channels-first.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, H, W) → (B, H, W, C) → norm → (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x


# ---------------------------------------------------------------------------
# Global Response Normalization (from ConvNeXt V2)
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """Global Response Normalization layer (ConvNeXt V2).

    Operates on channels-last (B, H, W, C) tensors.
    Learnable gamma and beta initialized to zero.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, C)
        gx = x.norm(p=2, dim=(1, 2), keepdim=True)           # (B, 1, 1, C)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)     # (B, 1, 1, C)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXt V2 Block
# ---------------------------------------------------------------------------

class ConvNeXtBlock(nn.Module):
    """ConvNeXt V2 block with depthwise conv, LayerNorm, GRN and DropPath.

    Layout (channels-first input/output):
      DWConv(7×7) → channels-last → LN → PW-expand → GELU → GRN
      → PW-project → channels-first → DropPath residual
    """

    def __init__(self, dim: int, kernel_size: int = 7, drop_path: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        # Depthwise — channels-first
        x = self.dwconv(x)
        # Switch to channels-last for MLP
        x = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        # Back to channels-first
        x = x.permute(0, 3, 1, 2)           # (B, C, H, W)
        x = residual + self.drop_path(x)
        return x


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------

class ConvNeXt32Backbone(nn.Module):
    """ConvNeXt backbone natively designed for 32×32 CIFAR images.

    Feature extractor contract:
      - ``self.out_dim = 320`` always
      - ``forward_features(x)`` returns ``(B, 320)``
      - ``forward(x)`` delegates to ``forward_features``

    Args:
        num_classes: When 0, operates as pure feature extractor.
            When > 0, attaches ``self.head = nn.Linear(320, num_classes)``.
        drop_path_rate: Stochastic depth rate. Linearly scheduled across
            all 12 blocks (0.0 at block 0, ``drop_path_rate`` at block 11).
    """

    out_dim: int = 320

    def __init__(
        self,
        num_classes: int = 0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()

        # Linearly spaced drop-path rates across all 12 blocks
        depths = [2, 2, 6, 2]   # blocks per stage
        total_blocks = sum(depths)
        dp_rates: list[float] = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()
        dp_idx = 0

        # Stem: 32×32 → 32×32, no spatial reduction
        self.stem = nn.Sequential(
            nn.Conv2d(3, 40, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(40),
        )

        # Stage 1  (32×32, dim=40)
        self.stage1 = nn.Sequential(
            *[ConvNeXtBlock(dim=40, kernel_size=7, drop_path=dp_rates[dp_idx + i])
              for i in range(depths[0])]
        )
        dp_idx += depths[0]

        # Downsample 1: 32×32 → 16×16
        self.down1 = nn.Sequential(
            LayerNorm2d(40),
            nn.Conv2d(40, 80, kernel_size=2, stride=2),
        )

        # Stage 2  (16×16, dim=80)
        self.stage2 = nn.Sequential(
            *[ConvNeXtBlock(dim=80, kernel_size=7, drop_path=dp_rates[dp_idx + i])
              for i in range(depths[1])]
        )
        dp_idx += depths[1]

        # Downsample 2: 16×16 → 8×8
        self.down2 = nn.Sequential(
            LayerNorm2d(80),
            nn.Conv2d(80, 160, kernel_size=2, stride=2),
        )

        # Stage 3  (8×8, dim=160)
        self.stage3 = nn.Sequential(
            *[ConvNeXtBlock(dim=160, kernel_size=7, drop_path=dp_rates[dp_idx + i])
              for i in range(depths[2])]
        )
        dp_idx += depths[2]

        # Downsample 3: 8×8 → 4×4
        self.down3 = nn.Sequential(
            LayerNorm2d(160),
            nn.Conv2d(160, 320, kernel_size=2, stride=2),
        )

        # Stage 4  (4×4, dim=320)
        self.stage4 = nn.Sequential(
            *[ConvNeXtBlock(dim=320, kernel_size=7, drop_path=dp_rates[dp_idx + i])
              for i in range(depths[3])]
        )

        # Neck: GAP → flatten → LayerNorm
        self.neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(320),
        )

        # Optional classification head
        if num_classes > 0:
            self.head: nn.Module = nn.Linear(320, num_classes)
        else:
            self.head = nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize Conv2d and Linear layers with trunc_normal_ / zeros."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        """Extract (B, 320) feature vector from a (B, 3, 32, 32) input."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.neck(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass; delegates to ``forward_features``."""
        return self.forward_features(x)
