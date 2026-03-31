"""ConvNeXtV2 backbone wrapper."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
from torch import nn

from cifar_100_benchmark.utils.logging import console


class ConvNeXtV2Backbone(nn.Module):
    def __init__(
        self, model_name: str = "convnextv2_atto", pretrained: bool = True
    ) -> None:
        super().__init__()
        try:
            self.encoder = timm.create_model(
                model_name, pretrained=pretrained, num_classes=0
            )
        except Exception as exc:
            if pretrained:
                console.print(
                    f"[yellow]Pretrained weight load failed ({exc}); falling back to random init for {model_name}.[/yellow]"
                )
                self.encoder = timm.create_model(
                    model_name, pretrained=False, num_classes=0
                )
            else:
                raise
        num_features = getattr(self.encoder, "num_features", None)
        if num_features is None:
            raise RuntimeError(f"Backbone {model_name} does not expose num_features")
        self.out_dim = int(num_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.ndim > 2:
            feats = feats.flatten(1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def load_backbone_weights(backbone: nn.Module, ckpt_path: str | Path) -> None:
    payload = torch.load(str(ckpt_path), map_location="cpu")
    state = payload.get("backbone_state_dict", payload)
    backbone.load_state_dict(state, strict=False)
