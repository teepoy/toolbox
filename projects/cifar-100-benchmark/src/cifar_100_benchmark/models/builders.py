"""Model builders for backbones and heads."""

from __future__ import annotations

from dataclasses import dataclass

from omegaconf import DictConfig
from torch import nn

from cifar_100_benchmark.models.backbone.convnext32 import ConvNeXt32Backbone
from cifar_100_benchmark.models.backbone.convnextv2 import ConvNeXtV2Backbone
from cifar_100_benchmark.models.backbone.yolo26 import YOLO26Backbone
from cifar_100_benchmark.models.head.classifier import LinearHead
from cifar_100_benchmark.models.head.view_fusion import ViewFusionHead


@dataclass(slots=True)
class BuiltModel:
    backbone: nn.Module
    head: nn.Module


class ClassifierModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)


def build_backbone(cfg: DictConfig) -> nn.Module:
    name = str(cfg.name)
    if name == "convnextv2_atto":
        return ConvNeXtV2Backbone(
            model_name=str(cfg.model_name),
            pretrained=bool(cfg.pretrained),
        )
    if name == "convnext32_atto":
        return ConvNeXt32Backbone(
            num_classes=0,
            drop_path_rate=float(getattr(cfg, "drop_path_rate", 0.1)),
        )
    if name == "yolo26n":
        return YOLO26Backbone(num_classes=int(cfg.num_classes))
    raise ValueError(f"Unknown backbone: {name}")


def build_head(cfg: DictConfig, in_dim: int, num_classes: int) -> nn.Module:
    name = str(cfg.name)
    if name == "linear":
        return LinearHead(in_dim=in_dim, num_classes=num_classes)
    if name == "view_fusion":
        return ViewFusionHead(
            in_dim=in_dim,
            num_classes=num_classes,
            attn_dim=int(cfg.attn_dim),
            num_views=int(cfg.num_views),
            num_heads=int(cfg.num_heads),
            num_layers=int(cfg.num_layers),
        )
    raise ValueError(f"Unknown head: {name}")


def build_classifier(cfg: DictConfig) -> ClassifierModel:
    backbone = build_backbone(cfg.backbone)
    in_dim = int(backbone.out_dim)
    head = build_head(cfg.head, in_dim=in_dim, num_classes=int(cfg.num_classes))
    return ClassifierModel(backbone=backbone, head=head)
