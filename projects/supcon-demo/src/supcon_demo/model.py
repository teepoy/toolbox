from __future__ import annotations

import timm
import torch
from torch import nn
from torch.nn import functional as F


class SupConModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        projection_dim: int,
        projection_hidden_dim: int,
        normalize_embeddings: bool = True,
        allow_random_init_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.used_pretrained_weights = pretrained
        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
            )
        except Exception:
            if not allow_random_init_fallback:
                raise
            self.backbone = timm.create_model(
                backbone_name, pretrained=False, num_classes=0, global_pool="avg"
            )
            self.used_pretrained_weights = False

        feature_dim = int(getattr(self.backbone, "num_features"))
        self.feature_dim = feature_dim
        self.normalize_embeddings = normalize_embeddings
        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, projection_hidden_dim),
            nn.GELU(),
            nn.Linear(projection_hidden_dim, projection_dim),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if self.normalize_embeddings:
            features = F.normalize(features, dim=-1)
        return features

    def project(self, features: torch.Tensor) -> torch.Tensor:
        projections = self.projection_head(features)
        return F.normalize(projections, dim=-1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(images)
        projections = self.project(features)
        return features, projections
