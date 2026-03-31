"""YOLO26 classification backbone wrapper via ultralytics."""

from __future__ import annotations

import torch
from torch import nn
from typing import Any


def _load_yolo_model() -> Any:
    from ultralytics import YOLO

    return YOLO("yolo26n-cls.yaml").model


class YOLO26Backbone(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        model = _load_yolo_model()
        if model is None:
            raise RuntimeError("Failed to initialize YOLO26 classification model")
        classify = model.model[-1]
        classify.linear = nn.Linear(classify.linear.in_features, num_classes)
        self.model = model
        self.out_dim = int(classify.linear.in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        head = self.model.model[-1]
        y = x
        for block in self.model.model[:-1]:
            y = block(y)
        y = head.conv(y)
        y = head.pool(y).flatten(1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
