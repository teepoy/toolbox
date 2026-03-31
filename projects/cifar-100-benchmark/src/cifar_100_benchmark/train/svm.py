"""SVM baseline training on frozen features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader


@dataclass(slots=True)
class SVMResult:
    top1: float


@torch.no_grad()
def _extract_features(
    backbone: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    backbone.eval()
    feats = []
    labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = backbone.forward_features(x)
        feats.append(z.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def run_svm(
    backbone: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> SVMResult:
    x_train, y_train = _extract_features(backbone, train_loader, device)
    x_test, y_test = _extract_features(backbone, test_loader, device)
    clf = LinearSVC(max_iter=5000)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    top1 = float((pred == y_test).mean() * 100.0)
    return SVMResult(top1=top1)
