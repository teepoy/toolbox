"""CIFAR-100 dataset loading and dataloader helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms


@dataclass(slots=True)
class Cifar100Bundle:
    train: Dataset
    test: Dataset
    class_names: list[str]


class HFDatasetWrapper(TorchDataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: Dataset,
        transform: transforms.Compose,
        label_key: str = "fine_label",
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.dataset[int(idx)]
        image = item["img"].convert("RGB")
        x = self.transform(image)
        y = torch.tensor(int(item[self.label_key]), dtype=torch.long)
        return x, y


class HFPairViewDataset(TorchDataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        dataset: Dataset,
        transform_a: transforms.Compose,
        transform_b: transforms.Compose,
        label_key: str = "fine_label",
    ) -> None:
        self.dataset = dataset
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.dataset[int(idx)]
        image = item["img"].convert("RGB")
        x1 = self.transform_a(image)
        x2 = self.transform_b(image)
        y = torch.tensor(int(item[self.label_key]), dtype=torch.long)
        return x1, x2, y


def load_cifar100() -> Cifar100Bundle:
    ds = load_dataset("cifar100")
    class_names = ds["train"].features["fine_label"].names
    return Cifar100Bundle(train=ds["train"], test=ds["test"], class_names=class_names)


def select_indices(dataset: Dataset, indices: list[int]) -> Dataset:
    return dataset.select(indices)


def make_transforms(image_size: int) -> dict[str, transforms.Compose]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return {"train": train_tf, "eval": eval_tf}


def make_ssl_pair_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    augment = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return augment, augment


def make_loader(
    dataset: TorchDataset[Any],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[Any]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
