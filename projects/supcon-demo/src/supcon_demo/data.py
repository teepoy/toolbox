from __future__ import annotations

import random
import json
from collections.abc import Sequence
from pathlib import Path
from urllib.request import urlopen, urlretrieve

from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from timm.data import resolve_model_data_config
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset as TorchDataset


class TwoViewFlowerDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        image_key: str,
        label_key: str,
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        sample = self.dataset[index]
        image = _to_rgb_image(sample[self.image_key])
        label = int(sample[self.label_key])
        view_one = self.transform(image)
        view_two = self.transform(image)
        return view_one, view_two, label


class SingleViewFlowerDataset(TorchDataset):
    def __init__(
        self,
        dataset: Dataset,
        image_key: str,
        label_key: str,
        transform: transforms.Compose,
    ) -> None:
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple:
        sample = self.dataset[index]
        image = _to_rgb_image(sample[self.image_key])
        label = int(sample[self.label_key])
        return self.transform(image), label


def _to_rgb_image(value: Image.Image) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    raise TypeError(f"Expected a PIL image, got {type(value)!r}")


def _download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    urlretrieve(url, destination)
    return destination


def _load_dataset_from_hf_parquet(dataset_name: str) -> DatasetDict:
    metadata_url = f"https://huggingface.co/api/datasets/{dataset_name}"
    with urlopen(metadata_url, timeout=60) as response:
        metadata = json.loads(response.read().decode("utf-8"))

    parquet_files = [
        sibling["rfilename"]
        for sibling in metadata.get("siblings", [])
        if sibling["rfilename"].endswith(".parquet")
    ]
    if not parquet_files:
        raise RuntimeError(f"No parquet files found for dataset {dataset_name}")

    cache_root = Path("outputs") / ".dataset_cache" / dataset_name.replace("/", "__")
    data_files: dict[str, list[str]] = {}
    for filename in parquet_files:
        split_name = Path(filename).name.split("-")[0]
        local_path = cache_root / filename
        remote_url = (
            f"https://huggingface.co/datasets/{dataset_name}/resolve/main/{filename}"
        )
        downloaded = _download_file(remote_url, local_path)
        data_files.setdefault(split_name, []).append(str(downloaded))

    return load_dataset("parquet", data_files=data_files)


def _load_dataset_dict(dataset_name: str) -> DatasetDict:
    try:
        return load_dataset(dataset_name)
    except Exception as error:
        print(f"[data] load_dataset failed for {dataset_name}: {error}")
        print("[data] falling back to direct parquet download from the dataset repo")
        return _load_dataset_from_hf_parquet(dataset_name)


def _subset_dataset(dataset: Dataset, subset_size: int | None, seed: int) -> Dataset:
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    generator = random.Random(seed)
    generator.shuffle(indices)
    return dataset.select(indices[:subset_size])


def _filter_dataset_by_classes(
    dataset: Dataset, label_key: str, allowed_classes: set[int]
) -> Dataset:
    if not allowed_classes:
        return dataset.select([])
    return dataset.filter(lambda sample: int(sample[label_key]) in allowed_classes)


def _resolve_class_split(
    train_source: Dataset, label_key: str, class_split_cfg
) -> tuple[set[int], set[int]]:
    classes = sorted({int(label) for label in train_source[label_key]})
    if len(classes) < 2:
        return set(classes), set(classes)

    train_fraction = float(class_split_cfg.train_fraction)
    train_fraction = min(max(train_fraction, 0.1), 0.9)
    train_count = max(1, min(len(classes) - 1, int(len(classes) * train_fraction)))

    mode = str(class_split_cfg.mode)
    if mode == "first_half":
        train_classes = set(classes[:train_count])
        eval_classes = set(classes[train_count:])
    elif mode == "random":
        rng = random.Random(int(class_split_cfg.seed))
        shuffled = classes[:]
        rng.shuffle(shuffled)
        train_classes = set(shuffled[:train_count])
        eval_classes = set(shuffled[train_count:])
    else:
        raise ValueError(
            f"Unsupported dataset.class_split.mode={mode!r}; expected 'first_half' or 'random'."
        )

    if not eval_classes:
        eval_classes = train_classes
    return train_classes, eval_classes


def _derive_missing_splits(
    dataset_dict: DatasetDict, val_fraction: float, test_fraction: float, seed: int
) -> DatasetDict:
    if {"train", "validation", "test"}.issubset(dataset_dict.keys()):
        return dataset_dict

    if "train" not in dataset_dict:
        first_split = next(iter(dataset_dict.keys()))
        train_source = dataset_dict[first_split]
    else:
        train_source = dataset_dict["train"]

    if "validation" not in dataset_dict and "test" not in dataset_dict:
        first_cut = train_source.train_test_split(
            test_size=val_fraction + test_fraction, seed=seed
        )
        second_fraction = test_fraction / (val_fraction + test_fraction)
        second_cut = first_cut["test"].train_test_split(
            test_size=second_fraction, seed=seed
        )
        return DatasetDict(
            {
                "train": first_cut["train"],
                "validation": second_cut["train"],
                "test": second_cut["test"],
            }
        )

    validation = dataset_dict.get("validation")
    test = dataset_dict.get("test")
    if validation is None and test is not None:
        split = train_source.train_test_split(test_size=val_fraction, seed=seed)
        return DatasetDict(
            {"train": split["train"], "validation": split["test"], "test": test}
        )
    if test is None and validation is not None:
        split = train_source.train_test_split(test_size=test_fraction, seed=seed)
        return DatasetDict(
            {"train": split["train"], "validation": validation, "test": split["test"]}
        )
    return dataset_dict


def build_transforms(
    image_size: int, model
) -> tuple[transforms.Compose, transforms.Compose]:
    data_cfg = resolve_model_data_config(model)
    mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
    std = data_cfg.get("std", (0.229, 0.224, 0.225))
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def load_data(config, model) -> dict:
    dataset_dict = _load_dataset_dict(str(config.dataset.name))
    dataset_dict = _derive_missing_splits(
        dataset_dict,
        val_fraction=float(config.dataset.val_fraction),
        test_fraction=float(config.dataset.test_fraction),
        seed=int(config.experiment.seed),
    )

    train_source = dataset_dict[config.dataset.train_split]
    val_source = dataset_dict[config.dataset.val_split]
    test_source = dataset_dict[config.dataset.test_split]

    class_split_cfg = config.dataset.class_split
    class_split_enabled = bool(class_split_cfg.enabled)
    class_split_info = {
        "enabled": class_split_enabled,
        "mode": str(class_split_cfg.mode),
    }
    if class_split_enabled:
        train_classes, eval_classes = _resolve_class_split(
            train_source, config.dataset.label_key, class_split_cfg
        )
        train_source = _filter_dataset_by_classes(
            train_source, config.dataset.label_key, train_classes
        )
        val_source = _filter_dataset_by_classes(
            val_source, config.dataset.label_key, eval_classes
        )
        test_source = _filter_dataset_by_classes(
            test_source, config.dataset.label_key, eval_classes
        )
        class_split_info.update(
            {
                "train_num_classes": len(train_classes),
                "eval_num_classes": len(eval_classes),
            }
        )

    train_ds = _subset_dataset(
        train_source,
        config.dataset.train_subset,
        int(config.experiment.seed),
    )
    val_ds = _subset_dataset(
        val_source,
        config.dataset.val_subset,
        int(config.experiment.seed) + 1,
    )
    test_ds = _subset_dataset(
        test_source,
        config.dataset.test_subset,
        int(config.experiment.seed) + 2,
    )

    train_transform, eval_transform = build_transforms(
        int(config.train.image_size), model
    )

    train_dataset = TwoViewFlowerDataset(
        train_ds, config.dataset.image_key, config.dataset.label_key, train_transform
    )
    eval_train_dataset = SingleViewFlowerDataset(
        train_ds, config.dataset.image_key, config.dataset.label_key, eval_transform
    )
    val_dataset = SingleViewFlowerDataset(
        val_ds, config.dataset.image_key, config.dataset.label_key, eval_transform
    )
    test_dataset = SingleViewFlowerDataset(
        test_ds, config.dataset.image_key, config.dataset.label_key, eval_transform
    )

    workers = int(config.dataset.num_workers)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.train.batch_size),
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    eval_train_loader = DataLoader(
        eval_train_dataset,
        batch_size=int(config.benchmark.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.benchmark.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config.benchmark.batch_size),
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    label_feature = train_ds.features.get(config.dataset.label_key)
    class_names: Sequence[str] | None = getattr(label_feature, "names", None)
    num_classes = (
        len(class_names)
        if class_names
        else len(set(train_ds[config.dataset.label_key]))
    )

    return {
        "train_loader": train_loader,
        "eval_train_loader": eval_train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": num_classes,
        "class_names": list(class_names) if class_names else None,
        "class_split": class_split_info,
        "split_sizes": {
            "train": len(train_ds),
            "validation": len(val_ds),
            "test": len(test_ds),
        },
    }
