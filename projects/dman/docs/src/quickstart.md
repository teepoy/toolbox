# dman Quickstart

A practical guide to managing ML datasets with the **dman** CLI and Rust/Python SDK.

This guide reflects the current extension model: **dataset formats are registry-backed format IDs**, not a built-in `custom` enum branch in core. Built-ins such as `yolo`, `coco`, and `huggingface` are just pre-registered format providers. User-defined providers can register their own format IDs such as `parquet-multi-image`.

## Installation

```bash
# Build from source
cargo build --release

# The Rust CLI binary is at target/release/dman-cli
# Optionally add it to your PATH:
export PATH="$PWD/target/release:$PATH"

# Python package users also get a `dman` console entrypoint
pip install .
dman --help
```

## 1. Initialize the Catalog

dman stores dataset metadata in a local SQLite catalog at `~/.dman/` (override with `$DMAN_HOME`).

```bash
dman-cli init
# ✓ Catalog initialized at /home/user/.dman
```

---

## 2. Register a Local Dataset (timm MNIST example)

Download the [timm MNIST](https://huggingface.co/datasets/timm/mnist) dataset in YOLO format, then register it with dman.

### Download the dataset

```bash
# Create a working directory
mkdir -p ~/datasets/mnist-yolo && cd ~/datasets/mnist-yolo

# Example YOLO layout:
# mnist-yolo/
# ├── data.yaml
# ├── images/
# │   └── train/
# └── labels/
#     └── train/
```

A minimal `data.yaml` looks like:

```yaml
path: .
train: images/train
nc: 10
names:
  0: zero
  1: one
  2: two
  3: three
  4: four
  5: five
  6: six
  7: seven
  8: eight
  9: nine
```

### Register with dman

```bash
dman-cli add mnist-yolo ~/datasets/mnist-yolo --format yolo
```

### Verify

```bash
dman-cli list
dman-cli inspect mnist-yolo
```

> **Note**: `dman-cli add` registers the dataset path and format ID. To parse/import the files, use `dman-cli import` or call an importer through the registry.

---

## 3. Import a YOLO Dataset

### Via CLI

```bash
dman-cli import ~/datasets/mnist-yolo --name mnist-imported
```

When `--format` is omitted, dman asks the registered providers to detect the format from the input path.

### Via Rust Library

```rust
use dman_core::catalog::Catalog;
use dman_core::formats::yolo::YoloImporter;
use dman_core::formats::FormatImporter;
use dman_core::storage::StorageManager;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let catalog = Catalog::open()?;
    let storage = StorageManager::new(Catalog::home_path().join("storage"));
    let importer = YoloImporter;
    let path = Path::new("/home/user/datasets/mnist-yolo");
    let dataset = importer.import(catalog.db(), &storage, path, "mnist-imported")?;
    println!("Imported '{}' — id={}", dataset.name, dataset.id);
    Ok(())
}
```

---

## 4. Export a Dataset as YOLO Format

### Via CLI

```bash
dman-cli export mnist-imported /tmp/mnist-yolo-export --format yolo
```

### Via Rust Library

```rust
use dman_core::catalog::Catalog;
use dman_core::dataset::DatasetService;
use dman_core::formats::yolo::YoloExporter;
use dman_core::formats::FormatExporter;
use dman_core::storage::StorageManager;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let catalog = Catalog::open()?;
    let storage = StorageManager::new(Catalog::home_path().join("storage"));
    let dataset = DatasetService::get(catalog.db(), "mnist-imported")?;
    let exporter = YoloExporter;
    let output = Path::new("/tmp/mnist-yolo-export");
    exporter.export(catalog.db(), &storage, &dataset, output)?;
    Ok(())
}
```

---

## 5. Import from Label Studio

```bash
dman-cli label-studio import \
  http://localhost:8080 \
  your-api-key \
  1 \
  my-labeled-dataset
```

```bash
dman-cli label-studio export \
  http://localhost:8080 \
  your-api-key \
  1 \
  my-labeled-dataset \
  --port 9090
```

---

## 6. Create an Empty Dataset

```bash
mkdir -p ~/datasets/my-empty-dataset
dman-cli add my-project ~/datasets/my-empty-dataset --format builder
```

```python
import dman

builder = dman.create_dataset("my-annotated-dataset")
builder.add_sample("scene_001")
builder.add_asset("scene_001", "image", "/path/to/image_001.jpg")
builder.add_annotation("scene_001", "cat", bbox=[100.0, 200.0, 50.0, 80.0])
builder.set_category("cat", supercategory="animal")
ds = builder.build()
print(f"Created '{ds.name}' with {len(ds)} samples")
```

---

## 7. Register a User-Defined Dataset Format

Formats are registry IDs. Providers can come from Rust or Python.

### Python plugin example

```python
dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}

def detect(path: str) -> bool:
    ...

def import_dataset(path: str):
    ...

def export_dataset(data, output_path: str):
    ...
```

```bash
dman-cli import ~/datasets/parquet-corpus --name parquet-demo --format parquet-multi-image
dman-cli export parquet-demo /tmp/parquet-export --format parquet-multi-image
```

For more detail, see [Format registry](./features/format-registry.md).
