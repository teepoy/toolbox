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
# ├── data.yaml          # class names and split paths
# ├── images/
# │   └── train/
# │       ├── 00001.jpg
# │       ├── 00002.jpg
# │       └── ...
# └── labels/
#     └── train/
#         ├── 00001.txt   # YOLO format: class_id x_center y_center width height
#         ├── 00002.txt
#         └── ...
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
# ✓ Registered dataset 'mnist-yolo' (id=1, format=yolo)
```

### Verify

```bash
dman-cli list
# ┌────┬────────────┬────────┬───────────────────────────┬─────────────────────┐
# │ ID │ Name       │ Format │ Path                      │ Created             │
# ├────┼────────────┼────────┼───────────────────────────┼─────────────────────┤
# │ 1  │ mnist-yolo │ yolo   │ /home/user/datasets/mnist │ 2026-04-05 12:00:00 │
# └────┴────────────┴────────┴───────────────────────────┴─────────────────────┘

dman-cli inspect mnist-yolo
# ──────────────────────────────────────────────────
#   Name:    mnist-yolo
#   ID:      1
#   Format:  yolo
#   Path:    /home/user/datasets/mnist-yolo
#   Created: 2026-04-05 12:00:00
# ──────────────────────────────────────────────────
#   Samples:     0 / Assets: 0
#   Annotations: 0
#   Disk size:   12345 B
# ──────────────────────────────────────────────────
```

> **Note**: `dman-cli add` _registers_ the dataset path and its format ID in the catalog. To parse/import the underlying files, use `dman-cli import` or call an importer through the `FormatRegistry`.

---

## 3. Import a YOLO Dataset

### Via CLI

```bash
dman-cli import ~/datasets/mnist-yolo --name mnist-imported
# ✓ Imported '/home/user/datasets/mnist-yolo' as dataset 'mnist-imported' (id=2, format=yolo)
```

When `--format` is omitted, dman asks the registered providers to detect the format from the input path.

### Via Rust Library

The `FormatImporter` trait provides full dataset parsing — samples, assets, labels, categories — directly into the catalog database.

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

    // Detect format automatically
    assert!(importer.detect(path)); // true if data.yaml exists

    // Import: parses data.yaml, reads image files as assets, inserts into DB
    let dataset = importer.import(catalog.db(), &storage, path, "mnist-imported")?;

    println!("Imported '{}' — id={}", dataset.name, dataset.id);
    Ok(())
}
```

After import, `dman-cli inspect mnist-imported` will show sample/asset and annotation counts.

---

## 4. Export a Dataset as YOLO Format

### Via CLI

```bash
dman-cli export mnist-imported /tmp/mnist-yolo-export --format yolo
# ✓ Exported dataset 'mnist-imported' to /tmp/mnist-yolo-export as yolo
```

### Via Rust Library

Export any dman dataset to a YOLO-compatible directory layout:

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

    // Output structure:
    // /tmp/mnist-yolo-export/
    // ├── data.yaml          # nc, names, path references
    // ├── images/train/      # image asset files
    // └── labels/train/      # YOLO label .txt files

    println!("Exported to {}", output.display());
    Ok(())
}
```

The exported directory is ready for use with Ultralytics YOLOv8, YOLOv11, etc.

---

## 5. Import from Label Studio

dman integrates directly with [Label Studio](https://labelstud.io/). Pull annotated data from a Label Studio project into your local catalog.

### Via CLI

```bash
dman-cli label-studio import \
    http://localhost:8080 \       # Label Studio URL
    your-api-key \                # API token (Settings → Account & API key)
    1 \                           # Project ID
    my-labeled-dataset            # Dataset name in dman
# ✓ Imported Label Studio project 1 into dataset 'my-labeled-dataset' (id=2)
```

### Export to Label Studio

Push a dman dataset back to Label Studio for further annotation:

```bash
dman-cli label-studio export \
    http://localhost:8080 \       # Label Studio URL
    your-api-key \                # API token
    1 \                           # Target project ID
    my-labeled-dataset \          # Dataset name in dman
    --port 9090                   # Local port for asset serving (default: 8080)
# ✓ Exported dataset 'my-labeled-dataset' to Label Studio project 1 using port 9090
```

> **How it works**: dman generates asset URLs pointing to `http://localhost:<port>/images/<dataset_id>/<filename>`. You need to run a dman asset server (or serve files yourself) so Label Studio can access them.

---

## 6. Create an Empty Dataset

Register an empty dataset and populate it programmatically via the Python SDK.

### Register an empty directory

```bash
mkdir -p ~/datasets/my-empty-dataset

dman-cli add my-project ~/datasets/my-empty-dataset --format builder
# ✓ Registered dataset 'my-project' (id=3, format=builder)
```

### Populate with the Python SDK

Build with Python support:

```bash
pip install .
# or: maturin develop
```

```python
import dman

# Create a new dataset using the sample/asset builder pattern
builder = dman.create_dataset("my-annotated-dataset")

# Add samples and assets — signature: add_asset(sample_name, asset_type, file_path, ...)
builder.add_sample("scene_001", metadata={"split": "train"})
builder.add_asset("scene_001", "image", "/path/to/image_001.jpg")

builder.add_sample("scene_002", metadata={"split": "train"})
builder.add_asset("scene_002", "image", "/path/to/image_002.jpg", metadata={"source": "camera_a"})

# Add annotations (bbox = [x, y, width, height])
builder.add_annotation("scene_001", "cat", bbox=[100.0, 200.0, 50.0, 80.0], asset_name="image_001.jpg")
builder.add_annotation("scene_001", "dog", bbox=[300.0, 150.0, 60.0, 90.0], asset_name="image_001.jpg")
builder.add_annotation("scene_002", "cat", bbox=[50.0, 50.0, 120.0, 100.0], asset_name="image_002.jpg")

# Convenience: add_image() creates a 1:1 Sample+Asset automatically
builder.add_image("/path/to/image_003.jpg")

# Define categories with supercategories
builder.set_category("cat", supercategory="animal")
builder.set_category("dog", supercategory="animal")

# Build: registers in catalog, inserts all samples + assets + annotations atomically
ds = builder.build()
print(f"Created '{ds.name}'")
```

### Load and iterate

```python
import dman

ds = dman.load_dataset("my-annotated-dataset")
print(f"Dataset: {ds.name}")

# Iterate samples
for sample in ds.samples():
    print(sample["name"], sample["assets"])

# Access a single sample by name
sample = ds.get_sample("scene_001")
print(sample["name"], sample["assets"])

# Get annotations for a sample
annotations = ds.annotations("scene_001")
print(f"  Annotations: {len(annotations)}")

# Convert to PyTorch Dataset
torch_ds = ds.to_torch_dataset()  # requires: pip install torch

# Convert to HuggingFace Dataset
hf_ds = ds.to_hf_dataset()  # requires: pip install datasets
```

### Update an existing dataset

```python
import dman

updater = dman.update_dataset("my-annotated-dataset")

# Add a new sample and asset — signature: add_asset(sample_name, asset_type, file_path, ...)
updater.add_sample("scene_003")
updater.add_asset("scene_003", "image", "/path/to/new_image.jpg")

# Add annotation to an existing sample — updater uses sample_id (integer), not name
updater.add_annotation(1, "bird", bbox=[10.0, 20.0, 30.0, 40.0])

# Apply all changes atomically
updater.apply()
```

---

## 7. Register a User-Defined Dataset Format

dman now treats format names as **registry IDs**. A provider can come from Rust or Python. The provider owns its own detection, serialization, and conversion logic.

### Python plugin layout

Place Python format providers under `~/.dman/plugins/` (or `$DMAN_HOME/plugins/`). Each provider declares a stable format ID using `dman_plugin["name"]`.

```python
# ~/.dman/plugins/parquet_multi_image.py
from __future__ import annotations

from pathlib import Path
import io
import json

import pyarrow.parquet as pq
from PIL import Image


dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}


def detect(path: str) -> bool:
    root = Path(path)
    return root.is_dir() and any(p.suffix == ".parquet" for p in root.glob("*.parquet"))


def _decode_image(raw: bytes, output_path: Path) -> None:
    image = Image.open(io.BytesIO(raw))
    image.save(output_path)


def import_dataset(path: str):
    root = Path(path)
    assets_dir = root / ".materialized-assets"
    assets_dir.mkdir(exist_ok=True)

    samples = []
    annotations = []

    for parquet_path in root.glob("*.parquet"):
        table = pq.read_table(parquet_path)
        columns = table.column_names
        image_columns = [name for name in columns if name.startswith("image_")]

        for row_idx in range(table.num_rows):
            row = table.slice(row_idx, 1).to_pylist()[0]
            label = row.get("label", "unknown")
            bbox = row.get("bbox", [0.0, 0.0, 1.0, 1.0])

            # Each row becomes one sample; each image column becomes one asset
            sample_name = f"{parquet_path.stem}_{row_idx}"
            assets = []

            for image_col in image_columns:
                raw = row.get(image_col)
                if raw is None:
                    continue

                file_name = f"{sample_name}_{image_col}.png"
                output_path = assets_dir / file_name
                _decode_image(raw, output_path)

                assets.append({
                    "file_name": str(Path(".materialized-assets") / file_name),
                    "asset_type": "image",
                })
                annotations.append(
                    {
                        "sample_name": sample_name,
                        "asset_file_name": str(Path(".materialized-assets") / file_name),
                        "category": label,
                        "bbox": bbox,
                    }
                )

            samples.append({"name": sample_name, "assets": assets})

    return {"samples": samples, "annotations": annotations}


def export_dataset(data, output_path: str):
    root = Path(output_path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
```

### Import it through the shipped CLI

```bash
dman-cli init
dman-cli import ~/datasets/parquet-corpus --name parquet-demo
# auto-detects parquet-multi-image from ~/.dman/plugins/

# or be explicit
dman-cli import ~/datasets/parquet-corpus --name parquet-demo --format parquet-multi-image

# export through the same provider
dman-cli export parquet-demo /tmp/parquet-export --format parquet-multi-image
```

The plugin decides how to deserialize and serialize. If your format needs MCP, gRPC, or another transport, keep the stable format ID the same and implement that transport inside the provider.

### Rust provider approach

Implement `FormatImporter` for full control:

```rust
use dman_core::db::Database;
use dman_core::formats::FormatImporter;
use dman_core::storage::StorageManager;
use dman_core::types::Dataset;
use dman_core::error::Result;
use std::path::Path;

pub struct MyCustomImporter;

impl FormatImporter for MyCustomImporter {
    fn name(&self) -> &str {
        "parquet-multi-image"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_dir() && path.join("train-00000-of-00001.parquet").exists()
    }

    fn import(
        &self,
        db: &Database,
        storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset> {
        // Parse parquet rows, materialize asset bytes to files, then insert
        // sample/asset/category/annotation records into dman.
        todo!("implement your format parsing")
    }
}
```

Register it with the `FormatRegistry`:

```rust
use dman_core::formats::FormatRegistry;
use std::path::Path;

let mut registry = FormatRegistry::default_registry();
registry.register_importer(Box::new(MyCustomImporter));

// Now you can auto-detect and import
if let Some(format_name) = registry.detect_format(Path::new("/path/to/data")) {
    let importer = registry.get_importer(format_name).unwrap();
    let dataset = importer.import(&db, &storage, path, "my-dataset")?;
}
```

---

## CLI Reference (Quick)

| Command | Description |
|---------|-------------|
| `dman-cli init` | Initialize catalog at `~/.dman` |
| `dman-cli add <name> <path> --format <format-id>` | Register a dataset using a canonical format ID |
| `dman-cli list [--format table\|json]` | List all datasets |
| `dman-cli inspect <name>` | Show dataset details (samples, assets, annotations) |
| `dman-cli remove <name> [-y]` | Remove a dataset (metadata only) |
| `dman-cli import <path> --name <dataset> [--format <format-id>]` | Import using the runtime format registry |
| `dman-cli export <dataset> <output> --format <format-id>` | Export using the runtime format registry |
| `dman-cli label-studio import <url> <api_key> <project_id> <name>` | Import from Label Studio |
| `dman-cli label-studio export <url> <api_key> <project_id> <dataset> [--port N]` | Export to Label Studio |
| `dman-cli embed <dataset> --model <path> [--batch-size N]` | Compute embeddings (requires `--features python`) |
| `dman-cli tui` | Launch interactive terminal UI |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DMAN_HOME` | `~/.dman` | Catalog storage location |

---

## Supported Formats

| Format ID | Import | Export | Source |
|-----------|--------|--------|--------|
| `yolo` | ✅ Full (image assets + labels + categories) | ✅ Full (data.yaml + images/ + labels/) | Built in |
| `coco` | ✅ Full (JSON annotations) | ✅ Full (instances JSON) | Built in |
| `huggingface` | ✅ Parquet-backed dataset loading | ✅ Dataset export | Built in |
| `label-studio` | ✅ Via `label-studio import` | ✅ Via `label-studio export` | Built in integration |
| `parquet-multi-image` | ✅ Plugin-defined | ✅ Plugin-defined | User plugin |
