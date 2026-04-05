# dman Quickstart

A practical guide to managing ML datasets with the **dman** CLI and Rust/Python SDK.

## Installation

```bash
# Build from source
cargo build --release

# The CLI binary is at target/release/dman-cli
# Optionally add it to your PATH:
export PATH="$PWD/target/release:$PATH"
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
# ✓ Registered dataset 'mnist-yolo' (id=1, format=Yolo)
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
#   Images:      0
#   Annotations: 0
#   Disk size:   12345 B
# ──────────────────────────────────────────────────
```

> **Note**: `dman-cli add` _registers_ the dataset path in the catalog but does not parse/import individual images and annotations. To import images and annotations into the catalog database, use the Rust library API (see Section 3 below).

---

## 3. Import a YOLO Dataset (Rust Library)

The `FormatImporter` trait provides full dataset parsing — images, labels, categories — directly into the catalog database.

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

    // Import: parses data.yaml, reads images + labels, inserts into DB
    let dataset = importer.import(catalog.db(), &storage, path, "mnist-imported")?;

    println!("Imported '{}' — id={}", dataset.name, dataset.id);
    Ok(())
}
```

After import, `dman-cli inspect mnist-imported` will show image/annotation counts.

---

## 4. Export a Dataset as YOLO Format (Rust Library)

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
    // ├── images/train/      # copied image files
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
    --port 9090                   # Local port for image serving (default: 8080)
# ✓ Exported dataset 'my-labeled-dataset' to Label Studio project 1 using port 9090
```

> **How it works**: dman generates image URLs pointing to `http://localhost:<port>/images/<dataset_id>/<filename>`. You need to run a dman image server (or serve images yourself) so Label Studio can access them.

---

## 6. Create an Empty Dataset

Register an empty dataset and populate it programmatically via the Python SDK.

### Register an empty directory

```bash
mkdir -p ~/datasets/my-empty-dataset

dman-cli add my-project ~/datasets/my-empty-dataset --format custom
# ✓ Registered dataset 'my-project' (id=3, format=Custom("custom"))
```

### Populate with the Python SDK

Build with Python support:

```bash
cargo build --release --features python
```

```python
import dman_python

# Create a new dataset with the builder pattern
builder = dman_python.create_dataset("my-annotated-dataset")

# Add images (returns an index for referencing in annotations)
idx0 = builder.add_image("/path/to/image_001.jpg")
idx1 = builder.add_image("/path/to/image_002.jpg", metadata={"source": "camera_a"})

# Add annotations (bbox = [x, y, width, height])
builder.add_annotation(idx0, "cat", [100.0, 200.0, 50.0, 80.0])
builder.add_annotation(idx0, "dog", [300.0, 150.0, 60.0, 90.0])
builder.add_annotation(idx1, "cat", [50.0, 50.0, 120.0, 100.0])

# Define categories with supercategories
builder.set_category("cat", supercategory="animal")
builder.set_category("dog", supercategory="animal")

# Build: registers in catalog, inserts all images + annotations atomically
ds = builder.build()
print(f"Created '{ds.name}' with {len(ds)} images")
```

### Load and iterate

```python
import dman_python

ds = dman_python.load_dataset("my-annotated-dataset")
print(f"Dataset: {ds.name}, Images: {len(ds)}")

# Access by index
sample = ds[0]
print(sample["file_name"], sample["image_path"])
print(f"  Annotations: {len(sample['annotations'])}")

# Get all image paths
paths = ds.images()

# Convert to PyTorch Dataset
torch_ds = ds.to_torch_dataset()  # requires: pip install torch

# Convert to HuggingFace Dataset
hf_ds = ds.to_hf_dataset()  # requires: pip install datasets
```

### Update an existing dataset

```python
import dman_python

updater = dman_python.update_dataset("my-annotated-dataset")

# Add new images
updater.add_image("/path/to/new_image.jpg")

# Add annotation to an existing image (by DB image id)
updater.add_annotation(image_id=1, category="bird", bbox=[10.0, 20.0, 30.0, 40.0])

# Remove an image and its annotations
updater.remove_image(image_id=2)

# Apply all changes atomically
updater.apply()
```

---

## 7. Import from a Custom Format

dman supports pluggable format importers. You can register a custom Python plugin or implement the `FormatImporter` trait in Rust.

### Python plugin approach

Create a Python file with a `dman_plugin` marker:

```python
# my_custom_format.py
dman_plugin = {
    "name": "my_format",
    "type": "format",
    "version": "1.0.0",
}

def import_dataset(path, dataset_name):
    """Your custom import logic here."""
    # Parse your format, yield (image_path, annotations) pairs
    pass
```

Place it in a plugin directory and dman will discover it automatically.

### Rust trait approach

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
        "my-custom"
    }

    fn detect(&self, path: &Path) -> bool {
        // Return true if this path looks like your format
        path.join("metadata.json").exists()
    }

    fn import(
        &self,
        db: &Database,
        storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset> {
        // 1. Parse your format files
        // 2. Register the dataset
        // 3. Insert images, categories, and annotations into the DB
        // 4. Return the Dataset record
        todo!("implement your format parsing")
    }
}
```

Register it with the `FormatRegistry`:

```rust
use dman_core::formats::FormatRegistry;

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
| `dman-cli add <name> <path> --format <yolo\|coco\|hf\|custom>` | Register a dataset |
| `dman-cli list [--format table\|json]` | List all datasets |
| `dman-cli inspect <name>` | Show dataset details |
| `dman-cli remove <name> [-y]` | Remove a dataset (metadata only) |
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

| Format | Import | Export | CLI `--format` |
|--------|--------|--------|----------------|
| YOLO | ✅ Full (images + labels + categories) | ✅ Full (data.yaml + images/ + labels/) | `yolo` |
| COCO | ✅ Full (JSON annotations) | ✅ Full (instances JSON) | `coco` |
| HuggingFace | ✅ (dataset loading) | ✅ (dataset export) | `hf` |
| Label Studio | ✅ Via `label-studio import` | ✅ Via `label-studio export` | — |
| Custom | ✅ Plugin system | ✅ Plugin system | `custom` |
