# Custom format registration

`dman` ships four built-in formats (`yolo`, `coco`, `huggingface`, `label-studio`) and a
registry-backed plugin system that lets you teach the CLI new dataset formats in plain
Python—no Rust required.

---

## Overview

Registering a custom format means placing a single `.py` file in `~/.dman/plugins/` (or
`$DMAN_HOME/plugins/`).  After that, the format ID you choose becomes available to
`dman import` and `dman export` as if it were a built-in:

```
dman import /path/to/dataset --format parquet-multi-image --name my_dataset
dman export my_dataset /out/dir --format parquet-multi-image
```

Plugins can also expose an optional `detect()` function so `dman import` can identify the
format automatically without `--format`.

---

## Canonical format IDs

`DatasetFormat` is a plain string wrapper—not a fixed enum.  Built-in IDs are:

| ID | Source |
|---|---|
| `yolo` | built-in Rust importer/exporter |
| `coco` | built-in Rust importer/exporter |
| `huggingface` | built-in Rust importer/exporter |
| `label-studio` | built-in Rust importer/exporter |

User-defined IDs are arbitrary strings such as `parquet-multi-image`, `my-sensor-log`,
or `roboflow-export`.  The string you put in `name` inside the plugin marker *is* the
format ID used everywhere in the CLI.

---

## Plugin file placement

```
~/.dman/
└── plugins/
    ├── parquet_multi_image.py   # discovered
    └── sensors/
        └── lidar_format.py      # also discovered — recursive walk
```

- Default directory: `~/.dman/plugins/`
- Override: set `DMAN_HOME` to point to a different catalog root; plugins are then
  expected at `$DMAN_HOME/plugins/`
- Discovery walks the directory **recursively** (subdirectories allowed)
- Only files with the `.py` extension are inspected
- A file is only loaded if it contains the `dman_plugin` marker (see next section)

---

## Plugin marker

Every plugin file must declare a top-level `dman_plugin` dictionary:

```python
dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}
```

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Canonical format ID—used by `--format` in the CLI |
| `type` | yes | Must be exactly `"format"` |
| `version` | yes | Metadata only; not used for logic |

Both single-quoted and double-quoted string values are accepted.

---

## Discovery lifecycle

Discovery is **text-based** — no Python code is ever executed during this phase.

When `dman` starts, `PluginManager::discover()` walks each `.py` file in the plugins
directory and calls `parse_plugin_marker()`, which:

1. Checks whether the file text contains the string `dman_plugin` (fast substring scan)
2. If present, extracts `name`, `type`, and `version` by searching for the quoted key
   names and reading the values after the `:` separator

No Python interpreter is invoked, no modules are imported, no side effects occur.
A file with broken Python syntax is silently accepted during discovery; the syntax error
will surface at import or export time when the plugin module is actually executed.

---

## Registry bootstrap sequence

Each time `dman` runs, the registry is built from scratch:

1. **`build_registry()` is called** — entry point in the CLI before dispatching any command
2. **`FormatRegistry::default_registry()`** — registers the built-in YOLO, COCO, and
   HuggingFace importers/exporters
3. **`Catalog::open()`** resolves the catalog root; `catalog.plugins_path()` returns the
   plugins directory
4. **`load_python_format_registry(plugin_dirs)`** — runs `PluginManager::discover()`;
   for every `PluginInfo` with `plugin_type == "format"` it creates one
   `PythonFormatImporter` and one `PythonFormatExporter`
5. **Importers and exporters are appended** to the registry (after built-ins)
6. **`dman import`** — if `--format` is supplied the exact ID is resolved; otherwise
   `detect_format(path)` calls every registered importer's `detect()` in order and
   returns the first one that returns `true`
7. **`dman export`** — always requires explicit `--format`; resolved by exact format ID

> **Note:** If the catalog has not been initialized (`dman init` not yet run), Python
> plugins are silently skipped and only built-in formats are available.

---

## `detect()` function (optional)

```python
def detect(path: str) -> bool:
    """Return True if path looks like a dataset this plugin can handle."""
    import os
    return any(f.endswith(".parquet") for f in os.listdir(path))
```

- **Signature:** `detect(path: str) -> bool`
- Called only during auto-detection (when `--format` is omitted)
- Must be **fast** — it is called for every registered importer
- Must have **no side effects** — do not mutate files, network, or state
- If the function is absent, the importer is never selected automatically; `--format` is
  required

---

## `import_dataset()` function

`import_dataset()` is the core of every import plugin.

### Signature

```python
def import_dataset(path: str) -> list:
    ...
```

`path` is the string form of the path passed to `dman import`.  The function must return
a **list of sample objects** — one element per logical sample in the dataset.

> **Critical:** `import_dataset` must return a `list`, not a `dict`.  An older dict-based
> API existed during development; it has been removed.  Returning anything other than a
> list will raise `PluginError: import_dataset must return a list`.

### Sample object

Each element of the returned list must expose these attributes:

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Unique name for this sample within the dataset |
| `assets` | `list` | List of asset objects (may be empty) |
| `annotations` | `list` | Sample-level annotations not tied to a specific asset (may be empty) |

### Asset object

Each element of the `assets` list must expose:

| Attribute | Type | Description |
|---|---|---|
| `asset_type` | `str` | One of `"image"`, `"depth"`, `"pointcloud"`, `"text"`, `"audio"`, `"video"`, `"mask"`, or any custom string |
| `file_name` | `str` | Filename unique within the dataset (e.g. `"sample001_front.jpg"`) |
| `file_path` | `str` | **Absolute** path to the file on disk |
| `width` | `int \| None` | Pixel width (or `None`) |
| `height` | `int \| None` | Pixel height (or `None`) |
| `annotations` | `list` | Asset-level annotations |

### Annotation object

Each element of any `annotations` list must expose:

| Attribute | Type | Description |
|---|---|---|
| `category` | `str` | Required; label class name |
| `bbox` | `dict \| None` | Bounding box (see below) or `None` |
| `segmentation` | `list[list[float]] \| None` | Polygon segmentation mask or `None` |
| `keypoints` | `list[float] \| None` | Flat keypoint list or `None` |
| `metadata` | `any \| None` | Arbitrary JSON-serialisable value or `None` |

**Bounding box format — dict only:**

```python
bbox = {"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0}
```

All four keys are required floats.  Do **not** use a 4-element list `[x, y, w, h]` —
Rust will raise `PluginError: 'bbox' must be a dict`.

### Defining your own classes

Rust reads all attributes **by name** using Python's attribute protocol.  You do not need
to import anything from `dman`.  Define simple classes directly in your plugin file:

```python
class Annotation:
    def __init__(self, category, bbox=None, segmentation=None, keypoints=None, metadata=None):
        self.category = category
        self.bbox = bbox
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.metadata = metadata

class Asset:
    def __init__(self, asset_type, file_name, file_path, width=None, height=None, annotations=None):
        self.asset_type = asset_type
        self.file_name = file_name
        self.file_path = file_path
        self.width = width
        self.height = height
        self.annotations = annotations or []

class Sample:
    def __init__(self, name, assets=None, annotations=None):
        self.name = name
        self.assets = assets or []
        self.annotations = annotations or []
```

---

## `export_dataset()` function

```python
def export_dataset(samples: list, output_path: str) -> None:
    ...
```

`samples` is a list of sample objects constructed by the Rust exporter, each with the
same attribute structure described in the import section above.  `output_path` is the
string form of the directory passed to `dman export`.

The function must return `None` (or nothing).  It is responsible for writing all output
files to `output_path`.

Attribute summary for received objects:

| Level | Attributes available |
|---|---|
| Sample | `.name`, `.assets`, `.annotations` |
| Asset | `.asset_type`, `.file_name`, `.file_path`, `.width`, `.height`, `.annotations` |
| Annotation | `.category`, `.bbox` (dict or `None`), `.segmentation`, `.keypoints`, `.metadata` |

> **Note:** `.metadata` is `None` in the current implementation for all three levels.

---

## Complete example — Parquet multi-image format

Assume a Parquet dataset where each row contains:

- `image_front: bytes` — JPEG bytes of the front camera
- `image_back: bytes` — JPEG bytes of the back camera
- `label: str` — class label
- `bbox: list[float]` — `[x, y, width, height]`

Place the following file at `~/.dman/plugins/parquet_multi_image.py`:

```python
import os
import pandas as pd

dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}


# ---------------------------------------------------------------------------
# Data classes (no import from dman needed)
# ---------------------------------------------------------------------------

class Annotation:
    def __init__(self, category, bbox=None, segmentation=None, keypoints=None, metadata=None):
        self.category = category
        self.bbox = bbox
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.metadata = metadata


class Asset:
    def __init__(self, asset_type, file_name, file_path, width=None, height=None, annotations=None):
        self.asset_type = asset_type
        self.file_name = file_name
        self.file_path = file_path
        self.width = width
        self.height = height
        self.annotations = annotations or []


class Sample:
    def __init__(self, name, assets=None, annotations=None):
        self.name = name
        self.assets = assets or []
        self.annotations = annotations or []


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect(path: str) -> bool:
    """Return True if the directory contains at least one .parquet file."""
    try:
        return any(f.endswith(".parquet") for f in os.listdir(path))
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_dataset(path: str) -> list:
    """Read all .parquet files in `path` and return a list of Sample objects."""
    mat_dir = os.path.join(path, ".materialized-images")
    os.makedirs(mat_dir, exist_ok=True)

    samples = []
    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".parquet"):
            continue
        fpath = os.path.join(path, fname)
        df = pd.read_parquet(fpath)

        for idx, row in df.iterrows():
            sample_id = f"{fname[:-8]}_row{idx}"

            # Materialise image bytes to disk so dman can store file paths
            front_name = f"{sample_id}_front.jpg"
            back_name = f"{sample_id}_back.jpg"
            front_path = os.path.join(mat_dir, front_name)
            back_path = os.path.join(mat_dir, back_name)

            with open(front_path, "wb") as f:
                f.write(bytes(row["image_front"]))
            with open(back_path, "wb") as f:
                f.write(bytes(row["image_back"]))

            # Build bbox annotation — dict format required
            raw_bbox = row["bbox"]  # [x, y, w, h]
            ann = Annotation(
                category=str(row["label"]),
                bbox={"x": float(raw_bbox[0]), "y": float(raw_bbox[1]),
                      "width": float(raw_bbox[2]), "height": float(raw_bbox[3])},
            )

            front_asset = Asset(
                asset_type="image",
                file_name=front_name,
                file_path=os.path.abspath(front_path),
                annotations=[ann],
            )
            back_asset = Asset(
                asset_type="image",
                file_name=back_name,
                file_path=os.path.abspath(back_path),
            )

            samples.append(Sample(name=sample_id, assets=[front_asset, back_asset]))

    return samples


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_dataset(samples: list, output_path: str) -> None:
    """Write dataset back to Parquet (one file per sample for illustration)."""
    import shutil
    os.makedirs(output_path, exist_ok=True)

    rows = []
    for sample in samples:
        row = {"sample_name": sample.name}
        for asset in sample.assets:
            key = "image_front" if "front" in asset.file_name else "image_back"
            with open(asset.file_path, "rb") as f:
                row[key] = f.read()
            for ann in asset.annotations:
                row["label"] = ann.category
                if ann.bbox:
                    row["bbox"] = [ann.bbox["x"], ann.bbox["y"],
                                   ann.bbox["width"], ann.bbox["height"]]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(output_path, "export.parquet"), index=False)
```

### Directory layout

```
~/.dman/plugins/
└── parquet_multi_image.py

/data/my_parquet_dataset/
├── batch_001.parquet
├── batch_002.parquet
└── .materialized-images/      ← created by import_dataset()
    ├── batch_001_row0_front.jpg
    ├── batch_001_row0_back.jpg
    └── ...
```

### Import command

```bash
dman import /data/my_parquet_dataset --format parquet-multi-image --name my_dataset
```

Or, because `detect()` looks for `.parquet` files, format auto-detection also works:

```bash
dman import /data/my_parquet_dataset --name my_dataset
```

---

## Current limitations

| Limitation | Detail |
|---|---|
| **Synchronous, file-based only** | Plugins cannot use HTTP, gRPC, subprocesses, or streaming; everything must be accessible as local files |
| **Discovery is text-based** | `parse_plugin_marker()` is a plain string scan; broken Python syntax is not caught until import/export time |
| **Auto-detection collision** | If multiple plugins return `true` from `detect()` for the same path, the first one registered wins; use `--format` to resolve ambiguity |
| **Non-UTF-8 paths rejected** | Plugin paths and dataset paths containing non-UTF-8 bytes will raise `PluginError: non-UTF8 plugin path` |
| **No persistent state** | The plugin module is executed from scratch on every import/export call; global variables are not preserved between calls |
| **No dependency isolation** | Plugins share the same Python process; conflicting package versions between plugins are not isolated |

---

## Error messages

| Error message | Cause |
|---|---|
| `PluginError: import_dataset must return a list` | `import_dataset()` returned something other than a `list` (e.g. a `dict`) |
| `PluginError: 'bbox' must be a dict` | A `bbox` field was set to a list `[x,y,w,h]` instead of `{"x":…, "y":…, "width":…, "height":…}` |
| `PluginError: annotation missing 'category'` | An annotation object does not have a `category` attribute |
| `PluginError: PySampleData missing 'name'` | A sample object in the returned list does not have a `name` attribute |
| `PluginError: PySampleData missing 'assets'` | A sample object is missing an `assets` attribute |
| `PluginError: PySampleData missing 'annotations'` | A sample object is missing an `annotations` attribute |
| `PluginError: PyAssetData missing 'asset_type'` | An asset object is missing the `asset_type` attribute |
| `PluginError: PyAssetData missing 'file_name'` | An asset object is missing the `file_name` attribute |
| `PluginError: PyAssetData missing 'file_path'` | An asset object is missing the `file_path` attribute |
| `PluginError: non-UTF8 plugin path` | The plugin `.py` file path contains non-UTF-8 bytes |
| `PluginError: non-UTF8 import path` | The path passed to `dman import` contains non-UTF-8 bytes |
| `PluginError: non-UTF8 output path` | The path passed to `dman export` contains non-UTF-8 bytes |
| `no registered format provider detected for <path>` | Auto-detection failed — no plugin's `detect()` returned `true`; use `--format` explicitly |
| `no importer registered for format '<id>'` | The format ID passed to `--format` does not match any registered plugin or built-in |
