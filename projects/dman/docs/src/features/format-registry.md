# Format Registry

The format system is registry-backed. Built-in formats and user-defined formats both participate through the same importer/exporter interfaces.

## Why this matters

Older dataset tools often hard-code a short list of formats into the core. `dman` instead treats formats as registry IDs, so the core can stay stable while users add new providers.

## Built-in format IDs

- `yolo`
- `coco`
- `huggingface`

These are pre-registered at startup.

## User-defined formats

User-defined formats can be registered through:

- Rust `FormatImporter` / `FormatExporter` implementations
- Python plugins discovered from `$DMAN_HOME/plugins`

## Python plugin contract

```python
dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}
```

Optional and required call points:

- `detect(path)` → optional; return `True` if this provider can handle the given path
- `import_dataset(path)` → required; return a dict with `samples` and `annotations` keys
- `export_dataset(data, output_path)` → required; serialize data back to disk

### Return type conventions

`import_dataset` should return a structure that maps to the `Dataset → Sample → Asset → Annotation` model:

```python
# PySampleData: one logical grouping (scene, timestamp, row)
# PyAssetData:  one file — image, depth map, point cloud, text, audio, video, mask
# PyAnnotationData: one label, attaches to a sample or a specific asset

def import_dataset(path: str) -> dict:
    # Each entry in 'samples' is a PySampleData dict
    # Each entry in 'annotations' is a PyAnnotationData dict
    return {
        "samples": [
            {
                "name": "scene_001",
                "assets": [
                    # Each asset is a PyAssetData dict
                    {"file_name": "images/frame_left.jpg",  "asset_type": "image"},
                    {"file_name": "depth/frame_left.npy",   "asset_type": "depth_map"},
                ],
                "metadata": {"split": "train"},
            }
        ],
        "annotations": [
            {
                "sample_name": "scene_001",
                "asset_file_name": "images/frame_left.jpg",  # None → sample-level
                "category": "car",
                "bbox": [100.0, 200.0, 50.0, 80.0],
            }
        ],
    }
```

Classic 1:1 formats (YOLO, COCO) auto-create one Sample per image asset. Multi-view and multi-modal providers place multiple assets under one sample.

## Example use case

A provider named `parquet-multi-image` can deserialize parquet files that contain multiple image-byte columns in each row, materialize those image files as assets, and register the resulting dataset inside dman.

```bash
dman-cli import ~/datasets/parquet-corpus --name parquet-demo --format parquet-multi-image
dman-cli export parquet-demo /tmp/out --format parquet-multi-image
```

## Registry behavior

- importers are looked up by format ID
- exporters are looked up by format ID
- import can auto-detect when one provider matches the path
- add/import/export all use the same canonical format ID model

See the [Quickstart](../quickstart.md) for a longer parquet provider walkthrough.
