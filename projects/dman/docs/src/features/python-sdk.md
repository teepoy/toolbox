# Python SDK

The Python SDK is for workflows where CLI commands are too rigid and you want direct control over dataset creation or mutation.

## Main entry points

```python
import dman

builder = dman.create_dataset("demo")
dataset = dman.load_dataset("demo")
updater = dman.update_dataset("demo")
```

## Data model

The SDK operates on the `Dataset → Sample → Asset → Annotation` hierarchy:

- **Sample**: a logical grouping such as a scene, timestamp, or input row
- **Asset**: an actual file attached to a sample — image, depth map, point cloud, text, audio, video, or mask
- **Annotation**: a label that attaches to a sample (sample-level) or to a specific asset (asset-level)

## Common workflows

### Create a dataset

```python
builder = dman.create_dataset("pets")

# Add a sample and its asset explicitly
builder.add_sample("scene_001", metadata={"split": "train"})
builder.add_asset("scene_001", "image", "/tmp/cat.jpg")
builder.add_annotation("scene_001", "cat", bbox=[10.0, 20.0, 30.0, 40.0], asset_name="cat.jpg")

# Convenience: add_image() creates a 1:1 Sample+Asset automatically
builder.add_image("/tmp/dog.jpg")

builder.set_category("cat", supercategory="animal")
builder.set_category("dog", supercategory="animal")
dataset = builder.build()
```

### Load a dataset

```python
dataset = dman.load_dataset("pets")

# Iterate all samples
for sample in dataset.samples():
    print(sample["name"], sample["assets"])

# Access a single sample by name
sample = dataset.get_sample("scene_001")
print(sample["name"])
print(sample["assets"])  # list of PyAssetData dicts

# Get annotations for a sample
annotations = dataset.annotations("scene_001")
print(len(annotations))
```

### Update a dataset

```python
updater = dman.update_dataset("pets")

# Add a new sample with an asset
updater.add_sample("scene_002")
updater.add_asset("scene_002", "image", "/tmp/new_cat.jpg")

# Add an annotation to an existing sample
updater.add_annotation(1, "bird", bbox=[10.0, 20.0, 30.0, 40.0])  # sample_id, not name

# Apply all changes atomically
updater.apply()
```

## Asset types

The `asset_type` parameter accepts the following values:

| Value | Description |
|-------|-------------|
| `"image"` | 2-D raster image (JPEG, PNG, etc.) |
| `"depth_map"` | Per-pixel depth data |
| `"point_cloud"` | 3-D point cloud (PLY, PCD, etc.) |
| `"text"` | Plain text or transcript file |
| `"audio"` | Audio recording |
| `"video"` | Video clip |
| `"mask"` | Segmentation or instance mask |
| `"other"` | Any other asset type |

## Framework integrations

```python
# Convert to PyTorch Dataset
torch_ds = dataset.to_torch_dataset()  # requires: pip install torch

# Convert to HuggingFace Dataset
hf_ds = dataset.to_hf_dataset()        # requires: pip install datasets
```

## Why use the SDK

- build datasets from scripts or notebooks
- integrate preprocessing pipelines directly
- create datasets from nonstandard sources before exporting them
- mix programmatic workflows with the CLI catalog
- work with multi-view and multi-modal data by adding multiple assets per sample
