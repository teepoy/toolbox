# Catalog and CLI

`dman` keeps dataset metadata in a local SQLite catalog under `~/.dman` by default. The catalog stores dataset names, paths, format IDs, samples, assets, annotations, categories, and related metadata.

## Core commands

```bash
dman-cli init
dman-cli add mnist ~/datasets/mnist --format yolo
dman-cli list
dman-cli inspect mnist
dman-cli remove mnist --yes
```

## Mental model

- `add` registers a dataset path and format ID in the catalog
- `import` parses an external dataset format into dman-managed records
- `export` writes a catalog dataset back out through a registered exporter
- `list` and `inspect` are read-only catalog views

## Data hierarchy

dman organizes data as `Dataset → Sample → Asset → Annotation`:

- **Sample**: a logical grouping (scene, timestamp, row); may contain multiple assets
- **Asset**: an actual file — image, depth map, point cloud, text, audio, video, mask
- **Annotation**: attaches to a sample or to a specific asset

Classic formats (YOLO, COCO, HuggingFace) auto-create a 1:1 Sample→Asset mapping. Multi-view and multi-modal datasets place multiple assets under one sample.

## Format IDs

Formats are identified by canonical strings such as:

- `yolo`
- `coco`
- `huggingface`
- `label-studio`
- user-defined IDs like `parquet-multi-image`

That means the CLI is not limited to a fixed enum of built-ins.

## Import and export

```bash
dman-cli import ~/datasets/yolo-sample --name sample
dman-cli export sample /tmp/sample-export --format yolo
```

If `--format` is omitted on import, dman asks the registered providers to detect the input format.

## Inspect output

`dman-cli inspect` shows sample and asset counts alongside annotation counts:

```
──────────────────────────────────────────────────
  Name:    my-dataset
  ID:      1
  Format:  yolo
  Path:    /path/to/data
  Created: 2026-04-05 12:00:00
──────────────────────────────────────────────────
  Samples:     42 / Assets: 42
  Annotations: 128
  Disk size:   54321 B
──────────────────────────────────────────────────
```

## Environment

- `DMAN_HOME` overrides the catalog root

```bash
DMAN_HOME=/tmp/my-dman-home dman-cli init
```
