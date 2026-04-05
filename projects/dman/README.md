# dman

`dman` is a dataset management toolkit for machine learning workflows. It combines a Rust core, a CLI, a Python SDK, a TUI, and a registry-backed format system so datasets can move between local directories, annotation tools, and custom providers without hard-coding every format into the core.

## What dman does

- registers datasets in a local SQLite catalog
- imports and exports built-in formats such as `yolo`, `coco`, and `huggingface`
- supports registry-based user-defined formats through Rust or Python providers
- integrates with Label Studio for import/export
- exposes a Python SDK for building, loading, and updating datasets
- ships an interactive terminal UI for browsing datasets
- handles multi-view, multi-modal, and segmentation datasets through the `Dataset → Sample → Asset → Annotation` model

## Data model

dman organizes data in a four-level hierarchy:

```
Dataset → Sample → Asset → Annotation
```

- **Dataset**: a named, versioned collection of samples
- **Sample**: a logical grouping (one scene, one timestamp, one row); may have multiple assets
- **Asset**: an actual file — image, depth map, point cloud, text, audio, video, mask
- **Annotation**: attaches to a Sample (sample-level) or to a specific Asset (asset-level)

Classic formats such as YOLO, COCO, and HuggingFace auto-create a 1:1 Sample→Asset mapping. Multi-view and multi-modal datasets place multiple assets (e.g. left + right camera) under one sample.

## Start here

- [Quickstart](./quickstart.md)
- [Docs index](./docs/index.md)

## Feature overview

- [Catalog and CLI](./docs/features/catalog-and-cli.md)
- [Format registry and custom providers](./docs/features/format-registry.md)
- [Python SDK](./docs/features/python-sdk.md)
- [Label Studio integration](./docs/features/label-studio.md)
- [Terminal UI](./docs/features/tui.md)

## Architecture at a glance

- `crates/core` — catalog, dataset services, formats, storage, schema, virtual datasets
- `crates/cli` — command-line interface and TUI subcommand
- `crates/server` — HTTP API and Label Studio integration helpers
- `crates/python` — Python bindings, plugin discovery, dataset SDK

## Install

```bash
cargo build --release

# Rust CLI binary
./target/release/dman-cli --help

# Python package with `dman` console entrypoint
pip install .
dman --help
```
