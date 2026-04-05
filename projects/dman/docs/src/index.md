# dman Docs

`dman` is a dataset management toolkit for machine learning workflows. It combines a Rust core, a CLI, a Python SDK, a TUI, and a registry-backed format system so datasets can move between local directories, annotation tools, and custom providers without hard-coding every format into the core.

## Start points

- [Quickstart](./quickstart.md)
- [Catalog and CLI](./features/catalog-and-cli.md)
- [Format registry](./features/format-registry.md)
- [Python SDK](./features/python-sdk.md)
- [Label Studio](./features/label-studio.md)
- [TUI](./features/tui.md)

## What dman does

- registers datasets in a local SQLite catalog
- imports and exports built-in formats such as `yolo`, `coco`, and `huggingface`
- supports registry-based user-defined formats through Rust or Python providers
- integrates with Label Studio for import/export
- exposes a Python SDK for building, loading, and updating datasets
- ships an interactive terminal UI for browsing datasets

## Recommended reading order

1. Quickstart
2. Catalog and CLI
3. Format registry
4. Python SDK or Label Studio, depending on workflow
5. TUI for interactive browsing
