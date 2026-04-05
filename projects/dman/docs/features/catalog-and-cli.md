# Catalog and CLI

`dman` keeps dataset metadata in a local SQLite catalog under `~/.dman` by default. The catalog stores dataset names, paths, format IDs, images, annotations, categories, and related metadata.

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

## Environment

- `DMAN_HOME` overrides the catalog root

Example:

```bash
DMAN_HOME=/tmp/my-dman-home dman-cli init
```
