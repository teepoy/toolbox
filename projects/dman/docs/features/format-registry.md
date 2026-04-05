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

A Python format plugin declares:

```python
dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}
```

Optional and required call points:

- `detect(path)` → optional, returns `True` when the provider recognizes the input
- `import_dataset(path)` → required
- `export_dataset(data, output_path)` → required

## Example use case

A provider named `parquet-multi-image` can deserialize parquet files that contain multiple image-byte columns in each row, materialize those images to files, and register the resulting dataset inside dman.

```bash
dman-cli import ~/datasets/parquet-corpus --name parquet-demo --format parquet-multi-image
dman-cli export parquet-demo /tmp/out --format parquet-multi-image
```

## Registry behavior

- importers are looked up by format ID
- exporters are looked up by format ID
- import can auto-detect when one provider matches the path
- add/import/export all use the same canonical format ID model

See the [quickstart custom provider section](../../quickstart.md) for a full parquet plugin example.
