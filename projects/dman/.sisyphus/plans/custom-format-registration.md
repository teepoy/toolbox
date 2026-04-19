# Custom Dataset Format Registration Plan

## Goal

Make custom dataset format registration understandable, reliable, and implementation-grade for both users and future contributors.

This plan focuses on explaining and hardening the current architecture for user-defined dataset formats in `dman`, especially Python-based providers under `$DMAN_HOME/plugins`.

---

## Why this plan is needed

The current docs describe the idea of a registry-backed format system, but they do not explain the actual runtime lifecycle in enough detail for someone to implement or debug a custom format.

Today, a reader can see that custom formats are “possible,” but they still lack answers to these practical questions:

1. Where exactly does the plugin file go?
2. How is the plugin discovered?
3. What functions are required versus optional?
4. What exact Python return payload shape does `dman` expect?
5. How does `dman-cli import` choose a provider?
6. How does `dman-cli export` find a provider?
7. What does the plugin receive and what must it return?
8. What are the current limitations of the Python plugin path?
9. How should a format that stores image bytes (for example, parquet with multiple image columns) map into dman’s file-path-based storage model?

This plan answers those gaps and lays out the changes required to make the system easy to use.

---

## Current architecture (as implemented today)

### 1. Format identity

- `crates/core/src/types/mod.rs`
- `DatasetFormat` is now a canonical string-backed type, not a fixed enum of built-ins
- built-ins are normalized to IDs such as `yolo`, `coco`, `huggingface`, and `label-studio`
- user-defined formats are also just string IDs, for example `parquet-multi-image`

### 2. Core registry

- `crates/core/src/formats/mod.rs`
- `FormatImporter` trait:
  - `name(&self) -> &str`
  - `detect(&self, path: &Path) -> bool`
  - `import(&self, db, storage, path, dataset_name) -> Result<Dataset>`
- `FormatExporter` trait:
  - `name(&self) -> &str`
  - `export(&self, db, storage, dataset, output_path) -> Result<()>`
- `FormatRegistry` stores `Vec<Box<dyn FormatImporter>>` and `Vec<Box<dyn FormatExporter>>`
- `FormatRegistry::default_registry()` registers built-in providers for YOLO, COCO, and HuggingFace

### 3. CLI registry usage

- `crates/cli/src/main.rs`
- CLI builds a registry at startup using `build_registry()`
- built-ins are always registered first
- when Python support is available, Python format plugins are loaded and appended
- `dman-cli import`:
  - uses explicit `--format` if provided
  - otherwise calls `registry.detect_format(path)`
  - then fetches the importer with `registry.get_importer(format_id)`
- `dman-cli export`:
  - requires `--format`
  - resolves provider with `registry.get_exporter(format_id)`

### 4. Python plugin discovery

- `crates/python/src/lib.rs`
- `PluginManager::discover()` walks configured plugin directories recursively
- every `.py` file is scanned as plain text for a `dman_plugin` marker
- discovery currently extracts only three metadata fields:
  - `name`
  - `type`
  - `version`
- the discovered plugin is represented by `PluginInfo`

### 5. Python format adapter

- `crates/python/src/plugins/format.rs`
- `load_python_format_registry(plugin_dirs)`:
  - discovers Python plugins
  - keeps only plugins where `plugin_type == "format"`
  - wraps each plugin in `PythonFormatImporter` and `PythonFormatExporter`
- plugin modules are loaded by reading the `.py` file and executing it through PyO3

### 6. Python importer contract (actual current behavior)

The importer path currently expects:

- optional function: `detect(path: str) -> bool`
- required function: `import_dataset(path: str) -> dict`

The `dict` returned by `import_dataset()` must contain:

- `images`: list of dicts containing at least:
  - `file_name: str`
- `annotations`: list of dicts containing at least:
  - `image_file_name: str`
  - `category: str`
  - `bbox: [x, y, width, height]`

Current importer behavior in Rust:

- registers the dataset with format ID = plugin name
- inserts image rows using `path.join(file_name)` as the stored image path
- inserts categories on demand
- inserts bbox annotations only
- assumes `image_file_name` matches one of the declared image `file_name` values

### 7. Python exporter contract (actual current behavior)

The exporter path exists, but the current docs do not explain its actual runtime payload clearly enough.

The exporter receives a dataset snapshot converted by Rust and calls Python `export_dataset(data, output_path)`.

This needs documentation and likely examples, because users cannot infer the exact `data` shape from the public docs today.

---

## Current limitations and confusion points

These are the main reasons the docs feel incomplete.

### A. Discovery is implicit

The docs mention plugins, but they do not explain that discovery is just:

- scan `$DMAN_HOME/plugins`
- find `.py` files
- look for a `dman_plugin` marker
- keep plugins whose `type` is `format`

### B. The import payload shape is underdocumented

The most important hidden detail right now is the required shape of `import_dataset()` output.

Without this, a user cannot implement a provider confidently.

### C. The current importer contract is path-oriented, not bytes-oriented

`dman` stores images as file paths in SQLite.

That means a custom importer for parquet-with-image-bytes must materialize those bytes to files first, then return file names that Rust can resolve under the dataset path.

This is not obvious from the current docs.

### D. Export payload shape is not documented enough

Users do not know what `export_dataset(data, output_path)` will receive.

### E. Registry bootstrap is only explained indirectly

The docs do not explicitly show the sequence:

1. CLI starts
2. built-in registry is created
3. Python plugin registry is loaded from `$DMAN_HOME/plugins`
4. providers are appended
5. import/export commands resolve providers by format ID

### F. Runtime precedence is not documented

If a custom provider reuses a built-in format name, behavior is ambiguous and should be documented or blocked.

### G. Failure modes are not documented

Examples:

- no provider detected
- provider exists but missing required Python function
- returned dict missing `images` or `annotations`
- annotation references unknown image
- non-UTF8 paths

---

## Desired user experience

After this plan is implemented, a user should be able to:

1. Create one Python file under `$DMAN_HOME/plugins/parquet_multi_image.py`
2. Add a `dman_plugin` marker with a unique format ID
3. Implement `detect`, `import_dataset`, and optionally `export_dataset`
4. Run:

```bash
dman-cli import ~/datasets/parquet-corpus --name parquet-demo
```

or:

```bash
dman-cli import ~/datasets/parquet-corpus --name parquet-demo --format parquet-multi-image
```

5. See clear errors if the plugin contract is violated
6. Understand how to materialize image bytes into files before returning records to dman

---

## Recommended scope

This should be handled in two tracks:

1. **Documentation track** — explain exactly how the current system works
2. **Hardening track** — improve runtime validation and lifecycle clarity

Do not redesign the whole plugin system first. Document and harden the current path before adding new transport layers like MCP or gRPC.

---

## Detailed implementation plan

## Phase 1 — Document the current runtime lifecycle

### 1.1 Create a dedicated docs page

Add a new page such as:

- `docs/src/features/custom-format-registration.md`

This page should explain:

- canonical format IDs
- where plugins live
- how discovery works
- how registry bootstrap works
- exact Python function contract
- exact input/output payload shape
- current limitations

### 1.2 Add a lifecycle diagram

Document this sequence explicitly:

1. `dman-cli` starts
2. `FormatRegistry::default_registry()` loads built-ins
3. CLI resolves plugin directory from `Catalog::plugins_path()`
4. `load_python_format_registry(...)` discovers Python `format` plugins
5. importer/exporter wrappers are added to the registry
6. `import` resolves by explicit ID or auto-detection
7. `export` resolves by explicit ID

### 1.3 Document plugin file placement

Be explicit:

- default path: `~/.dman/plugins`
- overridden path root: `$DMAN_HOME/plugins`
- recursive discovery is allowed
- any `.py` file with a valid marker is a candidate plugin

---

## Phase 2 — Specify the Python provider contract precisely

### 2.1 Document required metadata

Required marker:

```python
dman_plugin = {
    "name": "parquet-multi-image",
    "type": "format",
    "version": "1.0.0",
}
```

Clarify:

- `name` is the canonical format ID used by CLI and registry
- `type` must be `format`
- `version` is metadata only right now

### 2.2 Document `detect()`

- signature: `detect(path: str) -> bool`
- optional
- called during auto-detection only
- should be fast and side-effect-free
- should not mutate files
- should not do expensive parsing if a simple heuristic is enough

### 2.3 Document `import_dataset()`

Current actual signature:

- `import_dataset(path: str) -> dict`

Current required output shape:

```python
{
  "images": [
    {"file_name": "relative/or/materialized/file.png"}
  ],
  "annotations": [
    {
      "image_file_name": "relative/or/materialized/file.png",
      "category": "cat",
      "bbox": [x, y, width, height]
    }
  ]
}
```

Document all validation rules from the Rust side.

### 2.4 Document `export_dataset()`

Document the exact Python signature and the exact structure of `data` passed from Rust.

If the current structure is not stable or not documented in code, first inspect and codify it before publishing docs.

---

## Phase 3 — Add a complete parquet multi-image example

This should become the canonical example.

### 3.1 Example dataset assumption

Assume parquet rows like:

- `image_front: bytes`
- `image_back: bytes`
- `label: str`
- `bbox: list[float]`

### 3.2 Explain the required materialization step

Because dman stores image paths, not image bytes, the plugin must:

1. read parquet rows
2. decode image bytes
3. write those images to files under a deterministic directory
4. return `file_name` values that Rust can resolve under the dataset path

### 3.3 Show exact materialization layout

Recommend something like:

```text
parquet-corpus/
  train-00000.parquet
  .materialized-images/
    train-00000_0_image_front.png
    train-00000_0_image_back.png
```

### 3.4 Document constraints

- file names must be unique within the imported dataset
- `annotations[].image_file_name` must match `images[].file_name`
- bbox must have exactly 4 floats
- plugin is responsible for turning bytes into files

---

## Phase 4 — Harden runtime validation

The docs alone are not enough. The runtime should explain contract failures clearly.

### 4.1 Improve importer validation errors

Add explicit checks and user-facing errors for:

- missing `images`
- missing `annotations`
- non-list values
- missing `file_name`
- missing `image_file_name`
- missing `category`
- bbox length != 4
- duplicate image file names
- annotation references an unknown image

### 4.2 Improve exporter validation errors

Validate that the exporter callback exists and produce a clear error if it does not.

### 4.3 Document detection collisions

If multiple providers claim the same path, define current behavior explicitly.

Recommended short-term rule:

- first registered provider wins

Recommended longer-term improvement:

- surface a collision error rather than silently picking one

---

## Phase 5 — Add tests for the custom format lifecycle

### 5.1 Discovery tests

Cover:

- plugin file under nested `$DMAN_HOME/plugins/...`
- missing marker ignored
- non-`format` plugins ignored by format registry bootstrap

### 5.2 Importer contract tests

Use temp plugin files to verify:

- `detect()` present
- `detect()` absent
- valid `images`/`annotations` payload
- missing required keys
- invalid bbox length
- annotation/image mismatch

### 5.3 CLI integration tests

Add end-to-end tests that:

- create temp `DMAN_HOME`
- write plugin under `plugins/`
- run `dman-cli import ... --format parquet-multi-image`
- verify dataset format ID is stored correctly

### 5.4 Export tests

Add tests that verify `dman-cli export ... --format parquet-multi-image` reaches the plugin and writes expected output.

---

## Phase 6 — Clarify extension boundaries

The current Python provider path is file-based and synchronous.

Document that clearly.

If future interfaces are desired (MCP, gRPC, subprocess, HTTP), define them as separate provider backends rather than overloading the current Python contract.

Recommended future layering:

- core registry stays trait-based
- provider backends can be:
  - built-in Rust
  - Python file plugin
  - subprocess/MCP provider
  - remote gRPC provider

But do not mix this into the first docs pass unless implementation exists.

---

## Deliverables

1. A dedicated custom-format registration doc page in mdBook
2. A contract table for `dman_plugin`, `detect`, `import_dataset`, `export_dataset`
3. A full parquet multi-image example
4. Clear runtime limitation notes
5. Stronger runtime validation and tests

---

## Verification checklist

Documentation is done when:

- a user can implement a Python provider without reading Rust source
- the docs explain both discovery and invocation lifecycle
- the parquet example includes image-byte materialization

Runtime hardening is done when:

- contract violations produce specific errors
- CLI integration tests cover discovery/import/export with a temp plugin
- docs and behavior match each other exactly

---

## Recommended execution order

1. write the dedicated docs page
2. inspect and document the exact exporter payload
3. add the parquet example
4. harden runtime validation
5. add end-to-end tests

This order keeps the work grounded in the current implementation while making the system understandable before larger redesigns.
