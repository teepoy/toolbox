# AGENTS

Guide for coding agents working in `projects/dman`.

## Workspace layout

Rust workspace (`resolver = "3"`):

| Crate | Path | Purpose |
|-------|------|---------|
| `dman-core` | `crates/core` | Catalog, DB, formats, storage, schema, services, virtual datasets |
| `dman-cli` | `crates/cli` | CLI + TUI (`default = ["python"]`) |
| `dman-server` | `crates/server` | HTTP API, Label Studio integration; React frontend under `crates/server/frontend/` |
| `dman-python` | `crates/python` | PyO3 bindings, plugin discovery, Python SDK; gated by `feature = "python"` |

Core modules (`crates/core/src/`): `catalog`, `config`, `dataset`, `db`, `embeddings`, `error`, `formats`, `ops`, `patches`, `predictions`, `schema`, `storage`, `types`, `virtual_dataset`.

No `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md`. Inherited rule: always use timeouts for HTTP checks (`curl --max-time 5`).

## Data model

`Dataset → Sample → Asset → Annotation` (four-level hierarchy). Classic formats (YOLO, COCO, HuggingFace) create 1:1 Sample→Asset. Multi-view datasets put multiple assets under one sample. Annotations attach to a Sample or a specific Asset.

## Build / test / lint

Run from the workspace root. `dman-cli` enables `python` by default.

```bash
cargo build --workspace              # full build
cargo build -p dman-cli              # CLI only
pip install .                        # Python package (maturin-backed)
maturin develop                      # dev install into active venv

cargo test                           # all workspace tests
cargo test -p dman-core              # one crate
cargo test -p dman-core dataset_register_creates_record   # one test by name
cargo test -p dman-cli --test integration                 # integration target
cargo test -p dman-cli --test integration schema::import_with_explicit_format_succeeds  # single integration test
cargo test -p dman-python --features python               # Python crate tests

cargo fmt --all                      # format
cargo fmt --all -- --check           # CI check
cargo clippy --workspace --all-targets --all-features -- -D warnings
mdbook build                         # docs
```

Makefile shortcuts: `make build`, `make check`, `make test`, `make verify` (fmt-check + clippy + test + frontend-lint + frontend-build).

### Frontend (`crates/server/frontend`)

React 19 + Vite + Tailwind CSS + TypeScript. ESLint config in `eslint.config.js`.

```bash
npm --prefix crates/server/frontend install
npm --prefix crates/server/frontend run dev | build | lint
```

### Verification by area

| Area | Command |
|------|---------|
| Core logic | `cargo test -p dman-core` |
| CLI + integration | `cargo test -p dman-cli --test integration` |
| Format roundtrips | `cargo test -p dman-core --test roundtrip` |
| Virtual datasets | `cargo test -p dman-core --test virtual_integration` |
| Server/API | `cargo test -p dman-server` |
| Python plugins | `cargo test -p dman-python --features python` |
| Docs | `mdbook build` |
| Frontend | `npm --prefix crates/server/frontend run lint && npm --prefix crates/server/frontend run build` |

### Before finishing

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test
```

## Code style

**Formatting**: `cargo fmt --all`. Group imports: `std` → external crates → local crate (blank line between groups). Explicit imports, no globs.

**Naming**: types/enums/traits `PascalCase`; functions/modules/variables `snake_case`; constants `SCREAMING_SNAKE_CASE`; tests behavior-focused: `dataset_register_creates_record`.

**Types**: `DatasetFormat::new(s)` for format IDs (normalizes aliases). `Path`/`PathBuf` for filesystem paths, never `String`. Public model structs derive `Debug, Clone, Serialize, Deserialize`. Re-exports at `dman_core` root — use those, not deep paths.

**Error handling**: library code uses `dman_core::{DmanError, Result}`. No `unwrap()` in library/runtime code; `expect()` fine in tests. CLI may use `anyhow::{Result, Context}`. Use specific `DmanError` variants: `DatasetNotFound`, `DatasetAlreadyExists`, `SampleNotFound`, `AssetNotFound`, `FormatUnsupported`, `ImportFailed { path, reason }`, `ExportFailed { path, reason }`, `PluginError`, `StorageError`, `SchemaValidation`, `InvalidInput`. Auto-converted: `Io`, `Database`, `SerdeJson`, `SerdeYaml`, `Toml`.

**Lint suppressions**: `#[allow(dead_code)]` OK on fields used later. `#[allow(clippy::too_many_arguments)]` OK on internal service functions. Others need a comment.

## Database and storage

- Create connections via `Database::open(path)` or `Database::open_in_memory()` — enables WAL + FK pragmas automatically. Never use raw `rusqlite::Connection::open()`.
- SQL: explicit, readable, multiline. Bind with `rusqlite::params![]`.
- Store file paths in SQLite, never image bytes/BLOBs (except embeddings vectors as little-endian f32 BLOB).
- Metadata stored as JSON text where schema does so.
- Catalog root: `~/.dman` unless `DMAN_HOME` is set.

## Service patterns

Services are stateless structs (`pub struct XxxService;`) with static-like methods:

```rust
pub fn method(db: &Database, storage: &StorageManager, ...) -> Result<T>
```

Naming: `*Service` for query/CRUD (DatasetService, PatchService, EmbeddingService), `*Ops`/`*Transforms` for physical mutations (DatasetOps, DatasetTransforms).

**Transactions**: multi-statement ops use `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`:

```rust
db.conn.execute("BEGIN IMMEDIATE", [])?;
// ... inserts/updates ...
db.conn.execute("COMMIT", [])?;
// on error: db.conn.execute("ROLLBACK", []).ok();
```

### Key services

- **PatchService** (`patches/mod.rs`): `extract`, `extract_batch`, `get_by_asset`, `get_by_annotation`, `get_by_dataset`, `delete_patch`. Patches table has `annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL`.
- **EmbeddingService** (`embeddings/mod.rs`): `store`, `get`, `list_by_dataset`, `delete_by_asset`, `has_embeddings`, `get_embedding_models`. Vectors encoded as little-endian f32 bytes in BLOB column.
- **VirtualDatasetService** (`virtual_dataset/`): DSL-driven (`Filter`, `Merge`, `Sample`, `Split`, `SchemaTransform`, `Chain`). `evaluate` returns virtual samples; `materialize` creates a physical dataset. Deterministic sampling via `hash_id`. Circular dependency detection enforced on create.

## Python SDK conventions

- All PyO3 code behind `#[cfg(feature = "python")]` in `crates/python`.
- Module registered as `#[pymodule(name = "dman")]` in `lib.rs`. Exports: `create_dataset`, `update_dataset`, `load_dataset`, `DmanDataset`, `DmanDatasetBuilder`, `DmanDatasetUpdater`.
- Import as `import dman` (not `dman_python`).
- Classes use `#[pyclass(unsendable)]`. Method signatures use `#[pyo3(signature = (...))]` for default args.
- Python dicts accepted as `Bound<'_, PyDict>`, converted to JSON via `py.import("json")?.call_method1("dumps", ...)`.
- Internal logic in `*_internal` methods; `#[pymethods]` wrappers handle Python type conversion.
- `DmanDataset` is an eagerly-loaded snapshot — later DB changes not reflected.
- Builder/Updater use `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK` transactions in `build()` / `apply()`.

### Arrow zero-copy FFI

`pyo3-arrow = "0.17"` provides zero-copy RecordBatch export. Pattern:

```rust
let batch = crate::sdk::arrow::samples_to_record_batch(&self.sample_rows)?;
let pyarrow = pyo3_arrow::PyRecordBatch::new(batch).into_pyarrow(py)?;
Ok(pyarrow.unbind().into_any())
```

Four builders in `arrow.rs`: `samples_to_record_batch`, `assets_to_record_batch`, `annotations_to_record_batch`, `categories_to_record_batch`. `created_at` exported as `Utf8` (SQLite stores timestamps as TEXT).

### Plugin conventions

- Plugin-based custom formats resolve through `FormatRegistry`, not hard-coded branches.
- Python plugins define their own classes; Rust reads attributes by name — do **not** import from `dman`.
- `import_dataset` contract returns `list[SampleData-like]` (each with `.name`, `.metadata`, `.assets`, `.annotations`).
- Plugin integration tests in `crates/cli/tests/integration/python_plugin.rs`, gated with `#![cfg(feature = "python")]`.

## Format registry

Built-in formats via `FormatRegistry::default_registry()`: YOLO, COCO, HuggingFace. Custom formats via Rust `FormatImporter`/`FormatExporter` traits or Python plugins under `$DMAN_HOME/plugins`. Use canonical format IDs, not hard-coded enums. HuggingFace format uses `arrow`/`parquet` crates for Parquet I/O.

## Documentation

- Prefer self-explanatory code over comments. Comments only for non-obvious constraints.
- Keep `README.md`, `quickstart.md`, and `docs/src/` aligned with actual CLI/SDK surface.
- `docs/src/SUMMARY.md` must be updated when adding mdBook pages.
- `book/` is build output — do not commit.
- Test fixtures: reuse `crates/core/tests/fixtures/` (`yolo/`, `coco/`, `huggingface/`, `schema/`).

## Patterns to preserve

- Registry-based format dispatch (not fixed enum)
- Update stale tests to match behavior changes; don't revert features to stubs
- CLI output: concise and consistent (`"✓ Registered dataset 'name' (id=…, format=…)"`)
- `clap` derive macros for CLI commands
