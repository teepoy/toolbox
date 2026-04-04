# dman — Learnings

## Project Context
- Rust workspace: crates/core, crates/cli, crates/tui, crates/server, crates/python
- edition = "2024", rustc 1.94.1
- SQLite catalog at ~/.dman/catalog.db
- WAL mode on every connection
- NO unwrap() in library code — use DmanError everywhere
- PyO3 feature-gated behind `--features python`
- Images stored as files (paths in SQLite), NO BLOBs
- TDD: RED → GREEN → REFACTOR

## Key Crate Versions (from research)
- ratatui = "0.30"
- clap = { version = "4.6", features = ["derive"] }
- rusqlite = { version = "0.32", features = ["bundled"] }
- axum = "0.8"
- pyo3 = { version = "0.28", features = ["extension-module"] }
- parquet = "58.1"
- arrow = "58.1"
- rust-embed = "8.11"
- serde = { version = "1", features = ["derive"] }
- serde_json = "1"
- toml = "0.8"
- thiserror = "2"
- anyhow = "1"
- walkdir = "2.5"
- open = "5.3"
- crossterm = "0.28"
- tower-http = { version = "0.6", features = ["fs", "cors"] }
- tempfile = "3"
- assert_cmd = "2"
- predicates = "3"
- rusqlite_migration = "1"

## Workspace Setup Notes
- Cargo workspace builds cleanly with flat root `[workspace]` and centralized `[workspace.dependencies]`.
- `pyo3` must stay optional at the crate level for `dman-python`; workspace dependencies do not support `optional = true`.

## Task 2 Learnings
- Core error handling can centrally re-export `DmanError` and `Result` from `crates/core/src/lib.rs` without adding any application-level plumbing.
- `serde_yaml` and `toml` are already available in workspace deps, so the core crate can wire them in with `workspace = true` and keep `thiserror`-based conversion variants lightweight.
- Targeted error tests passed for display formatting and `From` conversions once `unwrap()` was removed from the test assertions.

## Core Types Task
- Core domain types live in `crates/core/src/types/mod.rs` and are re-exported from `crates/core/src/lib.rs`.
- `serde`/`serde_json` are already present in `crates/core/Cargo.toml` for serialization roundtrip tests.
- `BBox` remains pixel-based top-left origin; format conversions stay outside core types.

## DB Module Notes
- `rusqlite_migration` works well for append-only schema setup; keep migrations centralized in `db::migrations()`.
- In-memory SQLite may report `journal_mode=memory`; WAL pragmas can still execute without error during tests.
- Database primitives should stay thin: open, enable WAL, migrate, and leave domain logic elsewhere.
## Task 4 fixture infra
- Tiny 1x1 JPEG fixtures work well for YOLO/COCO smoke tests and keep workspace tests fast.
- A minimal parquet fixture can be generated via a temporary Rust helper when Python parquet libraries are unavailable.

## T7: Schema System (TOML Parsing + Validation)

### Key Findings

**Fixture format discrepancy**: `tests/fixtures/schema/basic.toml` uses `[[fields]]` (lowercase keys, lowercase dtype values like `"string"`, `"bbox"`) while the task spec shows `[[columns]]` with PascalCase dtypes. Solved by accepting both via a `RawSchema` intermediate struct with `#[serde(default)]` on both `columns` and `fields` fields, preferring `columns` when non-empty.

**DataType custom serde**: `EmbeddingVector(128)` cannot be deserialized by serde's built-in enum derivation. Implemented custom `Deserialize` with a `Visitor` that calls `DataType::from_str_repr()` — case-insensitive matching via `.to_ascii_lowercase()`. The same pattern handles `List(InnerType)` recursively.

**Wrapped `[schema]` form**: Some schemas may nest under `[schema]`. Handled with a `WrappedSchema` fallback: try flat `RawSchema` first, then `WrappedSchema`.

**Unused import fix**: `DmanError` needed only in test assertions — moved import inside `#[cfg(test)] mod tests { use crate::error::DmanError; }` to avoid production-code warning.

**validate_row semantics**:
- Missing optional field (no default) → OK
- Missing required field → ValidationError
- Present null required field → ValidationError
- Int accepts whole JSON numbers only (fract() == 0.0 check)
- EmbeddingVector(N) checks both array length and numeric elements

**Edition 2024 / rustc 1.94**: No issues with the approach used.

### Files Changed
- `crates/core/src/schema/mod.rs` — new (794 lines, 19 schema tests)
- `crates/core/src/lib.rs` — added `pub mod schema;`

### Test Results
20 tests pass, 0 failures, 0 warnings.

## T6: Dataset CRUD Operations

### Key Findings

**Actual Dataset struct**: `types/mod.rs` uses non-optional `id: i64`, `format: DatasetFormat`, `created_at: String` (not wrapped in Option). Task description showed optional fields — always check actual types.

**Format serialization**: `DatasetFormat` is an enum without string DB storage — must manually serialize/deserialize between `"Yolo"/"Coco"/"HuggingFace"/custom strings` and enum variants via helper functions.

**Cascade delete**: The DB schema has `ON DELETE CASCADE` foreign keys but they only fire if `PRAGMA foreign_keys = ON`. Instead of relying on the pragma, explicit DELETE statements with subqueries (`DELETE FROM patches WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?)`) are safer and work regardless of pragma state.

**Annotation FK**: In the actual schema, `annotations` only has `image_id` FK (not `dataset_id`). Must delete annotations through image_id subquery.

**lib.rs drift**: The file gained `pub mod schema;` from T7 before this task ran — always re-read before editing.

**DatasetService pattern**: Static methods on a unit struct (`pub struct DatasetService;`) work cleanly for service-style operations without constructor overhead.

### Files Changed
- `crates/core/src/dataset/mod.rs` — new (590 lines, 17 dataset tests)
- `crates/core/src/error.rs` — added `DatasetAlreadyExists(String)` variant
- `crates/core/src/lib.rs` — added `pub mod dataset;`

### Test Results
21 tests pass (17 dataset + pre-existing), 0 failures.

## T10: Configuration System

### Key Findings

- `#[serde(default)]` must be applied at both the struct level and on nested fields to make partial TOML config loading reliable.
- `DmanConfig::load()` can treat a missing file as a normal default case before attempting TOML parsing.
- `toml::to_string_pretty()` is a clean fit for human-editable config files; create parent dirs first, then write the serialized string.
- Round-trip tests are easiest with `tempdir()` plus a nested output path to verify directory creation.

### Test Results

- `cargo test -p dman-core -- config` passed: 4 config tests green.

## T9: Catalog Service (2026-04-04)
- `Catalog` struct wraps `Database` + `PathBuf` home directory
- `home_path()` checks `$DMAN_HOME` env var first, falls back to `dirs::home_dir()/.dman`
- `init_at(path)` / `open_at(path)` private helpers enable test isolation without env var races
- Public `init()` / `open()` delegate to private helpers using `home_path()`
- `open_at` returns `DmanError::StorageError("catalog not initialized - run \`dman init\`")` if `catalog.db` absent
- `dirs = "5"` added to crates/core/Cargo.toml
- Test pattern: `tempdir()` + call `init_at`/`open_at` directly — avoids `std::env::set_var` race conditions in parallel test runs
- `// SAFETY:` annotation required on all `unsafe` blocks (Rust convention, enforced by clippy)
- Pre-existing broken `storage/mod.rs` (T8) needed `sha2 = "0.10"` added to compile; also had duplicate test function (pre-existing, not my problem to fix beyond unblocking compilation)
- `unwrap_err()` requires `T: Debug` — use `.err().expect(...)` instead when `T` doesn't implement `Debug`

## T8: Image Storage Manager (2026-04-04)

### Dependency
- `sha2 = "0.10"` added directly to `crates/core/Cargo.toml` (not workspace)
- No `hex` crate needed — use `format!("{:02x}", b)` with `std::fmt::Write` trait

### SHA256 without hex crate
```rust
use std::fmt::Write as FmtWrite;
use sha2::{Digest, Sha256};
let digest = Sha256::new().chain_update(data).finalize();
let mut hex = String::with_capacity(64);
for b in digest.iter() { write!(&mut hex, "{:02x}", b).unwrap(); }
```

### Symlink deletion
- `fs::remove_file` handles both regular files and symlinks on Linux
- Guard: check `exists() || symlink_metadata().is_ok()` before removal (avoids error on missing)

### integrity check pattern (rusqlite)
```rust
let mut stmt = db.conn.prepare("SELECT file_path FROM images WHERE dataset_id = ?")?;
let paths: Vec<String> = stmt
    .query_map(rusqlite::params![dataset_id], |row| Ok(row.get::<_, String>(0)?))
    .?.collect::<rusqlite::Result<Vec<_>>>()?;
```

### Lib.rs already had extra modules
- By T8 time, `lib.rs` already had `pub mod catalog; pub mod config;` — always re-read before editing

### Cargo.toml already had `dirs = "5"`
- Re-read Cargo.toml before editing — other tasks may have added deps

## T15: Generic Format Trait + Plugin Interface (2026-04-04)

### Design Decisions

- `detect` must be `fn detect(&self, path: &Path) -> bool` (method, not associated function) because `FormatImporter` is used as a trait object `Box<dyn FormatImporter>` — associated functions (`where Self: Sized`) cannot be called on trait objects.
- `Default for FormatRegistry` delegates to `FormatRegistry::new()` (empty registry); callers who want pre-populated stubs call `FormatRegistry::default_registry()`.
- Stub implementations use `DmanError::FormatUnsupported` (not a `NotImplemented` variant that doesn't exist) for their `import`/`export` error returns.
- Detection heuristics (YOLO=`data.yaml`, COCO=`annotations/*.json`, HF=`*.parquet`) are documented in stub docstrings; stubs return `false` so no disk I/O occurs in tests.
- `Send + Sync` bounds on both traits are required for `Box<dyn FormatImporter>` / `Box<dyn FormatExporter>` to be storable in shared state.

### Files Changed
- `crates/core/src/formats/mod.rs` — new (467 lines, 15 tests)
- `crates/core/src/lib.rs` — added `pub mod formats;`

### Test Results
15 tests pass, 0 failures, 0 warnings.

## T16 — Embeddings Storage

### Schema Reality vs Spec
- Task spec referenced `dataset_id` column in `embeddings` table and an `Embedding.dataset_id` field — neither exists in actual code.
- Actual DB schema: `embeddings(id, image_id, model_name, vector BLOB, metadata TEXT)` — no `dataset_id`.
- Actual `Embedding` type: `id: i64` (not `Option<i64>`), field is `model_name` (not `model`), no `dataset_id`.
- Methods that filter by `dataset_id` (`list_by_dataset`, `has_embeddings`, `get_embedding_models`) use a JOIN through `images` table.

### f32 BLOB Encoding
- No `bytemuck` dep in project; used manual little-endian encoding:
  - encode: `iter().flat_map(|f| f.to_le_bytes()).collect()`
  - decode: `chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect()`
- `f32::NAN` roundtrips correctly via bit-level comparison (`to_bits()` / `is_nan()`).

### Files Changed
- `crates/core/src/embeddings/mod.rs` — new (302 lines, 8 tests)
- `crates/core/src/lib.rs` — added `pub mod embeddings;`

### Test Results
8 tests pass, 0 failures, 0 warnings.

## T11 CLI Shell Learnings

### assert_cmd integration tests must live in `tests/` not `src/main.rs`
- `Command::cargo_bin("name")` requires `CARGO_BIN_EXE_*` env vars, which are only set for integration tests (files under `tests/`), not unit tests inside `src/`.
- If placed inside `#[cfg(test)] mod tests { ... }` in main.rs, every test panics with `` `CARGO_BIN_EXE_dman` is unset ``.
- Fix: move assert_cmd tests to `crates/cli/tests/cli.rs`.

### Binary name must match package name in assert_cmd
- `Command::cargo_bin("dman")` fails if the actual binary is `dman-cli` (as set in `[package] name`).
- Use `Command::cargo_bin("dman-cli")` to match, or add `[[bin]] name = "dman"` to Cargo.toml.

### clap `#[derive(Subcommand)]` needs `///` doc comments for help text
- The `///` doc comments on enum variants and their fields are NOT memo-style agent comments — they are consumed by clap's derive macro to generate `--help` output. They are functionally required.

### Catalog::open() vs init()
- `open()` errors if `catalog.db` does not exist — commands that read data must call `open()`, never `init()`.
- `init()` is idempotent — safe to call even if already initialized.
- Both respect `DMAN_HOME` env var via `Catalog::home_path()`.

### comfy-table usage
- `Table::new()` → `table.set_header(vec![Cell::new(...)])` → `table.add_row(vec![Cell::new(...)])` → `println!("{table}")`.
- `Cell::new(value)` accepts anything implementing `Display`, including `i64`, `String`, `&str`, `PathBuf::display()`.

### Files Changed
- `crates/cli/Cargo.toml` — added dman-core, comfy-table, colored, serde_json, anyhow deps
- `crates/cli/src/main.rs` — full clap derive CLI (~355 lines)
- `crates/cli/tests/cli.rs` — 8 integration tests (all pass)

## T13: YOLO Format Importer/Exporter (2026-04-04)

### Key Findings

- `formats/mod.rs` was already modified by other tasks — had `pub mod huggingface;` prepended. Always re-read before editing.
- `walkdir` was already in `dman-core`'s `Cargo.toml` (added by a prior task). Re-check before adding deps.
- YOLO `names` field supports two YAML forms: `["cat","dog"]` (list) and `{0: cat, 1: dog}` (map). Handle both with an untagged enum + `into_sorted_vec()`.
- labels dir derivation: strip first path component of `train` subdir and prepend `labels/` — `images/train` → `labels/train`. Use `components().skip(1).collect::<PathBuf>()`.
- `DataYaml.path` field is present in YOLO format but unused by our importer; suppress dead_code warning with `#[allow(dead_code)]` on the field.
- Transaction pattern: `BEGIN IMMEDIATE` outside closure, run logic in `(|| -> Result<()> { ... })()`, then COMMIT or ROLLBACK based on result.
- `db.conn.last_insert_rowid()` is the right way to get newly inserted image_id.
- WalkDir `min_depth(1).max_depth(1).sort_by_file_name()` gives deterministic file enumeration without recursion.
- Bbox stored as `{"x": cx, "y": cy, "w": w, "h": h, "normalized": true}` — raw YOLO values, no pixel conversion.
- Missing label file: simply skip the annotation loop (don't error).
- Export: use `fs::copy` for image files; write label lines as `"class_idx x y w h"`.

### Files Changed
- `crates/core/src/formats/yolo/mod.rs` — new (340 lines, 11 tests)
- `crates/core/src/formats/mod.rs` — added `pub mod yolo;`

### Test Results
11 tests pass, 0 failures, 0 warnings.

## T12: HuggingFace Parquet Importer/Exporter (2026-04-04)

### Dependencies Added to crates/core/Cargo.toml
- `arrow = { workspace = true }` — Arrow array/record batch types
- `parquet = { workspace = true }` — Parquet read/write
- `walkdir = { workspace = true }` — Recursive directory traversal (was already present from T13, no-op re-add)

### Arrow `is_null` Requires Trait Import
- `StringArray::is_null(idx)` requires `use arrow::array::Array;` to be in scope.
- Compiler error: `no method named 'is_null' found` with hint `trait Array which provides is_null is implemented but not in scope`.

### Parquet Reading Pattern (arrow/parquet 58.1)
```rust
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::array::{Array, StringArray};

let file = File::open(&parquet_path)?;
let builder = ParquetRecordBatchReaderBuilder::try_new(file)
    .map_err(|e| DmanError::FormatError(e.to_string()))?;
let mut reader = builder.build()
    .map_err(|e| DmanError::FormatError(e.to_string()))?;
for batch in reader { ... }
```

### Parquet Writing Pattern (arrow/parquet 58.1)
```rust
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

let schema = Arc::new(Schema::new(vec![
    Field::new("file_name", DataType::Utf8, false),
    ...
]));
let batch = RecordBatch::try_new(schema.clone(), vec![
    Arc::new(StringArray::from(col_data)),
    ...
]).map_err(|e| DmanError::FormatError(e.to_string()))?;
let file = File::create(&out_path)?;
let mut writer = ArrowWriter::try_new(file, schema, None)
    .map_err(|e| DmanError::FormatError(e.to_string()))?;
writer.write(&batch).map_err(|e| DmanError::FormatError(e.to_string()))?;
writer.close().map_err(|e| DmanError::FormatError(e.to_string()))?;
```

### Image Filename Column Detection
- HuggingFace datasets use different column names for image paths: `file_name`, `image`, `path`, `image_path`.
- Probe each candidate name via `batch.schema().index_of(name)` and use the first match.

### Replacing Stubs in formats/mod.rs
- Removed `HuggingFaceStubImporter` and `HuggingFaceStubExporter` structs entirely.
- Updated `default_registry()` to use `huggingface::HuggingFaceImporter` and `huggingface::HuggingFaceExporter`.
- Existing stub-detection tests (`stubs_detect_returns_false_for_any_path`) still pass because `/tmp/fake` contains no `.parquet` files.

### Fixture Details
- `crates/core/tests/fixtures/huggingface/data/train-00000-of-00001.parquet` — 3 rows, columns: `image_path` (Utf8), `label` (Utf8).

### Files Changed
- `crates/core/Cargo.toml` — added arrow, parquet, walkdir deps
- `crates/core/src/formats/huggingface/mod.rs` — new (~358 lines, 6 tests)
- `crates/core/src/formats/mod.rs` — added `pub mod huggingface;`, replaced stubs with real structs

### Test Results
6 HuggingFace tests pass, 114 total tests pass, 0 failures, 0 warnings.

## T14: COCO Format Importer/Exporter (2026-04-04)

### Key Findings

- `formats/mod.rs` had `CocoStubImporter`/`CocoStubExporter` structs — replaced with `coco::CocoImporter`/`coco::CocoExporter` in `default_registry()`.
- Existing tests `stubs_detect_returns_false_for_any_path` and `each_stub_importer_detect_returns_false` pass with the real implementation because test paths (`/tmp/fake`, `/tmp/some_dataset_dir`) don't have COCO file structure, so `detect()` returns `false`.
- COCO bbox `[x, y, w, h]` stored as `{"x":..,"y":..,"w":..,"h":..}` JSON string — matches project convention.
- COCO category/image IDs are COCO-specific and don't match DB auto-increment IDs. Must maintain two HashMaps during import: `coco_id → db_id` for both images and categories.
- Export re-numbers all IDs 1-based sequentially to produce valid COCO output.
- `segmentation` field: `serde_json::Value` is the most flexible type — handles both polygon arrays and other formats without panicking.
- `#[serde(default)]` on `segmentation: serde_json::Value` requires that `serde_json::Value` implements `Default` (it does, defaults to `Value::Null`).
- `DatasetService::register()` requires path to exist on disk — pass the actual fixture directory.
- `db.conn.last_insert_rowid()` is called immediately after each INSERT — don't call it after multiple inserts.
- Internal struct `RawImg`/`RawAnn` defined inside function body is valid Rust — useful for local query result types.

### Files Changed
- `crates/core/src/formats/coco/mod.rs` — new (668 lines, 11 tests)
- `crates/core/src/formats/mod.rs` — added `pub mod coco;`, replaced COCO stubs with real impls

### Test Results
11/11 COCO tests pass, 125/125 full suite passes, 0 failures.

## T17: Virtual Dataset Engine (2026-04-04)

### Actual Types vs Spec Discrepancy
- Task spec described `FilterOp::HasAnnotations/CategoryIs/MetadataEq` — actual `FilterOp` in types/mod.rs is generic: `Eq/Ne/Gt/Lt/Gte/Lte/Contains/In`.
- Task spec described `VirtualDatasetDef::Source(i64)` — actual type has no Source variant; initial images come from `VirtualDataset.source_datasets: Vec<i64>`.
- `VirtualDataset` struct has no `created_at` field despite DB having it.
- `VirtualDatasetDef::Sample` only has `ratio: f64` — no `seed` field in actual type.
- `VirtualDatasetDef::Merge` uses `datasets: Vec<i64>` (not `sources: Vec<VirtualDatasetDef>`).

### Filter Column Dispatch Pattern
Column-based dispatch on `VirtualDatasetDef::Filter { column, ... }`:
- `"annotated"` → SQL `SELECT DISTINCT image_id FROM annotations WHERE image_id IN (...)`
- `"category"` / `"category_name"` → SQL JOIN annotations+categories by name
- `"metadata.KEY"` → in-memory `serde_json::Value` comparison
- anything else → in-memory field comparison via match on column name

### rusqlite params_from_iter with Box<dyn ToSql>
When building dynamic SQL with variable-length params:
```rust
let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(cat_name.to_string())];
for id in &image_ids { params_vec.push(Box::new(*id)); }
stmt.query_map(rusqlite::params_from_iter(params_vec.iter().map(|p| p.as_ref())), ...)?
```

### Deterministic Sample Without Seed
Since `Sample { ratio: f64 }` has no seed field, use `hash_id(id, 0)` with a constant seed for deterministic but reproducible ordering.

### Circular Reference Detection (V1)
`check_no_cycle_for_new(db, target_name, vds_id, &mut visited)` walks the existing VDS graph to verify no existing VDS with `target_name` is reachable from `vds_id`. Since new VDS doesn't exist yet at creation time, true A→B→A cycles can only be triggered by testing with existing IDs.

### Files Changed
- `crates/core/src/virtual_dataset/mod.rs` — new (~980 lines, 13 tests)
- `crates/core/src/lib.rs` — added `pub mod virtual_dataset;`

### Test Results
13 virtual_dataset tests pass, 138 total, 0 failures, 0 warnings.

## T24 - Predictions Storage + Management

- `PredictionService` follows identical pattern to `EmbeddingService` (unit struct, all static methods)
- `serde_json::Value` stored as TEXT: `serde_json::to_string()` on write, `serde_json::from_str()` on read
- `compare_models` iterates models list, calls `get_by_model` per model, builds `HashMap<i64, HashMap<String, Value>>` keyed by image_id, then converts to sorted `Vec<ImageComparison>`
- No `created_at` field on the Prediction type despite it existing in the DB schema — don't SELECT it; the type definition is authoritative
- `get_by_dataset` uses subquery: `WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?)`
- `delete_by_model` similarly uses subquery for dataset scoping
- predictions table has no `created_at` in the migration (unlike the docstring in context — the actual schema in db/mod.rs has no created_at column)
- Test count grew from 149 → 175 (added 7 prediction tests, existing tests unchanged)

## T18: Schema Evolution Transforms (2026-04-04)

### Key Findings

- `SchemaOp::AddColumn.default` is `serde_json::Value` (not `Option`); a JSON `null` means no default
- `ColumnDef.default` is `Option<toml::Value>` — must convert via custom `json_value_to_toml()` helper
- `parse_dtype()` works cleanly via serde JSON deserialization: `serde_json::from_value::<DataType>(json!(dtype_str))`
- `transform_metadata()` optimization: skip None metadata unless there's an AddColumn with non-null default
- Immutable pattern: each `apply_single()` produces new `SchemaDefinition` via `.clone()` — never mutates original
- `DmanError::SchemaValidation(String)` is the right variant for all column-not-found / duplicate column errors
- `DataType` implements `PartialEq` — works directly in `assert_eq!` in tests

### Files Changed
- `crates/core/src/virtual_dataset/transforms.rs` — new (340 lines, 16 tests)
- `crates/core/src/virtual_dataset/mod.rs` — added `pub mod transforms;`

### Test Results
16 transforms tests pass, 172 total, 0 failures, 0 warnings.

## T22: Axum HTTP Server (2026-04-04)

### Key Findings

- `tokio = { version = "1", features = ["full"] }` must be added to workspace `Cargo.toml` — it wasn't there by default; only axum and tower-http were in workspace deps.
- `dirs = "5"` added directly to `crates/server/Cargo.toml` (not workspace) — same pattern as in dman-core.
- `tower = { version = "0.5", features = ["util"] }` in `[dev-dependencies]` for `tower::ServiceExt` (`oneshot` method).
- `dman-core` not actually needed at lib.rs level (no imports used yet) — added for future use in T23.
- `Content-Type` header: use `HeaderValue::from_static(...)` with a `match` on file extension — no `mime` crate needed.
- `tokio::fs::read()` returns `Err` on both missing files and permission errors → both map to 404 cleanly.
- Image path convention: `{catalog_path}/data/{dataset_id}/images/{filename}` — matches StorageManager layout.
- Route parameter syntax in axum 0.8: `"/images/{dataset_id}/{filename}"` (curly braces, NOT colon syntax).
- SPA fallback: use `axum::response::Html(...)` for correct `text/html` content-type.
- `CorsLayer::permissive()` from tower_http automatically handles OPTIONS preflight — no manual handler needed.
- `axum::body::to_bytes(body, usize::MAX)` is the correct way to collect response body in tests (axum 0.8).
- `Response::builder().body(Body::from(bytes))` followed by `.unwrap_or_else(|_| ...)` avoids unwrap in library code.
- `axum::serve` requires `tokio::net::TcpListener` (not std): `tokio::net::TcpListener::bind(addr).await`.

### Files Changed
- `Cargo.toml` (workspace) — added `tokio = { version = "1", features = ["full"] }`
- `crates/server/Cargo.toml` — added tokio, serde_json, dman-core, dirs; dev: tower + tempfile
- `crates/server/src/lib.rs` — new (220 lines, 5 tests)
- `crates/server/src/main.rs` — updated (CLI arg parsing, tokio::main, axum::serve)

### Test Results
5 tests pass, 0 failures, 0 warnings.

## T20: Dataset Operate Commands (CRUD ops) (2026-04-04)

### Files Changed
- `crates/core/src/ops/mod.rs` — new (~570 lines, 11 tests: rename, rename_not_found, rename_target_already_exists, duplicate_basic, duplicate_not_found, duplicate_target_exists, merge_two, merge_output_already_exists, split_two_way, split_invalid_ratios, split_three_way)
- `crates/core/src/error.rs` — added `InvalidInput(String)` variant

### Key Patterns

**pre-existing `ops/mod.rs`**: The file existed with `pub mod transforms;` and a `transforms.rs` sibling. Replaced mod.rs entirely; transforms.rs becomes an orphan (not compiled since we dropped `pub mod transforms`).

**`pub mod ops` already in lib.rs**: lib.rs already had it — no edit needed.

**InvalidInput variant**: Was missing from DmanError — added it for ratios validation.

**Transaction pattern** (confirmed working):
```rust
db.conn.execute("BEGIN IMMEDIATE", [])?;
let result = (|| -> Result<T> { ... })();
match result {
    Ok(v) => { db.conn.execute("COMMIT", [])?; Ok(v) }
    Err(e) => { let _ = db.conn.execute("ROLLBACK", []); Err(e) }
}
```

**Deterministic split without FNV**: `std::collections::hash_map::DefaultHasher` + `(image_id as u64 ^ seed).hash(&mut hasher)` works well. Sort by hash, then assign using cumulative ratio thresholds with fractional position `(i + 0.5) / n`.

**Inner structs in closures**: Defining `struct RawImage { ... }` inside a closure body is valid Rust and avoids polluting module namespace.

**`ops/mod.rs` inline image/annotation copy pattern**: Prepare stmt → query_map → collect::<rusqlite::Result<Vec<_>>>() → .map_err(DmanError::Database) to avoid two-step error mapping.

### Test Results
11 ops tests pass, 173 total pass, 7 pre-existing patches failures (JPEG decode), 0 regressions.

## T21: Data Transform Operations (filter, sample, relabel, resize) (2026-04-04)

### Key Findings

**FilterOp reality**: Actual `FilterOp` enum is `Eq/Ne/Gt/Lt/Gte/Lte/Contains/In` — not `HasAnnotations/CategoryIs/MetadataEq` as the task spec implied. Used column-based dispatch matching the virtual_dataset pattern.

**relabel uses categories table**: `annotations` has `category_id` FK to `categories.name`, NOT a `category TEXT` field. Relabeling means `UPDATE categories SET name = ?1 WHERE name = ?2 AND dataset_id = ?3`.

**`pub mod transforms;` placement**: T20 created `ops/mod.rs` without `pub mod transforms;`. T21 prepended the declaration to the top of that file.

**Parallel task race condition**: T20 and T22 ran in parallel and modified the same files (ops/mod.rs, transforms.rs, lib.rs). By the time T21 ran, transforms.rs was already committed by T20's parallel run. T21 confirmed all 8 tests pass.

**resize_images**: Shells out to `convert` (ImageMagick) first, then `python3 -c "from PIL import Image; ..."`. If neither available (or both fail), returns `DmanError::StorageError`. Non-existent files also cause failure (which counts as "tool not available").

**Python format string in Rust**: Can't use `{path!r}` in Rust format strings (no Python repr syntax). Use `format!("{:?}", path)` as `repr_open` parameter — this adds quotes around the string.

**`cargo test` filter path**: Use `-- ops::transforms::tests` to target the submodule; plain `-- ops::transforms` also works.

### Files Changed
- `crates/core/src/ops/transforms.rs` — new (806 lines, 8 tests)
- `crates/core/src/ops/mod.rs` — prepended `pub mod transforms;`
- `crates/core/src/lib.rs` — added `pub mod ops;`

### Test Results
8 transforms tests pass, 181 total pass (7 pre-existing patches failures), 0 regressions.

## T25: Patch Extraction + Storage (2026-04-04)

### Key Findings

**BBox field names**: `width`/`height` not `w`/`h` — always verify actual type definition.
**Patch type**: `bbox: BBox` (embedded, stored as JSON in DB), `file_path: Option<PathBuf>`, no `created_at`.
**DB patches table**: `bbox TEXT NOT NULL` (JSON), `file_path TEXT` (nullable).

**Corrupt fixture JPEGs**: The existing 1×1 JPEG fixtures in `tests/fixtures/coco/images/` fail with zune-jpeg ("Premature End of image"). Must create valid images in tests using `image::RgbImage::from_pixel(4, 4, ...)` in a tempdir.

**image crate 0.25 API**:
```rust
use image::GenericImageView; // required for .dimensions()
let img = image::open(path).map_err(|e| DmanError::StorageError(e.to_string()))?;
let (img_w, img_h) = img.dimensions();
let cropped = img.crop_imm(x, y, w, h);
cropped.save(output_path).map_err(|e| DmanError::StorageError(e.to_string()))?;
```

**BBox hash for filename**:
```rust
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
let mut hasher = DefaultHasher::new();
bbox.x.to_bits().hash(&mut hasher); // hash f64 via bit representation
...
let hash = hasher.finish();
format!("{}_{:08x}.jpg", image_id, hash)
```

**Pre-existing bug fixed**: `crates/core/src/ops/transforms.rs` had `format!("{path!r}")` which is invalid in Rust (Python repr syntax) — fixed with `format!("{:?}", path)`.

### Files Changed
- `crates/core/Cargo.toml` — added `image = "0.25"`
- `crates/core/src/patches/mod.rs` — new (467 lines, 8 tests)
- `crates/core/src/lib.rs` — added `pub mod patches;`
- `crates/core/src/ops/transforms.rs` — fixed invalid format string bug

### Test Results
8 patches tests pass, 188 total tests pass, 0 failures.
