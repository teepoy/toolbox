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
