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
