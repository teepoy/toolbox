# dman — Dataset Management TUI Tool

## TL;DR

> **Quick Summary**: Build a Rust-based hybrid CLI+TUI dataset management tool with import/export (HF Parquet, YOLO, COCO), user-defined schemas, virtual datasets with lazy ETL, Python plugin system (PyO3), a React SPA frontend for visualization, and Label Studio API integration.
>
> **Deliverables**:
> - Cargo workspace with `crates/core`, `crates/cli`, `crates/tui`, `crates/server`, `crates/python`
> - CLI tool (`dman`) with subcommands for all dataset operations
> - Interactive TUI mode (`dman tui`) for browsing datasets
> - Built-in React SPA served by axum (`dman serve`)
> - Python SDK (pip wheel via PyO3/maturin) for PyTorch/HF datasets integration
> - Local HTTP server for serving images to frontend and Label Studio
> - SQLite-based global catalog at `~/.dman/catalog.db`
>
> **Estimated Effort**: XL (35+ tasks across 8 waves)
> **Parallel Execution**: YES — 8 waves, up to 7 concurrent tasks per wave
> **Critical Path**: Workspace setup → Core data model → CLI shell → Format importers → Virtual datasets → Server + Frontend → TUI → Python SDK

---

## Context

### Original Request
User wants to build "dman", a dataset management TUI tool with these features:
1. Find and manage all datasets added to dman's history
2. Import/export in HuggingFace Parquet, YOLO, COCO, and custom formats. Python plugins via PyO3.
3. Manage embeddings, patches (image crops), predictions (inference results)
4. Virtual dataset feature — lazy ETL, composable views + transforms, no sync burden
5. Operate commands — CRUD + merge/split + data-level transforms
6. User-defined schema per dataset (TOML)
7. Visualize by opening browser — built-in React SPA (primary) or Label Studio (API integration)
8. Local server to serve image files
9. Schema evolution via virtual dataset transforms + materialize
10. Python SDK for PyTorch/HF datasets interop
11. Label Studio live API integration (import/export)

### Interview Summary
**Key Discussions**:
- **CLI vs TUI**: Hybrid — CLI subcommands for scripting + `dman tui` for interactive mode
- **Operate commands**: CRUD + merge/split + data transforms (filter, sample, resize, augment, relabel)
- **Schema**: User-defined per dataset via TOML. Schema migration merged with virtual datasets.
- **Virtual datasets**: Composable views + transform pipelines, lazy until materialized via `materialize` command
- **Scale**: 100K–1M images per dataset
- **Embeddings**: Compute via Python plugins, store vectors in DB. Similarity search deferred to v2.
- **Patches**: Image crops/regions from larger images
- **Predictions**: Model inference results stored per-image
- **Visualization**: React SPA primary, Label Studio via REST API
- **Catalog**: Global at `~/.dman/catalog.db`
- **Python SDK**: PyO3 native bindings, pip wheel via maturin
- **Label Studio**: Live API connection for import/export
- **Tests**: TDD — RED-GREEN-REFACTOR

**Research Findings**:
- **Ratatui 0.30.x**: Active (7.5M dl/90d), rich widgets, official tui-rs successor (tui-rs DEPRECATED)
- **clap 4.6.x**: Derive API for type-safe subcommands
- **rusqlite**: Zero overhead, sufficient for local single-user catalog
- **axum 0.8.x**: Lightest for static files + API routes
- **PyO3 0.28.x**: Standard for Python↔Rust, supports plugin systems
- **parquet 58.1.0 + arrow 58.1.0**: Official Apache Arrow Rust impl
- **rust-embed 8.11.0**: Compile-time SPA embedding

### Metis Review
**Identified Gaps** (addressed):
- **Image storage model**: Resolved → file-based with paths in SQLite, no BLOBs
- **Concurrency**: WAL mode for SQLite, single-writer assumption
- **Virtual dataset persistence**: Stored in SQLite as part of catalog
- **PyO3 feature-gating**: `--features python` so core compiles without Python
- **Cargo.toml cleanup**: Remove stale `clippy = "0.0.302"` dependency
- **Workspace structure**: Essential at this scale — `crates/core`, `crates/cli`, `crates/tui`, `crates/server`, `crates/python`
- **SPA fallback routing**: Must handle deep links → index.html
- **Plugin discovery**: `~/.dman/plugins/` directory with manifest
- **Schema types**: string, int, float, bbox, polygon, embedding_vector, image_path
- **Idempotent imports**: Running same import twice must not duplicate data
- **Mid-operation failure**: Transactional imports with rollback on failure

---

## Work Objectives

### Core Objective
Build a production-quality Rust CLI+TUI tool that serves as a local-first ML dataset management platform — enabling import/export across formats, lazy virtual dataset composition, schema management, visual exploration, and Python SDK integration.

### Concrete Deliverables
- `dman` binary with 15+ subcommands
- `dman tui` interactive dataset browser
- `dman serve` web UI at localhost
- `dman-python` pip wheel for Python SDK
- SQLite catalog at `~/.dman/catalog.db`
- React SPA bundled into binary

### Definition of Done
- [ ] `cargo test --workspace` — all tests pass
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` — no warnings
- [ ] `cargo build --workspace --release` — compiles cleanly
- [ ] `dman init && dman import --format yolo ./test-data && dman list` — end-to-end works
- [ ] `dman serve` — opens browser with dataset gallery
- [ ] `dman tui` — interactive browser launches and displays data

### Must Have
- Global catalog at `~/.dman/`
- Import/export for HuggingFace Parquet, YOLO, COCO
- User-defined TOML schemas
- Virtual datasets with filter, merge, sample, split, schema transform
- Materialize command to make virtual datasets concrete
- Operate commands: add, remove, rename, duplicate, merge, split, filter, sample
- Python plugin system (feature-gated)
- Python SDK as pip wheel (PyTorch Dataset + HF datasets)
- React SPA frontend with image gallery + metadata inspector
- Local image server via axum
- Label Studio API integration (import/export projects)
- TUI mode with dataset browser
- Embeddings storage + computation via Python plugins
- Predictions and patches management
- TDD with comprehensive test coverage
- Transactional imports (rollback on failure)
- WAL mode for SQLite concurrency
- Idempotent imports (no duplicates)

### Must NOT Have (Guardrails)
- No cloud features (S3/GCS/Azure) — local only
- No GPU operations in Rust — compute-heavy work goes through Python plugins
- No custom query language/DSL — virtual datasets use Rust enum/struct-based definitions
- No real-time collaboration — single-user tool
- No web UI data mutation in V1 — read-only visualization
- No `unwrap()` in library crates — use `thiserror` for all error types
- No trait hierarchies deeper than 2 levels
- No image BLOBs in SQLite — file-based storage with path references
- No similarity search / nearest neighbor (V2)
- No train/val create command (deferred)
- No premature optimization — no custom allocators, SIMD, memory-mapped files unless proven necessary
- No over-abstraction — no generic type params unless used by 3+ concrete types

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (greenfield project)
- **Automated tests**: TDD — RED (failing test) → GREEN (minimal impl) → REFACTOR
- **Framework**: `cargo test` (built-in Rust test framework)
- **Additional crates**: `assert_cmd` for CLI testing, `tempfile` for temp directories, `mockall` for mocking
- **Test database**: In-memory SQLite (`:memory:`) for unit tests
- **Test fixtures**: Small datasets in `crates/core/tests/fixtures/` (3-image YOLO, small COCO JSON, tiny Parquet)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **CLI commands**: Use Bash — run command, assert exit code + stdout/stderr
- **Library code**: Use Bash (`cargo test`) — run specific test, assert pass
- **TUI**: Use interactive_bash (tmux) — launch TUI, send keystrokes, verify output
- **Web UI**: Use Playwright — navigate, interact, assert DOM, screenshot
- **API endpoints**: Use Bash (curl) — send requests, assert status + response JSON
- **Python SDK**: Use Bash (python -c) — import, call functions, verify output

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Foundation — start immediately):
├── Task 1: Workspace setup + Cargo.toml + .gitignore + README [quick]
├── Task 2: Error types + result types (thiserror) [quick]
├── Task 3: SQLite database module + migrations [quick]
├── Task 4: Test infrastructure + fixtures [quick]
└── Task 5: Core type definitions (Dataset, Schema, Annotation, etc.) [quick]

Wave 1 (Core Data Model — after Wave 0):
├── Task 6: Dataset CRUD operations (register, list, remove, inspect) [deep]
├── Task 7: Schema system — TOML parsing + validation [deep]
├── Task 8: Image storage manager (file-based, path tracking) [unspecified-high]
├── Task 9: Catalog service (global ~/.dman/ management) [unspecified-high]
└── Task 10: Configuration system (global + per-dataset config) [quick]

Wave 2 (CLI + Format Importers — after Wave 1):
├── Task 11: CLI shell — clap subcommands wired to core [unspecified-high]
├── Task 12: HuggingFace Parquet importer/exporter [deep]
├── Task 13: YOLO format importer/exporter [deep]
├── Task 14: COCO format importer/exporter [deep]
├── Task 15: Generic format trait + custom format plugin interface [unspecified-high]
└── Task 16: Embeddings storage + metadata columns [unspecified-high]

Wave 3 (Virtual Datasets + Operations — after Wave 2):
├── Task 17: Virtual dataset engine — filter, merge, sample, split [deep]
├── Task 18: Schema evolution transforms (rename/add/remove columns) [deep]
├── Task 19: Materialize command — make virtual datasets concrete [deep]
├── Task 20: Operate commands — CRUD (rename, duplicate, merge, split) [unspecified-high]
└── Task 21: Operate commands — data transforms (filter, sample, resize, relabel) [unspecified-high]

Wave 4 (Server + API — after Wave 2):
├── Task 22: Axum HTTP server — image serving + API routes [unspecified-high]
├── Task 23: REST API endpoints (datasets, images, metadata) [unspecified-high]
├── Task 24: Predictions storage + management [unspecified-high]
└── Task 25: Patches (image crop) extraction + storage [unspecified-high]

Wave 5 (Frontend + TUI — after Waves 3 & 4):
├── Task 26: React SPA scaffold + build pipeline + rust-embed [visual-engineering]
├── Task 27: Gallery view — image grid + pagination + filtering [visual-engineering]
├── Task 28: Detail view — image + metadata inspector + annotations overlay [visual-engineering]
├── Task 29: TUI shell — ratatui app + dataset browser [deep]
└── Task 30: TUI detail view + keyboard navigation [deep]

Wave 6 (Python + Label Studio — after Wave 4):
├── Task 31: PyO3 integration — feature-gated crate, plugin discovery [deep]
├── Task 32: Python format converter plugin API [deep]
├── Task 33: Python SDK — PyTorch Dataset + HF datasets loader [deep]
├── Task 34: Python SDK — dataset builder/updater [deep]
├── Task 35: Embeddings computation via Python plugins [unspecified-high]
└── Task 36: Label Studio API integration — import/export projects [unspecified-high]

Wave 7 (Integration + Polish — after all):
├── Task 37: CLI end-to-end integration tests [deep]
├── Task 38: Cross-format round-trip tests (import YOLO → export COCO → import back) [deep]
└── Task 39: Virtual dataset + materialize + export integration test [deep]

Wave FINAL (4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T3 → T6 → T11 → T12 → T17 → T19 → T22 → T26 → T37 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6 (Waves 2, 4, 6)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1-5 | — | 6-10 |
| 6 | 3, 5 | 11, 17, 20 |
| 7 | 5 | 12-15, 18 |
| 8 | 3, 5 | 12-14, 22, 25 |
| 9 | 3, 5 | 6, 10, 11 |
| 10 | 9 | 11 |
| 11 | 6, 9, 10 | 37 |
| 12 | 7, 8, 15 | 33, 38 |
| 13 | 7, 8, 15 | 38 |
| 14 | 7, 8, 15 | 38 |
| 15 | 5, 7 | 12-14, 32 |
| 16 | 3, 5 | 35 |
| 17 | 6, 7 | 19, 39 |
| 18 | 7, 17 | 19 |
| 19 | 17, 18 | 39 |
| 20 | 6 | 37 |
| 21 | 6, 8 | 37 |
| 22 | 8, 3 | 23, 26, 36 |
| 23 | 6, 22 | 26-28, 36 |
| 24 | 3, 5 | 37 |
| 25 | 8, 5 | 37 |
| 26 | 22, 23 | 27, 28 |
| 27 | 26 | 37 |
| 28 | 26 | 37 |
| 29 | 6 | 30 |
| 30 | 29 | 37 |
| 31 | 5 | 32, 33, 35 |
| 32 | 15, 31 | 37 |
| 33 | 6, 31 | 37 |
| 34 | 6, 31 | 37 |
| 35 | 16, 31 | 37 |
| 36 | 22, 23 | 37 |
| 37-39 | All impl tasks | F1-F4 |
| F1-F4 | 37-39 | user okay |

### Agent Dispatch Summary

- **Wave 0**: **5 tasks** — T1-T5 → `quick`
- **Wave 1**: **5 tasks** — T6 → `deep`, T7 → `deep`, T8 → `unspecified-high`, T9 → `unspecified-high`, T10 → `quick`
- **Wave 2**: **6 tasks** — T11 → `unspecified-high`, T12 → `deep`, T13 → `deep`, T14 → `deep`, T15 → `unspecified-high`, T16 → `unspecified-high`
- **Wave 3**: **5 tasks** — T17-T19 → `deep`, T20-T21 → `unspecified-high`
- **Wave 4**: **4 tasks** — T22-T25 → `unspecified-high`
- **Wave 5**: **5 tasks** — T26-T28 → `visual-engineering`, T29-T30 → `deep`
- **Wave 6**: **6 tasks** — T31-T35 → `deep`/`unspecified-high`, T36 → `unspecified-high`
- **Wave 7**: **3 tasks** — T37-T39 → `deep`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Workspace Setup + Cargo.toml + .gitignore + README

  **What to do**:
  - Remove stale `clippy = "0.0.302"` from root `Cargo.toml`
  - Convert to Cargo workspace with members: `crates/core`, `crates/cli`, `crates/tui`, `crates/server`, `crates/python`
  - Root `Cargo.toml`: `[workspace]` with `[workspace.dependencies]` for version centralization (ratatui, clap, rusqlite, axum, serde, etc.)
  - `crates/core/Cargo.toml`: shared library crate — rusqlite (bundled), serde, serde_json, serde_yaml, toml, parquet, arrow, walkdir, thiserror, anyhow
  - `crates/cli/Cargo.toml`: binary crate — depends on `dman-core`, clap with derive feature
  - `crates/tui/Cargo.toml`: binary crate — depends on `dman-core`, ratatui, crossterm
  - `crates/server/Cargo.toml`: binary crate — depends on `dman-core`, axum, tower-http, rust-embed, open
  - `crates/python/Cargo.toml`: cdylib crate — depends on `dman-core`, pyo3 (feature-gated behind workspace `python` feature)
  - Each crate gets a minimal `src/lib.rs` or `src/main.rs` with a placeholder
  - Add `.gitignore`: `target/`, `*.swp`, `.env`, `node_modules/`, `dist/`, `*.pyc`, `__pycache__/`
  - Add `README.md` with project name and one-line description
  - Verify: `cargo build --workspace` compiles, `cargo test --workspace` runs (even if 0 tests)

  **Must NOT do**:
  - No feature code — only scaffolding
  - No PyO3 compilation in default build (feature-gated)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure scaffolding — Cargo.toml files, directory structure, no complex logic
  - **Skills**: []
  - **Skills Evaluated but Omitted**:
    - None needed for scaffolding

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T4, T5 once dirs exist — but T1 creates dirs, so effectively first)
  - **Parallel Group**: Wave 0 (start immediately, others depend on this)
  - **Blocks**: T2, T3, T4, T5, and all subsequent tasks
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `Cargo.toml` (current, line 1-7) — existing file to replace. Note: `edition = "2024"` is correct for rustc 1.94.1
  - `src/main.rs` (current, line 1-3) — will be moved to `crates/cli/src/main.rs`

  **External References**:
  - Rust Cargo workspace docs: https://doc.rust-lang.org/cargo/reference/workspaces.html
  - `[workspace.dependencies]` pattern: https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#inheriting-a-dependency-from-a-workspace

  **WHY Each Reference Matters**:
  - Current `Cargo.toml` must be understood to know what to preserve (edition 2024) and what to remove (clippy dep)
  - Workspace docs ensure correct member declaration and dependency inheritance syntax

  **Acceptance Criteria**:

  - [ ] `cargo build --workspace` exits 0
  - [ ] `cargo test --workspace` exits 0
  - [ ] `ls crates/` shows `core/ cli/ tui/ server/ python/`
  - [ ] `grep -c "clippy" Cargo.toml` returns 0 (stale dep removed)
  - [ ] `.gitignore` exists with `target/` entry
  - [ ] `README.md` exists

  **QA Scenarios**:

  ```
  Scenario: Workspace compiles cleanly
    Tool: Bash
    Preconditions: Fresh checkout, rustc 1.94.1 available
    Steps:
      1. Run `cargo build --workspace 2>&1`
      2. Assert exit code = 0
      3. Assert no "error" in output
    Expected Result: Clean build with no errors
    Failure Indicators: Non-zero exit, "error[E" in output
    Evidence: .sisyphus/evidence/task-1-workspace-build.txt

  Scenario: Stale clippy dependency removed
    Tool: Bash
    Preconditions: Cargo.toml updated
    Steps:
      1. Run `grep "clippy" Cargo.toml`
      2. Assert exit code = 1 (not found)
    Expected Result: No "clippy" string in root Cargo.toml
    Failure Indicators: Exit code 0 (clippy still present)
    Evidence: .sisyphus/evidence/task-1-no-clippy.txt
  ```

  **Commit**: YES
  - Message: `chore: scaffold cargo workspace with core/cli/tui/server/python crates`
  - Files: `Cargo.toml`, `crates/*/Cargo.toml`, `crates/*/src/*`, `.gitignore`, `README.md`
  - Pre-commit: `cargo build --workspace`

- [x] 2. Error Types + Result Types (thiserror)

  **What to do**:
  - Create `crates/core/src/error.rs` with a comprehensive `DmanError` enum using `thiserror`
  - Error variants: `Database(#[from] rusqlite::Error)`, `Io(#[from] std::io::Error)`, `SerdeJson(#[from] serde_json::Error)`, `SerdeYaml(#[from] serde_yaml::Error)`, `Toml(#[from] toml::de::Error)`, `Parquet(#[from] parquet::errors::ParquetError)`, `SchemaValidation(String)`, `DatasetNotFound(String)`, `FormatUnsupported(String)`, `ImportFailed { path: PathBuf, reason: String }`, `ExportFailed { path: PathBuf, reason: String }`, `PluginError(String)`, `StorageError(String)`, `MigrationError(String)`
  - Create `pub type Result<T> = std::result::Result<T, DmanError>;`
  - Export from `crates/core/src/lib.rs`
  - Write RED tests first: test Display impl for each variant, test From conversions

  **Must NOT do**:
  - No `unwrap()` — this is the crate that prevents unwrap everywhere else
  - No `anyhow` in core — use `thiserror` for typed errors

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, well-defined pattern, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T3, T4, T5 after T1 completes)
  - **Parallel Group**: Wave 0
  - **Blocks**: T3, T6-T25 (everything uses error types)
  - **Blocked By**: T1

  **References**:

  **External References**:
  - `thiserror` crate docs: https://docs.rs/thiserror/latest/thiserror/
  - Rust error handling best practices: https://doc.rust-lang.org/book/ch09-02-recoverable-errors-with-result.html

  **WHY Each Reference Matters**:
  - `thiserror` derive macros for `#[error()]` and `#[from]` — defines the pattern for all error variants

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- error::tests` passes
  - [ ] Each error variant has a Display impl (test asserts non-empty message)
  - [ ] `From<rusqlite::Error>` for `DmanError` compiles and converts

  **QA Scenarios**:

  ```
  Scenario: Error types compile and display correctly
    Tool: Bash
    Preconditions: crates/core/src/error.rs exists
    Steps:
      1. Run `cargo test -p dman-core -- error::tests 2>&1`
      2. Assert exit code = 0
      3. Assert output contains "test result: ok"
    Expected Result: All error type tests pass
    Failure Indicators: "FAILED" in output, non-zero exit
    Evidence: .sisyphus/evidence/task-2-error-types.txt

  Scenario: From conversions work for all wrapped types
    Tool: Bash
    Preconditions: Tests include From conversion tests
    Steps:
      1. Run `cargo test -p dman-core -- error::tests::test_from 2>&1`
      2. Assert exit code = 0
    Expected Result: All From conversion tests pass
    Failure Indicators: Compilation error or test failure
    Evidence: .sisyphus/evidence/task-2-from-conversions.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add error types and result aliases`
  - Files: `crates/core/src/error.rs`, `crates/core/src/lib.rs`
  - Pre-commit: `cargo test -p dman-core`

- [x] 3. SQLite Database Module + Migrations

  **What to do**:
  - Create `crates/core/src/db/mod.rs` with `Database` struct wrapping `rusqlite::Connection`
  - Enable WAL mode on every connection open: `PRAGMA journal_mode=WAL;`
  - Use `rusqlite_migration` crate for versioned, append-only migrations
  - Initial migration (v1): Create tables:
    - `datasets` (id INTEGER PK, name TEXT UNIQUE NOT NULL, path TEXT NOT NULL, format TEXT, schema_path TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, updated_at TEXT, metadata TEXT)
    - `images` (id INTEGER PK, dataset_id INTEGER FK, file_name TEXT NOT NULL, file_path TEXT NOT NULL, width INTEGER, height INTEGER, hash TEXT, metadata TEXT)
    - `annotations` (id INTEGER PK, image_id INTEGER FK, category_id INTEGER, bbox TEXT, segmentation TEXT, keypoints TEXT, metadata TEXT)
    - `categories` (id INTEGER PK, dataset_id INTEGER FK, name TEXT NOT NULL, supercategory TEXT)
    - `embeddings` (id INTEGER PK, image_id INTEGER FK, model_name TEXT, vector BLOB, metadata TEXT)
    - `predictions` (id INTEGER PK, image_id INTEGER FK, model_version TEXT, result TEXT, score REAL)
    - `patches` (id INTEGER PK, image_id INTEGER FK, bbox TEXT NOT NULL, file_path TEXT, metadata TEXT)
    - `virtual_datasets` (id INTEGER PK, name TEXT UNIQUE NOT NULL, source_datasets TEXT NOT NULL, definition TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)
  - Add indices: `images(dataset_id)`, `annotations(image_id)`, `categories(dataset_id)`, `embeddings(image_id)`, `images(hash)` for dedup
  - Implement `Database::open(path)`, `Database::open_in_memory()`, `Database::migrate()`
  - RED tests: open in-memory → migrate → verify tables exist → insert row → query back

  **Must NOT do**:
  - No business logic — only database operations
  - No `unwrap()` — use `DmanError::Database`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Well-defined database schema, standard migration pattern
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T4, T5 after T1)
  - **Parallel Group**: Wave 0
  - **Blocks**: T6, T8, T9, T16, T22, T24
  - **Blocked By**: T1, T2

  **References**:

  **External References**:
  - `rusqlite` docs: https://docs.rs/rusqlite/latest/rusqlite/
  - `rusqlite_migration` docs: https://docs.rs/rusqlite_migration/latest/rusqlite_migration/
  - SQLite WAL mode: https://www.sqlite.org/wal.html

  **WHY Each Reference Matters**:
  - `rusqlite` API for Connection::open, execute, prepare, query_map — core patterns for all DB operations
  - `rusqlite_migration` for versioned migrations pattern — ensures schema upgrades work on existing databases
  - WAL mode docs to understand concurrency implications (multiple readers, single writer)

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- db::tests` passes
  - [ ] WAL mode verified: test queries `PRAGMA journal_mode` and asserts "wal"
  - [ ] All 8 tables exist after migration
  - [ ] Indices exist on `images(dataset_id)`, `annotations(image_id)`, `images(hash)`
  - [ ] Round-trip: insert dataset → query → verify all fields match

  **QA Scenarios**:

  ```
  Scenario: Database migration creates all tables
    Tool: Bash
    Preconditions: dman-core compiles
    Steps:
      1. Run `cargo test -p dman-core -- db::tests::test_migration_creates_tables 2>&1`
      2. Assert exit code = 0
    Expected Result: Test passes — all 8 tables verified
    Failure Indicators: "FAILED" or missing table error
    Evidence: .sisyphus/evidence/task-3-migration.txt

  Scenario: WAL mode is enabled
    Tool: Bash
    Preconditions: Database module exists
    Steps:
      1. Run `cargo test -p dman-core -- db::tests::test_wal_mode 2>&1`
      2. Assert exit code = 0
    Expected Result: PRAGMA journal_mode returns "wal"
    Failure Indicators: Returns "delete" or other non-WAL mode
    Evidence: .sisyphus/evidence/task-3-wal-mode.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add SQLite database module with versioned migrations`
  - Files: `crates/core/src/db/mod.rs`, `crates/core/src/db/migrations/`
  - Pre-commit: `cargo test -p dman-core -- db`

- [x] 4. Test Infrastructure + Fixtures

  **What to do**:
  - Add test dependencies to `crates/core/Cargo.toml`: `assert_cmd`, `tempfile`, `predicates`
  - Create `crates/core/tests/fixtures/` directory with small test datasets:
    - `fixtures/yolo/`: `data.yaml` + `images/train/` (3 tiny .jpg) + `labels/train/` (3 .txt files)
    - `fixtures/coco/`: `annotations.json` (3 images, 5 annotations, 2 categories)
    - `fixtures/huggingface/`: `README.md` + `train.parquet` (3-row Parquet with image_path + label columns)
    - `fixtures/schema/`: `basic.toml` (sample schema definition)
  - Create `crates/core/tests/common/mod.rs` with test helper functions:
    - `setup_test_db() -> Database` (in-memory, migrated)
    - `fixture_path(name: &str) -> PathBuf`
    - `create_temp_dataset_dir() -> TempDir`
  - Create `crates/cli/tests/` directory with `assert_cmd` boilerplate for CLI integration tests
  - Verify: `cargo test --workspace` runs with the test helpers importable

  **Must NOT do**:
  - No actual tests for features (those come with feature tasks) — only infrastructure
  - Test fixtures must be TINY (3 images max, small files)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File creation + small dataset generation, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T5 after T1)
  - **Parallel Group**: Wave 0
  - **Blocks**: T6-T14 (all tests depend on fixtures), T37-T39
  - **Blocked By**: T1

  **References**:

  **External References**:
  - `assert_cmd` crate: https://docs.rs/assert_cmd/latest/assert_cmd/
  - `tempfile` crate: https://docs.rs/tempfile/latest/tempfile/
  - YOLO format spec (from research): data.yaml + images/ + labels/ with normalized xywh coordinates
  - COCO format spec (from research): JSON with images/annotations/categories arrays, bbox as [x, y, w, h] in pixels
  - HuggingFace Parquet (from research): README.md + .parquet files with split inference

  **WHY Each Reference Matters**:
  - `assert_cmd` docs show how to build CLI integration tests — `Command::cargo_bin("dman")...`
  - Format specs ensure test fixtures are valid and representative of real data

  **Acceptance Criteria**:

  - [ ] `ls crates/core/tests/fixtures/yolo/` shows data.yaml, images/, labels/
  - [ ] `ls crates/core/tests/fixtures/coco/` shows annotations.json
  - [ ] `ls crates/core/tests/fixtures/huggingface/` shows README.md, train.parquet
  - [ ] `cargo test --workspace` passes (test helpers compile, fixtures accessible)

  **QA Scenarios**:

  ```
  Scenario: Test fixtures exist and are valid
    Tool: Bash
    Preconditions: Fixture directory created
    Steps:
      1. Run `ls -R crates/core/tests/fixtures/ 2>&1`
      2. Assert output contains "yolo", "coco", "huggingface", "schema"
      3. Run `python3 -c "import json; json.load(open('crates/core/tests/fixtures/coco/annotations.json'))" 2>&1`
      4. Assert exit code = 0 (valid JSON)
    Expected Result: All fixture dirs exist, COCO JSON is valid
    Failure Indicators: Missing directories, invalid JSON
    Evidence: .sisyphus/evidence/task-4-fixtures.txt

  Scenario: Test helpers compile and work
    Tool: Bash
    Preconditions: common/mod.rs exists
    Steps:
      1. Run `cargo test --workspace 2>&1`
      2. Assert exit code = 0
    Expected Result: All workspace tests pass including helper compilation
    Failure Indicators: Compilation errors in test helpers
    Evidence: .sisyphus/evidence/task-4-helpers.txt
  ```

  **Commit**: YES
  - Message: `test(core): add test infrastructure and fixture datasets`
  - Files: `crates/core/tests/`, `crates/cli/tests/`
  - Pre-commit: `cargo test --workspace`

- [x] 5. Core Type Definitions

  **What to do**:
  - Create `crates/core/src/types/mod.rs` with core domain types:
    - `Dataset { id: i64, name: String, path: PathBuf, format: DatasetFormat, schema_path: Option<PathBuf>, created_at: String, updated_at: Option<String>, metadata: Option<serde_json::Value> }`
    - `DatasetFormat` enum: `Yolo`, `Coco`, `HuggingFace`, `Custom(String)`
    - `Image { id: i64, dataset_id: i64, file_name: String, file_path: PathBuf, width: Option<u32>, height: Option<u32>, hash: Option<String>, metadata: Option<serde_json::Value> }`
    - `Annotation { id: i64, image_id: i64, category_id: Option<i64>, bbox: Option<BBox>, segmentation: Option<Vec<Vec<f64>>>, keypoints: Option<Vec<f64>>, metadata: Option<serde_json::Value> }`
    - `BBox { x: f64, y: f64, width: f64, height: f64 }` (pixel coords for internal representation)
    - `Category { id: i64, dataset_id: i64, name: String, supercategory: Option<String> }`
    - `Embedding { id: i64, image_id: i64, model_name: String, vector: Vec<f32>, metadata: Option<serde_json::Value> }`
    - `Prediction { id: i64, image_id: i64, model_version: String, result: serde_json::Value, score: Option<f64> }`
    - `Patch { id: i64, image_id: i64, bbox: BBox, file_path: Option<PathBuf>, metadata: Option<serde_json::Value> }`
    - `VirtualDataset { id: i64, name: String, source_datasets: Vec<i64>, definition: VirtualDatasetDef }`
    - `VirtualDatasetDef` enum: `Filter { column: String, op: FilterOp, value: serde_json::Value }`, `Merge { datasets: Vec<i64> }`, `Sample { ratio: f64 }`, `Split { ratios: HashMap<String, f64> }`, `SchemaTransform { transforms: Vec<SchemaOp> }`, `Chain(Vec<VirtualDatasetDef>)`
    - `SchemaOp` enum: `RenameColumn { from: String, to: String }`, `AddColumn { name: String, dtype: String, default: serde_json::Value }`, `RemoveColumn(String)`, `CastColumn { name: String, dtype: String }`
    - `FilterOp` enum: `Eq`, `Ne`, `Gt`, `Lt`, `Gte`, `Lte`, `Contains`, `In`
  - All types derive `Debug, Clone, Serialize, Deserialize`
  - `BBox` also derives `PartialEq` for test assertions
  - RED tests: construct each type, serialize to JSON, deserialize back, assert equality
  - Export all types from `crates/core/src/lib.rs`

  **Must NOT do**:
  - No database operations — types only
  - No business logic — these are data containers
  - No trait impls beyond derives (no custom Display, no conversion methods)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure type definitions with derive macros, straightforward
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T3, T4 after T1)
  - **Parallel Group**: Wave 0
  - **Blocks**: T6, T7, T8, T12-T18, T24, T25, T31
  - **Blocked By**: T1

  **References**:

  **External References**:
  - COCO bbox format: `[x, y, width, height]` in pixels, top-left origin
  - YOLO bbox format: `[x_center, y_center, width, height]` normalized 0-1
  - serde derive: https://serde.rs/derive.html

  **WHY Each Reference Matters**:
  - COCO/YOLO bbox specs ensure internal `BBox` representation can convert to/from both formats
  - serde derive docs for `#[serde(rename_all)]`, `#[serde(skip_serializing_if)]` options

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- types::tests` passes
  - [ ] All types serialize/deserialize to/from JSON correctly
  - [ ] `VirtualDatasetDef::Chain` can nest other definitions
  - [ ] `SchemaOp` covers rename, add, remove, cast

  **QA Scenarios**:

  ```
  Scenario: Core types serialize/deserialize round-trip
    Tool: Bash
    Preconditions: types module exists
    Steps:
      1. Run `cargo test -p dman-core -- types::tests::test_serde_roundtrip 2>&1`
      2. Assert exit code = 0
    Expected Result: All round-trip tests pass
    Failure Indicators: Serialization mismatch, missing fields
    Evidence: .sisyphus/evidence/task-5-types-serde.txt

  Scenario: VirtualDatasetDef supports chained transforms
    Tool: Bash
    Preconditions: types module exists
    Steps:
      1. Run `cargo test -p dman-core -- types::tests::test_virtual_dataset_chain 2>&1`
      2. Assert exit code = 0
    Expected Result: Chain of Filter → SchemaTransform → Sample serializes correctly
    Failure Indicators: Nested enum serialization failure
    Evidence: .sisyphus/evidence/task-5-virtual-chain.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add core type definitions`
  - Files: `crates/core/src/types/mod.rs`, `crates/core/src/lib.rs`
  - Pre-commit: `cargo test -p dman-core -- types`

- [ ] 6. Dataset CRUD Operations

  **What to do**:
  - Create `crates/core/src/dataset/mod.rs` with `DatasetService` struct
  - Implement methods:
    - `register(db: &Database, name: &str, path: &Path, format: DatasetFormat) -> Result<Dataset>` — insert into `datasets` table, verify path exists
    - `list(db: &Database) -> Result<Vec<Dataset>>` — query all datasets
    - `get(db: &Database, name: &str) -> Result<Dataset>` — get by name, error if not found
    - `get_by_id(db: &Database, id: i64) -> Result<Dataset>`
    - `remove(db: &Database, name: &str) -> Result<()>` — delete dataset + cascade (images, annotations, etc.)
    - `update_metadata(db: &Database, name: &str, metadata: serde_json::Value) -> Result<()>`
    - `inspect(db: &Database, name: &str) -> Result<DatasetInfo>` — returns stats: image count, annotation count, categories, disk size
  - `DatasetInfo` struct: `{ dataset: Dataset, image_count: u64, annotation_count: u64, categories: Vec<Category>, disk_size_bytes: u64 }`
  - Idempotency: `register` checks if dataset name already exists, returns error if duplicate
  - Cascade delete: removing a dataset removes all images, annotations, embeddings, predictions, patches
  - RED tests: register → list → get → inspect → remove → list (empty). Test duplicate name error. Test remove cascade.

  **Must NOT do**:
  - No file operations (that's the storage manager T8)
  - No format-specific logic

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core business logic with database interactions, cascade deletes, transactional integrity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T7, T8, T9, T10)
  - **Parallel Group**: Wave 1
  - **Blocks**: T11, T17, T20, T22, T29, T33, T34
  - **Blocked By**: T1, T2, T3, T5

  **References**:

  **Pattern References**:
  - `crates/core/src/db/mod.rs` (T3) — Database struct, Connection access pattern
  - `crates/core/src/types/mod.rs` (T5) — Dataset, Category, DatasetInfo types
  - `crates/core/src/error.rs` (T2) — DmanError::DatasetNotFound for get failures

  **WHY Each Reference Matters**:
  - Database module defines how to execute queries — use `db.conn.prepare()` / `query_map()` patterns
  - Types define exact struct fields that map to SQL columns
  - Error types ensure get() returns typed DatasetNotFound, not generic error

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- dataset::tests` passes
  - [ ] CRUD cycle: register → list (1) → get → inspect (counts correct) → remove → list (0)
  - [ ] Duplicate name returns `DmanError::DatasetAlreadyExists` (or similar)
  - [ ] Cascade: after remove, `SELECT COUNT(*) FROM images WHERE dataset_id = ?` returns 0

  **QA Scenarios**:

  ```
  Scenario: Full CRUD lifecycle
    Tool: Bash
    Preconditions: Database module and types ready
    Steps:
      1. Run `cargo test -p dman-core -- dataset::tests::test_crud_lifecycle 2>&1`
      2. Assert exit code = 0
    Expected Result: register, list, get, inspect, remove all work in sequence
    Failure Indicators: Any step fails or returns wrong count
    Evidence: .sisyphus/evidence/task-6-crud-lifecycle.txt

  Scenario: Duplicate name rejected
    Tool: Bash
    Preconditions: Dataset registered
    Steps:
      1. Run `cargo test -p dman-core -- dataset::tests::test_duplicate_name 2>&1`
      2. Assert exit code = 0
    Expected Result: Second register with same name returns error
    Failure Indicators: No error on duplicate
    Evidence: .sisyphus/evidence/task-6-duplicate.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement dataset CRUD operations`
  - Files: `crates/core/src/dataset/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- dataset`

- [ ] 7. Schema System — TOML Parsing + Validation

  **What to do**:
  - Create `crates/core/src/schema/mod.rs` with schema types and parser
  - `SchemaDefinition { name: String, version: String, columns: Vec<ColumnDef>, annotation_format: Option<AnnotationFormat> }`
  - `ColumnDef { name: String, dtype: DataType, required: bool, default: Option<toml::Value>, description: Option<String> }`
  - `DataType` enum: `String`, `Int`, `Float`, `Bool`, `BBox`, `Polygon`, `Keypoints`, `EmbeddingVector(usize)`, `ImagePath`, `Json`, `List(Box<DataType>)`
  - `AnnotationFormat` enum: `BoundingBox`, `Polygon`, `Keypoints`, `Custom(String)`
  - Implement `Schema::from_toml(path: &Path) -> Result<SchemaDefinition>` — parse TOML file
  - Implement `Schema::validate_row(schema: &SchemaDefinition, row: &serde_json::Value) -> Result<Vec<ValidationError>>` — check types, required fields
  - Example TOML schema:
    ```toml
    [schema]
    name = "object-detection"
    version = "1.0"

    [[columns]]
    name = "image_path"
    dtype = "ImagePath"
    required = true

    [[columns]]
    name = "bbox"
    dtype = "BBox"
    required = true

    [[columns]]
    name = "label"
    dtype = "String"
    required = true

    [[columns]]
    name = "confidence"
    dtype = "Float"
    required = false
    default = 1.0
    ```
  - RED tests: parse valid TOML → SchemaDefinition. Parse invalid TOML → error. Validate conforming row → OK. Validate non-conforming row → list errors.

  **Must NOT do**:
  - No runtime enforcement on queries (validation is explicit, called on import)
  - No schema migration logic here (that's T18)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Type system design + TOML parsing + validation logic requires careful thought
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T8, T9, T10)
  - **Parallel Group**: Wave 1
  - **Blocks**: T12, T13, T14, T15, T18
  - **Blocked By**: T1, T5

  **References**:

  **Pattern References**:
  - `crates/core/tests/fixtures/schema/basic.toml` (T4) — test fixture to parse against

  **External References**:
  - `toml` crate: https://docs.rs/toml/latest/toml/
  - Serde TOML patterns: https://docs.rs/toml/latest/toml/#deserialization

  **WHY Each Reference Matters**:
  - TOML crate docs for `toml::from_str` and custom deserialize patterns — needed for enum variants like DataType
  - Schema fixture ensures parsing tests have a concrete, representative input file

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- schema::tests` passes
  - [ ] Valid TOML parses to SchemaDefinition with correct columns
  - [ ] Invalid TOML (missing required fields) returns parse error
  - [ ] Row validation catches type mismatches and missing required fields

  **QA Scenarios**:

  ```
  Scenario: Parse valid schema TOML
    Tool: Bash
    Preconditions: Schema module and test fixture exist
    Steps:
      1. Run `cargo test -p dman-core -- schema::tests::test_parse_valid 2>&1`
      2. Assert exit code = 0
    Expected Result: SchemaDefinition has expected columns and types
    Failure Indicators: Parse error on valid TOML
    Evidence: .sisyphus/evidence/task-7-parse-valid.txt

  Scenario: Validate row against schema catches errors
    Tool: Bash
    Preconditions: Schema module exists
    Steps:
      1. Run `cargo test -p dman-core -- schema::tests::test_validate_row 2>&1`
      2. Assert exit code = 0
    Expected Result: Conforming row passes, non-conforming row returns specific errors
    Failure Indicators: Valid row rejected or invalid row accepted
    Evidence: .sisyphus/evidence/task-7-validate-row.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement TOML schema system`
  - Files: `crates/core/src/schema/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- schema`

- [ ] 8. Image Storage Manager

  **What to do**:
  - Create `crates/core/src/storage/mod.rs` with `StorageManager` struct
  - Storage layout: images stored as files with paths tracked in SQLite
  - On import: copy/symlink images to `{dataset_root}/images/{original_filename}` (configurable: copy vs symlink)
  - Implement:
    - `StorageManager::new(base_path: PathBuf) -> Self`
    - `store_image(dataset_id: i64, source_path: &Path, strategy: StorageStrategy) -> Result<PathBuf>` — copy or symlink, return stored path
    - `get_image_path(dataset_id: i64, file_name: &str) -> PathBuf`
    - `delete_image(stored_path: &Path) -> Result<()>`
    - `delete_dataset_images(dataset_id: i64) -> Result<()>` — remove all images for dataset
    - `calculate_hash(path: &Path) -> Result<String>` — SHA256 for dedup detection
    - `check_integrity(db: &Database, dataset_id: i64) -> Result<IntegrityReport>` — verify all referenced images exist on disk
  - `StorageStrategy` enum: `Copy`, `Symlink`, `Reference` (don't move, just track path)
  - `IntegrityReport { total: u64, ok: u64, missing: Vec<String>, corrupted: Vec<String> }`
  - RED tests: store image (copy) → verify file exists → get path → delete → verify gone. Hash consistency. Integrity check with missing file.

  **Must NOT do**:
  - No BLOBs in database
  - No image processing (resize, thumbnail) — that's transforms/patches

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: File system operations with integrity checks, hashing, multiple strategies
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T9, T10)
  - **Parallel Group**: Wave 1
  - **Blocks**: T12, T13, T14, T22, T25
  - **Blocked By**: T1, T2, T3, T5

  **References**:

  **External References**:
  - `sha2` crate for SHA256: https://docs.rs/sha2/latest/sha2/
  - `std::fs::copy`, `std::os::unix::fs::symlink` for storage strategies

  **WHY Each Reference Matters**:
  - SHA256 hashing for content-addressable dedup detection
  - Symlink support is OS-specific — Unix straightforward, Windows needs special handling

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- storage::tests` passes
  - [ ] Copy strategy: file copied, original untouched
  - [ ] Symlink strategy: symlink created pointing to original
  - [ ] Reference strategy: no file operation, path stored as-is
  - [ ] Integrity check detects missing files correctly

  **QA Scenarios**:

  ```
  Scenario: Store and retrieve image with copy strategy
    Tool: Bash
    Preconditions: Storage module exists, temp dir available
    Steps:
      1. Run `cargo test -p dman-core -- storage::tests::test_copy_strategy 2>&1`
      2. Assert exit code = 0
    Expected Result: Image copied to storage, retrievable by path
    Failure Indicators: File not found at stored path
    Evidence: .sisyphus/evidence/task-8-copy-strategy.txt

  Scenario: Integrity check detects missing files
    Tool: Bash
    Preconditions: Storage module exists
    Steps:
      1. Run `cargo test -p dman-core -- storage::tests::test_integrity_missing 2>&1`
      2. Assert exit code = 0
    Expected Result: IntegrityReport.missing contains the deleted file
    Failure Indicators: Missing file not detected
    Evidence: .sisyphus/evidence/task-8-integrity.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement file-based image storage manager`
  - Files: `crates/core/src/storage/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- storage`

- [ ] 9. Catalog Service (Global ~/.dman/ Management)

  **What to do**:
  - Create `crates/core/src/catalog/mod.rs` with `Catalog` struct
  - Manage the global `~/.dman/` directory:
    - `Catalog::init() -> Result<Catalog>` — create `~/.dman/` if not exists, open/create `catalog.db`, run migrations
    - `Catalog::open() -> Result<Catalog>` — open existing, error if not initialized
    - `Catalog::db(&self) -> &Database` — access underlying database
    - `Catalog::config_path() -> PathBuf` — `~/.dman/config.toml`
    - `Catalog::plugins_path() -> PathBuf` — `~/.dman/plugins/`
    - `Catalog::data_path() -> PathBuf` — `~/.dman/data/` (default image storage root)
  - Directory structure:
    ```
    ~/.dman/
    ├── catalog.db          # SQLite database
    ├── config.toml         # Global configuration
    ├── plugins/            # Python plugins directory
    └── data/               # Default image storage
        └── {dataset_id}/   # Per-dataset image storage
    ```
  - Handle `$DMAN_HOME` env var override (default `~/.dman/`)
  - RED tests: use `tempfile` for `$DMAN_HOME` override → init → verify dirs created → open → verify db accessible

  **Must NOT do**:
  - No dataset operations (that's T6)
  - No config parsing (that's T10)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: File system management, env vars, directory creation, error handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T8, T10)
  - **Parallel Group**: Wave 1
  - **Blocks**: T6, T10, T11
  - **Blocked By**: T1, T2, T3

  **References**:

  **Pattern References**:
  - `crates/core/src/db/mod.rs` (T3) — Database::open() pattern
  - `crates/core/src/error.rs` (T2) — DmanError for catalog failures

  **External References**:
  - `dirs` crate for home directory: https://docs.rs/dirs/latest/dirs/ — `dirs::home_dir()`

  **WHY Each Reference Matters**:
  - Database module for opening/creating the catalog.db file
  - `dirs` crate for cross-platform home directory resolution

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- catalog::tests` passes
  - [ ] `Catalog::init()` creates `~/.dman/`, `catalog.db`, `plugins/`, `data/` directories
  - [ ] `Catalog::open()` on non-existent dir returns error
  - [ ] `$DMAN_HOME` override works

  **QA Scenarios**:

  ```
  Scenario: Catalog initialization creates directory structure
    Tool: Bash
    Preconditions: tempfile available for DMAN_HOME override
    Steps:
      1. Run `cargo test -p dman-core -- catalog::tests::test_init 2>&1`
      2. Assert exit code = 0
    Expected Result: All expected directories and catalog.db created
    Failure Indicators: Missing directories or database
    Evidence: .sisyphus/evidence/task-9-init.txt

  Scenario: DMAN_HOME env var override
    Tool: Bash
    Preconditions: Catalog module exists
    Steps:
      1. Run `cargo test -p dman-core -- catalog::tests::test_env_override 2>&1`
      2. Assert exit code = 0
    Expected Result: Catalog created at custom path, not ~/.dman/
    Failure Indicators: Files created at default path despite override
    Evidence: .sisyphus/evidence/task-9-env-override.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement global catalog service`
  - Files: `crates/core/src/catalog/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- catalog`

- [ ] 10. Configuration System

  **What to do**:
  - Create `crates/core/src/config/mod.rs` with `DmanConfig` struct
  - Parse `~/.dman/config.toml`:
    ```toml
    [storage]
    strategy = "copy"  # copy | symlink | reference
    base_path = "~/.dman/data"

    [server]
    host = "127.0.0.1"
    port = 8080

    [python]
    interpreter = "python3"
    plugin_paths = ["~/.dman/plugins"]

    [ui]
    default_page_size = 50
    thumbnail_size = 128
    ```
  - `DmanConfig` with nested structs: `StorageConfig`, `ServerConfig`, `PythonConfig`, `UiConfig`
  - All fields have sensible defaults via `impl Default`
  - `DmanConfig::load(path: &Path) -> Result<DmanConfig>` — parse file, fall back to defaults for missing fields
  - `DmanConfig::save(path: &Path) -> Result<()>` — write current config
  - RED tests: load default config → all fields have defaults. Load partial config → missing fields use defaults. Load full config → all fields populated.

  **Must NOT do**:
  - No env var merging (keep it simple — file only)
  - No runtime config changes (restart to pick up changes)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward TOML config parsing with defaults
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T8, T9)
  - **Parallel Group**: Wave 1
  - **Blocks**: T11
  - **Blocked By**: T1, T9

  **References**:

  **External References**:
  - `toml` crate for parsing config: https://docs.rs/toml/latest/toml/
  - serde `#[serde(default)]` for default values

  **WHY Each Reference Matters**:
  - TOML parsing with `#[serde(default)]` allows partial config files with fallback defaults

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- config::tests` passes
  - [ ] Default config has sensible values for all fields
  - [ ] Partial TOML file merges with defaults correctly
  - [ ] Config round-trips: load → save → load → same values

  **QA Scenarios**:

  ```
  Scenario: Default config has all fields set
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- config::tests::test_defaults 2>&1`
      2. Assert exit code = 0
    Expected Result: DmanConfig::default() has non-empty values for all fields
    Evidence: .sisyphus/evidence/task-10-defaults.txt

  Scenario: Partial config merges with defaults
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- config::tests::test_partial 2>&1`
      2. Assert exit code = 0
    Expected Result: Specified fields override defaults, unspecified fields use defaults
    Evidence: .sisyphus/evidence/task-10-partial.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add configuration system`
  - Files: `crates/core/src/config/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- config`

- [ ] 11. CLI Shell — Clap Subcommands

  **What to do**:
  - Create `crates/cli/src/main.rs` with clap derive-based CLI structure
  - Define subcommands:
    - `dman init` — initialize catalog (`Catalog::init()`)
    - `dman add <name> <path> [--format yolo|coco|hf|custom]` — register dataset
    - `dman list [--format table|json]` — list all datasets
    - `dman inspect <name>` — show dataset details (image count, categories, schema, disk size)
    - `dman remove <name> [--yes]` — remove dataset (confirm prompt unless --yes)
    - `dman import <path> [--format yolo|coco|hf] [--name <name>] [--schema <path>]` — import dataset
    - `dman export <name> <output-path> --format <format>` — export dataset
    - `dman operate <subcommand>` — operate commands (stubbed, wired in T20/T21)
    - `dman virtual <subcommand>` — virtual dataset commands (stubbed, wired in T17-T19)
    - `dman serve [--port 8080]` — start web server (stubbed, wired in T22)
    - `dman tui` — launch TUI mode (stubbed, wired in T29)
    - `dman materialize <virtual-dataset-name> [--output <path>]` — materialize virtual dataset (stubbed)
  - Wire `init`, `add`, `list`, `inspect`, `remove` to core library immediately
  - Stubbed commands print "Not yet implemented" with exit code 0
  - Use `colored` or `console` crate for pretty output (tables for `list`, structured for `inspect`)
  - RED tests via `assert_cmd`: `dman --help` exits 0, `dman list` on initialized catalog returns table

  **Must NOT do**:
  - No import/export logic (that's T12-T14)
  - No virtual dataset logic (that's T17-T19)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Many subcommands to wire, output formatting, integration with core library
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T12-T16)
  - **Parallel Group**: Wave 2
  - **Blocks**: T37
  - **Blocked By**: T6, T9, T10

  **References**:

  **Pattern References**:
  - `crates/core/src/catalog/mod.rs` (T9) — Catalog::init(), Catalog::open()
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService CRUD methods
  - `crates/core/src/config/mod.rs` (T10) — DmanConfig for server port, etc.

  **External References**:
  - `clap` derive API: https://docs.rs/clap/latest/clap/_derive/index.html
  - `comfy-table` or `tabled` for terminal tables: https://docs.rs/tabled/latest/tabled/

  **WHY Each Reference Matters**:
  - Catalog/Dataset modules define the API surface the CLI calls
  - clap derive docs for subcommand nesting, argument types, validation

  **Acceptance Criteria**:

  - [ ] `cargo build -p dman-cli` compiles
  - [ ] `dman --help` exits 0, shows all subcommands
  - [ ] `dman init` creates `~/.dman/` (or `$DMAN_HOME`)
  - [ ] `dman add test-ds /path --format yolo && dman list` shows the dataset
  - [ ] `dman inspect test-ds` shows structured output

  **QA Scenarios**:

  ```
  Scenario: CLI help shows all subcommands
    Tool: Bash
    Preconditions: dman-cli compiled
    Steps:
      1. Run `cargo run -p dman-cli -- --help 2>&1`
      2. Assert exit code = 0
      3. Assert output contains "init", "add", "list", "inspect", "remove", "import", "export", "serve", "tui"
    Expected Result: All subcommands listed in help
    Failure Indicators: Missing subcommands
    Evidence: .sisyphus/evidence/task-11-help.txt

  Scenario: Init + add + list workflow
    Tool: Bash
    Preconditions: Clean environment (no existing catalog)
    Steps:
      1. Set DMAN_HOME to temp dir
      2. Run `cargo run -p dman-cli -- init 2>&1` — assert exit 0
      3. Run `cargo run -p dman-cli -- add test-ds ./crates/core/tests/fixtures/yolo --format yolo 2>&1` — assert exit 0
      4. Run `cargo run -p dman-cli -- list 2>&1` — assert output contains "test-ds"
    Expected Result: Dataset registered and visible in list
    Failure Indicators: Init fails, add fails, list doesn't show dataset
    Evidence: .sisyphus/evidence/task-11-workflow.txt
  ```

  **Commit**: YES
  - Message: `feat(cli): implement clap subcommands shell`
  - Files: `crates/cli/src/main.rs`, `crates/cli/src/commands/`
  - Pre-commit: `cargo test -p dman-cli`

- [ ] 12. HuggingFace Parquet Importer/Exporter

  **What to do**:
  - Create `crates/core/src/formats/huggingface/mod.rs`
  - Implement `HuggingFaceImporter`:
    - Parse directory structure: detect README.md, find .parquet files, infer splits from filenames/dirs
    - Read Parquet via `arrow`/`parquet` crate: `ArrowReaderBuilder` → `RecordBatch`
    - Map Parquet schema to dman schema: infer column types, handle image columns (path-based and bytes-based)
    - Insert images + annotations into database within a transaction
    - Handle split detection: filename patterns (train-*.parquet), directory names (train/, test/)
  - Implement `HuggingFaceExporter`:
    - Query images + annotations from database
    - Build Arrow RecordBatches with correct schema
    - Write Parquet files per split
    - Generate README.md with YAML config header
  - Transaction-based import: if any row fails, rollback entire import
  - Idempotency: check if dataset already imported (by name), error on duplicate
  - RED tests: import fixture Parquet → verify DB rows → export → verify output files → re-import exported → compare

  **Must NOT do**:
  - No downloading from HuggingFace Hub (out of scope for V1)
  - No image-as-bytes column handling (defer to V2, only path-based)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Parquet/Arrow API complexity, schema inference, transactional imports
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T13, T14, T15, T16)
  - **Parallel Group**: Wave 2
  - **Blocks**: T33, T38
  - **Blocked By**: T7, T8, T15

  **References**:

  **Pattern References**:
  - `crates/core/tests/fixtures/huggingface/` (T4) — test fixture Parquet file
  - `crates/core/src/formats/mod.rs` (T15) — FormatImporter/FormatExporter trait

  **External References**:
  - `parquet` crate: https://docs.rs/parquet/latest/parquet/ — `ArrowReaderBuilder`, `ArrowWriter`
  - `arrow` crate: https://docs.rs/arrow/latest/arrow/ — `RecordBatch`, `Schema`, `DataType`
  - HuggingFace dataset structure (from research): README.md with YAML config, split inference rules

  **WHY Each Reference Matters**:
  - Parquet/Arrow crate APIs are complex — ArrowReaderBuilder for reading, ArrowWriter for writing
  - HF split inference rules determine how files map to train/val/test

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- formats::huggingface::tests` passes
  - [ ] Import fixture → DB has correct image count and column values
  - [ ] Export produces valid .parquet file + README.md
  - [ ] Round-trip: import → export → re-import → same data
  - [ ] Transaction rollback on malformed data

  **QA Scenarios**:

  ```
  Scenario: Import HuggingFace Parquet fixture
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::huggingface::tests::test_import 2>&1`
      2. Assert exit code = 0
    Expected Result: Fixture imported, DB contains 3 images with correct metadata
    Evidence: .sisyphus/evidence/task-12-hf-import.txt

  Scenario: Round-trip import → export → re-import
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::huggingface::tests::test_roundtrip 2>&1`
      2. Assert exit code = 0
    Expected Result: Data matches after full round-trip
    Evidence: .sisyphus/evidence/task-12-hf-roundtrip.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement HuggingFace Parquet importer/exporter`
  - Files: `crates/core/src/formats/huggingface/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- formats::huggingface`

- [ ] 13. YOLO Format Importer/Exporter

  **What to do**:
  - Create `crates/core/src/formats/yolo/mod.rs`
  - Implement `YoloImporter`:
    - Parse `data.yaml`: extract path, train/val/test dirs, class names
    - Walk `images/` directory, match with corresponding `labels/` .txt files
    - Parse label files: `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)
    - Convert normalized coords to pixel coords using image dimensions (read via `image` crate or store as normalized)
    - Handle: missing label files (image with no annotations), empty label files, extra label files (no matching image)
    - Insert into DB within transaction
  - Implement `YoloExporter`:
    - Query images + annotations from database
    - Convert pixel coords back to normalized YOLO format
    - Write `data.yaml`, `images/train/`, `labels/train/` structure
    - Copy/symlink images to export directory
  - RED tests: import YOLO fixture → verify → export → compare file-by-file with fixture

  **Must NOT do**:
  - No YOLOv8 segmentation/pose format (V1 = bounding boxes only)
  - No OBB (oriented bounding box) support

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Format parsing, coordinate conversion, file system operations, edge cases
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T12, T14, T15, T16)
  - **Parallel Group**: Wave 2
  - **Blocks**: T38
  - **Blocked By**: T7, T8, T15

  **References**:

  **Pattern References**:
  - `crates/core/tests/fixtures/yolo/` (T4) — test fixture with data.yaml + images + labels
  - `crates/core/src/formats/mod.rs` (T15) — FormatImporter/FormatExporter trait

  **External References**:
  - YOLO format spec (from research): data.yaml structure, normalized xywh, one .txt per image

  **WHY Each Reference Matters**:
  - YOLO coord format is normalized (0-1), dman internal BBox is pixels — conversion logic needed
  - data.yaml defines split directories and class names — parser must handle relative/absolute paths

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- formats::yolo::tests` passes
  - [ ] Import fixture → DB has 3 images with correct bbox annotations
  - [ ] Missing label file → image imported with 0 annotations (not error)
  - [ ] Export produces valid data.yaml + images/ + labels/ structure
  - [ ] Round-trip preserves annotation coordinates within floating-point tolerance

  **QA Scenarios**:

  ```
  Scenario: Import YOLO fixture
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::yolo::tests::test_import 2>&1`
      2. Assert exit code = 0
    Expected Result: 3 images, correct annotations with bbox coordinates
    Evidence: .sisyphus/evidence/task-13-yolo-import.txt

  Scenario: Handle missing label files gracefully
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::yolo::tests::test_missing_labels 2>&1`
      2. Assert exit code = 0
    Expected Result: Image imported with 0 annotations, no error
    Evidence: .sisyphus/evidence/task-13-missing-labels.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement YOLO format importer/exporter`
  - Files: `crates/core/src/formats/yolo/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- formats::yolo`

- [ ] 14. COCO Format Importer/Exporter

  **What to do**:
  - Create `crates/core/src/formats/coco/mod.rs`
  - Implement `CocoImporter`:
    - Parse COCO JSON: `images`, `annotations`, `categories` arrays
    - Map COCO image entries to dman Image records
    - Map COCO annotations (bbox as `[x, y, w, h]` pixels, segmentation polygons, keypoints) to dman Annotation records
    - Map COCO categories to dman Category records
    - Handle: images without annotations, annotations without images (skip/warn), multiple annotations per image
    - Transaction-based insert
  - Implement `CocoExporter`:
    - Build COCO JSON structure from database records
    - Include `info`, `licenses` (with defaults), `categories`, `images`, `annotations`
    - Calculate `area` from bbox for annotations
    - Write single JSON file per split
  - RED tests: import COCO fixture → verify → export → parse exported JSON → compare structures

  **Must NOT do**:
  - No RLE segmentation decoding (polygon only)
  - No caption annotations (object detection only in V1)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Complex JSON parsing with cross-references (image_id → annotation → category), area calculation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T12, T13, T15, T16)
  - **Parallel Group**: Wave 2
  - **Blocks**: T38
  - **Blocked By**: T7, T8, T15

  **References**:

  **Pattern References**:
  - `crates/core/tests/fixtures/coco/annotations.json` (T4) — test fixture
  - `crates/core/src/formats/mod.rs` (T15) — FormatImporter/FormatExporter trait

  **External References**:
  - COCO format spec (from research): JSON with images/annotations/categories, bbox = [x, y, w, h] pixels

  **WHY Each Reference Matters**:
  - COCO uses cross-referenced IDs (image_id in annotations → id in images) — parser must resolve references
  - Fixture file is the concrete input for import tests

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- formats::coco::tests` passes
  - [ ] Import fixture → correct image count, annotation count, category count
  - [ ] Annotations have correct bbox in pixel coordinates
  - [ ] Export produces valid COCO JSON (parseable, correct structure)
  - [ ] Round-trip preserves all annotation data

  **QA Scenarios**:

  ```
  Scenario: Import COCO fixture
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::coco::tests::test_import 2>&1`
      2. Assert exit code = 0
    Expected Result: 3 images, 5 annotations, 2 categories in DB
    Evidence: .sisyphus/evidence/task-14-coco-import.txt

  Scenario: Exported COCO JSON is valid
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::coco::tests::test_export_valid_json 2>&1`
      2. Assert exit code = 0
    Expected Result: Output JSON has all required top-level keys, correct cross-references
    Evidence: .sisyphus/evidence/task-14-coco-export.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement COCO format importer/exporter`
  - Files: `crates/core/src/formats/coco/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- formats::coco`

- [ ] 15. Generic Format Trait + Plugin Interface

  **What to do**:
  - Create `crates/core/src/formats/mod.rs` with trait definitions:
    ```rust
    pub trait FormatImporter: Send + Sync {
        fn name(&self) -> &str;
        fn detect(path: &Path) -> bool; // Auto-detect format from directory structure
        fn import(&self, db: &Database, storage: &StorageManager, path: &Path, dataset_name: &str) -> Result<Dataset>;
    }

    pub trait FormatExporter: Send + Sync {
        fn name(&self) -> &str;
        fn export(&self, db: &Database, storage: &StorageManager, dataset: &Dataset, output_path: &Path) -> Result<()>;
    }
    ```
  - Create `FormatRegistry`:
    - `register_importer(importer: Box<dyn FormatImporter>)`
    - `register_exporter(exporter: Box<dyn FormatExporter>)`
    - `detect_format(path: &Path) -> Option<&str>` — try each importer's detect()
    - `get_importer(name: &str) -> Option<&dyn FormatImporter>`
    - `get_exporter(name: &str) -> Option<&dyn FormatExporter>`
  - Register built-in formats: YOLO, COCO, HuggingFace
  - This trait is what Python plugins will implement (via T32)
  - `detect` logic: YOLO = has data.yaml, COCO = has annotations/*.json, HF = has *.parquet
  - RED tests: register mock importer → detect → retrieve by name

  **Must NOT do**:
  - No Python plugin loading (that's T32)
  - No actual import/export logic (that's T12-T14)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Trait design, registry pattern, auto-detection heuristics
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T16)
  - **Parallel Group**: Wave 2 (but should be started early as T12-T14 depend on it)
  - **Blocks**: T12, T13, T14, T32
  - **Blocked By**: T5, T7

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — DatasetFormat enum maps to importer names
  - `crates/core/src/schema/mod.rs` (T7) — SchemaDefinition used during import validation

  **External References**:
  - Rust trait object patterns: https://doc.rust-lang.org/book/ch17-02-trait-objects.html

  **WHY Each Reference Matters**:
  - Trait object pattern (`Box<dyn FormatImporter>`) enables runtime polymorphism for plugin system
  - DatasetFormat enum links CLI --format flag to registry lookup

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- formats::tests` passes
  - [ ] FormatRegistry stores and retrieves importers/exporters by name
  - [ ] Auto-detect: YOLO dir → "yolo", COCO dir → "coco", HF dir → "huggingface"
  - [ ] Unknown format → None

  **QA Scenarios**:

  ```
  Scenario: Format auto-detection works
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::tests::test_auto_detect 2>&1`
      2. Assert exit code = 0
    Expected Result: Each fixture dir detected as correct format
    Evidence: .sisyphus/evidence/task-15-autodetect.txt

  Scenario: Registry lookup by name
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- formats::tests::test_registry 2>&1`
      2. Assert exit code = 0
    Expected Result: Registered importers retrievable by name, unknown returns None
    Evidence: .sisyphus/evidence/task-15-registry.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add generic format trait and plugin interface`
  - Files: `crates/core/src/formats/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- formats`

- [ ] 16. Embeddings Storage + Metadata Columns

  **What to do**:
  - Create `crates/core/src/embeddings/mod.rs` with `EmbeddingService`
  - Implement:
    - `store(db: &Database, image_id: i64, model_name: &str, vector: &[f32]) -> Result<i64>` — store as BLOB (f32 bytes)
    - `get(db: &Database, image_id: i64, model_name: &str) -> Result<Option<Embedding>>`
    - `list_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Embedding>>`
    - `delete_by_image(db: &Database, image_id: i64) -> Result<()>`
    - `has_embeddings(db: &Database, dataset_id: i64) -> Result<bool>`
    - `get_embedding_models(db: &Database, dataset_id: i64) -> Result<Vec<String>>` — list distinct model names
  - Store vectors as BLOB (raw f32 bytes) — efficient storage, no JSON overhead
  - Decode: read BLOB → cast to `&[f32]` slice
  - RED tests: store embedding → retrieve → compare vectors. List by dataset. Delete.

  **Must NOT do**:
  - No similarity search / nearest neighbor (V2)
  - No vector indexing (no HNSW, no usearch)
  - No computation (that's T35)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: BLOB encoding/decoding, f32 byte handling, database operations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11-T15)
  - **Parallel Group**: Wave 2
  - **Blocks**: T35
  - **Blocked By**: T3, T5

  **References**:

  **Pattern References**:
  - `crates/core/src/db/mod.rs` (T3) — Database access pattern, BLOB handling with rusqlite
  - `crates/core/src/types/mod.rs` (T5) — Embedding type definition

  **External References**:
  - rusqlite BLOB handling: https://docs.rs/rusqlite/latest/rusqlite/blob/index.html

  **WHY Each Reference Matters**:
  - rusqlite BLOB API for efficient binary storage/retrieval of float vectors
  - Embedding type defines the struct that maps to/from DB rows

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- embeddings::tests` passes
  - [ ] Store 128-dim f32 vector → retrieve → exact match
  - [ ] List by dataset returns only embeddings for that dataset
  - [ ] Delete cascades correctly

  **QA Scenarios**:

  ```
  Scenario: Embedding storage round-trip
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- embeddings::tests::test_store_retrieve 2>&1`
      2. Assert exit code = 0
    Expected Result: Stored vector matches retrieved vector exactly
    Evidence: .sisyphus/evidence/task-16-embedding-roundtrip.txt

  Scenario: List embeddings by dataset
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- embeddings::tests::test_list_by_dataset 2>&1`
      2. Assert exit code = 0
    Expected Result: Only embeddings for target dataset returned
    Evidence: .sisyphus/evidence/task-16-list-by-dataset.txt
  ```

  **Commit**: YES
  - Message: `feat(core): add embeddings storage and metadata columns`
  - Files: `crates/core/src/embeddings/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- embeddings`

- [ ] 17. Virtual Dataset Engine — Filter, Merge, Sample, Split

  **What to do**:
  - Create `crates/core/src/virtual_dataset/mod.rs` with `VirtualDatasetService`
  - Implement:
    - `create(db: &Database, name: &str, source_datasets: Vec<i64>, definition: VirtualDatasetDef) -> Result<VirtualDataset>` — store definition in DB
    - `list(db: &Database) -> Result<Vec<VirtualDataset>>`
    - `get(db: &Database, name: &str) -> Result<VirtualDataset>`
    - `delete(db: &Database, name: &str) -> Result<()>`
    - `evaluate(db: &Database, vds: &VirtualDataset) -> Result<Vec<Image>>` — lazily resolve: apply filters, merges, samples to produce image list WITHOUT copying data
    - `preview(db: &Database, vds: &VirtualDataset, limit: usize) -> Result<Vec<Image>>` — evaluate first N images
  - Filter evaluation: translate `FilterOp` to SQL WHERE clauses on images/annotations tables
  - Merge evaluation: UNION SELECT across source datasets
  - Sample evaluation: random sampling with seed for reproducibility (`ORDER BY RANDOM()` with seed via hash)
  - Split evaluation: deterministic split by hash(image_id) % bucket
  - Chain evaluation: apply definitions in sequence, each operating on the output of the previous
  - Detect circular references: before creating, verify no cycles in source_datasets → virtual_dataset → source chain
  - RED tests: create virtual dataset → evaluate → verify image list matches expected. Chain two transforms. Circular ref detection.

  **Must NOT do**:
  - No materialization (that's T19)
  - No schema transforms (that's T18)
  - No file operations — virtual datasets operate on DB records only

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Query generation, lazy evaluation, chain composition, cycle detection — core algorithmic complexity
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T18, T20, T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: T19, T39
  - **Blocked By**: T6, T7

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — VirtualDatasetDef, FilterOp, VirtualDataset types
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService for querying source datasets
  - `crates/core/src/db/mod.rs` (T3) — SQL query building patterns

  **WHY Each Reference Matters**:
  - VirtualDatasetDef enum defines all possible operations — evaluate() must handle each variant
  - DatasetService provides the image listing API that virtual datasets compose over
  - Database patterns for building dynamic SQL WHERE clauses from FilterOp

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- virtual_dataset::tests` passes
  - [ ] Filter: create filter on class=X → evaluate → only images with class X returned
  - [ ] Merge: merge two datasets → evaluate → images from both datasets present
  - [ ] Sample: sample 50% → evaluate → approximately half the images
  - [ ] Chain: filter → sample → evaluate → correct subset
  - [ ] Circular reference detected and rejected with error

  **QA Scenarios**:

  ```
  Scenario: Filter virtual dataset
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::tests::test_filter 2>&1`
      2. Assert exit code = 0
    Expected Result: Only images matching filter criteria returned
    Evidence: .sisyphus/evidence/task-17-filter.txt

  Scenario: Circular reference detection
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::tests::test_circular_ref 2>&1`
      2. Assert exit code = 0
    Expected Result: Error returned when circular dependency detected
    Evidence: .sisyphus/evidence/task-17-circular.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement virtual dataset engine`
  - Files: `crates/core/src/virtual_dataset/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- virtual_dataset`

- [ ] 18. Schema Evolution Transforms

  **What to do**:
  - Create `crates/core/src/virtual_dataset/transforms.rs`
  - Implement schema transforms as VirtualDatasetDef::SchemaTransform operations:
    - `RenameColumn { from, to }` — rename a column in the virtual view
    - `AddColumn { name, dtype, default }` — add new column with default value
    - `RemoveColumn(name)` — exclude column from virtual view
    - `CastColumn { name, dtype }` — convert column type (e.g., String → Int)
  - Schema transforms produce a new SchemaDefinition that reflects the changes
  - Transforms operate on the metadata/annotations — they modify how data is presented, not the underlying storage
  - Composable: multiple SchemaOps can be chained in a single SchemaTransform
  - Validation: verify column exists before rename/remove/cast, verify new column name doesn't conflict
  - This IS the migration mechanism: user creates a virtual dataset with schema transforms, then materializes it (T19) to apply the migration
  - RED tests: rename column → evaluate → new column name present. Add column → evaluate → default populated. Remove → evaluate → column absent. Invalid transform → error.

  **Must NOT do**:
  - No in-place schema modification — always through virtual dataset layer
  - No type coercion logic (cast just changes declared type, actual conversion on materialize)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Schema manipulation logic, validation, composability
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T17, T20, T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: T19
  - **Blocked By**: T7, T17

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — SchemaOp enum variants
  - `crates/core/src/schema/mod.rs` (T7) — SchemaDefinition, ColumnDef, DataType
  - `crates/core/src/virtual_dataset/mod.rs` (T17) — VirtualDatasetService.evaluate()

  **WHY Each Reference Matters**:
  - SchemaOp variants define the exact transform operations to implement
  - SchemaDefinition is the input/output — transforms modify it to produce new schema
  - evaluate() integration — schema transforms must work within the virtual dataset chain

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- virtual_dataset::transforms::tests` passes
  - [ ] RenameColumn: old name absent, new name present in evaluated schema
  - [ ] AddColumn: new column present with default value
  - [ ] RemoveColumn: column absent from evaluated schema
  - [ ] Invalid column name → descriptive error

  **QA Scenarios**:

  ```
  Scenario: Schema migration via virtual dataset
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::transforms::tests::test_migration_flow 2>&1`
      2. Assert exit code = 0
    Expected Result: Schema transform chain produces correct new schema
    Evidence: .sisyphus/evidence/task-18-migration.txt

  Scenario: Invalid transform rejected
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::transforms::tests::test_invalid 2>&1`
      2. Assert exit code = 0
    Expected Result: Rename non-existent column returns error
    Evidence: .sisyphus/evidence/task-18-invalid.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement schema evolution transforms`
  - Files: `crates/core/src/virtual_dataset/transforms.rs`
  - Pre-commit: `cargo test -p dman-core -- virtual_dataset::transforms`

- [ ] 19. Materialize Command

  **What to do**:
  - Create `crates/core/src/virtual_dataset/materialize.rs`
  - Implement `materialize(db: &Database, storage: &StorageManager, vds: &VirtualDataset, output_name: &str) -> Result<Dataset>`:
    - Evaluate virtual dataset → get full image list
    - Create new physical dataset in catalog with output_name
    - Copy/link images from source datasets to new dataset storage
    - Apply schema transforms: rename columns in annotations, add defaults, remove columns
    - Apply data transforms: if virtual dataset includes filter/sample/merge, only include matching rows
    - Insert all resolved records into new dataset's DB entries
    - Transaction: if any step fails, rollback new dataset creation
  - Wire into CLI: `dman materialize <virtual-dataset-name> [--output-name <name>]`
  - Support optional `--output-path <dir>` to also export to a directory format
  - RED tests: create virtual dataset (filter + schema transform) → materialize → verify new physical dataset exists with correct data

  **Must NOT do**:
  - No streaming/chunk-based materialization (V1 = load all into memory, process, write)
  - No incremental materialization (always full rebuild)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Combines virtual dataset evaluation + storage management + transaction handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on T17 and T18)
  - **Parallel Group**: Wave 3 (after T17, T18)
  - **Blocks**: T39
  - **Blocked By**: T17, T18

  **References**:

  **Pattern References**:
  - `crates/core/src/virtual_dataset/mod.rs` (T17) — evaluate() returns image list
  - `crates/core/src/virtual_dataset/transforms.rs` (T18) — schema transform application
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager for copying images
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService.register() for creating new dataset

  **WHY Each Reference Matters**:
  - evaluate() produces the list of images to include — materialize iterates over this
  - Schema transforms define how columns are renamed/added/removed during materialization
  - StorageManager handles the actual file copies from source to new dataset
  - DatasetService creates the new physical dataset entry

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- virtual_dataset::materialize::tests` passes
  - [ ] Materialized dataset appears in `dman list`
  - [ ] Materialized dataset has correct image count (matching virtual dataset evaluation)
  - [ ] Schema transforms applied: new column names, defaults populated
  - [ ] Transaction rollback on failure: partial dataset cleaned up

  **QA Scenarios**:

  ```
  Scenario: Materialize filtered virtual dataset
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::materialize::tests::test_materialize_filter 2>&1`
      2. Assert exit code = 0
    Expected Result: New physical dataset with only filtered images
    Evidence: .sisyphus/evidence/task-19-materialize.txt

  Scenario: Rollback on failure
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- virtual_dataset::materialize::tests::test_rollback 2>&1`
      2. Assert exit code = 0
    Expected Result: Failed materialization leaves no partial dataset
    Evidence: .sisyphus/evidence/task-19-rollback.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement materialize command`
  - Files: `crates/core/src/virtual_dataset/materialize.rs`
  - Pre-commit: `cargo test -p dman-core -- virtual_dataset::materialize`

- [ ] 20. Operate Commands — CRUD (rename, duplicate, merge, split)

  **What to do**:
  - Create `crates/core/src/ops/mod.rs` with dataset-level operations
  - Implement:
    - `rename(db: &Database, old_name: &str, new_name: &str) -> Result<()>` — update dataset name
    - `duplicate(db: &Database, storage: &StorageManager, name: &str, new_name: &str) -> Result<Dataset>` — copy dataset + images to new name
    - `merge(db: &Database, names: &[&str], output_name: &str) -> Result<Dataset>` — combine multiple datasets (union images + annotations)
    - `split(db: &Database, name: &str, ratios: HashMap<String, f64>, seed: u64) -> Result<Vec<Dataset>>` — deterministic split into train/val/test by ratio
  - Wire into CLI: `dman operate rename <old> <new>`, `dman operate duplicate <src> <dst>`, etc.
  - Merge handles schema conflicts: if datasets have different schemas, use union of columns (missing values = null)
  - Split uses deterministic hashing for reproducibility
  - RED tests: rename → verify. duplicate → two datasets with same data. merge → combined. split → ratios approximately correct.

  **Must NOT do**:
  - No data transforms (that's T21)
  - No virtual dataset interaction (these are physical operations)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple operations with DB + storage coordination, schema conflict handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T17-T19, T21)
  - **Parallel Group**: Wave 3
  - **Blocks**: T37
  - **Blocked By**: T6

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService for creating/querying datasets
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager for duplicating images

  **WHY Each Reference Matters**:
  - DatasetService provides the CRUD foundation that operate commands extend
  - StorageManager handles file duplication during dataset duplicate/merge

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- ops::tests` passes
  - [ ] Rename: old name gone, new name present
  - [ ] Duplicate: both datasets exist with same image count
  - [ ] Merge: output has union of images from all inputs
  - [ ] Split: ratios approximately correct (within 5% for small datasets)

  **QA Scenarios**:

  ```
  Scenario: Dataset operations lifecycle
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- ops::tests::test_rename 2>&1`
      2. Run `cargo test -p dman-core -- ops::tests::test_merge 2>&1`
      3. Assert both exit code = 0
    Expected Result: Rename and merge work correctly
    Evidence: .sisyphus/evidence/task-20-ops.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement dataset CRUD operate commands`
  - Files: `crates/core/src/ops/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- ops`

- [ ] 21. Operate Commands — Data Transforms (filter, sample, resize, relabel)

  **What to do**:
  - Create `crates/core/src/ops/transforms.rs` with data-level transform operations
  - These are PHYSICAL transforms (unlike virtual datasets which are lazy):
    - `filter_dataset(db: &Database, name: &str, predicate: FilterOp, value: &str, output_name: &str) -> Result<Dataset>` — create new dataset with only matching images
    - `sample_dataset(db: &Database, name: &str, ratio: f64, seed: u64, output_name: &str) -> Result<Dataset>` — random sample
    - `relabel(db: &Database, name: &str, mapping: HashMap<String, String>) -> Result<()>` — rename categories in-place
    - `resize_images(storage: &StorageManager, name: &str, width: u32, height: u32) -> Result<()>` — shell out to Python/ImageMagick for actual resize
  - resize_images: use `std::process::Command` to call `python3 -c "from PIL import Image; ..."` or `convert` (ImageMagick). Fail gracefully if neither available.
  - RED tests: filter → correct subset. sample → correct ratio. relabel → categories renamed. resize with mock.

  **Must NOT do**:
  - No native Rust image processing — shell out for resize
  - No augmentation (V2, via Python plugins)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple transform types, external process invocation, in-place mutation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T17-T20)
  - **Parallel Group**: Wave 3
  - **Blocks**: T37
  - **Blocked By**: T6, T8

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — FilterOp enum
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService for querying
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager for image paths

  **WHY Each Reference Matters**:
  - FilterOp defines the filtering language used by transform operations
  - DatasetService provides image iteration for filter/sample
  - StorageManager provides image paths for resize operations

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- ops::transforms::tests` passes
  - [ ] Filter: output dataset has only matching images
  - [ ] Sample: output dataset size ≈ ratio × input size (within tolerance)
  - [ ] Relabel: category names changed in-place

  **QA Scenarios**:

  ```
  Scenario: Filter and sample transforms
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- ops::transforms::tests::test_filter 2>&1`
      2. Run `cargo test -p dman-core -- ops::transforms::tests::test_sample 2>&1`
      3. Assert both exit code = 0
    Expected Result: Correct subsets produced
    Evidence: .sisyphus/evidence/task-21-transforms.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement data transform operations`
  - Files: `crates/core/src/ops/transforms.rs`
  - Pre-commit: `cargo test -p dman-core -- ops::transforms`

- [ ] 22. Axum HTTP Server — Image Serving + API Routes

  **What to do**:
  - Create `crates/server/src/main.rs` and `crates/server/src/lib.rs`
  - Set up axum HTTP server:
    - `GET /api/health` — health check
    - `GET /images/{dataset_id}/{filename}` — serve image file from storage (with correct Content-Type)
    - `GET /thumbnails/{dataset_id}/{filename}` — serve thumbnail (generate on-the-fly or cache)
    - Static file serving for SPA frontend (via rust-embed, placeholder for now)
    - SPA fallback: non-API routes → index.html (for React Router)
    - CORS headers for local development
  - Server startup: `serve(config: &ServerConfig, catalog: &Catalog) -> Result<()>`
  - Image serving reads from StorageManager paths, sets Content-Type based on extension
  - Wire into CLI: `dman serve [--port 8080]` starts the server and opens browser via `open` crate
  - RED tests: start server in test → request health → 200. Request image → 200 with correct content-type. Request missing image → 404.

  **Must NOT do**:
  - No actual SPA content yet (that's T26)
  - No full API endpoints yet (that's T23)
  - No authentication (local tool, single user)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: HTTP server setup, routing, static file serving, content-type handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T23, T24, T25)
  - **Parallel Group**: Wave 4
  - **Blocks**: T23, T26, T36
  - **Blocked By**: T8, T3

  **References**:

  **Pattern References**:
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager.get_image_path() for file lookup
  - `crates/core/src/config/mod.rs` (T10) — ServerConfig for host/port

  **External References**:
  - axum docs: https://docs.rs/axum/latest/axum/
  - tower-http ServeDir: https://docs.rs/tower-http/latest/tower_http/services/struct.ServeDir.html
  - rust-embed: https://docs.rs/rust-embed/latest/rust_embed/

  **WHY Each Reference Matters**:
  - axum Router pattern for organizing routes + middleware (CORS, logging)
  - tower-http ServeDir for static SPA files
  - StorageManager provides the file paths that the image endpoint serves

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-server` passes
  - [ ] `GET /api/health` returns 200 with `{"status": "ok"}`
  - [ ] `GET /images/{id}/{file}` returns image with correct Content-Type
  - [ ] Missing image returns 404

  **QA Scenarios**:

  ```
  Scenario: Server starts and serves health endpoint
    Tool: Bash
    Steps:
      1. Start server in background: `cargo run -p dman-server -- --port 8099 &`
      2. Wait 2s for startup
      3. Run `curl -s http://localhost:8099/api/health`
      4. Assert response contains "ok"
      5. Kill background server
    Expected Result: Health endpoint returns 200
    Evidence: .sisyphus/evidence/task-22-health.txt

  Scenario: Missing image returns 404
    Tool: Bash
    Steps:
      1. Start server, run `curl -s -o /dev/null -w "%{http_code}" http://localhost:8099/images/999/nonexistent.jpg`
      2. Assert response code = 404
    Expected Result: 404 for non-existent image
    Evidence: .sisyphus/evidence/task-22-404.txt
  ```

  **Commit**: YES
  - Message: `feat(server): implement axum HTTP server with image serving`
  - Files: `crates/server/src/main.rs`, `crates/server/src/lib.rs`, `crates/server/src/routes/`
  - Pre-commit: `cargo test -p dman-server`

- [ ] 23. REST API Endpoints

  **What to do**:
  - Add REST API routes to the axum server:
    - `GET /api/datasets` — list all datasets (paginated: `?page=1&per_page=50`)
    - `GET /api/datasets/{name}` — get dataset details (same as inspect)
    - `GET /api/datasets/{name}/images` — list images (paginated, filterable: `?category=X&page=1`)
    - `GET /api/datasets/{name}/images/{id}` — get single image metadata + annotations
    - `GET /api/datasets/{name}/categories` — list categories
    - `GET /api/datasets/{name}/schema` — get schema definition
    - `GET /api/datasets/{name}/stats` — get statistics (counts, distribution)
    - `GET /api/virtual-datasets` — list virtual datasets
    - `GET /api/virtual-datasets/{name}` — get virtual dataset definition
  - All responses as JSON with consistent envelope: `{ "data": ..., "pagination": { "page": 1, "per_page": 50, "total": 1000 } }`
  - Error responses: `{ "error": { "code": "NOT_FOUND", "message": "..." } }` with appropriate HTTP status
  - RED tests: register dataset → request /api/datasets → verify JSON structure. Request /api/datasets/nonexistent → 404.

  **Must NOT do**:
  - No mutation endpoints (read-only API in V1)
  - No authentication

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple endpoints, pagination, JSON serialization, error handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T24, T25)
  - **Parallel Group**: Wave 4
  - **Blocks**: T26, T27, T28, T36
  - **Blocked By**: T6, T22

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService methods the API wraps
  - `crates/server/src/routes/` (T22) — existing route structure

  **External References**:
  - axum extractors: https://docs.rs/axum/latest/axum/extract/index.html — Path, Query, State

  **WHY Each Reference Matters**:
  - DatasetService is the backend the API calls — each endpoint maps to one or more service methods
  - axum extractors for parsing path params, query params, and shared state

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-server -- api::tests` passes
  - [ ] `GET /api/datasets` returns paginated JSON list
  - [ ] `GET /api/datasets/{name}/images?category=X` filters correctly
  - [ ] 404 for non-existent dataset
  - [ ] Pagination: `page=2&per_page=10` returns correct slice

  **QA Scenarios**:

  ```
  Scenario: List datasets API
    Tool: Bash
    Steps:
      1. Start server with test data
      2. Run `curl -s http://localhost:8099/api/datasets | python3 -m json.tool`
      3. Assert JSON has "data" array and "pagination" object
    Expected Result: Valid paginated JSON response
    Evidence: .sisyphus/evidence/task-23-list-api.txt

  Scenario: Filter images by category
    Tool: Bash
    Steps:
      1. Run `curl -s "http://localhost:8099/api/datasets/test-ds/images?category=person"`
      2. Assert all returned images have annotations with category "person"
    Expected Result: Only matching images returned
    Evidence: .sisyphus/evidence/task-23-filter.txt
  ```

  **Commit**: YES
  - Message: `feat(server): implement REST API endpoints`
  - Files: `crates/server/src/api/`
  - Pre-commit: `cargo test -p dman-server -- api`

- [ ] 24. Predictions Storage + Management

  **What to do**:
  - Create `crates/core/src/predictions/mod.rs` with `PredictionService`
  - Implement:
    - `store(db: &Database, image_id: i64, model_version: &str, result: serde_json::Value, score: Option<f64>) -> Result<i64>`
    - `get_by_image(db: &Database, image_id: i64) -> Result<Vec<Prediction>>`
    - `get_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Prediction>>`
    - `get_by_model(db: &Database, dataset_id: i64, model_version: &str) -> Result<Vec<Prediction>>`
    - `delete_by_model(db: &Database, dataset_id: i64, model_version: &str) -> Result<()>`
    - `compare_models(db: &Database, dataset_id: i64, models: &[&str]) -> Result<ComparisonReport>`
  - `ComparisonReport { models: Vec<String>, per_image: Vec<ImageComparison> }` — basic comparison (which model predicted what per image)
  - Predictions stored as JSON `result` field — flexible for different model output formats (YOLO, COCO, custom)
  - RED tests: store predictions → retrieve by image → retrieve by model → compare → delete

  **Must NOT do**:
  - No metrics computation (mAP, precision, recall) — V2
  - No model versioning/experiment tracking

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: DB operations, flexible JSON storage, comparison logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T23, T25)
  - **Parallel Group**: Wave 4
  - **Blocks**: T37
  - **Blocked By**: T3, T5

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — Prediction type
  - `crates/core/src/db/mod.rs` (T3) — Database patterns, JSON column handling

  **WHY Each Reference Matters**:
  - Prediction type defines the struct, DB module defines storage patterns
  - JSON column allows flexible model output formats without schema changes

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- predictions::tests` passes
  - [ ] Store + retrieve round-trip preserves JSON result exactly
  - [ ] Filter by model version returns only matching predictions
  - [ ] Compare returns per-image comparison across models

  **QA Scenarios**:

  ```
  Scenario: Predictions CRUD
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- predictions::tests::test_crud 2>&1`
      2. Assert exit code = 0
    Expected Result: Store, retrieve, filter, delete all work
    Evidence: .sisyphus/evidence/task-24-predictions.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement predictions storage and management`
  - Files: `crates/core/src/predictions/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- predictions`

- [ ] 25. Patches (Image Crop) Extraction + Storage

  **What to do**:
  - Create `crates/core/src/patches/mod.rs` with `PatchService`
  - Implement:
    - `extract(db: &Database, storage: &StorageManager, image_id: i64, bbox: BBox, output_dir: &Path) -> Result<Patch>` — crop image region, save to output, record in DB
    - `extract_batch(db: &Database, storage: &StorageManager, dataset_id: i64, annotations: bool, output_dir: &Path) -> Result<Vec<Patch>>` — extract patches for all annotations in dataset
    - `get_by_image(db: &Database, image_id: i64) -> Result<Vec<Patch>>`
    - `get_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Patch>>`
    - `delete_patch(db: &Database, storage: &StorageManager, patch_id: i64) -> Result<()>` — remove file + DB record
  - Image cropping: use `image` crate (`image::open(path)?.crop(x, y, w, h)`) for Rust-native crop
  - Patch file naming: `{image_id}_{bbox_hash}.{ext}`
  - RED tests: create test image → extract patch → verify crop dimensions → verify DB record

  **Must NOT do**:
  - No polygon-based crop (bbox only in V1)
  - No batch processing optimization (process images one at a time)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Image I/O, crop operations, file management + DB coordination
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T23, T24)
  - **Parallel Group**: Wave 4
  - **Blocks**: T37
  - **Blocked By**: T8, T5

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs` (T5) — BBox, Patch types
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager for image paths

  **External References**:
  - `image` crate: https://docs.rs/image/latest/image/ — `DynamicImage::crop()`, `GenericImageView`

  **WHY Each Reference Matters**:
  - `image` crate provides native Rust image loading and cropping — no external dependency
  - BBox type defines crop coordinates, Patch type defines the output record

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-core -- patches::tests` passes
  - [ ] Extract: output image has correct dimensions matching bbox
  - [ ] DB record: patch record links to correct image_id and bbox
  - [ ] Delete: file removed from disk + DB record deleted

  **QA Scenarios**:

  ```
  Scenario: Extract patch from image
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- patches::tests::test_extract 2>&1`
      2. Assert exit code = 0
    Expected Result: Patch file created with correct crop dimensions
    Evidence: .sisyphus/evidence/task-25-extract.txt

  Scenario: Batch extraction from annotations
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-core -- patches::tests::test_batch 2>&1`
      2. Assert exit code = 0
    Expected Result: One patch per annotation, all files created
    Evidence: .sisyphus/evidence/task-25-batch.txt
  ```

  **Commit**: YES
  - Message: `feat(core): implement patch extraction and storage`
  - Files: `crates/core/src/patches/mod.rs`
  - Pre-commit: `cargo test -p dman-core -- patches`

- [ ] 26. React SPA Scaffold + Build Pipeline + rust-embed

  **What to do**:
  - Create `crates/server/frontend/` directory with React SPA scaffold:
    - Use Vite + React + TypeScript: `npm create vite@latest frontend -- --template react-ts`
    - Add Tailwind CSS for styling
    - Add React Router for client-side routing
    - Configure Vite to output to `crates/server/frontend/dist/`
  - Set up rust-embed in `crates/server/`:
    - `#[derive(RustEmbed)] #[folder = "frontend/dist/"] struct Frontend;`
    - Axum handler serves embedded files, falls back to `index.html` for SPA routing
  - Build pipeline:
    - `build.rs` in `crates/server/` checks if `frontend/dist/` exists. If missing, prints cargo warning but doesn't fail build.
    - `Makefile` or `justfile` recipe: `build-frontend` runs `npm run build` in frontend/
    - Document: `cd crates/server/frontend && npm install && npm run build` before `cargo build --release`
  - Placeholder pages: Home (dataset list), Dataset (placeholder), About
  - Verify: `npm run build` produces dist/, `cargo build -p dman-server` embeds it, serve → browser shows React app

  **Must NOT do**:
  - No actual feature UI (that's T27, T28)
  - No backend calls yet (just scaffolding)
  - Don't require Node.js for Rust compilation (build.rs just warns if dist/ missing)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: React + Vite + Tailwind scaffolding, SPA routing, build pipeline
  - **Skills**: [`frontend-design`]
    - `frontend-design`: React SPA scaffold with production build setup

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T29, T30)
  - **Parallel Group**: Wave 5
  - **Blocks**: T27, T28
  - **Blocked By**: T22, T23

  **References**:

  **Pattern References**:
  - `crates/server/src/routes/` (T22) — axum routes where SPA handler is added
  - `crates/server/src/lib.rs` (T22) — server setup where rust-embed is integrated

  **External References**:
  - rust-embed with axum: https://github.com/pyrossh/rust-embed/tree/master/examples/axum
  - Vite React template: https://vitejs.dev/guide/

  **WHY Each Reference Matters**:
  - rust-embed axum example shows exact pattern for serving SPA with fallback routing
  - Server routes need the SPA handler integrated alongside API routes

  **Acceptance Criteria**:

  - [ ] `cd crates/server/frontend && npm install && npm run build` succeeds
  - [ ] `cargo build -p dman-server` compiles (with or without dist/)
  - [ ] Server serves React app at `http://localhost:8080/`
  - [ ] Non-API routes (e.g., `/datasets/foo`) serve index.html (SPA routing)

  **QA Scenarios**:

  ```
  Scenario: SPA serves and routes correctly
    Tool: Playwright
    Preconditions: Server running with built frontend
    Steps:
      1. Navigate to `http://localhost:8080/`
      2. Assert page title contains "dman" or app renders
      3. Navigate to `http://localhost:8080/datasets`
      4. Assert page doesn't 404 (SPA routing works)
    Expected Result: React app loads on all routes
    Evidence: .sisyphus/evidence/task-26-spa-routing.png

  Scenario: Build pipeline works
    Tool: Bash
    Steps:
      1. Run `cd crates/server/frontend && npm install && npm run build 2>&1`
      2. Assert exit code = 0
      3. Assert `ls crates/server/frontend/dist/index.html` exists
    Expected Result: Frontend builds to dist/
    Evidence: .sisyphus/evidence/task-26-build.txt
  ```

  **Commit**: YES
  - Message: `feat(server): scaffold React SPA with build pipeline and rust-embed`
  - Files: `crates/server/frontend/`, `crates/server/src/spa.rs`, `crates/server/build.rs`
  - Pre-commit: `cargo build -p dman-server`

- [ ] 27. Gallery View — Image Grid + Pagination + Filtering

  **What to do**:
  - Implement the main gallery view in React SPA:
    - Dataset selector (dropdown or sidebar) showing all datasets from `/api/datasets`
    - Image grid: thumbnail images loaded from `/thumbnails/{dataset_id}/{filename}`
    - Pagination controls: page number, per-page selector (25/50/100)
    - Category filter: dropdown populated from `/api/datasets/{name}/categories`
    - Search bar: filter by image filename
    - Dataset stats summary: image count, annotation count, category distribution (bar chart or numbers)
  - Image grid uses CSS grid with responsive columns
  - Lazy loading: images load as they enter viewport (IntersectionObserver)
  - Click on image → navigates to detail view (T28)
  - Loading states + error handling for API calls
  - RED test: hard to unit test React, but QA scenarios cover it

  **Must NOT do**:
  - No annotation overlay on gallery thumbnails (that's detail view T28)
  - No editing/mutation
  - No complex state management (React hooks + fetch sufficient)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: React UI components, grid layout, responsive design, API integration
  - **Skills**: [`frontend-design`]
    - `frontend-design`: Production-quality React components with Tailwind

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T28, T29, T30)
  - **Parallel Group**: Wave 5
  - **Blocks**: T37
  - **Blocked By**: T26

  **References**:

  **Pattern References**:
  - `crates/server/src/api/` (T23) — API endpoints the frontend calls
  - `crates/server/frontend/` (T26) — React scaffold, router setup

  **External References**:
  - React Router: https://reactrouter.com/
  - Tailwind CSS grid: https://tailwindcss.com/docs/grid-template-columns

  **WHY Each Reference Matters**:
  - API endpoints define the data contract — frontend must match request/response format
  - React scaffold provides the router and layout the gallery plugs into

  **Acceptance Criteria**:

  - [ ] Gallery shows images from selected dataset
  - [ ] Pagination: changing page shows different images
  - [ ] Category filter: selecting category reduces visible images
  - [ ] Loading state shown during API calls
  - [ ] 404/empty state for datasets with no images

  **QA Scenarios**:

  ```
  Scenario: Gallery loads and displays images
    Tool: Playwright
    Preconditions: Server running with imported dataset
    Steps:
      1. Navigate to `http://localhost:8080/`
      2. Select dataset from dropdown (selector: `select[data-testid="dataset-selector"]`)
      3. Wait for images to load (selector: `img[data-testid="gallery-image"]`)
      4. Assert at least 1 image visible
      5. Screenshot
    Expected Result: Image grid populated with thumbnails
    Evidence: .sisyphus/evidence/task-27-gallery.png

  Scenario: Category filter works
    Tool: Playwright
    Steps:
      1. Select category from filter (selector: `select[data-testid="category-filter"]`)
      2. Wait for grid to update
      3. Assert image count changed
    Expected Result: Fewer images shown after filtering
    Evidence: .sisyphus/evidence/task-27-filter.png
  ```

  **Commit**: YES
  - Message: `feat(frontend): implement gallery view with pagination and filtering`
  - Files: `crates/server/frontend/src/`
  - Pre-commit: `cd crates/server/frontend && npm run build`

- [ ] 28. Detail View — Image + Metadata Inspector + Annotations Overlay

  **What to do**:
  - Implement detail view in React SPA:
    - Full-size image display from `/images/{dataset_id}/{filename}`
    - Bounding box overlay: render annotations as colored rectangles on canvas over image
    - Category labels on bboxes with color coding per category
    - Metadata panel: show all image metadata, annotations, predictions
    - Navigation: prev/next image arrows, keyboard shortcuts (← →)
    - Predictions tab: if predictions exist, show model outputs alongside ground truth
    - Patches tab: if patches exist, show extracted crops
    - Zoom controls for large images
  - Use HTML5 Canvas or SVG for annotation overlay
  - Responsive: works on various screen sizes
  - RED test: QA scenarios via Playwright

  **Must NOT do**:
  - No editing annotations (read-only)
  - No annotation creation
  - No polygon rendering (bbox only in V1)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
    - Reason: Canvas/SVG rendering, interactive overlays, responsive layout
  - **Skills**: [`frontend-design`]
    - `frontend-design`: Complex interactive UI with canvas rendering

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T27, T29, T30)
  - **Parallel Group**: Wave 5
  - **Blocks**: T37
  - **Blocked By**: T26

  **References**:

  **Pattern References**:
  - `crates/server/src/api/` (T23) — API for image metadata, annotations, predictions
  - `crates/server/frontend/` (T26, T27) — existing React components and routing

  **WHY Each Reference Matters**:
  - API provides annotations as JSON — frontend must parse bbox coordinates and render them
  - Gallery view's routing sends user to detail view with image ID parameter

  **Acceptance Criteria**:

  - [ ] Full-size image displays correctly
  - [ ] Bounding boxes render at correct positions over image
  - [ ] Category labels visible on bboxes
  - [ ] Prev/next navigation works
  - [ ] Metadata panel shows all fields

  **QA Scenarios**:

  ```
  Scenario: Annotation overlay renders correctly
    Tool: Playwright
    Preconditions: Server running with annotated dataset
    Steps:
      1. Navigate to detail view: `http://localhost:8080/datasets/test-ds/images/1`
      2. Wait for image to load (selector: `img[data-testid="detail-image"]`)
      3. Assert bbox overlays present (selector: `[data-testid="bbox-overlay"]`)
      4. Assert at least one category label visible
      5. Screenshot
    Expected Result: Image with colored bboxes and labels
    Evidence: .sisyphus/evidence/task-28-overlay.png

  Scenario: Keyboard navigation works
    Tool: Playwright
    Steps:
      1. On detail view, press ArrowRight
      2. Assert URL changed to next image
      3. Press ArrowLeft
      4. Assert URL back to original image
    Expected Result: Navigation between images via keyboard
    Evidence: .sisyphus/evidence/task-28-navigation.png
  ```

  **Commit**: YES
  - Message: `feat(frontend): implement detail view with annotations overlay`
  - Files: `crates/server/frontend/src/`
  - Pre-commit: `cd crates/server/frontend && npm run build`

- [ ] 29. TUI Shell — Ratatui App + Dataset Browser

  **What to do**:
  - Create `crates/tui/src/main.rs` with ratatui application:
    - Main event loop with crossterm backend
    - Application state: `AppState { datasets: Vec<Dataset>, selected: usize, view: View, search: String }`
    - `View` enum: `DatasetList`, `DatasetDetail(String)`, `Help`
    - Dataset list view: table with columns (Name, Format, Images, Annotations, Size, Created)
    - Sorted by name (default), toggleable sort by other columns
    - Keyboard bindings:
      - `j`/`k` or `↓`/`↑`: navigate list
      - `Enter`: open dataset detail
      - `/`: search/filter
      - `q`: quit
      - `?`: help
    - Footer: status bar with key hints
    - Header: "dman — Dataset Manager" title bar
  - Wire into CLI: `dman tui` launches the TUI app
  - Use `Catalog::open()` to load datasets on startup
  - RED tests: hard to unit test TUI, QA via tmux

  **Must NOT do**:
  - No detail view yet (that's T30)
  - No inline editing
  - No async operations from TUI (load data synchronously on startup)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Ratatui event loop, state management, keyboard handling, widget composition
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T26, T27, T28)
  - **Parallel Group**: Wave 5
  - **Blocks**: T30
  - **Blocked By**: T6

  **References**:

  **Pattern References**:
  - `crates/core/src/catalog/mod.rs` (T9) — Catalog::open() for loading data
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService.list(), .inspect()

  **External References**:
  - ratatui examples: https://github.com/ratatui/ratatui/tree/main/examples
  - ratatui Table widget: https://docs.rs/ratatui/latest/ratatui/widgets/struct.Table.html

  **WHY Each Reference Matters**:
  - Ratatui examples show event loop + widget composition patterns
  - Catalog provides the data source that TUI displays

  **Acceptance Criteria**:

  - [ ] `cargo build -p dman-tui` compiles
  - [ ] `dman tui` launches without crash (even with 0 datasets)
  - [ ] Dataset list displays with correct columns
  - [ ] Keyboard navigation works (j/k/Enter/q)

  **QA Scenarios**:

  ```
  Scenario: TUI launches and displays datasets
    Tool: interactive_bash (tmux)
    Preconditions: Catalog initialized with at least 1 dataset
    Steps:
      1. Create tmux session: `new-session -d -s dman-tui`
      2. Send: `DMAN_HOME=/tmp/test-dman cargo run -p dman-tui`
      3. Wait 3s for startup
      4. Capture pane: `capture-pane -t dman-tui -p`
      5. Assert output contains "dman" and dataset name
      6. Send 'q' to quit
    Expected Result: TUI renders dataset table
    Evidence: .sisyphus/evidence/task-29-tui-launch.txt

  Scenario: TUI handles empty catalog
    Tool: interactive_bash (tmux)
    Steps:
      1. Launch TUI with empty catalog
      2. Assert "No datasets found" or empty table displayed
      3. Send 'q' to quit
    Expected Result: Graceful empty state, no crash
    Evidence: .sisyphus/evidence/task-29-tui-empty.txt
  ```

  **Commit**: YES
  - Message: `feat(tui): implement ratatui app shell and dataset browser`
  - Files: `crates/tui/src/main.rs`, `crates/tui/src/app.rs`, `crates/tui/src/ui/`
  - Pre-commit: `cargo build -p dman-tui`

- [ ] 30. TUI Detail View + Keyboard Navigation

  **What to do**:
  - Add dataset detail view to TUI:
    - Triggered by pressing `Enter` on a dataset in list view
    - Tabs: Info | Images | Categories | Schema
    - Info tab: dataset name, path, format, image count, annotation count, created date
    - Images tab: scrollable list of images with filename + annotation count
    - Categories tab: list of categories with image count per category
    - Schema tab: display schema definition (TOML formatted)
    - Tab switching: `Tab` key or `1-4` number keys
    - Back to list: `Esc` or `Backspace`
  - Image list supports scrolling with j/k, shows basic metadata per image
  - Add `Ctrl+R` to refresh data (re-query database)
  - RED test: QA via tmux

  **Must NOT do**:
  - No image preview in terminal (out of scope for V1)
  - No annotation editing
  - No TUI for virtual datasets (view only real datasets)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Multi-tab widget layout, state management, keyboard handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T26-T28)
  - **Parallel Group**: Wave 5
  - **Blocks**: T37
  - **Blocked By**: T29

  **References**:

  **Pattern References**:
  - `crates/tui/src/app.rs` (T29) — AppState, View enum, event loop
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService.inspect() for detail data
  - `crates/core/src/schema/mod.rs` (T7) — SchemaDefinition for schema tab

  **External References**:
  - ratatui Tabs widget: https://docs.rs/ratatui/latest/ratatui/widgets/struct.Tabs.html

  **WHY Each Reference Matters**:
  - T29's app state needs extension for detail view mode
  - DatasetService.inspect() provides the stats shown in detail info tab

  **Acceptance Criteria**:

  - [ ] `cargo build -p dman-tui` compiles
  - [ ] Enter on dataset → detail view with tabs
  - [ ] Tab switching between Info/Images/Categories/Schema works
  - [ ] Esc returns to list view
  - [ ] Images tab scrolls through image list

  **QA Scenarios**:

  ```
  Scenario: Detail view navigation
    Tool: interactive_bash (tmux)
    Steps:
      1. Launch TUI with dataset
      2. Press Enter on first dataset
      3. Capture pane — assert detail view visible
      4. Press Tab — assert tab switched
      5. Press Esc — assert back to list
    Expected Result: Detail view with working tab navigation
    Evidence: .sisyphus/evidence/task-30-detail.txt
  ```

  **Commit**: YES
  - Message: `feat(tui): implement detail view and keyboard navigation`
  - Files: `crates/tui/src/ui/detail.rs`, `crates/tui/src/app.rs`
  - Pre-commit: `cargo build -p dman-tui`

- [ ] 31. PyO3 Integration — Feature-Gated Crate + Plugin Discovery

  **What to do**:
  - Set up `crates/python/` as a PyO3 crate:
    - `Cargo.toml`: `crate-type = ["cdylib", "rlib"]`, `pyo3 = { version = "0.28", features = ["extension-module"] }`
    - Feature-gated: root workspace `[features] python = ["dman-python"]`
    - Without `--features python`, this crate is excluded from build
  - Implement plugin discovery:
    - `PluginManager::new(plugin_dirs: Vec<PathBuf>) -> Self`
    - `discover() -> Result<Vec<PluginInfo>>` — scan plugin_dirs for Python files with `dman_plugin` marker
    - `load_plugin(path: &Path) -> Result<Box<dyn FormatImporter>>` — load Python module, wrap as FormatImporter
  - Plugin convention: Python files in `~/.dman/plugins/` with a `dman_plugin` dict at module level:
    ```python
    dman_plugin = {
        "name": "my-custom-format",
        "type": "format",  # format | transform | embeddings
        "version": "1.0"
    }

    def import_dataset(path: str) -> dict:
        """Return dict with 'images' and 'annotations' keys"""
        ...

    def export_dataset(data: dict, output_path: str) -> None:
        ...
    ```
  - PyO3 wrapping: call Python functions from Rust, convert between Rust types and Python dicts
  - Maturin config for building pip wheel: `pyproject.toml`
  - RED tests (with feature gate): discover plugin from test fixture → load → verify callable

  **Must NOT do**:
  - No sandboxing (trust user plugins)
  - No plugin dependency management
  - No hot-reload (restart to pick up new plugins)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: PyO3 FFI, Python↔Rust type conversion, feature gating, maturin setup
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T32-T36)
  - **Parallel Group**: Wave 6
  - **Blocks**: T32, T33, T34, T35
  - **Blocked By**: T5

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/mod.rs` (T15) — FormatImporter trait that plugins implement
  - `crates/core/src/config/mod.rs` (T10) — PythonConfig for interpreter path, plugin dirs

  **External References**:
  - PyO3 guide: https://pyo3.rs/v0.28/
  - maturin docs: https://www.maturin.rs/

  **WHY Each Reference Matters**:
  - FormatImporter trait is the contract Python plugins must satisfy — PyO3 wrapper bridges the gap
  - maturin builds the pip-installable wheel from the Rust+Python crate

  **Acceptance Criteria**:

  - [ ] `cargo build -p dman-python --features python` compiles (with Python available)
  - [ ] `cargo build --workspace` compiles WITHOUT Python (feature not enabled)
  - [ ] Plugin discovery finds .py files with dman_plugin marker
  - [ ] Loaded plugin callable from Rust

  **QA Scenarios**:

  ```
  Scenario: Build with and without Python feature
    Tool: Bash
    Steps:
      1. Run `cargo build --workspace 2>&1` — assert exit 0 (no python required)
      2. Run `cargo build -p dman-python --features python 2>&1` — assert exit 0 (if Python available)
    Expected Result: Both builds succeed
    Evidence: .sisyphus/evidence/task-31-feature-gate.txt

  Scenario: Plugin discovery
    Tool: Bash
    Steps:
      1. Create test plugin in temp dir with dman_plugin marker
      2. Run `cargo test -p dman-python --features python -- plugin::tests::test_discover 2>&1`
      3. Assert exit code = 0
    Expected Result: Plugin found and info extracted
    Evidence: .sisyphus/evidence/task-31-discover.txt
  ```

  **Commit**: YES
  - Message: `feat(python): add PyO3 integration with feature gate and plugin discovery`
  - Files: `crates/python/`, `pyproject.toml`
  - Pre-commit: `cargo build --workspace && cargo build -p dman-python --features python`

- [ ] 32. Python Format Converter Plugin API

  **What to do**:
  - Create `crates/python/src/plugins/format.rs` with the bridge between Python plugins and Rust FormatImporter/Exporter traits
  - `PythonFormatImporter` struct implementing `FormatImporter`:
    - `name()` → reads from Python plugin's `dman_plugin["name"]`
    - `detect(path)` → calls Python plugin's `detect(path)` if it exists, else false
    - `import(db, storage, path, name)` → calls Python `import_dataset(path)`, converts returned dict to Rust types, inserts into DB
  - `PythonFormatExporter` struct implementing `FormatExporter`:
    - Queries data from DB, converts to Python dict, calls `export_dataset(data, output_path)`
  - Type conversion bridge:
    - Python dict `{"images": [...], "annotations": [...]}` → Rust Vec<Image>, Vec<Annotation>
    - Rust Dataset/Image/Annotation → Python dict for export
  - Error handling: Python exceptions → DmanError::PluginError with traceback
  - Register discovered plugins into FormatRegistry
  - RED tests: create a minimal Python plugin → import via bridge → verify DB records

  **Must NOT do**:
  - No transform or embedding plugins (those are T35)
  - No async Python calls

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: PyO3 FFI complexity, type conversion bridge, error handling across language boundary
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T33, T34, T35, T36)
  - **Parallel Group**: Wave 6
  - **Blocks**: T37
  - **Blocked By**: T15, T31

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/mod.rs` (T15) — FormatImporter/FormatExporter traits to implement
  - `crates/python/src/` (T31) — PluginManager, PyO3 setup

  **External References**:
  - PyO3 calling Python from Rust: https://pyo3.rs/v0.28/python-from-rust.html

  **WHY Each Reference Matters**:
  - FormatImporter trait defines the exact methods the Python bridge must implement
  - PyO3 Python-from-Rust docs show how to call Python functions and extract return values

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-python --features python -- plugins::format::tests` passes
  - [ ] Python plugin import → Rust receives correct data types
  - [ ] Python exception → DmanError::PluginError with message

  **QA Scenarios**:

  ```
  Scenario: Python format plugin import
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-python --features python -- plugins::format::tests::test_import_bridge 2>&1`
      2. Assert exit code = 0
    Expected Result: Python plugin imports data correctly via bridge
    Evidence: .sisyphus/evidence/task-32-format-bridge.txt
  ```

  **Commit**: YES
  - Message: `feat(python): implement format converter plugin API`
  - Files: `crates/python/src/plugins/format.rs`
  - Pre-commit: `cargo test -p dman-python --features python`

- [ ] 33. Python SDK — PyTorch Dataset + HF Datasets Loader

  **What to do**:
  - Create `crates/python/src/sdk/loader.rs` — expose Rust dataset loading to Python
  - Python-callable functions (via `#[pyfunction]`):
    - `load_dataset(name: str, split: Optional[str]) -> DmanDataset` — load from catalog
    - `DmanDataset` Python class with:
      - `__len__()` → image count
      - `__getitem__(idx)` → returns dict with image path, annotations, metadata
      - `to_torch_dataset()` → returns a wrapper compatible with `torch.utils.data.Dataset`
      - `to_hf_dataset()` → returns a `datasets.Dataset` object (HuggingFace)
      - `images()` → iterator over images
      - `annotations()` → iterator over annotations
  - For `to_torch_dataset()`: create Python class that inherits from `torch.utils.data.Dataset` (defined in Python, instantiated from Rust)
  - For `to_hf_dataset()`: build Arrow table from data, construct `datasets.Dataset.from_dict()`
  - Handle: dataset not found → Python exception. Missing torch/datasets → ImportError with helpful message.
  - RED tests: load dataset → check len → getitem → verify dict structure

  **Must NOT do**:
  - No image loading/transforms (user handles that in their PyTorch transform pipeline)
  - No distributed loading (single-process)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: PyO3 class definitions, Python interop with torch/datasets, type bridging
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T32, T34, T35, T36)
  - **Parallel Group**: Wave 6
  - **Blocks**: T37
  - **Blocked By**: T6, T31

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService for loading data
  - `crates/core/src/catalog/mod.rs` (T9) — Catalog::open() for finding datasets
  - `crates/python/src/` (T31) — PyO3 setup, module definition

  **External References**:
  - PyO3 classes: https://pyo3.rs/v0.28/class.html
  - torch Dataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
  - HF datasets.Dataset.from_dict: https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.from_dict

  **WHY Each Reference Matters**:
  - PyO3 class docs for #[pyclass] + #[pymethods] pattern
  - torch/HF docs define the exact interface the SDK must produce

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-python --features python -- sdk::loader::tests` passes
  - [ ] `import dman; ds = dman.load_dataset("test")` works from Python
  - [ ] `len(ds)` returns correct count
  - [ ] `ds[0]` returns dict with expected keys

  **QA Scenarios**:

  ```
  Scenario: Python SDK loads dataset
    Tool: Bash
    Steps:
      1. Build wheel: `maturin develop --features python`
      2. Run `python3 -c "import dman; ds = dman.load_dataset('test'); print(len(ds)); print(ds[0].keys())" 2>&1`
      3. Assert output contains expected count and keys
    Expected Result: Dataset loads and is iterable from Python
    Evidence: .sisyphus/evidence/task-33-python-sdk.txt
  ```

  **Commit**: YES
  - Message: `feat(python): implement PyTorch Dataset and HF datasets loader`
  - Files: `crates/python/src/sdk/loader.rs`
  - Pre-commit: `cargo test -p dman-python --features python -- sdk::loader`

- [ ] 34. Python SDK — Dataset Builder/Updater

  **What to do**:
  - Create `crates/python/src/sdk/builder.rs` — Python API for creating/updating datasets
  - Python-callable functions:
    - `create_dataset(name: str, schema_path: Optional[str]) -> DmanDatasetBuilder`
    - `DmanDatasetBuilder` class with:
      - `add_image(path: str, metadata: Optional[dict]) -> int` — add image, return image_id
      - `add_annotation(image_id: int, category: str, bbox: list, metadata: Optional[dict])` — add annotation
      - `set_category(name: str, supercategory: Optional[str])` — define category
      - `build() -> DmanDataset` — finalize, register in catalog, return loadable dataset
    - `update_dataset(name: str) -> DmanDatasetUpdater`
    - `DmanDatasetUpdater` class with:
      - `add_image(...)`, `add_annotation(...)` — same as builder
      - `remove_image(image_id: int)`
      - `apply()` — commit changes
  - Transaction semantics: `build()`/`apply()` wraps all changes in DB transaction
  - RED tests: create dataset from Python → add images → build → verify in catalog

  **Must NOT do**:
  - No schema inference from data (user provides schema or uses defaults)
  - No batch operations beyond sequential add (V1)

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Builder pattern in PyO3, transaction management, multi-method class
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T32, T33, T35, T36)
  - **Parallel Group**: Wave 6
  - **Blocks**: T37
  - **Blocked By**: T6, T31

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs` (T6) — DatasetService.register() for creating datasets
  - `crates/core/src/storage/mod.rs` (T8) — StorageManager for image storage during build
  - `crates/python/src/sdk/loader.rs` (T33) — DmanDataset class that build() returns

  **WHY Each Reference Matters**:
  - DatasetService is the Rust backend the Python builder calls
  - build() returns a DmanDataset (from T33), so the types must be compatible

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-python --features python -- sdk::builder::tests` passes
  - [ ] Python: create_dataset → add_image → add_annotation → build → dataset visible in catalog
  - [ ] Transaction: if build fails midway, no partial dataset in catalog

  **QA Scenarios**:

  ```
  Scenario: Build dataset from Python
    Tool: Bash
    Steps:
      1. Run `python3 -c "
         import dman
         builder = dman.create_dataset('python-test')
         img_id = builder.add_image('/path/to/img.jpg')
         builder.add_annotation(img_id, 'cat', [10, 20, 50, 50])
         ds = builder.build()
         print(len(ds))
         " 2>&1`
      2. Assert output = "1"
    Expected Result: Dataset created with 1 image
    Evidence: .sisyphus/evidence/task-34-builder.txt
  ```

  **Commit**: YES
  - Message: `feat(python): implement dataset builder/updater SDK`
  - Files: `crates/python/src/sdk/builder.rs`
  - Pre-commit: `cargo test -p dman-python --features python -- sdk::builder`

- [ ] 35. Embeddings Computation via Python Plugins

  **What to do**:
  - Create `crates/python/src/embeddings.rs` — bridge for computing embeddings via Python
  - Implement:
    - `compute_embeddings(db: &Database, dataset_id: i64, model_script: &Path, batch_size: usize) -> Result<u64>` — call Python script for each batch of images, store results
  - Python embedding plugin convention:
    ```python
    dman_plugin = {
        "name": "clip-embeddings",
        "type": "embeddings",
        "model": "openai/clip-vit-base-patch32",
        "dimension": 512
    }

    def compute(image_paths: list[str]) -> list[list[float]]:
        """Return list of embedding vectors for each image"""
        ...
    ```
  - Batch processing: send `batch_size` image paths to Python, receive vectors back
  - Store each vector via EmbeddingService (T16)
  - Progress tracking: return count of processed images
  - Wire into CLI: `dman embed <dataset-name> --model <plugin-path> [--batch-size 32]`
  - RED tests: mock Python plugin that returns random vectors → compute → verify stored in DB

  **Must NOT do**:
  - No model loading in Rust
  - No GPU management
  - No incremental computation (recomputes all images every time)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: PyO3 batch processing, progress tracking, plugin convention
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T31-T34, T36)
  - **Parallel Group**: Wave 6
  - **Blocks**: T37
  - **Blocked By**: T16, T31

  **References**:

  **Pattern References**:
  - `crates/core/src/embeddings/mod.rs` (T16) — EmbeddingService.store() for saving vectors
  - `crates/python/src/` (T31) — PyO3 setup, plugin loading

  **WHY Each Reference Matters**:
  - EmbeddingService defines how vectors are stored — compute bridge calls store() per batch
  - Plugin loading from T31 provides the mechanism to load Python embedding scripts

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-python --features python -- embeddings::tests` passes
  - [ ] Mock plugin computes vectors → all stored in DB
  - [ ] Batch processing: 10 images in batches of 3 → all 10 get embeddings

  **QA Scenarios**:

  ```
  Scenario: Compute embeddings via Python plugin
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-python --features python -- embeddings::tests::test_compute 2>&1`
      2. Assert exit code = 0
    Expected Result: All images get embedding vectors stored
    Evidence: .sisyphus/evidence/task-35-embeddings.txt
  ```

  **Commit**: YES
  - Message: `feat(python): implement embeddings computation via plugins`
  - Files: `crates/python/src/embeddings.rs`
  - Pre-commit: `cargo test -p dman-python --features python -- embeddings`

- [ ] 36. Label Studio API Integration — Import/Export Projects

  **What to do**:
  - Create `crates/server/src/label_studio/mod.rs` with Label Studio client
  - Implement `LabelStudioClient`:
    - `new(base_url: &str, api_key: &str) -> Self`
    - `list_projects() -> Result<Vec<LSProject>>` — `GET /api/projects/`
    - `import_project(project_id: i64, dataset_name: &str) -> Result<Dataset>` — fetch tasks + annotations from LS, convert to dman format, store in catalog
    - `export_to_project(dataset_name: &str, project_id: i64) -> Result<()>` — push dman images as tasks to LS project (image URLs point to dman's local server)
    - `create_project(title: &str, label_config: &str) -> Result<LSProject>` — create new LS project
  - REST API calls via `reqwest` crate (async HTTP client)
  - Import conversion: LS task format → dman Image + Annotation
  - Export: dman Image → LS task with data.image pointing to `http://localhost:{port}/images/{dataset_id}/{filename}`
  - Wire into CLI: `dman label-studio import --url <url> --api-key <key> --project <id> --name <name>`, `dman label-studio export --url <url> --api-key <key> --project <id> --dataset <name>`
  - RED tests: mock LS API (mock HTTP server or recorded responses) → import → verify. Export → verify request body.

  **Must NOT do**:
  - No Label Studio server management (user runs LS separately)
  - No label config generation (user provides or uses LS defaults)
  - No real-time sync (one-shot import/export)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: HTTP client, API integration, format conversion, mock testing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T31-T35)
  - **Parallel Group**: Wave 6
  - **Blocks**: T37
  - **Blocked By**: T22, T23

  **References**:

  **External References**:
  - Label Studio API docs: https://labelstud.io/api
  - `reqwest` crate: https://docs.rs/reqwest/latest/reqwest/
  - LS task format (from research): `{"data": {"image": "url"}, "annotations": [...]}`

  **WHY Each Reference Matters**:
  - LS API defines the exact endpoints and request/response formats
  - reqwest is the HTTP client for making LS API calls
  - Task format defines the conversion between LS and dman data models

  **Acceptance Criteria**:

  - [ ] `cargo test -p dman-server -- label_studio::tests` passes
  - [ ] Import: LS project tasks → dman dataset with images + annotations
  - [ ] Export: dman dataset → LS tasks with correct image URLs
  - [ ] API errors handled gracefully (connection refused, 401, 404)

  **QA Scenarios**:

  ```
  Scenario: Import from Label Studio (mocked)
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-server -- label_studio::tests::test_import 2>&1`
      2. Assert exit code = 0
    Expected Result: Mocked LS responses converted to dman dataset
    Evidence: .sisyphus/evidence/task-36-ls-import.txt

  Scenario: Export generates correct LS task format
    Tool: Bash
    Steps:
      1. Run `cargo test -p dman-server -- label_studio::tests::test_export_format 2>&1`
      2. Assert exit code = 0
    Expected Result: Task JSON has correct image URL and structure
    Evidence: .sisyphus/evidence/task-36-ls-export.txt
  ```

  **Commit**: YES
  - Message: `feat(server): implement Label Studio API integration`
  - Files: `crates/server/src/label_studio/mod.rs`
  - Pre-commit: `cargo test -p dman-server -- label_studio`

- [ ] 37. CLI End-to-End Integration Tests

  **What to do**:
  - Create `crates/cli/tests/integration/` directory with end-to-end test modules
  - Write integration tests that exercise the CLI binary as a subprocess (`std::process::Command`)
  - Test the full dataset lifecycle: `dman init` → `dman import` → `dman list` → `dman info` → `dman operate` → `dman export`
  - Test `dman schema show`, `dman schema validate` on a real dataset
  - Test `dman virtual create` and `dman virtual list` flows
  - Test `dman materialize` end-to-end
  - Use a temp directory (`tempdir` crate) for each test to isolate state
  - Set `DMAN_HOME` env var to temp dir so tests don't pollute real catalog
  - Include tests for error cases: invalid paths, missing datasets, bad format flags
  - Verify JSON output mode (`--json`) for machine-parseable results where supported

  **Must NOT do**:
  - Do NOT test TUI mode (not feasible in integration tests)
  - Do NOT test `dman serve` (server tests are separate)
  - Do NOT add any new features — tests only exercise existing CLI commands

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration tests require understanding the full CLI interface and orchestrating multi-step flows. Deep reasoning needed to cover edge cases.
  - **Skills**: []
    - No specialized skills needed — pure Rust testing
  - **Skills Evaluated but Omitted**:
    - `webapp-testing`: Not applicable — CLI tests, not web UI

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 38, 39)
  - **Blocks**: F1-F4 (Final Verification)
  - **Blocked By**: T11 (CLI shell), T12-T14 (importers), T17 (virtual datasets), T19 (materialize), T20 (operate)

  **References**:

  **Pattern References** (existing code to follow):
  - `crates/core/src/lib.rs` — Core API surface that CLI wraps (understand available operations)
  - `crates/cli/src/main.rs` — CLI entry point and subcommand structure (know exact command names and flags)

  **API/Type References** (contracts to test against):
  - `crates/core/src/types.rs` — Dataset, VirtualDataset, Schema types (expected shapes in JSON output)
  - `crates/core/src/error.rs` — Error types (expected error messages)

  **Test References** (testing patterns to follow):
  - `crates/core/tests/` — Unit test patterns from earlier tasks (assertion style, setup helpers)

  **External References**:
  - `assert_cmd` crate docs: https://docs.rs/assert_cmd/latest/assert_cmd/ — CLI testing helpers
  - `predicates` crate docs: https://docs.rs/predicates/latest/predicates/ — Output assertion predicates

  **WHY Each Reference Matters**:
  - CLI main.rs gives exact subcommand names and argument flags to invoke
  - Core types define expected JSON shapes for `--json` output assertions
  - Error types define expected error messages for negative test cases
  - `assert_cmd` provides `Command::cargo_bin()` for testing the compiled binary

  **Acceptance Criteria**:

  **Tests (TDD):**
  - [ ] Test file created: `crates/cli/tests/integration/lifecycle.rs`
  - [ ] Test file created: `crates/cli/tests/integration/schema.rs`
  - [ ] Test file created: `crates/cli/tests/integration/virtual_dataset.rs`
  - [ ] `cargo test -p dman-cli --test integration` → PASS (≥10 tests, 0 failures)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Full dataset lifecycle via CLI
    Tool: Bash
    Preconditions: `cargo build -p dman-cli` succeeds, temp dir created
    Steps:
      1. Run `DMAN_HOME=/tmp/dman-test-$$ cargo run -p dman-cli -- init test-ds --path /tmp/dman-test-$$/images --schema /tmp/dman-test-$$/schema.toml 2>&1`
      2. Assert stdout contains "Dataset 'test-ds' initialized"
      3. Run `DMAN_HOME=/tmp/dman-test-$$ cargo run -p dman-cli -- list --json 2>&1`
      4. Assert JSON output contains `"name": "test-ds"`
      5. Run `DMAN_HOME=/tmp/dman-test-$$ cargo run -p dman-cli -- info test-ds --json 2>&1`
      6. Assert JSON output contains `"image_count"` field
    Expected Result: All commands exit 0 with correct output
    Failure Indicators: Non-zero exit code, missing fields in JSON, "error" in stderr
    Evidence: .sisyphus/evidence/task-37-cli-lifecycle.txt

  Scenario: CLI error handling for invalid input
    Tool: Bash
    Preconditions: `cargo build -p dman-cli` succeeds
    Steps:
      1. Run `DMAN_HOME=/tmp/dman-test-$$ cargo run -p dman-cli -- info nonexistent-dataset 2>&1`
      2. Assert exit code ≠ 0
      3. Assert stderr contains "not found" or "does not exist"
      4. Run `DMAN_HOME=/tmp/dman-test-$$ cargo run -p dman-cli -- import --format invalid-fmt /tmp/nothing 2>&1`
      5. Assert exit code ≠ 0
      6. Assert stderr contains "unsupported format" or usage help
    Expected Result: Graceful error messages, non-zero exit codes, no panics
    Failure Indicators: Panic/backtrace in output, exit code 0 on invalid input
    Evidence: .sisyphus/evidence/task-37-cli-errors.txt
  ```

  **Commit**: YES
  - Message: `test(cli): add end-to-end integration tests for full CLI lifecycle`
  - Files: `crates/cli/tests/integration/*.rs`, `crates/cli/Cargo.toml` (dev-dependencies)
  - Pre-commit: `cargo test -p dman-cli --test integration`

- [ ] 38. Cross-Format Round-Trip Tests (Import → Export → Re-Import)

  **What to do**:
  - Create `crates/core/tests/roundtrip/` directory with cross-format test modules
  - Write tests that import a dataset from one format, export to another, and re-import — verifying data integrity
  - Test matrix (minimum):
    - YOLO → COCO → YOLO (verify bounding boxes survive coordinate conversion)
    - COCO → HF Parquet → COCO (verify annotations survive serialization)
    - HF Parquet → YOLO → HF Parquet (verify metadata preservation)
  - For each round-trip:
    - Create minimal but representative test fixtures (5-10 images with annotations)
    - Import into dman dataset
    - Export to target format into temp dir
    - Re-import from exported files into new dataset
    - Compare: image count, annotation count, bounding box coordinates (within epsilon for float conversion), class labels, metadata fields
  - Test that format-specific fields that have no equivalent in the target format are gracefully dropped (not error)
  - Include test fixtures as small test images (1x1 PNGs are fine for format testing)

  **Must NOT do**:
  - Do NOT test Python plugin formats (those are separate)
  - Do NOT test with real large datasets — keep fixtures minimal
  - Do NOT add new format features — only test existing import/export paths

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Round-trip testing requires deep understanding of format-specific field mappings and lossy vs lossless conversions. Careful reasoning about coordinate systems.
  - **Skills**: []
    - No specialized skills needed — pure Rust testing
  - **Skills Evaluated but Omitted**:
    - `webapp-testing`: Not applicable — data format tests, not web UI

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 37, 39)
  - **Blocks**: F1-F4 (Final Verification)
  - **Blocked By**: T12 (HF Parquet importer), T13 (YOLO importer), T14 (COCO importer), T15 (Format trait + export)

  **References**:

  **Pattern References** (existing code to follow):
  - `crates/core/src/format/mod.rs` — Format trait definition (import/export API surface)
  - `crates/core/src/format/yolo.rs` — YOLO format implementation (coordinate normalization pattern)
  - `crates/core/src/format/coco.rs` — COCO format implementation (JSON annotation structure)
  - `crates/core/src/format/hf_parquet.rs` — HF Parquet implementation (column mapping)

  **API/Type References** (contracts to test against):
  - `crates/core/src/types.rs:Annotation` — Annotation type (what fields must survive round-trip)
  - `crates/core/src/types.rs:BBox` — Bounding box type (coordinate format for comparison)

  **Test References** (testing patterns to follow):
  - Unit tests in each format module — Per-format test patterns (fixture creation, assertion style)

  **External References**:
  - YOLO format spec: normalized xywh coordinates (0.0-1.0)
  - COCO format spec: absolute xywh pixel coordinates
  - Epsilon comparison for float round-trips: `(a - b).abs() < 1e-6`

  **WHY Each Reference Matters**:
  - Format implementations show exact coordinate conversion logic that round-trip tests validate
  - Annotation/BBox types define the comparison contract for data integrity checks
  - Coordinate system differences (YOLO normalized vs COCO absolute) are the primary source of round-trip precision loss

  **Acceptance Criteria**:

  **Tests (TDD):**
  - [ ] Test file created: `crates/core/tests/roundtrip/yolo_coco.rs`
  - [ ] Test file created: `crates/core/tests/roundtrip/coco_parquet.rs`
  - [ ] Test file created: `crates/core/tests/roundtrip/parquet_yolo.rs`
  - [ ] `cargo test -p dman-core --test roundtrip` → PASS (≥6 tests, 0 failures)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: YOLO → COCO → YOLO round-trip preserves bounding boxes
    Tool: Bash
    Preconditions: Test fixtures exist in crates/core/tests/fixtures/yolo_sample/
    Steps:
      1. Run `cargo test -p dman-core --test roundtrip -- yolo_coco::test_roundtrip_bbox_preservation 2>&1`
      2. Assert exit code = 0
      3. Assert test output contains "test yolo_coco::test_roundtrip_bbox_preservation ... ok"
    Expected Result: All bounding box coordinates match within epsilon (1e-6) after full round-trip
    Failure Indicators: Test failure mentioning coordinate mismatch, assertion errors with float values
    Evidence: .sisyphus/evidence/task-38-yolo-coco-roundtrip.txt

  Scenario: Format-specific fields gracefully dropped on cross-format export
    Tool: Bash
    Preconditions: Test fixtures with format-specific metadata
    Steps:
      1. Run `cargo test -p dman-core --test roundtrip -- test_lossy_field_handling 2>&1`
      2. Assert exit code = 0
      3. Verify test checks that COCO `segmentation` field (not in YOLO) is dropped without error
      4. Verify test checks image count and class labels still match
    Expected Result: Non-universal fields dropped gracefully, core data preserved
    Failure Indicators: Panic on unknown field, error during export, data loss beyond expected fields
    Evidence: .sisyphus/evidence/task-38-lossy-fields.txt
  ```

  **Commit**: YES
  - Message: `test(core): add cross-format round-trip integration tests`
  - Files: `crates/core/tests/roundtrip/*.rs`, `crates/core/tests/fixtures/`
  - Pre-commit: `cargo test -p dman-core --test roundtrip`

- [ ] 39. Virtual Dataset + Materialize + Export Integration Test

  **What to do**:
  - Create `crates/core/tests/virtual_integration.rs` with end-to-end virtual dataset tests
  - Test the full virtual dataset pipeline:
    1. Create a base dataset with known data (images + annotations)
    2. Create a virtual dataset with filter transform (e.g., only class "dog")
    3. Verify virtual dataset iteration only yields filtered items
    4. Chain a second transform (e.g., sample 50%)
    5. Verify combined pipeline works correctly
    6. Materialize the virtual dataset into a real dataset
    7. Verify materialized dataset has correct image count and annotations
    8. Export materialized dataset to YOLO format
    9. Verify exported files match expected filtered+sampled subset
  - Test schema evolution via virtual dataset:
    1. Create dataset with schema v1
    2. Create virtual dataset with transform that adds a new field (schema v2)
    3. Materialize — verify new schema is applied
    4. Verify old fields preserved, new field populated
  - Test error cases:
    - Materialize with conflicting dataset name → error
    - Virtual dataset referencing deleted base dataset → graceful error
    - Circular virtual dataset references → detected and rejected

  **Must NOT do**:
  - Do NOT test Python-based transforms (those depend on PyO3)
  - Do NOT test with large datasets — keep fixtures small (10-20 images)
  - Do NOT add new transform types — only test existing transform pipeline

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Virtual dataset pipeline testing requires deep understanding of lazy evaluation, transform composition, and materialization semantics. Complex multi-step test scenarios.
  - **Skills**: []
    - No specialized skills needed — pure Rust testing
  - **Skills Evaluated but Omitted**:
    - `webapp-testing`: Not applicable — data pipeline tests, not web UI

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 7 (with Tasks 37, 38)
  - **Blocks**: F1-F4 (Final Verification)
  - **Blocked By**: T17 (virtual dataset engine), T18 (schema evolution), T19 (materialize), T21 (data transforms), T15 (format export)

  **References**:

  **Pattern References** (existing code to follow):
  - `crates/core/src/virtual_dataset/mod.rs` — Virtual dataset engine (pipeline composition API)
  - `crates/core/src/virtual_dataset/transforms.rs` — Available transforms (filter, sample, map)
  - `crates/core/src/virtual_dataset/materialize.rs` — Materialization logic

  **API/Type References** (contracts to test against):
  - `crates/core/src/types.rs:VirtualDataset` — Virtual dataset type (creation API, pipeline config)
  - `crates/core/src/types.rs:Transform` — Transform enum (available transform variants)
  - `crates/core/src/types.rs:Schema` — Schema type (evolution fields)

  **Test References** (testing patterns to follow):
  - Unit tests in `crates/core/src/virtual_dataset/` — Per-transform unit tests (setup patterns)
  - `crates/core/tests/roundtrip/` — Integration test patterns from T38 (fixture management)

  **External References**:
  - No external references needed — all internal APIs

  **WHY Each Reference Matters**:
  - Virtual dataset module shows how pipelines are composed (chaining API)
  - Transform module lists available transforms and their config (test inputs)
  - Materialize module shows the conversion API (expected behavior of `materialize()`)
  - Schema type shows evolution fields (what changes between v1 and v2)

  **Acceptance Criteria**:

  **Tests (TDD):**
  - [ ] Test file created: `crates/core/tests/virtual_integration.rs`
  - [ ] `cargo test -p dman-core --test virtual_integration` → PASS (≥8 tests, 0 failures)

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Virtual dataset filter → materialize → export pipeline
    Tool: Bash
    Preconditions: Base dataset with 20 images (10 "dog", 10 "cat") created in temp dir
    Steps:
      1. Run `cargo test -p dman-core --test virtual_integration -- test_filter_materialize_export 2>&1`
      2. Assert exit code = 0
      3. Assert test verifies: virtual dataset iteration yields exactly 10 "dog" images
      4. Assert test verifies: materialized dataset has exactly 10 images
      5. Assert test verifies: exported YOLO directory has 10 .txt label files
    Expected Result: Full pipeline produces correct filtered subset at every stage
    Failure Indicators: Wrong image count at any stage, missing annotations, export directory empty
    Evidence: .sisyphus/evidence/task-39-vds-pipeline.txt

  Scenario: Schema evolution via virtual dataset transform + materialize
    Tool: Bash
    Preconditions: Dataset with schema v1 (fields: class, bbox) created
    Steps:
      1. Run `cargo test -p dman-core --test virtual_integration -- test_schema_evolution 2>&1`
      2. Assert exit code = 0
      3. Assert test verifies: virtual dataset has schema with new field added
      4. Assert test verifies: materialized dataset schema includes new field
      5. Assert test verifies: old field values preserved in materialized data
    Expected Result: Schema evolution through virtual dataset produces valid v2 schema with all data preserved
    Failure Indicators: Schema mismatch, missing old fields, new field not populated
    Evidence: .sisyphus/evidence/task-39-schema-evolution.txt

  Scenario: Error handling — circular virtual dataset reference
    Tool: Bash
    Preconditions: Two virtual datasets set up to reference each other
    Steps:
      1. Run `cargo test -p dman-core --test virtual_integration -- test_circular_reference_error 2>&1`
      2. Assert exit code = 0
      3. Assert test verifies: creating circular reference returns Err with descriptive message
      4. Assert test verifies: error message contains "circular" or "cycle"
    Expected Result: Circular references detected and rejected with clear error
    Failure Indicators: Infinite loop, stack overflow, panic instead of error
    Evidence: .sisyphus/evidence/task-39-circular-error.txt
  ```

  **Commit**: YES
  - Message: `test(core): add virtual dataset pipeline integration tests with schema evolution`
  - Files: `crates/core/tests/virtual_integration.rs`
  - Pre-commit: `cargo test -p dman-core --test virtual_integration`

---

## Final Verification Wave

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace`. Review all changed files for: `unwrap()` in library code, empty catches, dead code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names.
  Output: `Build [PASS/FAIL] | Clippy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill for web UI)
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (import → virtual dataset → materialize → export). Test edge cases: empty datasets, corrupted files, concurrent access.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Wave | Commit | Message | Files |
|------|--------|---------|-------|
| 0 | T1 | `chore: scaffold cargo workspace with core/cli/tui/server/python crates` | Cargo.toml, crates/*/Cargo.toml, .gitignore, README.md |
| 0 | T2 | `feat(core): add error types and result aliases` | crates/core/src/error.rs |
| 0 | T3 | `feat(core): add SQLite database module with versioned migrations` | crates/core/src/db/ |
| 0 | T4 | `test(core): add test infrastructure and fixture datasets` | crates/core/tests/ |
| 0 | T5 | `feat(core): add core type definitions` | crates/core/src/types/ |
| 1 | T6 | `feat(core): implement dataset CRUD operations` | crates/core/src/dataset/ |
| 1 | T7 | `feat(core): implement TOML schema system` | crates/core/src/schema/ |
| 1 | T8 | `feat(core): implement file-based image storage manager` | crates/core/src/storage/ |
| 1 | T9 | `feat(core): implement global catalog service` | crates/core/src/catalog/ |
| 1 | T10 | `feat(core): add configuration system` | crates/core/src/config/ |
| 2 | T11 | `feat(cli): implement clap subcommands shell` | crates/cli/src/ |
| 2 | T12 | `feat(core): implement HuggingFace Parquet importer/exporter` | crates/core/src/formats/huggingface/ |
| 2 | T13 | `feat(core): implement YOLO format importer/exporter` | crates/core/src/formats/yolo/ |
| 2 | T14 | `feat(core): implement COCO format importer/exporter` | crates/core/src/formats/coco/ |
| 2 | T15 | `feat(core): add generic format trait and plugin interface` | crates/core/src/formats/mod.rs |
| 2 | T16 | `feat(core): add embeddings storage and metadata columns` | crates/core/src/embeddings/ |
| 3 | T17 | `feat(core): implement virtual dataset engine` | crates/core/src/virtual_dataset/ |
| 3 | T18 | `feat(core): implement schema evolution transforms` | crates/core/src/virtual_dataset/transforms/ |
| 3 | T19 | `feat(core): implement materialize command` | crates/core/src/virtual_dataset/materialize.rs |
| 3 | T20 | `feat(core): implement dataset CRUD operate commands` | crates/core/src/ops/ |
| 3 | T21 | `feat(core): implement data transform operations` | crates/core/src/ops/transforms/ |
| 4 | T22 | `feat(server): implement axum HTTP server with image serving` | crates/server/src/ |
| 4 | T23 | `feat(server): implement REST API endpoints` | crates/server/src/api/ |
| 4 | T24 | `feat(core): implement predictions storage and management` | crates/core/src/predictions/ |
| 4 | T25 | `feat(core): implement patch extraction and storage` | crates/core/src/patches/ |
| 5 | T26 | `feat(server): scaffold React SPA with build pipeline and rust-embed` | crates/server/frontend/, crates/server/src/spa.rs |
| 5 | T27 | `feat(frontend): implement gallery view with pagination and filtering` | crates/server/frontend/src/ |
| 5 | T28 | `feat(frontend): implement detail view with annotations overlay` | crates/server/frontend/src/ |
| 5 | T29 | `feat(tui): implement ratatui app shell and dataset browser` | crates/tui/src/ |
| 5 | T30 | `feat(tui): implement detail view and keyboard navigation` | crates/tui/src/ |
| 6 | T31 | `feat(python): add PyO3 integration with feature gate and plugin discovery` | crates/python/src/ |
| 6 | T32 | `feat(python): implement format converter plugin API` | crates/python/src/plugins/ |
| 6 | T33 | `feat(python): implement PyTorch Dataset and HF datasets loader` | crates/python/src/sdk/ |
| 6 | T34 | `feat(python): implement dataset builder/updater SDK` | crates/python/src/sdk/ |
| 6 | T35 | `feat(python): implement embeddings computation via plugins` | crates/python/src/embeddings/ |
| 6 | T36 | `feat(server): implement Label Studio API integration` | crates/server/src/label_studio/ |
| 7 | T37 | `test: add CLI end-to-end integration tests` | crates/cli/tests/ |
| 7 | T38 | `test: add cross-format round-trip tests` | crates/core/tests/ |
| 7 | T39 | `test: add virtual dataset integration tests` | crates/core/tests/ |

---

## Success Criteria

### Verification Commands
```bash
cargo test --workspace                                      # Expected: all tests pass
cargo clippy --workspace --all-targets -- -D warnings       # Expected: no warnings
cargo build --workspace --release                           # Expected: clean build
./target/release/dman init                                  # Expected: creates ~/.dman/
./target/release/dman import --format yolo ./test-data      # Expected: imports dataset
./target/release/dman list                                  # Expected: shows imported dataset
./target/release/dman virtual create --filter "class=dog"   # Expected: creates virtual dataset
./target/release/dman materialize <vds-name>                # Expected: materializes virtual dataset
./target/release/dman serve                                 # Expected: opens browser with gallery
./target/release/dman tui                                   # Expected: interactive browser
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All tests pass (`cargo test --workspace`)
- [ ] No clippy warnings
- [ ] Import/export round-trips work for YOLO, COCO, HuggingFace Parquet
- [ ] Virtual datasets compose and materialize correctly
- [ ] Web UI displays images and metadata
- [ ] TUI browses datasets interactively
- [ ] Python SDK loads datasets into PyTorch/HF
- [ ] Label Studio API integration imports/exports
