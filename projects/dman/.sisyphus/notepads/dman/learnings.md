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
