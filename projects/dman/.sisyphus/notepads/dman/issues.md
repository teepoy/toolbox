# dman — Issues

## Known Issues / Gotchas
- Stale `clippy = "0.0.302"` in root Cargo.toml — must be removed in T1
- edition = "2024" is correct for rustc 1.94.1 — keep it
- PyO3 must NOT compile in default build — feature-gated only
- SPA deep links must fall back to index.html (axum catch-all route)
## Task 4 fixture infra
- pyarrow was not available in the environment, so the HuggingFace parquet fixture was generated with a small temporary Rust helper using the workspace `parquet` dependency instead.

## F4 Scope Fidelity Check — Cross-Task Contamination (RESOLVED, non-blocking)
These are commit-hygiene observations; the code itself is correct and within overall spec:
1. T18 commit (2222a3f) bundled T20/T21/T22/T25 files (ops/mod.rs, ops/transforms.rs, patches/mod.rs, server/src/lib.rs, server/src/main.rs)
2. T19 commit (21695d8) bundled server/src/api.rs and tui/src/main.rs (T23/T29 files)
3. T22 commit (3ce168f) and T23 commit (0a6401f) had near-empty diffs — actual code landed in earlier commits
4. T35 commit (4ece0c7) modified crates/cli/src/main.rs and crates/cli/Cargo.toml (CLI territory)
5. T31 commit (fdb20e9) modified crates/tui/Cargo.toml (TUI territory, fixed pre-existing broken dep)

## F4 Scope Fidelity Check — Uncommitted Changes (2026-04-05)
- .sisyphus/boulder.json — orchestrator bookkeeping only, not code
- .sisyphus/plans/dman.md — T27 checkbox marked [x] by prior agent (plan-write violation, not by F4 audit)
- crates/python/src/plugins/format.rs — minor correctness refactors:
  * Removed 3 #[cfg(feature = "python")] import lines that are redundant (they are already inside mod python_impl which is cfg-gated)
  * Changed c_mod from "plugin" to path_str to fix Python module cache collision bug (cross-test contamination fix)
  * Removed explicit `: PyErr` type annotations on .map_err closures after .cast() calls (required by compiler: cast() returns CastError not PyErr)
  These changes are correctness fixes from T32 follow-up work, not new scope. All within T32 boundary.
