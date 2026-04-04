# dman — Issues

## Known Issues / Gotchas
- Stale `clippy = "0.0.302"` in root Cargo.toml — must be removed in T1
- edition = "2024" is correct for rustc 1.94.1 — keep it
- PyO3 must NOT compile in default build — feature-gated only
- SPA deep links must fall back to index.html (axum catch-all route)
## Task 4 fixture infra
- pyarrow was not available in the environment, so the HuggingFace parquet fixture was generated with a small temporary Rust helper using the workspace `parquet` dependency instead.
