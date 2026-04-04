# dman — Decisions

## Architecture Decisions
- Cargo workspace (not monolith)
- SQLite for catalog — rusqlite with bundled feature
- WAL mode for concurrency
- File-based image storage (paths in DB)
- Virtual datasets stored in SQLite as JSON definition
- React SPA embedded via rust-embed
- PyO3 behind feature flag

## Convention Decisions
- Internal BBox representation: pixel coords (x, y, width, height), top-left origin
- YOLO conversion: normalized → pixel on import, pixel → normalized on export
- All public API returns Result<T, DmanError>
- No unwrap() in library crates
- Test DB: in-memory (":memory:")
