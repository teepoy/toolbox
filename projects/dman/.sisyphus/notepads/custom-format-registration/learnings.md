## [2026-04-05] Plan initialized

### Key architectural facts (from sample-asset-migration)
- `DatasetFormat` is string-backed, not a fixed enum
- Built-in format IDs: `yolo`, `coco`, `huggingface`, `label-studio`
- Python plugins live under `$DMAN_HOME/plugins` (default: `~/.dman/plugins`)
- Discovery: scan `.py` files for `dman_plugin` marker with `type = "format"`
- Registry bootstrap: built-ins first, then Python plugins appended
- Images stored as file paths in SQLite, NEVER as bytes
- WAL mode + FK pragmas on every SQLite connection
- `FormatImporter` trait: `name()`, `detect()`, `import()`
- `FormatExporter` trait: `name()`, `export()`
- CLI uses `build_registry()` → explicit `--format` or `detect_format(path)` for import

### Current Python importer contract (from plan doc)
- Required: `import_dataset(path: str) -> dict`
- Optional: `detect(path: str) -> bool`
- Return shape: `{"images": [{"file_name": str}], "annotations": [{"image_file_name": str, "category": str, "bbox": [x,y,w,h]}]}`

### Key files
- `crates/core/src/formats/mod.rs` — FormatRegistry, traits
- `crates/core/src/types/mod.rs` — DatasetFormat
- `crates/python/src/plugins/format.rs` — PythonFormatImporter/Exporter, load_python_format_registry
- `crates/python/src/lib.rs` — PluginManager::discover()
- `crates/cli/src/main.rs` — build_registry(), CLI import/export dispatch
- `docs/src/features/` — mdBook source
- `docs/src/SUMMARY.md` — must be updated to include new page

## [2026-04-05] Actual importer/exporter contracts (from source inspection)

### CRITICAL: The plan doc is outdated re: import_dataset return type
The plan says import_dataset returns a dict like `{"images": [...], "annotations": [...]}`.
That was the PRE-migration contract. The CURRENT contract (post-migration) is:

**import_dataset(path: str) -> list[PySampleData]**
- Returns a list of PySampleData objects (NOT a dict)
- Each PySampleData has: name, metadata (opt), assets (list[PyAssetData]), annotations (list[PyAnnotationData])
- Each PyAssetData has: asset_type (str), file_name (str), file_path (str), width (opt int), height (opt int), metadata (opt), annotations (list[PyAnnotationData])
- Each PyAnnotationData has: category (str), bbox (opt dict {x,y,width,height}), segmentation (opt list[list[float]]), keypoints (opt list[float]), metadata (opt)
- bbox MUST be a dict with keys x, y, width, height (all floats) — NOT a 4-element list

### CRITICAL: Plugin classes are not imported from dman
Plugins define their own Python classes (PySampleData, PyAssetData, PyAnnotationData)
or return plain Python objects with the required attributes.
The Rust code reads attributes by name — it does NOT require import of dman classes.

### Exporter contract:
**export_dataset(samples: list[PySampleData], output_path: str) -> None**
- samples: list of PySampleData objects (same structure as importer output)
- Each PySampleData: .name, .metadata (None in current impl), .assets (list[PyAssetData]), .annotations (list[PyAnnotationData])
- Each PyAssetData: .asset_type (str), .file_name (str), .file_path (str, absolute), .width (int|None), .height (int|None), .metadata (None in current impl), .annotations (list[PyAnnotationData])
- Each PyAnnotationData: .category (str), .bbox (dict|None), .segmentation (list[list[float]]|None), .keypoints (list[float]|None), .metadata (None in current impl)

### Discovery (parse_plugin_marker is text-based, NOT Python execution):
- Scans file text for `dman_plugin` string
- Extracts name/type/version by simple string search for quoted fields
- Does NOT execute Python during discovery — safe and fast

### Registry bootstrap sequence (from CLI main.rs):
- build_registry() → FormatRegistry::default_registry() → loads YOLO/COCO/HuggingFace
- If python feature: load_python_format_registry(plugin_dirs) → appends Python providers
- detect_format(path) → tries importers in registration order, returns first match
- get_importer(format_id) → exact format ID match

## [2026-04-05] Phase 4-5: Runtime validation hardening + tests

### Validation additions in `crates/python/src/plugins/format.rs`

- `FormatImporter::import()`: capture `self.info.name` as `plugin_name` before entering the `with_plugin_module` closure, then call `module.getattr("import_dataset").map_err(|_| DmanError::PluginError(...))?` before the actual `call_method1`
- `FormatExporter::export()`: same pattern — capture `plugin_name`, then `module.getattr("export_dataset").map_err(...).map(|_| ())?` before calling `export_dataset`
  - The `.map(|_| ())?` is needed because `getattr` returns `Result<Bound<'_, PyAny>>` and type inference requires the discarded value to be made `()`

### Test patterns that work reliably

- `tempfile::NamedTempFile::with_suffix(".py")` + `writeln!` for single-function minimal plugins
- `tempfile::TempDir` + `std::fs::write` for multi-class plugins where Python class defs span many lines
- Use `{{` `}}` escaping inside `writeln!(... r#"..."#)` for literal `{` `}` in Python dict literals
- Use raw string literals `r#"..."#` with `std::fs::write` for multi-line plugins (no escaping needed)
- `dman_core::db::Database::open_in_memory()` works in all plugin tests for isolation
- `dman_core::dataset::DatasetService::register()` can be called directly in tests to set up export preconditions

### Gotchas

- The `with_plugin_module` closure signature is `|_py, module|` — if you need the plugin name in the error, capture it BEFORE the closure (not inside)
- `cargo test -p dman-python` (without `--features python`) runs 13 tests; `--features python` runs 42 tests
- All 5 new tests are gated with `#[cfg(feature = "python")]` so they only run with the feature enabled
