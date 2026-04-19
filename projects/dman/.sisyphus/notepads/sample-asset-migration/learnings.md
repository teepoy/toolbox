# Learnings — sample-asset-migration

## [2026-04-05] Session ses_2a347fd2fffehN935qasR1YJiv

### Codebase patterns
- Rust workspace with crates: core, cli, server, python
- `rusqlite_migration` crate used for DB schema: `Migrations::new(vec![M::up("SQL...")])`
- WAL mode enabled on every SQLite connection — must preserve
- Error handling: `DmanError` enum in `crates/core/src/error.rs` with `thiserror` derive
- No `unwrap()` in library crates — use `expect()` in tests with specific messages
- Images stored as file paths (TEXT), never BLOBs
- `DatasetFormat` is string-backed, not a fixed enum
- CLI binary: `dman-cli`, user-facing command: `dman`
- PyO3 feature-gated behind `--features python`

### Current data model (being replaced)
- `Dataset → Image → Annotation` (image-centric)
- `images` table: id, dataset_id, file_name, file_path, width, height, hash, metadata
- `annotations.image_id NOT NULL REFERENCES images(id)`
- `embeddings.image_id NOT NULL REFERENCES images(id)`
- `predictions.image_id NOT NULL REFERENCES images(id)`
- `patches.image_id NOT NULL REFERENCES images(id)`
- Storage path: `{dataset_id}/images/`

### Target data model
- `Dataset → Sample → Asset → Annotation` (multi-modal)
- `Sample`: id, dataset_id, name, metadata, created_at
- `Asset`: id, sample_id, asset_type (TEXT), file_name, file_path, width, height, hash, metadata
- `AssetType` enum: Image, DepthMap, PointCloud, Text, Audio, Video, Mask, Other(String)
- `Annotation`: sample_id NOT NULL + optional asset_id with CHECK constraint
- Storage path: `{dataset_id}/assets/`

### Clean break strategy
- No backward compatibility migration — replace single M::up with new combined schema
- Simplest approach: single combined migration (avoids SQLite ALTER TABLE complexity)

## T1: DB Schema Migration — samples/assets schema

**Date**: 2026-04-05

### What changed
- Replaced `images` table with `samples` + `assets` tables
- `samples`: group of related assets, references `datasets`
- `assets`: individual files (images, video frames, etc), references `samples`
- `annotations` now has `sample_id NOT NULL` + `asset_id` (optional FK)
- `embeddings` and `patches` now reference `assets(id)` not `images(id)`
- `predictions` has `sample_id NOT NULL` + optional `asset_id`

### New indices
- `idx_samples_dataset_id`, `idx_assets_sample_id`, `idx_assets_hash`
- `idx_annotations_sample_id`, `idx_annotations_asset_id`
- `idx_categories_dataset_id`, `idx_embeddings_asset_id`

### Test approach
- `can_insert_and_read_dataset` extended to walk full FK chain: dataset → sample → asset → annotation (with and without asset_id)
- All 5 db:: tests pass

### Side effect
- `patches::tests::extract_db_record` fails because `crates/core/src/patches/mod.rs` still references the old `images` table — that module needs updating in a follow-on task.

### Patterns
- `rusqlite::params![]` needed when mixing types (i64, &str); array literal `["str", "str"]` works when all params are &str
- Single combined `M::up(...)` is the right pattern for a clean-break schema (no ALTER TABLE, no data migration)
## 2026-04-05

- `DmanError` variants are appended cleanly near related not-found cases; keep display strings exactly aligned with the existing lowercase/uppercase style used in each variant.
- Targeted `cargo test -p dman-core error` can still surface unrelated core test failures, so capture the full output as evidence even when the new display tests pass.

## T4: Storage layout update

- Storage paths now use `{dataset_id}/assets/` instead of `{dataset_id}/images/`.
- `check_integrity()` must query `assets JOIN samples` rather than the old `images` table.
- Storage tests that used to seed `images` now need a `samples` row first, then an `assets` row with `sample_id`.
- `cargo test -p dman-core storage` is currently blocked by unrelated compile errors from earlier `Image`→`Asset` type migration fallout outside `storage/mod.rs`.

## T3: New Rust Types (types/mod.rs)

### Completed changes
- `Image` struct removed entirely from `crates/core/src/types/mod.rs`
- `AssetType` enum added with `Display` + `FromStr` (infallible, unknown → `Other(String)`)
- `Sample` struct added: `id, dataset_id, name, metadata, created_at`
- `Asset` struct added: `id, sample_id, asset_type, file_name, file_path, width, height, hash, metadata`
- `Annotation`: `image_id: i64` → `sample_id: i64` + `asset_id: Option<i64>`
- `Embedding`: `image_id: i64` → `asset_id: i64`
- `Prediction`: `image_id: i64` → `sample_id: i64` + `asset_id: Option<i64>`
- `Patch`: `image_id: i64` → `asset_id: i64`
- Tests updated: removed `Image` construction, added `test_asset_type_display_and_fromstr`, `test_sample_serde_roundtrip`, `test_asset_serde_roundtrip`

### Downstream errors (expected, to fix in later tasks)
- `crates/core/src/lib.rs:18` — re-exports `Image` (need to remove, add `Sample`, `Asset`, `AssetType`)
- `crates/core/src/ops/transforms.rs:11` — imports `Image`
- `crates/core/src/virtual_dataset/transforms.rs:4` — imports `Image`
- `crates/core/src/virtual_dataset/mod.rs:10` — imports `Image`
- `crates/core/src/embeddings/mod.rs` — uses `image_id` field on `Embedding`
- `crates/core/src/patches/mod.rs` — uses `image_id` field on `Patch`
- `crates/core/src/predictions/mod.rs` — uses `image_id` field on `Prediction`
- `crates/core/src/virtual_dataset/materialize.rs` — calls `storage.get_image_path()`

### Key patterns
- `AssetType` uses `std::fmt::Display` (already imported as `fmt`) so `fmt::Display` is the impl form
- `FromStr` with `Err = std::convert::Infallible` means it never fails — unknown strings become `Other(s)`
- The `VirtualDatasetDef::Sample { ratio }` variant is a different `Sample` (not the new struct) — no conflict since it's an enum variant, not a type

## T5: Virtual Dataset Type Updates

**Date**: 2026-04-05

### What changed

- `crates/core/src/virtual_dataset/mod.rs`:
  - `row_to_image()` → `row_to_sample()`, SELECT from `samples` table (id, dataset_id, name, metadata, created_at)
  - `fetch_images_for_datasets()` → `fetch_samples_for_datasets()` returning `Vec<Sample>`
  - `filter_has_annotations()` queries `annotations.sample_id` (not `image_id`)
  - `filter_by_category()` queries `annotations.sample_id`
  - `filter_by_metadata()` / `filter_by_field()` operate on `sample.metadata` and `sample.name/id/dataset_id`
  - `get_image_field_value()` → `get_sample_field_value()`
  - `VirtualDatasetService::evaluate()` and `preview()` return `Vec<Sample>`
  - Internal tests: `insert_image()` → `insert_sample()`, annotations use `sample_id` column

- `crates/core/src/virtual_dataset/transforms.rs`:
  - `apply_to_images()` → `apply_to_samples()` with `&[Sample]` parameter
  - Constructs `Sample { id, dataset_id, name, metadata, created_at }` (not `Image`)
  - Test helpers `make_image_with_metadata()` → `make_sample_with_metadata()`

- `crates/core/src/virtual_dataset/materialize.rs`:
  - New `copy_assets()` function: copies asset rows from old `sample_id` to new `sample_id`
  - `copy_annotations()` uses `sample_id` instead of `image_id`
  - Main loop: `INSERT INTO samples (dataset_id, name, metadata, created_at)` instead of images
  - `storage_path_for_output()` calls `storage.get_asset_path()` (was `get_image_path()`)
  - Internal tests: `insert_image()` → `insert_sample()`, `count_images()` → `count_samples()`

- `crates/core/tests/virtual_integration.rs`:
  - `insert_image()` / `insert_image_with_meta()` → `insert_sample()` / `insert_sample_with_meta()`
  - `count_images()` → `count_samples()`
  - `count_annotations_for_dataset()` queries via `sample_id IN (SELECT id FROM samples ...)`
  - All test assertions use `s.name` (not `img.file_name`)
  - `DELETE FROM images` → `DELETE FROM samples`

### Key patterns

- `copy_assets()` is a new helper needed in materialize.rs because assets must be copied before annotations (FK chain: samples → assets → annotations)
- `row_to_sample()` uses indices 0–4: id, dataset_id, name, metadata (Option<String> → Option<Value>), created_at
- `VirtualDatasetDef::Sample { ratio }` enum variant has no name clash with `Sample` struct because Rust resolves them in different namespaces
- `materialize.rs` must insert sample first, then copy assets, then copy annotations (ordering matters for FK integrity)

### Compile status

- 0 LSP errors in all 4 modified files
- `cargo test -p dman-core virtual` cannot compile yet because other modules (patches, embeddings, predictions, ops/transforms, lib.rs) still reference the old `Image` type — those are T7/T8 scope

## T7: DatasetService + remaining core files rewrite

**Date**: 2026-04-05

### What changed

- `crates/core/src/lib.rs`: removed `Image` from `pub use types::{...}`, added `Asset`, `AssetType`, `Sample`
- `crates/core/src/ops/transforms.rs`: full rewrite from `Image`-centric to `Sample`/`Asset`-centric
  - `row_to_image()` → `row_to_sample()`, SELECT from `samples` table
  - `list_images()` → `list_samples()`
  - `filter_has_annotations()` / `filter_by_category()` use `annotations.sample_id`
  - `filter_by_metadata_eq()` / `get_sample_field_value()` operate on `Sample` struct fields
  - `insert_image_ref()` → `insert_sample_ref()` (INSERT INTO `samples`)
  - `copy_annotations()` uses `sample_id` column (not `image_id`)
  - `resize_images()` now queries `assets JOIN samples` for `file_path` filtered by `asset_type = 'image'`
  - All tests: `insert_image()` → `insert_sample()`, `count_images()` → `count_samples()`
  - Test SQL: `INSERT INTO samples`, `INSERT INTO annotations (sample_id, ...)`

- Previously completed in this session (see plan T7 full list):
  - `dataset/mod.rs`: `DatasetInfo` fields updated, full Sample/Asset CRUD added
  - `embeddings/mod.rs`: `image_id` → `asset_id` throughout
  - `patches/mod.rs`: `image_id` → `asset_id`, SQL updated with `assets` table
  - `predictions/mod.rs`: `image_id` → `sample_id`, `ImageComparison` → `SampleComparison`

### Verification

- `cargo check -p dman-core` → 0 errors ✓
- `cargo test -p dman-core dataset` → 66 unit tests + 1 integration test, all pass ✓
- Evidence: `.sisyphus/evidence/task-7-cargo-check.txt`, `.sisyphus/evidence/task-7-dataset-crud.txt`

### Remaining out-of-scope errors

- `crates/server/src/api.rs` still references old `Prediction` field `image_id` — T8/server scope

## T8: DatasetOps Rewrite (ops/mod.rs)

**Date**: 2026-04-05

### What changed

- `crates/core/src/ops/mod.rs`: complete rewrite from `images`-centric to `samples`/`assets`-centric
  - Removed `RawImage` struct → replaced with `RawSample` + `RawAsset` + `RawAnnotation`
  - Added helper functions: `fetch_samples()`, `fetch_assets()`, `fetch_annotations()`, `copy_sample()`
  - `copy_sample()` handles full FK chain: insert sample → insert assets (building old→new asset_id map) → insert annotations with mapped asset_id
  - `duplicate()`: loops `fetch_samples()` → `copy_sample()` for each
  - `merge()`: for each source dataset, `fetch_samples()` → `copy_sample()` each
  - `split()`: `fetch_samples()`, deterministic sort by `hash(id XOR seed)`, bucket assignment, `copy_sample()` per bucket
  - Test helpers: `insert_image()` removed → `insert_sample()` (inserts into `samples` + one `assets` row)
  - `count_images()` → `count_samples()` (queries `samples WHERE dataset_id = ?1`)
  - `count_annotations_for_dataset()` queries via `sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)`

### Key patterns

- `copy_sample()` builds `asset_id_map: HashMap<i64, i64>` to remap old `asset_id` FK on copied annotations
- annotations INSERT uses `sample_id` + optional `asset_id` (NULL if original had no asset_id or asset wasn't in the map)
- Split distributes whole samples to buckets — never splits a sample's assets across datasets
- Bucket assignment uses `(i as f64 + 0.5) / n as f64` fraction against cumulative ratio thresholds

### Verification

- `cargo check -p dman-core` → 0 errors ✓
- `cargo test -p dman-core ops` → 20 passed, 0 failed ✓ (was 4 failing before: duplicate_basic, merge_two, split_two_way, split_three_way)
- Evidence: `.sisyphus/evidence/task-8-ops-tests.txt`

## [2026-04-05] T12: YOLO adapter rewrite

### Task

Rewrite `crates/core/src/formats/yolo/mod.rs` (and all dependent format adapters/tests) to use the new `Dataset → Sample → Asset → Annotation` schema instead of the old `images` table.

### Files changed

- `crates/core/src/formats/yolo/mod.rs` — primary rewrite
- `crates/core/src/formats/coco/mod.rs` — collateral fix
- `crates/core/src/formats/huggingface/mod.rs` — collateral fix
- `crates/core/tests/roundtrip/yolo_coco.rs` — updated queries
- `crates/core/tests/roundtrip/parquet_yolo.rs` — updated queries
- `crates/core/tests/roundtrip/coco_parquet.rs` — updated queries

### Key patterns used

**Importer pattern:**
```rust
db.conn.execute("INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)", params![dataset_id, stem])?;
let sample_id = db.conn.last_insert_rowid();
let asset_type_str = AssetType::Image.to_string(); // = "image"
db.conn.execute("INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, ?2, ?3, ?4)", params![sample_id, asset_type_str, file_name, file_path])?;
let asset_id = db.conn.last_insert_rowid();
// Then INSERT INTO annotations (sample_id, asset_id, category_id, bbox)
```

**Exporter query pattern:**
```sql
SELECT a.id, s.id, a.file_name, a.file_path
FROM assets a JOIN samples s ON a.sample_id = s.id
WHERE s.dataset_id = ?1 ORDER BY a.id
```

**Annotation query pattern:**
```sql
SELECT category_id, bbox FROM annotations
WHERE sample_id = ?1 AND asset_id = ?2
```

**Test helper pattern:**
```sql
-- Old: SELECT COUNT(*) FROM images WHERE dataset_id = ?
-- New: SELECT COUNT(*) FROM samples WHERE dataset_id = ?
-- Old: image_id IN (SELECT id FROM images WHERE dataset_id = ?)
-- New: sample_id IN (SELECT id FROM samples WHERE dataset_id = ?)
```

### COCO-specific notes

- `img_id_map` now maps `coco_image_id → (sample_id, asset_id)` (was `coco_image_id → db_image_id`)
- Annotation INSERT uses both `sample_id` and `asset_id`
- Exporter `img_id_remap` maps `asset_id → export_id` (sequential integer for COCO JSON)

### YOLO bbox format preserved

Bbox stored as JSON: `{"x": x, "y": y, "w": w, "h": h, "normalized": true}` — unchanged per T25 standardization constraint.

### Verification

- `cargo test -p dman-core` → 235 tests (202 unit + 3 fixture + 18 roundtrip + 12 virtual), 0 failures ✓
- Evidence: `.sisyphus/evidence/task-12-yolo-tests.txt`

## T15: Multi-Asset Roundtrip Test Infrastructure

**Date**: 2026-04-05

### File created
- `crates/core/tests/multi_asset.rs` — 4 integration tests, all passing

### Test patterns confirmed

**register_dataset helper** uses `DatasetService::register(db, name, tmp.path(), DatasetFormat::yolo())` — path must exist, so always use `TempDir`.

**DatasetOps::duplicate** signature: `DatasetOps::duplicate(db, src_name, new_name) -> Result<Dataset>` — no StorageManager needed, pure DB operation.

**DatasetOps::split** signature: `DatasetOps::split(db, src_name, ratios: HashMap<String, f64>, seed: u64) -> Result<Vec<Dataset>>` — no StorageManager needed; output datasets named `{src_name}_{key}`.

**Hierarchy integrity after duplicate**: `copy_sample()` in `ops/mod.rs` builds `asset_id_map: HashMap<i64, i64>` to remap annotation FKs from old asset_ids to new asset_ids. Verified via `get_annotations_for_sample` + checking `ann.asset_id` is in the copied asset id set.

**Split integrity**: each sample's assets are copied atomically (via `copy_sample()`) — split distributes whole samples to buckets, so `asset_count == sample_count * assets_per_sample` holds for every partition.

**Dual-level annotation scoping**:
- `add_annotation(db, sample_id, None, ...)` → sample-level, `asset_id IS NULL`
- `add_annotation(db, sample_id, Some(asset_id), ...)` → asset-level
- `get_annotations_for_sample(db, sample_id)` returns both kinds (WHERE sample_id = ?)
- `get_annotations_for_asset(db, asset_id)` returns only asset-scoped ones (WHERE asset_id = ?)

### Verification
- `cargo test -p dman-core --test multi_asset` → 4 passed, 0 failed ✓
- Evidence: `.sisyphus/evidence/task-15-multi-asset-tests.txt`

## T20: Server API Update (api.rs)

**Date**: 2026-04-05

### What changed
- `crates/server/src/api.rs`: replaced all image-centric code with sample/asset equivalents
- `crates/server/src/lib.rs`: updated routes from `/images` → `/samples`, `/images/{id}` → `/assets/{id}`

### Compile fixes (4 errors)
1. `use dman_core::Image` → `use dman_core::{Asset, Category, Annotation, ...}` (no `Image`, no `Sample` needed)
2. `info.image_count` → `info.sample_count` + `info.asset_count`
3. `info.categories` → `info.category_count` (scalar, not a vec)
4. `Annotation.image_id` → `Annotation.sample_id` + `Annotation.asset_id`

### Function renames
- `row_to_image` → `row_to_asset` — reads from `assets` table columns (id, sample_id, asset_type, file_name, file_path, width, height, hash, metadata)
- `fetch_images_for_dataset` → `fetch_assets_for_dataset` — JOINs `assets JOIN samples` instead of `FROM images`
- `fetch_annotations_for_image` → `fetch_annotations_for_sample` — uses `sample_id` column
- `list_images` → `list_samples`, `get_image` → `get_asset`

### SQL patterns
- Assets query: `SELECT a.id, a.sample_id, a.asset_type, ... FROM assets a JOIN samples s ON s.id = a.sample_id WHERE s.dataset_id = ?1`
- Category-filtered assets: JOIN annotations on `ann.sample_id = s.id` (not `a.image_id`)
- Annotations fetch: `SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata FROM annotations WHERE sample_id = ?1`
- Column indices in `row_to_annotation`: bbox=4, seg=5, keypoints=6, metadata=7 (after id=0, sample_id=1, asset_id=2, category_id=3)

### Test updates
- `make_router`: routes updated to `/samples` and `/assets/{id}`
- Tests 4,5: `INSERT INTO images` → `INSERT INTO samples` + `INSERT INTO assets`
- Tests 4,5: `INSERT INTO annotations (image_id, ...)` → `INSERT INTO annotations (sample_id, ...)`
- Test 6: endpoint `/images` → `/samples`
- Test 8: stats assertions updated: `image_count`/`categories[]` → `sample_count`/`asset_count`/`category_count`
- Test 12: endpoint `/images/9999` → `/assets/9999`

### Test results
- 27/29 pass; 2 label_studio failures are pre-existing/T21 scope (no such table: images)
- All 13 `api::tests::*` pass

### Note on static file handler
- `/images/{dataset_id}/{filename}` route in `lib.rs` is a file-serving handler (not a dataset metadata API) — left unchanged, out of T20 scope

## T16: Python Plugin Contract Redesign

**Date**: 2026-04-05

### What changed
- `crates/python/src/plugins/format.rs`: complete rewrite
- Removed `PyImgData` and `PyAnnData` entirely
- Added three new PyO3 `#[pyclass]` structs: `PySampleData`, `PyAssetData`, `PyAnnotationData`
- Import contract: `import_dataset(path) → list[PySampleData]`
- Export contract: `export_dataset(samples: list[PySampleData], path: str) → None`
- BBox format: `{"x": f64, "y": f64, "width": f64, "height": f64}`
- `PythonFormatImporter::import()` calls `DatasetService` (register, add_sample, add_asset, add_annotation)
- `PythonFormatExporter::export()` calls `DatasetService` (get_samples, get_assets, get_annotations_for_sample, get_annotations_for_asset)
- Category names resolved via `category_name_by_id()` helper for export (avoids empty string placeholders)
- Transactional: BEGIN IMMEDIATE / COMMIT / ROLLBACK around DB writes in importer

### Key patterns

**PyO3 0.28 gotchas:**
- `PyObject` no longer exists — use `Py<PyAny>` everywhere
- `#[pyo3(signature = (...))]` required on `#[new]` when any param has a default
- Cannot chain `getattr(...)?. cast::<PyList>()` — the intermediate `Bound<'py, PyAny>` is a temporary; must split into two `let` bindings
- `serde_json::Value` does not implement `FromPyObject` — extract metadata as Python repr string instead

**DB helpers:**
- `get_or_create_category(db, dataset_id, name) → Result<i64>`: INSERT OR IGNORE + SELECT
- `category_name_by_id(db, category_id) → Result<String>`: SELECT name FROM categories WHERE id = ?

**FormatImporter pattern:**
1. Call `with_plugin_module(plugin_path, |_py, module| { module.call_method1("import_dataset", ...) })`
2. Parse all `PySampleData` / `PyAssetData` / `PyAnnotationData` fields inside the GIL closure into plain Rust structs
3. After the closure, open a transaction and write via `DatasetService`

**FormatExporter pattern:**
1. `DatasetService::get_samples` then `get_assets` then `get_annotations_for_sample/asset`
2. Build `PySampleData` objects inside `with_plugin_module` closure
3. Call `export_dataset(py_samples, output_path_str)`

### Verification
- `cargo check -p dman-python --features python` → 0 errors ✓
- `cargo test -p dman-python --features python` → all plugin format tests pass (test_import_inserts_samples_and_annotations ✓)
- 13 pre-existing sdk::builder/loader failures (no such table: images) are T17/T18 scope, not T16
- `grep PyImgData|PyAnnData crates/python/src/` → empty ✓
- Evidence: `.sisyphus/evidence/task-16-python-plugin-check.txt`

## [2026-04-05] T25: BBox Serialization Check
- `crates/core/src/formats/coco/mod.rs` previously emitted bbox JSON with `w`/`h`; updated to `width`/`height` for consistency with `BBox` serialization.
- `crates/core/src/formats/yolo/mod.rs` previously wrote and read stored bbox JSON using `w`/`h`; updated to `width`/`height` while leaving YOLO text label format unchanged.
- Post-fix grep checks over `crates/python/src/` and `crates/core/src/` returned zero `"w":` / `"h":` matches for BBox JSON.

## [2026-04-05] T18: Python SDK Loader Rewrite

- Replaced `ImageRow` with `SampleRow` + `AssetRow` structs in `crates/python/src/sdk/loader.rs`
- `AnnotationRow` now has `sample_id`, `asset_id: Option<i64>`, `segmentation`, `keypoints` fields
- `DmanDataset` struct now holds `sample_rows: Vec<SampleRow>` + `asset_rows: Vec<AssetRow>` (pre-loaded on construction)
- Added `samples()` → list of sample dicts with nested assets
- Added `get_sample(name)` → single sample dict or None
- Added `annotations(sample_name, asset_name=None)` → filtered annotations via `load_annotations_for_sample` / `load_annotations_for_asset`
- `images()` pymethods fn delegates to `pub(crate) image_paths()` so tests can call it from Rust test modules
- `__len__` / `__getitem__` updated to sample-level semantics
- Pre-existing `embeddings::tests::embeddings_compute_embeddings_processes_all_images` still fails (uses old `images` table) — outside T18 scope
- All 9 loader tests pass (36 total pass, 1 pre-existing failure in embeddings.rs)
- Key: `#[pymethods]` fn methods are Rust-private; expose a `pub(crate)` helper for test access

## [2026-04-05] T17: Python SDK Builder Rewrite

### What was done
- Rewrote `crates/python/src/sdk/builder.rs` completely:
  - Replaced `PendingImage` with `PendingSample` + `PendingAsset`
  - `DmanDatasetBuilder` now uses `pending_samples: Vec<PendingSample>` instead of `pending_images`
  - `build_inner` inserts into `samples`/`assets`/`annotations` tables (no `images` table)
  - Added `add_sample`, `add_asset`, `add_annotation` pymethods; kept `add_image` as backward-compat convenience
  - BBox uses `{"x", "y", "width", "height"}` keys via `BBox` struct from `dman_core::types`
  - `UpdateOp` variants updated: `AddSample`, `AddAsset`, `AddAnnotation`, `RemoveSample`
  - No `unwrap()` calls — all errors converted to `PyRuntimeError`

### Additional fix required
- `crates/python/src/embeddings.rs` also queried the old `images` table. Fixed:
  - Production query: `SELECT a.id, a.file_path FROM assets a JOIN samples s ON a.sample_id = s.id WHERE s.dataset_id = ?1 AND a.asset_type = 'image'`
  - Test fixture: now inserts into `samples` + `assets` (asset_type='image') instead of `images`

### Key insight
- `EmbeddingService::store` already accepted `asset_id` (not `image_id`) — only the caller (embeddings.rs) was still using the old query

### Result
- All 37 `dman-python` tests pass (including embeddings test)
- 0 references to `images` table in python crate production code
## [2026-04-05] T19: Python lib.rs + plugin_info Updates
Verified `crates/python/src/lib.rs` and `crates/python/src/plugin_info.rs` have no remaining `Image`/`PyImgData`/`PyAnnData` references. Confirmed the Python module registers `PySampleData`, `PyAssetData`, and `PyAnnotationData` with `m.add_class::<...>()?`, and `cargo check -p dman-python --features python` passes with warnings only.

## [2026-04-05] T21: Label Studio Integration Update

### Changes made
- `import_tasks_to_db`: replaced single `INSERT INTO images` with two inserts: `INSERT INTO samples (dataset_id, name)` (using file stem as name) + `INSERT INTO assets (sample_id, asset_type, file_name, file_path, width, height)` (asset_type = "image")
- Tracked both `sample_id` and `asset_id` from `last_insert_rowid()` after each insert
- `INSERT INTO annotations` updated from `(image_id, category_id, bbox)` to `(sample_id, asset_id, category_id, bbox)`
- `build_export_tasks`: replaced `SELECT file_name FROM images WHERE dataset_id = ?1` with `SELECT a.file_name FROM assets a JOIN samples s ON s.id = a.sample_id WHERE s.dataset_id = ?1 ORDER BY a.id`

### Test fixture updates
- `test_import_task_conversion`: `COUNT(*) FROM images` → `COUNT(*) FROM samples`; annotation count query now uses `sample_id IN (SELECT id FROM samples ...)`; file_name select now JOINs assets+samples
- `test_export_task_format`: two `INSERT INTO images` replaced with `INSERT INTO samples` + `INSERT INTO assets` pairs

### Key pattern
- Label Studio is 1:1 sample→asset (one image per task)
- `build_image_url` URL format (`/images/{dataset_id}/{filename}`) is a static file route, NOT a DB query — left unchanged
- File stem extraction from URL filename using `std::path::Path::new(&file_name).file_stem()` for sample name

### Result
All 29 dman-server tests pass (up from 27/29).

## [2026-04-05] T22: CLI Update

- `DatasetInfo.image_count` → `sample_count` + `asset_count` (two separate fields)
- `DatasetInfo.categories: Vec<Category>` → `category_count: u64` (scalar)
- `tui.rs` also had `DetailData` struct with `image_count`/`categories` fields — needed same migration even though task said T23 scope; compile errors were blocking the build
- `DetailData` in tui.rs updated: `image_count` → `sample_count` + `asset_count`, `categories: Vec<String>` → `category_count: u64`
- `draw_detail_categories` in tui.rs: replaced `data.categories.iter()` with scalar `data.category_count` display
- Integration tests in lifecycle.rs and schema.rs had no assertions on "Images:" label — no test changes needed
- All 32 CLI integration tests pass after changes

## [2026-04-05] T23: TUI tui.rs — Sample/Asset migration

### What changed
- `DetailData.images: Vec<(String, u64)>` → `assets: Vec<(String, String, u64)>` (file_name, asset_type, ann_count)
- `load_detail()` SQL rewrote FROM `images` (old) to `assets JOIN samples` with `LEFT JOIN annotations ON ann.asset_id = a.id`
- `draw_detail_images()` renamed to `draw_detail_assets()`; tuple destructuring updated to include `asset_type`
- Tab label `"2:Images"` → `"2:Assets"`
- Status bar `"Scroll (Images)"` → `"Scroll (Assets)"`
- Strings `"No images..."` / `"Loading images..."` / `" Images "` → assets equivalents
- Scroll saturation: `d.images.len()` → `d.assets.len()`
- Test fixtures: `images: vec![("a.jpg".to_string(), 1), ...]` → `assets: vec![("a.jpg".to_string(), "image".to_string(), 1), ...]`

### SQL pattern used
```sql
SELECT a.file_name, a.asset_type, COUNT(ann.id) as ann_count
FROM assets a
JOIN samples s ON a.sample_id = s.id
LEFT JOIN annotations ann ON ann.asset_id = a.id
WHERE s.dataset_id = ?1
GROUP BY a.id
ORDER BY a.file_name
```

### Verification
- `cargo check -p dman-cli` → 0 errors ✓
- `cargo test -p dman-cli --test integration` → 32 passed, 0 failed ✓
- Evidence: `.sisyphus/evidence/task-23-tui-check.txt`

## [2026-04-05] T24: Documentation Update

### What changed
- `README.md`: added `Dataset → Sample → Asset → Annotation` hierarchy explanation; added multi-view/multi-modal feature bullet; removed images-only wording
- `quickstart.md`: replaced `Images: 0` in inspect output with `Samples: 0 / Assets: 0`; `builder.add_image()` / `add_annotation(idx, ...)` pattern replaced with `add_sample()` / `add_asset()` / `add_annotation(sample_name, ...)`; `ds.images()` / `ds[0]` replaced with `ds.samples()` / `ds.get_sample()`; Rust comment "reads images + labels" → "reads image files as assets"; `update_dataset` section updated; plugin example uses `samples`/`assets` dict return; Supported Formats table updated
- `docs/src/features/catalog-and-cli.md`: added Data hierarchy section; added Inspect output block showing `Samples: 42 / Assets: 42`; catalog stores "samples, assets, annotations" not "images, annotations"
- `docs/src/features/format-registry.md`: Python plugin contract extended with `PySampleData`/`PyAssetData`/`PyAnnotationData` types and example return structure; description of multi-view multi-modal use case
- `docs/src/features/python-sdk.md`: full rewrite with new Data model section, `add_sample()`/`add_asset()`/`add_annotation()` builder API, `samples()`/`get_sample()`/`annotations()` loader API, `asset_type` table
- `docs/src/features/label-studio.md`: added Data model mapping section describing 1:1 Sample+Asset; "image URL" → "asset URL"; static file route clarification
- `docs/src/features/tui.md`: Tab table with `2:Assets`; "tabbed views for info, images, categories" → "info, assets, categories"

### Verification
- `grep -rn "image_count" docs/ README.md quickstart.md` → zero matches ✓
- `mdbook build` → INFO Book building has started → HTML book written → exit 0 ✓
- Evidence: `.sisyphus/evidence/task-24-no-stale-refs.txt`, `.sisyphus/evidence/task-24-mdbook-build.txt`

## [2026-04-05] T26: Final Cleanup Pass (cargo fmt / clippy / test)

### What was done
- `cargo fmt --all` — no formatting changes needed (exit 0)
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — all warnings fixed (exit 0)
- `cargo test` — all tests pass: 18 cli unit + 8 cli.rs + 32 integration + 202 core unit + 3 fixture + 4 multi_asset + 18 roundtrip + 12 virtual_integration + 37 python + 29 server = 363 total

### Clippy fixes applied (T26)
- `crates/core/src/dataset/mod.rs`: added `#[allow(clippy::too_many_arguments)]` to 3 DB CRUD helpers (`add_asset`, `add_sample_with_single_asset`, `add_annotation`) — these map directly to DB columns; arg restructuring would require 40+ call-site changes
- `crates/python/src/sdk/builder.rs`:
  - Removed dead struct fields: `metadata` from `PendingAsset`, `PendingSample`, `PendingAnnotation`, `UpdateOp::AddSample`, `UpdateOp::AddAsset`
  - Removed `schema_path` from `DmanDatasetBuilder` (stored but never read)
  - Removed `dataset_name` from `DmanDatasetUpdater` (stored but never read)
  - Removed `metadata` param from `add_annotation_internal` (unused, 8→7 args)
  - Fixed 3× `collapsible_if` → combined `&&` conditions
  - Added `#[allow(clippy::too_many_arguments)]` to pymethod `add_annotation` (public Python API, removing args would be a breaking change)
- `crates/python/src/sdk/loader.rs`: converted outer `///` doc to inner `//!` module doc; replaced `CStr::from_bytes_with_nul(b"...\0")` with `c"..."` literal
- `crates/python/src/plugins/format.rs`: added `type ImporterExporterParts = (...)` type alias for complex return type of `into_parts()`
- `crates/cli/src/main.rs`: changed `cmd_embed` param from `&PathBuf` to `&Path`; fixed `collapsible_if` in `build_format_registry`

### Test fixes (stale bbox key names)
Two roundtrip tests used stale `"w"`/`"h"` bbox JSON keys — fixed to `"width"`/`"height"` (the current storage format):
- `crates/core/src/formats/coco/mod.rs`: `test_import_bbox_stored_correctly` (fixed in earlier session)
- `crates/core/tests/roundtrip/yolo_coco.rs`: `yolo_import_coco_export_reimport_bbox_values_within_epsilon` (`for key in &["x", "y", "w", "h"]` → `["x", "y", "width", "height"]`)

### Key insight
When bbox JSON format changes (standardized to `width`/`height`), all tests that directly inspect stored bbox JSON keys must be updated to match. Tests that use numeric comparisons without key access are unaffected.

### Evidence
- `.sisyphus/evidence/task-26-clippy.txt` — clippy exit 0
- `.sisyphus/evidence/task-26-tests.txt` — all 363 tests pass

## [2026-04-05] Task-F: F-wave Reviewer Rejection Fixes

### Summary of fixes applied

**Fix 1 — DB: PRAGMA foreign_keys + UNIQUE constraint (`crates/core/src/db/mod.rs`)**
- Added `PRAGMA foreign_keys=ON` to `enable_wal()` so it runs on every connection open (not inside the migration SQL string)
- Added `UNIQUE (dataset_id, name)` constraint to `categories` table in migration SQL
- For existing DBs the UNIQUE constraint only applies to NEW databases (migration won't re-run with `rusqlite_migration`) — acceptable since tests use in-memory or temp DBs

**Fix 2 — JSON error propagation in row_to_* (`crates/core/src/dataset/mod.rs`)**
- Replaced all `.and_then(|s| serde_json::from_str(s).ok())` patterns with `.map(|s| serde_json::from_str(s).map_err(|e| rusqlite::Error::FromSqlConversionFailure(col, rusqlite::types::Type::Text, Box::new(e)))).transpose()?`
- Affected functions: `row_to_dataset`, `row_to_sample`, `row_to_asset`, `row_to_annotation`

**Fix 3 — COCO unknown image_id error propagation (`crates/core/src/formats/coco/mod.rs`)**
- Replaced `eprintln! + continue` for unknown `image_id` with proper `DmanError::ImportFailed` via `.ok_or_else(|| DmanError::ImportFailed { ... })?`
- `CocoAnnotation.image_id` field name was NOT renamed (it is the COCO JSON wire-format field)

**Fix 4 — Python plugin metadata serialization (`crates/python/src/plugins/format.rs`)**
- Replaced `__repr__`-based metadata serialization with Python's `json.dumps` called via PyO3
- Pattern: `ann_obj.py().import("json").and_then(|json_mod| json_mod.call_method1("dumps", (&meta_obj,))).and_then(|r| r.extract::<String>())`
- Also fixed two `.and_then(|s| serde_json::from_str(s).ok())` to use `.map(...).transpose()?`

**Fix 5 — Python SDK builder metadata serialization (`crates/python/src/sdk/builder.rs`)**
- Fixed all 7 occurrences of `d.to_string()` (Python dict repr) for metadata PyDict serialization
- All now use `d.py().import("json").and_then(|j| j.call_method1("dumps", (d.as_any(),))).and_then(|r| r.extract::<String>())`
- Affected pymethods: `add_sample`, `add_asset`, `add_image`, `add_annotation` in both `DmanDatasetBuilder` and `DmanDatasetUpdater`

**Fix 6 — Docs: add_asset() argument order (`docs/src/features/python-sdk.md`)**
- Fixed wrong `add_asset("scene_001", "/tmp/cat.jpg", asset_type="image")` → `add_asset("scene_001", "image", "/tmp/cat.jpg")`
- Same fix applied to the updater example

### Key PyO3 0.28 pattern for calling Python json.dumps from Rust
```rust
let json_str = d
    .py()
    .import("json")
    .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
    .and_then(|r| r.extract::<String>())
    .map_err(|e| PyRuntimeError::new_err(format!("json.dumps failed: {e}")))?;
```

### rusqlite JSON error wrapping pattern
```rust
let meta: Option<serde_json::Value> = row
    .get::<_, Option<String>>(col_idx)?
    .map(|s| {
        serde_json::from_str(&s).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                col_idx,
                rusqlite::types::Type::Text,
                Box::new(e),
            )
        })
    })
    .transpose()?;
```

### Verification
- `cargo fmt --all` → exit 0 (no changes)
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` → exit 0
- `cargo test` → 363 tests pass, 0 failed
- Evidence: `.sisyphus/evidence/task-F-clippy.txt`, `.sisyphus/evidence/task-F-tests.txt`
