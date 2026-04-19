# Sample → Asset → Annotation Model Migration

## TL;DR

> **Quick Summary**: Replace dman's image-centric data model (`Dataset → Image → Annotation`) with a multi-modal `Dataset → Sample → Asset → Annotation` model that natively supports multi-view, multi-modal, and segmentation datasets. Clean break — no backward compatibility migration needed.
>
> **Deliverables**:
> - New DB schema with `samples` and `assets` tables, updated FKs across all related tables
> - New Rust types: `Sample`, `Asset`, `AssetType` enum
> - Updated `FormatImporter`/`FormatExporter` traits using sample/asset vocabulary
> - Rewritten YOLO, COCO, HuggingFace adapters (auto-creating 1:1 Sample→Asset for classic formats)
> - Redesigned Python plugin contract supporting multi-asset samples
> - Updated Python SDK builder/loader with sample/asset API
> - Updated CLI inspect/list output, TUI, Label Studio integration, HTTP API
> - Updated docs and mdBook
> - All existing tests updated; new tests for multi-asset scenarios
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: T1 (schema) → T3 (types) → T7 (DatasetService) → T10 (format traits) → T12 (YOLO adapter) → T18 (Python plugin) → T22 (CLI) → F1–F4

---

## Context

### Original Request
User wants dman to support multi-view (stereo cameras, surround-view), multi-modal (image+depth+lidar+text), and segmentation datasets natively. The current `Image`-centric model only supports single-image detection/classification workflows. The user explicitly requested a full model evolution rather than bolting on workarounds.

### Interview Summary
**Key Discussions**:
- **Backward compatibility**: Not needed — no actual users yet. Clean break preferred.
- **Annotation attachment**: Annotations attach to BOTH samples (sample-level labels) AND individual assets (per-view/per-modal annotations). `sample_id NOT NULL` + optional `asset_id`.
- **Python plugin contract**: Full redesign, not incremental extension. The current bbox-only `PyImgData`/`PyAnnData` structs are the biggest bottleneck.
- **Format adapter strategy**: Update adapters directly — no compatibility shim layer.

**Research Findings**:
- Current schema (db/mod.rs) uses single `rusqlite_migration` with all tables in one migration — new migration will add a second `M::up` entry.
- `FormatImporter`/`FormatExporter` traits pass `&Database` + `&StorageManager` — same pattern continues.
- `DatasetService::get_info()` returns `image_count`/`annotation_count` — must become `sample_count`/`asset_count`/`annotation_count`.
- `DatasetOps` (ops/mod.rs, 829 lines) iterates images/annotations for rename, duplicate, merge, split — all need sample/asset rewrites.
- Python plugin format.rs (646 lines) has `PyImgData`/`PyAnnData` — must become `PySampleData`/`PyAssetData`/`PyAnnotationData`.
- BBox JSON serialization inconsistency: format.rs uses `{x, y, width, height}`, builder.rs uses `{x, y, w, h}` — must fix to use consistent `{x, y, width, height}`.

### Metis Review
**Identified Gaps** (addressed):
- **Asset type representation**: Resolved → Rust `AssetType` enum serialized to TEXT column in SQLite
- **Annotation dual-level**: Resolved → `sample_id NOT NULL` + optional `asset_id` with CHECK constraint
- **Storage path layout**: Resolved → Rename `images/` to `assets/` in storage layout
- **Single-image ergonomics**: Resolved → Importers auto-create 1:1 Sample→Asset for classic single-image formats
- **Virtual dataset filters**: Resolved → Filter on samples (assets come along with their sample)
- **BBox JSON inconsistency**: Will fix as part of Python crate tasks (standardize to `{x, y, width, height}`)

---

## Work Objectives

### Core Objective
Replace the image-centric data model with a sample/asset/annotation hierarchy that supports multi-view, multi-modal, and segmentation datasets while maintaining all existing detection/classification functionality through automatic 1:1 sample→asset mapping.

### Concrete Deliverables
- `crates/core/src/db/mod.rs` — New migration adding `samples`, `assets` tables; dropping `images`; updating FKs in `annotations`, `embeddings`, `predictions`, `patches`
- `crates/core/src/types/mod.rs` — New `Sample`, `Asset`, `AssetType` types; update `Annotation`, `Embedding`, `Prediction`, `Patch`
- `crates/core/src/formats/mod.rs` — Updated `FormatImporter`/`FormatExporter` traits
- `crates/core/src/formats/{yolo,coco,huggingface}/mod.rs` — Rewritten adapters
- `crates/core/src/dataset/mod.rs` — Updated `DatasetService` with sample/asset operations
- `crates/core/src/ops/mod.rs` + `ops/transforms.rs` — Updated operations
- `crates/core/src/virtual_dataset/` — Updated evaluation to filter on samples
- `crates/core/src/storage/mod.rs` — Updated path layout (`assets/` instead of `images/`)
- `crates/python/src/plugins/format.rs` — Redesigned Python plugin contract
- `crates/python/src/sdk/builder.rs` — Updated SDK builder with sample/asset API
- `crates/python/src/sdk/loader.rs` — Updated SDK loader
- `crates/cli/src/main.rs` — Updated inspect/list output
- `crates/cli/src/tui.rs` — Updated TUI
- `crates/server/src/api.rs` + `label_studio/mod.rs` — Updated API and Label Studio integration
- `docs/src/` — Updated documentation

### Definition of Done
- [ ] `cargo test` passes with 0 failures across all crates
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes clean
- [ ] `cargo fmt --all` produces no changes
- [ ] Multi-asset import roundtrip test proves sample→asset hierarchy works
- [ ] Classic YOLO/COCO/HF import still works via automatic 1:1 Sample→Asset

### Must Have
- `Sample` type with `id`, `dataset_id`, `name`, `metadata`
- `Asset` type with `id`, `sample_id`, `asset_type` (TEXT from enum), `file_name`, `file_path`, `width`, `height`, `hash`, `metadata`
- `AssetType` enum: `Image`, `DepthMap`, `PointCloud`, `Text`, `Audio`, `Video`, `Mask`, `Other(String)`
- Annotation `sample_id NOT NULL` + optional `asset_id` with CHECK constraint
- Auto 1:1 Sample→Asset creation in YOLO/COCO/HF importers
- Python plugin contract that can express multi-asset samples
- Storage layout using `{dataset_id}/assets/` directory
- BBox serialization standardized to `{x, y, width, height}` everywhere

### Must NOT Have (Guardrails)
- No image BLOBs in SQLite — file paths only
- No `unwrap()` in library crates — `DmanError` throughout
- No backward-compatible migration layer (clean break confirmed)
- No compatibility shim between old and new format traits
- No hardcoded format enum — `DatasetFormat` remains string-backed
- No `as any` or `@ts-ignore` equivalents — proper Rust types
- No half-migrated code — every reference to `Image` or `image_id` must be fully replaced
- No stale documentation — every doc/mdbook page must reflect new model

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES — `cargo test` with existing test suite (343+ tests)
- **Automated tests**: YES (Tests-alongside) — each task includes tests as part of implementation
- **Framework**: `cargo test` (built-in Rust test harness)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Core crate**: Use `cargo test -p dman-core` + specific test filters
- **CLI crate**: Use `cargo test -p dman-cli --test integration` + specific scenarios
- **Python crate**: Use `cargo test -p dman-python` + `pip install . && python -c "..."` where feasible
- **Server crate**: Use `cargo test -p dman-server`
- **Compilation**: Use `cargo check`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — types, schema, error, storage):
├── Task 1: DB schema migration (new tables, drop images, update FKs) [deep]
├── Task 2: Error type additions [quick]
├── Task 3: New Rust types (Sample, Asset, AssetType, updated Annotation etc.) [deep]
├── Task 4: Storage layout update (images/ → assets/) [quick]
└── Task 5: Virtual dataset type updates [quick]

Wave 2 (Core services — depends on Wave 1):
├── Task 6: Category service update (no image dependency) [quick]
├── Task 7: DatasetService rewrite (sample/asset CRUD) [deep]
├── Task 8: DatasetOps rewrite (rename, duplicate, merge, split) [deep]
├── Task 9: Virtual dataset evaluation update [unspecified-high]
└── Task 10: Format trait evolution (FormatImporter/FormatExporter) [deep]

Wave 3 (Format adapters + schema — depends on Wave 2):
├── Task 11: Schema module update [unspecified-high]
├── Task 12: YOLO adapter rewrite [unspecified-high]
├── Task 13: COCO adapter rewrite [unspecified-high]
├── Task 14: HuggingFace adapter rewrite [unspecified-high]
└── Task 15: Multi-asset roundtrip test infrastructure [deep]

Wave 4 (Python + Server — depends on Wave 2/3):
├── Task 16: Python plugin contract redesign (format.rs) [deep]
├── Task 17: Python SDK builder rewrite [deep]
├── Task 18: Python SDK loader rewrite [unspecified-high]
├── Task 19: Python lib.rs + plugin_info updates [quick]
├── Task 20: Server API update [unspecified-high]
└── Task 21: Label Studio integration update [unspecified-high]

Wave 5 (CLI, TUI, docs — depends on Waves 2-4):
├── Task 22: CLI main.rs update (inspect, list, import, export) [unspecified-high]
├── Task 23: TUI update [unspecified-high]
├── Task 24: Documentation update (README, quickstart, mdBook) [writing]
├── Task 25: BBox serialization consistency fix [quick]
└── Task 26: Final clippy + fmt + full test pass [quick]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| T1 (DB schema) | — | T3, T7, T8 | 1 |
| T2 (Error types) | — | T7, T8, T10 | 1 |
| T3 (Rust types) | T1 | T7, T8, T10, T11-T15, T16-T18 | 1 |
| T4 (Storage layout) | — | T7, T12-T14 | 1 |
| T5 (VirtualDataset types) | T3 | T9 | 1 |
| T6 (Category svc) | T1, T3 | T12-T14 | 2 |
| T7 (DatasetService) | T1, T2, T3, T4 | T8, T10, T12-T14, T16, T22 | 2 |
| T8 (DatasetOps) | T1, T2, T3, T7 | T22 | 2 |
| T9 (Virtual eval) | T5, T7 | T22 | 2 |
| T10 (Format traits) | T2, T3, T7 | T12-T14, T16 | 2 |
| T11 (Schema module) | T3 | — | 3 |
| T12 (YOLO adapter) | T4, T6, T7, T10 | T15 | 3 |
| T13 (COCO adapter) | T4, T6, T7, T10 | T15 | 3 |
| T14 (HF adapter) | T4, T6, T7, T10 | T15 | 3 |
| T15 (Roundtrip tests) | T12, T13, T14 | — | 3 |
| T16 (Py plugin contract) | T3, T7, T10 | T17, T18, T19 | 4 |
| T17 (Py SDK builder) | T16 | — | 4 |
| T18 (Py SDK loader) | T16 | — | 4 |
| T19 (Py lib.rs) | T16 | — | 4 |
| T20 (Server API) | T7 | — | 4 |
| T21 (Label Studio) | T7, T20 | — | 4 |
| T22 (CLI) | T7, T8, T9, T10 | — | 5 |
| T23 (TUI) | T7, T22 | — | 5 |
| T24 (Docs) | T22 | — | 5 |
| T25 (BBox fix) | T16, T17 | — | 5 |
| T26 (Final check) | T1-T25 | — | 5 |

### Agent Dispatch Summary

- **Wave 1**: **5 tasks** — T1 → `deep`, T2 → `quick`, T3 → `deep`, T4 → `quick`, T5 → `quick`
- **Wave 2**: **5 tasks** — T6 → `quick`, T7 → `deep`, T8 → `deep`, T9 → `unspecified-high`, T10 → `deep`
- **Wave 3**: **5 tasks** — T11 → `unspecified-high`, T12 → `unspecified-high`, T13 → `unspecified-high`, T14 → `unspecified-high`, T15 → `deep`
- **Wave 4**: **6 tasks** — T16 → `deep`, T17 → `deep`, T18 → `unspecified-high`, T19 → `quick`, T20 → `unspecified-high`, T21 → `unspecified-high`
- **Wave 5**: **5 tasks** — T22 → `unspecified-high`, T23 → `unspecified-high`, T24 → `writing`, T25 → `quick`, T26 → `quick`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. DB Schema Migration — New `samples` and `assets` tables, update all FKs

  **What to do**:
  - Add a second `M::up(...)` entry to the `migrations()` function in `crates/core/src/db/mod.rs`
  - The migration must:
    - CREATE TABLE `samples` (`id` INTEGER PRIMARY KEY AUTOINCREMENT, `dataset_id` INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE, `name` TEXT NOT NULL, `metadata` TEXT, `created_at` TEXT DEFAULT CURRENT_TIMESTAMP)
    - CREATE TABLE `assets` (`id` INTEGER PRIMARY KEY AUTOINCREMENT, `sample_id` INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE, `asset_type` TEXT NOT NULL, `file_name` TEXT NOT NULL, `file_path` TEXT NOT NULL, `width` INTEGER, `height` INTEGER, `hash` TEXT, `metadata` TEXT)
    - DROP TABLE `images` (since no backward compat needed and the new migration runs from scratch for all users)
    - Update `annotations`: change `image_id` to `sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE` + add `asset_id INTEGER REFERENCES assets(id) ON DELETE SET NULL` + add CHECK constraint `CHECK(sample_id IS NOT NULL)`
    - Update `embeddings`: change `image_id` to `asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE`
    - Update `predictions`: change `image_id` to `sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE` + add optional `asset_id`
    - Update `patches`: change `image_id` to `asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE`
    - CREATE INDEX `idx_samples_dataset_id` ON samples(dataset_id)
    - CREATE INDEX `idx_assets_sample_id` ON assets(sample_id)
    - CREATE INDEX `idx_assets_hash` ON assets(hash)
    - CREATE INDEX `idx_annotations_sample_id` ON annotations(sample_id)
    - CREATE INDEX `idx_annotations_asset_id` ON annotations(asset_id)
    - CREATE INDEX `idx_embeddings_asset_id` ON embeddings(asset_id)
    - Remove old indices: `idx_images_dataset_id`, `idx_images_hash`, `idx_annotations_image_id`, `idx_embeddings_image_id`
  - IMPORTANT: Since there are no real users, the simplest approach is to REPLACE the single existing `M::up` migration with a new one containing all tables from scratch (samples/assets instead of images). This avoids complex ALTER TABLE logic in SQLite. The `rusqlite_migration` crate tracks migration versions — a single combined migration is cleaner.
  - Update the DB test `creates_expected_tables` to check for `samples`, `assets` instead of `images`
  - Update `creates_expected_indices` to check new indices
  - Update `can_insert_and_read_dataset` to also insert a sample+asset to verify FK chains

  **Must NOT do**:
  - Do NOT store image data as BLOBs
  - Do NOT add a backward-compatible migration path
  - Do NOT use `unwrap()` — use `expect()` only in tests with specific messages

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Schema migration is foundational — errors here cascade to every other task. Needs careful SQLite understanding.
  - **Skills**: []
    - No special skills needed — pure Rust + SQL
  - **Skills Evaluated but Omitted**:
    - None relevant

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T2, T4, T5; partially with T3 since T3 needs schema awareness)
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5)
  - **Blocks**: T3, T6, T7, T8
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `crates/core/src/db/mod.rs:6-78` — Current migration using `rusqlite_migration::Migrations` and `M::up`. Shows the exact pattern for defining schema. The entire existing migration will be REPLACED with the new schema.
  - `crates/core/src/db/mod.rs:86-112` — `Database::open`, `enable_wal`, `migrate` methods. WAL mode must remain enabled.

  **API/Type References**:
  - `crates/core/src/db/mod.rs:81-83` — `Database` struct with public `conn: Connection`. All DB access goes through this.

  **Test References**:
  - `crates/core/src/db/mod.rs:116-201` — All existing DB tests. These test table existence, index existence, insert/read, and WAL. All must be updated for the new schema.

  **External References**:
  - `rusqlite_migration` crate: `Migrations::new(vec![M::up("SQL...")])` pattern
  - SQLite CHECK constraints: `CHECK(sample_id IS NOT NULL)` syntax
  - SQLite foreign keys: `REFERENCES table(id) ON DELETE CASCADE`

  **WHY Each Reference Matters**:
  - `db/mod.rs:6-78`: This is THE file being rewritten. The executor must understand the existing migration pattern to replace it correctly.
  - `db/mod.rs:86-112`: The Database open/migrate flow must NOT change — only the SQL inside the migration changes.
  - `db/mod.rs:116-201`: Tests are the acceptance criteria. Every test must pass with the new schema.

  **Acceptance Criteria**:

  - [ ] `crates/core/src/db/mod.rs` contains new schema with `samples` and `assets` tables
  - [ ] No `images` table exists in schema
  - [ ] `annotations` table has `sample_id NOT NULL` and optional `asset_id`
  - [ ] `embeddings` table references `assets(id)` not `images(id)`
  - [ ] `predictions` table references `samples(id)` with optional `asset_id`
  - [ ] `patches` table references `assets(id)` not `images(id)`
  - [ ] `cargo test -p dman-core db::tests` → all pass

  **QA Scenarios**:

  ```
  Scenario: Database opens and creates all expected tables
    Tool: Bash (cargo test)
    Preconditions: Clean workspace, no existing DB
    Steps:
      1. Run: cargo test -p dman-core db::tests::creates_expected_tables
      2. Verify test passes — checks for tables: datasets, samples, assets, annotations, categories, embeddings, predictions, patches, virtual_datasets
    Expected Result: Test passes (0 failures)
    Failure Indicators: Test fails with "table 'X' should exist" assertion
    Evidence: .sisyphus/evidence/task-1-tables-exist.txt

  Scenario: FK chain works — insert dataset → sample → asset → annotation
    Tool: Bash (cargo test)
    Preconditions: In-memory DB
    Steps:
      1. Run: cargo test -p dman-core db::tests::can_insert_and_read_dataset
      2. Verify test inserts dataset, then sample referencing dataset, then asset referencing sample, then annotation referencing sample+asset
    Expected Result: All inserts succeed, FK constraints satisfied
    Failure Indicators: FOREIGN KEY constraint failure, or NULL violation
    Evidence: .sisyphus/evidence/task-1-fk-chain.txt

  Scenario: Cascade delete removes children
    Tool: Bash (cargo test)
    Preconditions: In-memory DB with dataset → sample → asset → annotation chain
    Steps:
      1. Delete the dataset row
      2. Verify samples, assets, annotations are all cascade-deleted
    Expected Result: All child rows removed
    Failure Indicators: Orphaned rows remain after parent deletion
    Evidence: .sisyphus/evidence/task-1-cascade-delete.txt
  ```

  **Commit**: YES (groups with T2, T3)
  - Message: `refactor(core): replace image model with sample/asset/annotation schema`
  - Files: `crates/core/src/db/mod.rs`
  - Pre-commit: `cargo test -p dman-core db`

- [x] 2. Error Type Additions

  **What to do**:
  - Add new error variants to `DmanError` in `crates/core/src/error.rs`:
    - `SampleNotFound(String)` — when a sample ID/name doesn't exist
    - `AssetNotFound(String)` — when an asset ID doesn't exist
    - `InvalidAssetType(String)` — when an unrecognized asset type string is encountered
  - Add corresponding tests for Display output

  **Must NOT do**:
  - Do NOT remove existing error variants — they're still used
  - Do NOT use `unwrap()` in any added code

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small additive change — 3 new enum variants + tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5)
  - **Blocks**: T7, T8, T10
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `crates/core/src/error.rs:5-57` — Existing `DmanError` enum. Follow exact same pattern: `#[error("...")]` attribute + variant. Examples: `DatasetNotFound(String)`, `FormatUnsupported(String)`.

  **Test References**:
  - `crates/core/src/error.rs:62-113` — Existing error tests. Follow same pattern: test Display output contains expected text.

  **WHY Each Reference Matters**:
  - `error.rs:5-57`: Must match existing derive macros (`Debug, Error`) and `#[error(...)]` format strings exactly.
  - `error.rs:62-113`: Test style must be consistent — `assert!(e.to_string().contains("..."))` pattern.

  **Acceptance Criteria**:
  - [ ] `DmanError::SampleNotFound`, `AssetNotFound`, `InvalidAssetType` variants exist
  - [ ] Each has `#[error("...")]` with descriptive message
  - [ ] Tests verify Display output
  - [ ] `cargo test -p dman-core error` → all pass

  **QA Scenarios**:

  ```
  Scenario: New error variants display correctly
    Tool: Bash (cargo test)
    Preconditions: Error variants added
    Steps:
      1. Run: cargo test -p dman-core error::tests
      2. Verify all error display tests pass
    Expected Result: 0 failures, new variants show correct messages
    Failure Indicators: Assertion failures on error message content
    Evidence: .sisyphus/evidence/task-2-error-display.txt
  ```

  **Commit**: YES (groups with T1, T3)
  - Message: `refactor(core): replace image model with sample/asset/annotation schema`
  - Files: `crates/core/src/error.rs`
  - Pre-commit: `cargo test -p dman-core error`

- [x] 3. New Rust Types — `Sample`, `Asset`, `AssetType`; Update Existing Types

  **What to do**:
  - In `crates/core/src/types/mod.rs`:
    - Add `AssetType` enum:
      ```rust
      #[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
      pub enum AssetType {
          Image,
          DepthMap,
          PointCloud,
          Text,
          Audio,
          Video,
          Mask,
          Other(String),
      }
      ```
      Implement `Display` (lowercase: "image", "depth_map", etc.) and `FromStr` / `From<&str>` for TEXT column serialization.
    - Add `Sample` struct:
      ```rust
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Sample {
          pub id: i64,
          pub dataset_id: i64,
          pub name: String,
          pub metadata: Option<serde_json::Value>,
          pub created_at: String,
      }
      ```
    - Add `Asset` struct:
      ```rust
      #[derive(Debug, Clone, Serialize, Deserialize)]
      pub struct Asset {
          pub id: i64,
          pub sample_id: i64,
          pub asset_type: AssetType,
          pub file_name: String,
          pub file_path: PathBuf,
          pub width: Option<u32>,
          pub height: Option<u32>,
          pub hash: Option<String>,
          pub metadata: Option<serde_json::Value>,
      }
      ```
    - Remove the `Image` struct entirely
    - Update `Annotation`: replace `image_id: i64` with `sample_id: i64` + `asset_id: Option<i64>`
    - Update `Embedding`: replace `image_id: i64` with `asset_id: i64`
    - Update `Prediction`: replace `image_id: i64` with `sample_id: i64` + `asset_id: Option<i64>`
    - Update `Patch`: replace `image_id: i64` with `asset_id: i64`
    - Update ALL tests in the module:
      - Remove `Image` serde test → add `Sample` and `Asset` serde tests
      - Update `Annotation` test to use `sample_id` + `asset_id`
      - Add `AssetType` roundtrip test (including `Other("custom")` variant)
      - Update `test_serde_roundtrip_all_types` to use new types

  **Must NOT do**:
  - Do NOT leave any reference to the `Image` struct
  - Do NOT change `DatasetFormat` — it stays string-backed
  - Do NOT change `BBox`, `Category`, `FilterOp`, `SchemaOp`, `VirtualDatasetDef`, `VirtualDataset` in this task

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core type changes affect the entire codebase. Must be precise about field names, derives, serialization.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: Partially (needs schema awareness from T1 for field names, but can start once schema design is agreed)
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5)
  - **Blocks**: T5, T6, T7, T8, T10, T11-T18
  - **Blocked By**: T1 (field names must match schema)

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs:84-94` — Current `Dataset` struct pattern. Shows derives: `Debug, Clone, Serialize, Deserialize`. Follow same pattern for `Sample` and `Asset`.
  - `crates/core/src/types/mod.rs:104-114` — Current `Image` struct. This is being REPLACED by `Asset`. Note the fields that carry over: `file_name`, `file_path`, `width`, `height`, `hash`, `metadata`.
  - `crates/core/src/types/mod.rs:116-125` — Current `Annotation` struct. Fields `image_id` → `sample_id` + `asset_id`.
  - `crates/core/src/types/mod.rs:6-8` — `DatasetFormat` transparent string pattern. `AssetType` uses a different approach (proper enum with `Display`/`FromStr`), NOT transparent string.

  **Test References**:
  - `crates/core/src/types/mod.rs:222-328` — All existing type tests. Every test referencing `Image` must be updated.

  **WHY Each Reference Matters**:
  - `types/mod.rs:84-94`: Derive pattern (`Debug, Clone, Serialize, Deserialize`) must be consistent across all new types.
  - `types/mod.rs:104-114`: This is the struct being replaced — executor needs to know exactly what fields migrate to `Asset`.
  - `types/mod.rs:116-125`: Annotation's FK change is the most delicate — `sample_id` (required) vs `asset_id` (optional).

  **Acceptance Criteria**:
  - [ ] `Sample`, `Asset`, `AssetType` types exist in types/mod.rs
  - [ ] `Image` struct is removed
  - [ ] `Annotation` has `sample_id: i64` and `asset_id: Option<i64>`
  - [ ] `Embedding` has `asset_id: i64` (not `image_id`)
  - [ ] `Prediction` has `sample_id: i64` and `asset_id: Option<i64>`
  - [ ] `Patch` has `asset_id: i64` (not `image_id`)
  - [ ] `AssetType::from_str("image")` → `AssetType::Image`; `AssetType::Image.to_string()` → `"image"`
  - [ ] `cargo test -p dman-core types` → all pass
  - [ ] `cargo check -p dman-core` → compiles (may have warnings from unused types until Wave 2)

  **QA Scenarios**:

  ```
  Scenario: AssetType roundtrip serialization
    Tool: Bash (cargo test)
    Preconditions: AssetType enum with Display/FromStr implemented
    Steps:
      1. Run: cargo test -p dman-core types::tests::test_asset_type_serde
      2. Verify Image, DepthMap, PointCloud, Text, Audio, Video, Mask, Other("lidar") all roundtrip correctly
    Expected Result: All variants serialize to lowercase TEXT and deserialize back
    Failure Indicators: Serde error or value mismatch
    Evidence: .sisyphus/evidence/task-3-asset-type-roundtrip.txt

  Scenario: No Image struct references remain
    Tool: Bash (grep)
    Preconditions: Task complete
    Steps:
      1. Run: grep -n "pub struct Image" crates/core/src/types/mod.rs
      2. Verify zero matches
    Expected Result: No matches — Image struct is fully removed
    Failure Indicators: grep returns any match
    Evidence: .sisyphus/evidence/task-3-no-image-struct.txt

  Scenario: Sample and Asset serde roundtrip
    Tool: Bash (cargo test)
    Preconditions: New types added
    Steps:
      1. Run: cargo test -p dman-core types::tests::test_serde_roundtrip_all_types
      2. Verify Sample, Asset, updated Annotation all serialize/deserialize correctly
    Expected Result: All roundtrips succeed
    Failure Indicators: Serde error, missing field
    Evidence: .sisyphus/evidence/task-3-serde-roundtrip.txt
  ```

  **Commit**: YES (groups with T1, T2)
  - Message: `refactor(core): replace image model with sample/asset/annotation schema`
  - Files: `crates/core/src/types/mod.rs`
  - Pre-commit: `cargo test -p dman-core types`

- [x] 4. Storage Layout Update — `images/` → `assets/`

  **What to do**:
  - In `crates/core/src/storage/mod.rs`:
    - Rename `store_image()` → `store_asset()`. Update the subdirectory from `"images"` to `"assets"` in both Copy and Symlink strategies (lines 46, 53).
    - Rename `get_image_path()` → `get_asset_path()`. Update subdirectory from `"images"` to `"assets"` (line 66).
    - Rename `delete_image()` → `delete_asset()`. No path changes needed (operates on stored_path directly).
    - Rename `delete_dataset_images()` → `delete_dataset_storage()`. This deletes the whole dataset directory — no path change needed.
    - Update `check_integrity()`: change SQL from `SELECT file_path FROM images WHERE dataset_id = ?` to `SELECT a.file_path FROM assets a JOIN samples s ON a.sample_id = s.id WHERE s.dataset_id = ?`. This now checks asset paths instead of image paths.
    - Update ALL tests:
      - `test_get_image_path` → `test_get_asset_path` — verify path contains `"assets"` not `"images"`
      - `test_store_image_copy` → `test_store_asset_copy`
      - `test_store_image_symlink` → `test_store_asset_symlink`
      - `test_store_image_reference` → `test_store_asset_reference`
      - `test_delete_image` → `test_delete_asset`
      - `test_delete_dataset_images` → `test_delete_dataset_storage`
      - `test_check_integrity*` tests — update to use samples/assets tables

  **Must NOT do**:
  - Do NOT store BLOBs
  - Do NOT change `StorageStrategy` enum — it stays as-is
  - Do NOT change the `calculate_hash()` method — it's file-agnostic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mostly renaming + path string changes. Low complexity.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5)
  - **Blocks**: T7, T12, T13, T14
  - **Blocked By**: None (can start immediately, but tests need T1 schema to pass DB queries)

  **References**:

  **Pattern References**:
  - `crates/core/src/storage/mod.rs:33-61` — `store_image()` method. The `"images"` subdirectory string at lines 46 and 53 must become `"assets"`.
  - `crates/core/src/storage/mod.rs:63-68` — `get_image_path()`. Path construction uses `"images"` at line 66.
  - `crates/core/src/storage/mod.rs:70-83` — `delete_image()` and `delete_dataset_images()`. Delete operates on path directly, no format-specific logic.
  - `crates/core/src/storage/mod.rs:104-132` — `check_integrity()`. SQL queries `images` table — must be rewritten to join `assets` and `samples`.

  **Test References**:
  - `crates/core/src/storage/mod.rs:135-354` — All storage tests. Every test name with `image` must be renamed to `asset`. Tests that insert into `images` table must insert into `samples` + `assets`.

  **WHY Each Reference Matters**:
  - `storage/mod.rs:33-61`: Core method being renamed + path updated. Executor must change both the method name and the `"images"` string literal.
  - `storage/mod.rs:104-132`: SQL query must be rewritten — this is the most complex change since it moves from single-table to join query.

  **Acceptance Criteria**:
  - [ ] No method or path referencing `"images"` directory remains in storage/mod.rs
  - [ ] `store_asset()`, `get_asset_path()`, `delete_asset()`, `delete_dataset_storage()` methods exist
  - [ ] `check_integrity()` queries `assets` joined with `samples`
  - [ ] `cargo test -p dman-core storage` → all pass

  **QA Scenarios**:

  ```
  Scenario: Asset stored in correct directory
    Tool: Bash (cargo test)
    Preconditions: Storage module updated
    Steps:
      1. Run: cargo test -p dman-core storage::tests::test_store_asset_copy
      2. Verify the test file is stored under {base}/{dataset_id}/assets/ not {base}/{dataset_id}/images/
    Expected Result: Test passes, file exists at assets/ path
    Failure Indicators: File stored under images/ or test assertion fails
    Evidence: .sisyphus/evidence/task-4-store-asset-path.txt

  Scenario: No "images" string literal in storage paths
    Tool: Bash (grep)
    Preconditions: Task complete
    Steps:
      1. Run: grep -n '"images"' crates/core/src/storage/mod.rs
      2. Verify zero matches in non-test code (test fixtures may reference old fixture paths)
    Expected Result: No matches for "images" as a directory name in runtime code
    Failure Indicators: grep returns matches
    Evidence: .sisyphus/evidence/task-4-no-images-string.txt
  ```

  **Commit**: YES (groups with T5)
  - Message: `refactor(core): update storage layout and virtual dataset types for sample model`
  - Files: `crates/core/src/storage/mod.rs`
  - Pre-commit: `cargo test -p dman-core storage`

- [x] 5. Virtual Dataset Type Updates

  **What to do**:
  - In `crates/core/src/types/mod.rs`: The `VirtualDatasetDef::Filter` variant currently filters on generic columns. No structural change needed here — filters will target sample-level columns. The `VirtualDatasetDef::Sample` variant (ratio-based sampling) will sample at the sample level, not image level. No type changes needed since these are runtime semantics, not type changes.
  - In `crates/core/src/virtual_dataset/mod.rs`:
    - Replace `row_to_image()` helper with `row_to_sample()` that constructs a `Sample` instead of `Image`
    - Update all SQL queries that reference `images` table to reference `samples` and `assets` tables
    - The `evaluate()` function should return `Vec<Sample>` (with assets accessible via separate queries) instead of `Vec<Image>`
    - Update `evaluate_filter()`, `evaluate_merge()`, `evaluate_sample_ratio()`, `evaluate_split()` to work on samples
  - In `crates/core/src/virtual_dataset/transforms.rs`: Update any `Image` references to `Sample`/`Asset`
  - In `crates/core/src/virtual_dataset/materialize.rs`: Update materialization to copy samples+assets instead of images

  **Must NOT do**:
  - Do NOT change `VirtualDatasetDef` or `VirtualDataset` type structures — only change the runtime evaluation
  - Do NOT add asset-level filtering (design decision: filter on samples, assets come along)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mostly search-and-replace of `Image` → `Sample` and SQL query updates. The logic flow stays the same.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: Partially
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4)
  - **Blocks**: T9
  - **Blocked By**: T3 (needs `Sample` type), T1 (needs new schema for SQL)

  **References**:

  **Pattern References**:
  - `crates/core/src/virtual_dataset/mod.rs:17-40` — `row_to_image()` helper. Must become `row_to_sample()` constructing `Sample` type.
  - `crates/core/src/virtual_dataset/mod.rs:1-11` — Imports `Image` from types. Must change to `Sample` (and possibly `Asset`).

  **API/Type References**:
  - `crates/core/src/types/mod.rs` — New `Sample` struct (from T3). The `row_to_sample()` must construct this type.

  **Test References**:
  - `crates/core/tests/virtual_integration.rs` — Integration tests for virtual datasets. These insert images — must insert samples+assets instead.

  **WHY Each Reference Matters**:
  - `virtual_dataset/mod.rs:17-40`: This is the row mapper being rewritten. Executor must know the current structure to replace it correctly.
  - `virtual_dataset/mod.rs:1-11`: Import changes are needed for compilation. Missing imports cause hard-to-diagnose errors.

  **Acceptance Criteria**:
  - [ ] No `Image` type references in `virtual_dataset/` directory
  - [ ] `evaluate()` returns sample-level results
  - [ ] SQL queries reference `samples`/`assets` tables, not `images`
  - [ ] `cargo test -p dman-core virtual` → all pass

  **QA Scenarios**:

  ```
  Scenario: Virtual dataset evaluate returns samples
    Tool: Bash (cargo test)
    Preconditions: Virtual dataset module updated
    Steps:
      1. Run: cargo test -p dman-core --test virtual_integration
      2. Verify all virtual dataset tests pass with sample-based evaluation
    Expected Result: All integration tests pass
    Failure Indicators: Type mismatch errors, SQL errors, assertion failures
    Evidence: .sisyphus/evidence/task-5-virtual-integration.txt

  Scenario: No Image references in virtual_dataset/
    Tool: Bash (grep)
    Preconditions: Task complete
    Steps:
      1. Run: grep -rn "Image" crates/core/src/virtual_dataset/
      2. Verify zero matches for the `Image` type (comments mentioning "image" conceptually are okay)
    Expected Result: No `Image` struct references
    Failure Indicators: grep returns struct/type references to Image
    Evidence: .sisyphus/evidence/task-5-no-image-refs.txt
  ```

  **Commit**: YES (groups with T4)
  - Message: `refactor(core): update storage layout and virtual dataset types for sample model`
  - Files: `crates/core/src/virtual_dataset/mod.rs`, `crates/core/src/virtual_dataset/transforms.rs`, `crates/core/src/virtual_dataset/materialize.rs`
  - Pre-commit: `cargo test -p dman-core virtual`

- [x] 6. Category Service Update

  **What to do**:
  - Categories are already dataset-scoped (not image-scoped), so the `categories` table and `Category` type need NO structural changes.
  - Review `crates/core/src/dataset/mod.rs` for any category-related methods that reference `images` — update them.
  - The category insert/query pattern in format adapters will continue to work as-is. This task is a verification pass, not a rewrite.
  - If any category query joins to `images`, update it to join through `samples`/`assets` or remove the join if unnecessary.

  **Must NOT do**:
  - Do NOT change the `Category` struct
  - Do NOT change the `categories` table schema

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification pass with minimal changes. Categories are already dataset-scoped.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10)
  - **Blocks**: T12, T13, T14
  - **Blocked By**: T1, T3

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs` — Search for `category` in all methods. Categories are inserted during import and queried during export.
  - `crates/core/src/types/mod.rs:127-133` — `Category` struct. Note it references `dataset_id` not `image_id` — already correct.

  **WHY Each Reference Matters**:
  - `dataset/mod.rs`: Must verify no category method indirectly references `images` table through a JOIN or subquery.

  **Acceptance Criteria**:
  - [ ] No category-related code references `images` table or `Image` type
  - [ ] `cargo check -p dman-core` compiles cleanly

  **QA Scenarios**:

  ```
  Scenario: Category queries work with new schema
    Tool: Bash (cargo check)
    Preconditions: Schema updated (T1), types updated (T3)
    Steps:
      1. Run: cargo check -p dman-core
      2. Verify no compilation errors related to categories
    Expected Result: Clean compilation
    Failure Indicators: Type errors mentioning Category or images
    Evidence: .sisyphus/evidence/task-6-category-check.txt
  ```

  **Commit**: YES (groups with T7)
  - Message: `refactor(core): rewrite DatasetService for sample/asset CRUD`
  - Files: (minimal changes, if any)
  - Pre-commit: `cargo check -p dman-core`

- [x] 7. DatasetService Rewrite — Sample/Asset CRUD

  **What to do**:
  - This is a MAJOR rewrite of `crates/core/src/dataset/mod.rs` (currently 571 lines).
  - Replace ALL image-centric CRUD with sample/asset-centric operations:
    - `add_image()` → split into `add_sample()` and `add_asset()` (asset references a sample)
    - `get_images()` → `get_samples()` (returns `Vec<Sample>`) and `get_assets()` (returns `Vec<Asset>` for a sample or dataset)
    - `get_image_count()` → `get_sample_count()` + `get_asset_count()`
    - `get_annotation_count()` — update SQL to reference `sample_id` instead of `image_id`
    - `get_annotations_for_image()` → `get_annotations_for_sample()` + `get_annotations_for_asset()`
    - `remove_image()` → `remove_sample()` (cascade deletes assets) + `remove_asset()`
  - Update `DatasetInfo` struct:
    - Replace `image_count` → `sample_count` + `asset_count`
    - Keep `annotation_count`, `category_count`
  - Update `get_info()` to populate new counts
  - Update row helper functions: `row_to_image()` → `row_to_sample()` + `row_to_asset()`
  - Add convenience method: `add_sample_with_single_asset()` — creates a sample and one asset in one call (used by classic format importers for 1:1 mapping)
  - Update ALL tests in this module

  **Must NOT do**:
  - Do NOT use `unwrap()` — use `DmanError` variants
  - Do NOT leave any `Image` type references
  - Do NOT change `register()`, `remove()`, `list()`, `get()` (dataset-level ops) — only image→sample/asset operations

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Core service rewrite touching ~400+ lines of SQL + Rust. Must be careful with FK references and query correctness.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T8, T9, T10 if sequenced correctly)
  - **Parallel Group**: Wave 2
  - **Blocks**: T8, T10, T12-T14, T16, T20, T22
  - **Blocked By**: T1, T2, T3, T4

  **References**:

  **Pattern References**:
  - `crates/core/src/dataset/mod.rs:1-571` — ENTIRE file is being rewritten for image→sample/asset. Key methods:
    - Lines ~50-100: `register()`, `remove()`, `list()`, `get()` — KEEP these mostly intact (dataset-level)
    - Lines ~100-200: `add_image()`, `get_images()`, row helpers — REPLACE with sample/asset equivalents
    - Lines ~200-300: `get_annotations_for_image()`, `add_annotation()` — UPDATE FK references
    - Lines ~300-400: `DatasetInfo`, `get_info()` — UPDATE counts
    - Lines ~400-571: Tests — UPDATE all

  **API/Type References**:
  - `crates/core/src/types/mod.rs` — New `Sample`, `Asset`, `AssetType` types (from T3)
  - `crates/core/src/error.rs` — New `SampleNotFound`, `AssetNotFound` errors (from T2)
  - `crates/core/src/db/mod.rs` — New schema (from T1)

  **WHY Each Reference Matters**:
  - `dataset/mod.rs`: This IS the file being rewritten. Every method must be examined.
  - Types: The new `Sample`/`Asset` structs define what the row helpers must construct.
  - Errors: `SampleNotFound`/`AssetNotFound` are used in the new get/remove methods.

  **Acceptance Criteria**:
  - [ ] `add_sample()`, `add_asset()`, `add_sample_with_single_asset()` exist
  - [ ] `get_samples()`, `get_assets()` return correct types
  - [ ] `get_sample_count()`, `get_asset_count()` return correct counts
  - [ ] `DatasetInfo` has `sample_count` and `asset_count` (not `image_count`)
  - [ ] No `Image` type reference in dataset/mod.rs
  - [ ] `cargo test -p dman-core dataset` → all pass

  **QA Scenarios**:

  ```
  Scenario: Add sample with single asset and retrieve
    Tool: Bash (cargo test)
    Preconditions: Schema (T1), types (T3) in place
    Steps:
      1. Run: cargo test -p dman-core dataset::tests
      2. Verify test creates dataset → adds sample with single asset → retrieves sample → retrieves asset → counts match
    Expected Result: All CRUD operations work, counts are correct
    Failure Indicators: SQL error, type mismatch, wrong counts
    Evidence: .sisyphus/evidence/task-7-dataset-crud.txt

  Scenario: DatasetInfo returns correct sample/asset/annotation counts
    Tool: Bash (cargo test)
    Preconditions: Dataset with samples, assets, annotations inserted
    Steps:
      1. Run: cargo test -p dman-core dataset::tests (filter for info test)
      2. Verify get_info() returns correct sample_count, asset_count, annotation_count
    Expected Result: Counts match inserted data
    Failure Indicators: Wrong count values
    Evidence: .sisyphus/evidence/task-7-dataset-info.txt

  Scenario: Multi-asset sample — one sample with two assets
    Tool: Bash (cargo test)
    Preconditions: Service methods exist
    Steps:
      1. Create dataset, add sample, add 2 assets to that sample
      2. get_assets(sample_id) returns 2 assets
      3. get_sample_count() returns 1, get_asset_count() returns 2
    Expected Result: Multi-asset sample works correctly
    Failure Indicators: Only 1 asset returned, or count mismatch
    Evidence: .sisyphus/evidence/task-7-multi-asset.txt
  ```

  **Commit**: YES (groups with T6)
  - Message: `refactor(core): rewrite DatasetService for sample/asset CRUD`
  - Files: `crates/core/src/dataset/mod.rs`
  - Pre-commit: `cargo test -p dman-core dataset`

- [x] 8. DatasetOps Rewrite — rename, duplicate, merge, split

  **What to do**:
  - Rewrite `crates/core/src/ops/mod.rs` (829 lines) and `ops/transforms.rs` to use samples/assets instead of images:
    - `rename()` — update any SQL referencing images
    - `duplicate()` — copy samples+assets instead of images. For each sample, copy all its assets.
    - `merge()` — merge samples from multiple datasets. Deduplicate by sample name or asset hash.
    - `split()` — split by samples (not by images). Assets stay with their sample.
    - `filter()` / transform operations in `transforms.rs` — filter on sample-level criteria
  - Update all row helpers that construct `Image` → construct `Sample`/`Asset`
  - Update all SQL: `images` → `samples`/`assets` with proper JOINs
  - Update all tests

  **Must NOT do**:
  - Do NOT split a sample's assets across different datasets during split operations
  - Do NOT change the public API signatures unnecessarily — keep the same function names where possible
  - Do NOT use `unwrap()` in library code

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 829 lines of complex SQL + Rust logic with merge/split algorithms. High complexity.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T9, T10 — but needs T7 done first)
  - **Parallel Group**: Wave 2
  - **Blocks**: T22
  - **Blocked By**: T1, T2, T3, T7

  **References**:

  **Pattern References**:
  - `crates/core/src/ops/mod.rs:1-829` — ENTIRE file. Key sections:
    - `rename()`: Updates dataset name, may touch image paths → now sample/asset paths
    - `duplicate()`: Creates new dataset, copies images → now copies samples+assets
    - `merge()`: Combines datasets, deduplicates images by hash → dedup assets by hash, group into samples
    - `split()`: Splits dataset by ratio, moves images → moves samples (with all their assets)
  - `crates/core/src/ops/transforms.rs` — Transform operations on images → on samples/assets

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — Updated `DatasetService` methods (from T7): `add_sample()`, `add_asset()`, `get_samples()`, `get_assets()`
  - `crates/core/src/types/mod.rs` — `Sample`, `Asset` types (from T3)

  **WHY Each Reference Matters**:
  - `ops/mod.rs`: This IS the file being rewritten. The merge/split algorithms are the most complex — executor must understand the current image-based logic to translate it to sample/asset logic.
  - `dataset/mod.rs`: Ops uses DatasetService for CRUD — must use the new sample/asset methods.

  **Acceptance Criteria**:
  - [ ] `duplicate()` copies samples+assets, not images
  - [ ] `merge()` deduplicates at asset level, groups into samples
  - [ ] `split()` keeps all assets with their sample — never splits a sample
  - [ ] No `Image` type references in ops/
  - [ ] `cargo test -p dman-core ops` → all pass

  **QA Scenarios**:

  ```
  Scenario: Duplicate preserves sample-asset hierarchy
    Tool: Bash (cargo test)
    Preconditions: Dataset with 2 samples (one with 2 assets, one with 1 asset)
    Steps:
      1. Run duplicate
      2. Verify new dataset has 2 samples, 3 total assets
      3. Verify multi-asset sample still has 2 assets in the copy
    Expected Result: Hierarchy preserved exactly
    Failure Indicators: Asset count mismatch, sample-asset mapping broken
    Evidence: .sisyphus/evidence/task-8-duplicate-hierarchy.txt

  Scenario: Split keeps samples intact
    Tool: Bash (cargo test)
    Preconditions: Dataset with 10 samples, split 70/30
    Steps:
      1. Run split with ratios {"train": 0.7, "val": 0.3}
      2. Verify train has 7 samples, val has 3 samples
      3. Verify no sample appears in both splits
      4. Verify each sample's assets are all in the same split
    Expected Result: Clean split at sample level
    Failure Indicators: Sample appears in both splits, or assets split across datasets
    Evidence: .sisyphus/evidence/task-8-split-samples.txt
  ```

  **Commit**: YES
  - Message: `refactor(core): update DatasetOps for sample/asset model`
  - Files: `crates/core/src/ops/mod.rs`, `crates/core/src/ops/transforms.rs`
  - Pre-commit: `cargo test -p dman-core ops`

- [x] 9. Virtual Dataset Evaluation Update

  **What to do**:
  - Complete the runtime evaluation rewrite in `crates/core/src/virtual_dataset/mod.rs` (started structurally in T5):
    - `evaluate()` now returns `Vec<Sample>` — each sample's assets are accessible by querying with the sample ID
    - `evaluate_filter()` — filter on sample-level columns (name, metadata fields). Assets come along with their sample.
    - `evaluate_merge()` — merge samples from multiple source datasets
    - `evaluate_sample_ratio()` — sample N% of samples randomly
    - `evaluate_split()` — split samples into groups
    - `evaluate_schema_transform()` — apply schema ops to sample/annotation metadata
  - Update `materialize.rs`: when materializing a virtual dataset, copy samples+assets (not images)
  - Update all SQL to use `samples`/`assets` tables
  - Update all integration tests in `crates/core/tests/virtual_integration.rs`

  **Must NOT do**:
  - Do NOT add asset-level filtering (design decision: filter on samples)
  - Do NOT change `VirtualDatasetDef` type structure

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex SQL rewrite with multiple evaluation strategies. Not quite deep-level architecture, but significant.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T8, T10)
  - **Parallel Group**: Wave 2
  - **Blocks**: T22
  - **Blocked By**: T5, T7

  **References**:

  **Pattern References**:
  - `crates/core/src/virtual_dataset/mod.rs:1-909` — Entire evaluation engine. Key functions: `evaluate()`, `evaluate_filter()`, `evaluate_merge()`, `evaluate_sample_ratio()`, `evaluate_split()`, `evaluate_schema_transform()`
  - `crates/core/src/virtual_dataset/materialize.rs` — Materialization copies images → now copies samples+assets
  - `crates/core/src/virtual_dataset/transforms.rs` — Transform helpers

  **Test References**:
  - `crates/core/tests/virtual_integration.rs` — Integration tests that create datasets with images, then apply virtual operations

  **WHY Each Reference Matters**:
  - `virtual_dataset/mod.rs`: All evaluation functions use SQL queries against `images` — every query must be updated.
  - `materialize.rs`: Physical materialization copies files — must copy assets not images.

  **Acceptance Criteria**:
  - [ ] `evaluate()` returns `Vec<Sample>` (or similar sample-level result)
  - [ ] Filter, merge, sample, split all operate on samples
  - [ ] Materialization copies samples+assets
  - [ ] `cargo test -p dman-core --test virtual_integration` → all pass

  **QA Scenarios**:

  ```
  Scenario: Filter virtual dataset returns matching samples
    Tool: Bash (cargo test)
    Preconditions: Dataset with samples having different metadata
    Steps:
      1. Create virtual dataset with filter on sample metadata
      2. Evaluate → get samples
      3. Verify only matching samples returned
    Expected Result: Correct sample filtering
    Failure Indicators: Wrong samples returned, SQL error
    Evidence: .sisyphus/evidence/task-9-virtual-filter.txt

  Scenario: Materialize virtual dataset copies sample+asset files
    Tool: Bash (cargo test)
    Preconditions: Virtual dataset defined and evaluated
    Steps:
      1. Materialize the virtual dataset
      2. Verify new dataset has correct samples and assets
      3. Verify asset files exist at new paths
    Expected Result: Materialized dataset is a proper copy
    Failure Indicators: Missing files, wrong paths
    Evidence: .sisyphus/evidence/task-9-materialize.txt
  ```

  **Commit**: YES (groups with T10)
  - Message: `refactor(core): update virtual dataset eval and format traits`
  - Files: `crates/core/src/virtual_dataset/mod.rs`, `crates/core/src/virtual_dataset/materialize.rs`, `crates/core/src/virtual_dataset/transforms.rs`
  - Pre-commit: `cargo test -p dman-core --test virtual_integration`

- [x] 10. Format Trait Evolution — `FormatImporter`/`FormatExporter`

  **What to do**:
  - In `crates/core/src/formats/mod.rs`:
    - The `FormatImporter::import()` signature stays the same: `fn import(&self, db: &Database, storage: &StorageManager, path: &Path, dataset_name: &str) -> Result<Dataset>`. The trait contract doesn't change — what changes is what importers DO inside (they create samples+assets instead of images).
    - The `FormatExporter::export()` signature stays the same: `fn export(&self, db: &Database, storage: &StorageManager, dataset: &Dataset, output_path: &Path) -> Result<()>`. Exporters read samples+assets instead of images.
    - This means the trait definitions in `formats/mod.rs` may need minimal changes. The main work is documenting the new contract in doc comments.
    - Update trait doc comments to describe sample/asset model instead of image model.
    - Update `FormatRegistry` if any method references `Image` type — it shouldn't, but verify.
    - Update tests in this module (they don't directly test import/export behavior, just registry).
  - This task is primarily about updating the CONTRACTS (documentation) and verifying the trait signatures work. Actual adapter rewrites are T12-T14.

  **Must NOT do**:
  - Do NOT rewrite YOLO/COCO/HF adapters here — that's T12-T14
  - Do NOT change the trait function signatures unless absolutely necessary
  - Do NOT change `FormatRegistry` structure

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Trait contract design affects all downstream adapters. Must be thoughtful about documentation.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T6, T7, T8, T9)
  - **Parallel Group**: Wave 2
  - **Blocks**: T12, T13, T14, T16
  - **Blocked By**: T2, T3, T7

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/mod.rs:15-52` — `FormatImporter` and `FormatExporter` trait definitions. Doc comments must be updated to describe sample/asset model.
  - `crates/core/src/formats/mod.rs:54-126` — `FormatRegistry`. Verify no `Image` references.

  **Test References**:
  - `crates/core/src/formats/mod.rs:128-284` — Registry tests. The `can_register_custom_importer` test constructs a dummy `Dataset` — update if needed.

  **WHY Each Reference Matters**:
  - `formats/mod.rs:15-52`: The trait contract is the API that all adapters (built-in + Python plugins) must implement. Getting the documentation right is critical.

  **Acceptance Criteria**:
  - [ ] Trait doc comments describe sample/asset model
  - [ ] No `Image` type references in formats/mod.rs
  - [ ] `FormatRegistry` compiles and tests pass
  - [ ] `cargo test -p dman-core formats` → all pass

  **QA Scenarios**:

  ```
  Scenario: Format registry still works with updated types
    Tool: Bash (cargo test)
    Preconditions: Traits updated, adapters temporarily broken (expected)
    Steps:
      1. Run: cargo test -p dman-core formats::tests
      2. Verify registry tests (registration, lookup, detection) still pass
    Expected Result: All registry-level tests pass
    Failure Indicators: Compilation error from Image type reference
    Evidence: .sisyphus/evidence/task-10-format-registry.txt
  ```

  **Commit**: YES (groups with T9)
  - Message: `refactor(core): update virtual dataset eval and format traits`
  - Files: `crates/core/src/formats/mod.rs`
  - Pre-commit: `cargo test -p dman-core formats`

- [x] 11. Schema Module Update

  **What to do**:
  - In `crates/core/src/schema/mod.rs` (795 lines):
    - The schema system validates dataset structure using TOML schema definitions. It knows about BBox, Polygon, Keypoints annotation types.
    - Update any references to `Image` or `image` in schema validation logic to use `Sample`/`Asset` vocabulary
    - Schema definitions should validate at the sample level (e.g., "each sample must have at least one asset") and asset level (e.g., "each image asset must have width/height")
    - Update `validate()` function if it iterates over images → iterate over samples/assets
    - If schema validation queries the DB, update SQL from `images` → `samples`/`assets`
  - Update tests

  **Must NOT do**:
  - Do NOT redesign the TOML schema format — just update internal references
  - Do NOT change BBox, Polygon, Keypoints validation logic — those still apply to annotations

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 795 lines of schema validation logic. Moderate complexity — mostly renaming with some SQL updates.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T12, T13, T14, T15)
  - **Parallel Group**: Wave 3
  - **Blocks**: None directly
  - **Blocked By**: T3

  **References**:

  **Pattern References**:
  - `crates/core/src/schema/mod.rs:1-795` — Entire schema validation module. Search for `Image`, `image`, `images` to find all update points.

  **WHY Each Reference Matters**:
  - `schema/mod.rs`: Contains all validation logic. The executor must search-and-replace `image` references systematically.

  **Acceptance Criteria**:
  - [ ] No `Image` type references in schema/mod.rs
  - [ ] Schema validation works with sample/asset model
  - [ ] `cargo test -p dman-core schema` → all pass

  **QA Scenarios**:

  ```
  Scenario: Schema validation with sample/asset model
    Tool: Bash (cargo test)
    Preconditions: Schema module updated
    Steps:
      1. Run: cargo test -p dman-core schema
      2. Verify all schema tests pass
    Expected Result: All pass
    Failure Indicators: Type errors or validation failures
    Evidence: .sisyphus/evidence/task-11-schema-validation.txt
  ```

  **Commit**: YES
  - Message: `refactor(core): update schema module for asset-aware validation`
  - Files: `crates/core/src/schema/mod.rs`
  - Pre-commit: `cargo test -p dman-core schema`

- [x] 12. YOLO Adapter Rewrite

  **What to do**:
  - Rewrite `crates/core/src/formats/yolo/mod.rs`:
    - **Importer**: Currently reads image files + `.txt` label files, creates `Image` + `Annotation` records.
      - New behavior: For each image file, create a `Sample` (name = image file stem) with a single `Asset` (type = `Image`). Create `Annotation` records referencing `sample_id` + `asset_id`.
      - Use `DatasetService::add_sample_with_single_asset()` for 1:1 mapping.
    - **Exporter**: Currently reads `Image` + `Annotation` records, writes image files + `.txt` label files.
      - New behavior: Read `Sample` → `Asset` → `Annotation` records. For each sample with a single image asset, write the YOLO format as before.
      - For multi-asset samples: warn/skip (YOLO format doesn't support multi-view) or export only the primary image asset.
    - Update detection logic if it references `images` table.
  - Update tests

  **Must NOT do**:
  - Do NOT change the YOLO on-disk format (directories, label files, classes.txt)
  - Do NOT add multi-asset YOLO support — YOLO is inherently single-image

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Format adapter rewrite with DB interaction. Follows known pattern but needs SQL precision.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T13, T14)
  - **Parallel Group**: Wave 3
  - **Blocks**: T15
  - **Blocked By**: T4, T6, T7, T10

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/yolo/mod.rs` — ENTIRE current adapter. Shows image iteration, label parsing, category creation pattern.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — `add_sample_with_single_asset()` (from T7), `get_samples()`, `get_assets()`
  - `crates/core/src/types/mod.rs` — `Sample`, `Asset`, `AssetType::Image`, `Annotation` with `sample_id` + `asset_id`
  - `crates/core/src/storage/mod.rs` — `store_asset()` (from T4)

  **Test References**:
  - `crates/core/tests/roundtrip.rs` — YOLO roundtrip test: `yolo_import_then_coco_export_preserves_image_count`

  **WHY Each Reference Matters**:
  - `yolo/mod.rs`: This is the file being rewritten. Executor needs the current structure to translate.
  - `dataset/mod.rs`: The new CRUD API is the primary interface for the adapter.

  **Acceptance Criteria**:
  - [ ] YOLO importer creates Sample+Asset for each image
  - [ ] YOLO exporter reads Sample→Asset→Annotation chain
  - [ ] `cargo test -p dman-core yolo` → all pass
  - [ ] Roundtrip: import YOLO → export YOLO produces equivalent output

  **QA Scenarios**:

  ```
  Scenario: YOLO import creates 1:1 sample-asset mapping
    Tool: Bash (cargo test)
    Preconditions: YOLO test fixtures exist under crates/core/tests/fixtures/
    Steps:
      1. Run: cargo test -p dman-core --test roundtrip yolo
      2. Verify import creates N samples and N assets (one per image)
      3. Verify each annotation references both sample_id and asset_id
    Expected Result: 1:1 mapping preserved
    Failure Indicators: sample_count ≠ asset_count, or NULL asset_id on annotations
    Evidence: .sisyphus/evidence/task-12-yolo-import.txt

  Scenario: YOLO export from sample model
    Tool: Bash (cargo test)
    Preconditions: Dataset imported via YOLO importer
    Steps:
      1. Export to YOLO format
      2. Verify output directory has images/ and labels/ subdirectories
      3. Verify label files contain correct bbox annotations
    Expected Result: Valid YOLO format output
    Failure Indicators: Missing label files, wrong bbox format
    Evidence: .sisyphus/evidence/task-12-yolo-export.txt
  ```

  **Commit**: YES (groups with T13, T14)
  - Message: `refactor(core): rewrite YOLO, COCO, HuggingFace adapters for sample model`
  - Files: `crates/core/src/formats/yolo/mod.rs`
  - Pre-commit: `cargo test -p dman-core yolo`

- [x] 13. COCO Adapter Rewrite

  **What to do**:
  - Rewrite `crates/core/src/formats/coco/mod.rs`:
    - **Importer**: Currently reads COCO JSON (images array + annotations array). For each COCO image, create a Sample + Asset. Map COCO annotation `image_id` to both `sample_id` and `asset_id`.
    - **Exporter**: Read samples → assets → annotations. Generate COCO JSON with `images` array (from assets), `annotations` array (from annotations), `categories` array.
    - COCO format natively has image_id linking — the 1:1 Sample→Asset mapping aligns well.
  - Update tests

  **Must NOT do**:
  - Do NOT change the COCO JSON schema — it's a standard
  - Do NOT add multi-asset COCO support — standard COCO is single-image

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Same pattern as YOLO adapter, different format specifics.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T12, T14)
  - **Parallel Group**: Wave 3
  - **Blocks**: T15
  - **Blocked By**: T4, T6, T7, T10

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/coco/mod.rs` — Current COCO adapter. COCO JSON structure: `{"images": [...], "annotations": [...], "categories": [...]}`

  **API/Type References**:
  - Same as T12: `add_sample_with_single_asset()`, `get_samples()`, `get_assets()`, `store_asset()`

  **Test References**:
  - `crates/core/tests/roundtrip.rs` — COCO roundtrip tests

  **WHY Each Reference Matters**:
  - `coco/mod.rs`: File being rewritten. COCO's `image_id` in annotations maps naturally to asset_id.

  **Acceptance Criteria**:
  - [ ] COCO importer creates Sample+Asset for each COCO image
  - [ ] COCO exporter produces valid COCO JSON from sample model
  - [ ] `cargo test -p dman-core coco` → all pass

  **QA Scenarios**:

  ```
  Scenario: COCO roundtrip preserves data
    Tool: Bash (cargo test)
    Preconditions: COCO test fixtures exist
    Steps:
      1. Run: cargo test -p dman-core --test roundtrip coco
      2. Verify import → export → reimport preserves image count, annotation count, categories
    Expected Result: Roundtrip preserves data
    Failure Indicators: Count mismatch, missing categories
    Evidence: .sisyphus/evidence/task-13-coco-roundtrip.txt
  ```

  **Commit**: YES (groups with T12, T14)
  - Message: `refactor(core): rewrite YOLO, COCO, HuggingFace adapters for sample model`
  - Files: `crates/core/src/formats/coco/mod.rs`
  - Pre-commit: `cargo test -p dman-core coco`

- [x] 14. HuggingFace Adapter Rewrite

  **What to do**:
  - Rewrite `crates/core/src/formats/huggingface/mod.rs`:
    - Same 1:1 Sample→Asset pattern as YOLO and COCO.
    - HuggingFace datasets have a specific directory layout (arrow files, dataset_info.json). Import creates Sample+Asset per row.
    - Export generates HF-compatible layout from samples+assets.
  - Update tests

  **Must NOT do**:
  - Do NOT change the HuggingFace on-disk format
  - Same constraints as T12/T13

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Same pattern as T12/T13.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T11, T12, T13)
  - **Parallel Group**: Wave 3
  - **Blocks**: T15
  - **Blocked By**: T4, T6, T7, T10

  **References**:

  **Pattern References**:
  - `crates/core/src/formats/huggingface/mod.rs` — Current HF adapter.

  **API/Type References**:
  - Same as T12: `add_sample_with_single_asset()`, etc.

  **Acceptance Criteria**:
  - [ ] HF importer creates Sample+Asset per HF row
  - [ ] HF exporter produces valid HF layout from sample model
  - [ ] `cargo test -p dman-core huggingface` → all pass

  **QA Scenarios**:

  ```
  Scenario: HuggingFace roundtrip
    Tool: Bash (cargo test)
    Preconditions: HF test fixtures exist
    Steps:
      1. Run: cargo test -p dman-core --test roundtrip huggingface
    Expected Result: All pass
    Failure Indicators: Format parsing error, count mismatch
    Evidence: .sisyphus/evidence/task-14-hf-roundtrip.txt
  ```

  **Commit**: YES (groups with T12, T13)
  - Message: `refactor(core): rewrite YOLO, COCO, HuggingFace adapters for sample model`
  - Files: `crates/core/src/formats/huggingface/mod.rs`
  - Pre-commit: `cargo test -p dman-core huggingface`

- [x] 15. Multi-Asset Roundtrip Test Infrastructure

  **What to do**:
  - Create new integration test in `crates/core/tests/` that verifies multi-asset scenarios end-to-end:
    - **Test 1**: Create a dataset programmatically with 2 samples: sample A has 2 assets (left-camera, right-camera images), sample B has 1 asset. Add annotations at both sample and asset level. Verify full hierarchy is stored and retrievable.
    - **Test 2**: Create a multi-asset dataset, duplicate it, verify hierarchy preserved.
    - **Test 3**: Create a multi-asset dataset, split it, verify no sample is torn apart.
    - **Test 4**: Test annotation dual-level: annotation on sample only (no asset_id), annotation on sample+asset. Both must work.
  - Create test fixtures if needed (small test images, or use empty files with correct extensions).
  - This task creates the PROOF that the entire model evolution works end-to-end.

  **Must NOT do**:
  - Do NOT test format-specific behavior — that's covered in T12-T14
  - Do NOT create large test fixtures — minimal data is sufficient

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Integration tests that exercise the full stack. Must design test scenarios carefully.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (needs T12-T14 complete)
  - **Parallel Group**: Wave 3 (after T12-T14)
  - **Blocks**: None
  - **Blocked By**: T12, T13, T14

  **References**:

  **Pattern References**:
  - `crates/core/tests/roundtrip.rs` — Existing roundtrip test pattern. Shows how to create DB, import, export, verify.
  - `crates/core/tests/virtual_integration.rs` — Another integration test pattern.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — All CRUD methods
  - `crates/core/src/types/mod.rs` — Sample, Asset, AssetType, Annotation

  **WHY Each Reference Matters**:
  - `roundtrip.rs`: Template for test file structure, fixture usage, assertion patterns.

  **Acceptance Criteria**:
  - [ ] Multi-asset sample creation and retrieval test passes
  - [ ] Duplicate preserves hierarchy test passes
  - [ ] Split preserves sample integrity test passes
  - [ ] Dual-level annotation test passes
  - [ ] `cargo test -p dman-core --test multi_asset` → all pass

  **QA Scenarios**:

  ```
  Scenario: Multi-asset integration tests all pass
    Tool: Bash (cargo test)
    Preconditions: All Wave 1-3 tasks complete
    Steps:
      1. Run: cargo test -p dman-core --test multi_asset
      2. Verify all 4+ test cases pass
    Expected Result: 0 failures
    Failure Indicators: Any test failure — indicates a gap in the model implementation
    Evidence: .sisyphus/evidence/task-15-multi-asset-tests.txt
  ```

  **Commit**: YES
  - Message: `test(core): add multi-asset roundtrip test infrastructure`
  - Files: `crates/core/tests/multi_asset.rs`
  - Pre-commit: `cargo test -p dman-core --test multi_asset`

- [x] 16. Python Plugin Contract Redesign

  **What to do**:
  - MAJOR rewrite of `crates/python/src/plugins/format.rs` (646 lines):
    - Replace `PyImgData` with `PySampleData` + `PyAssetData`:
      ```python
      # What Python plugins will see:
      class SampleData:
          name: str
          metadata: dict | None
          assets: list[AssetData]  # one or more assets
          annotations: list[AnnotationData]  # sample-level annotations

      class AssetData:
          asset_type: str  # "image", "depth_map", "point_cloud", etc.
          file_name: str
          file_path: str
          width: int | None
          height: int | None
          metadata: dict | None
          annotations: list[AnnotationData]  # asset-level annotations

      class AnnotationData:
          category: str
          bbox: dict | None  # {x, y, width, height}
          segmentation: list[list[float]] | None
          keypoints: list[float] | None
          metadata: dict | None
      ```
    - Replace `PyAnnData` with `PyAnnotationData` that includes segmentation, keypoints, metadata (not just bbox)
    - Update `PythonFormatImporter::import()`:
      - Call Python plugin's `import_dataset(path)` which now returns `list[SampleData]`
      - For each SampleData: create Sample via DatasetService, create Assets, create Annotations
    - Update `PythonFormatExporter::export()`:
      - Read samples+assets+annotations from DB
      - Convert to PySampleData/PyAssetData/PyAnnotationData
      - Call Python plugin's `export_dataset(samples, output_path)`
    - Fix BBox serialization: standardize to `{x, y, width, height}` (not `{x, y, w, h}`)
  - This is the most impactful change for plugin authors — the entire Python plugin API changes.

  **Must NOT do**:
  - Do NOT keep backward compatibility with old `PyImgData`/`PyAnnData` contract
  - Do NOT use `unwrap()` — use `DmanError::PluginError` for Python interop errors
  - Do NOT use `{x, y, w, h}` for BBox — use `{x, y, width, height}` consistently

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Python-Rust interop via PyO3 is complex. Contract redesign affects all plugins.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T20, T21)
  - **Parallel Group**: Wave 4
  - **Blocks**: T17, T18, T19, T25
  - **Blocked By**: T3, T7, T10

  **References**:

  **Pattern References**:
  - `crates/python/src/plugins/format.rs:1-646` — ENTIRE file being rewritten. Key structures:
    - `PyImgData` struct (~line 30-50): fields `file_name`, `file_path`. Must become `PySampleData` + `PyAssetData`.
    - `PyAnnData` struct (~line 50-70): fields `image_file_name`, `category`, `bbox`. Must become `PyAnnotationData` with segmentation, keypoints.
    - `PythonFormatImporter::import()` (~line 100-300): Calls Python, converts result to dman types. Major rewrite.
    - `PythonFormatExporter::export()` (~line 300-500): Reads from DB, calls Python. Major rewrite.
    - `img_data_to_py()`, `ann_data_to_py()` helpers: Replace with `sample_data_to_py()`, `asset_data_to_py()`, `annotation_data_to_py()`

  **API/Type References**:
  - `crates/core/src/types/mod.rs` — `Sample`, `Asset`, `AssetType`, `Annotation` (from T3)
  - `crates/core/src/dataset/mod.rs` — `add_sample()`, `add_asset()`, `get_samples()`, `get_assets()`, `get_annotations_for_sample()`, `get_annotations_for_asset()` (from T7)
  - `crates/core/src/error.rs` — `DmanError::PluginError` for Python errors

  **External References**:
  - PyO3 docs: `https://pyo3.rs/` — Python↔Rust type conversion patterns

  **WHY Each Reference Matters**:
  - `format.rs`: This IS the file being rewritten. The current structure shows the PyO3 conversion patterns that must be adapted.
  - `dataset/mod.rs`: The new DatasetService API defines how plugin data flows into the DB.
  - `error.rs`: All Python interop errors must use `PluginError`, not panic.

  **Acceptance Criteria**:
  - [ ] `PySampleData`, `PyAssetData`, `PyAnnotationData` structs exist
  - [ ] `PyImgData`, `PyAnnData` are fully removed
  - [ ] Import creates samples+assets+annotations from Python plugin output
  - [ ] Export provides samples+assets+annotations to Python plugin
  - [ ] BBox uses `{x, y, width, height}` format
  - [ ] `cargo check -p dman-python` compiles
  - [ ] `cargo test -p dman-python` → all pass

  **QA Scenarios**:

  ```
  Scenario: Python plugin contract compiles
    Tool: Bash (cargo check)
    Preconditions: Format.rs rewritten
    Steps:
      1. Run: cargo check -p dman-python --features python
      2. Verify clean compilation
    Expected Result: 0 errors
    Failure Indicators: PyO3 type errors, missing struct fields
    Evidence: .sisyphus/evidence/task-16-python-plugin-check.txt

  Scenario: No PyImgData/PyAnnData references remain
    Tool: Bash (grep)
    Preconditions: Task complete
    Steps:
      1. Run: grep -rn "PyImgData\|PyAnnData" crates/python/src/
      2. Verify zero matches
    Expected Result: Old types completely removed
    Failure Indicators: Any match found
    Evidence: .sisyphus/evidence/task-16-no-old-types.txt
  ```

  **Commit**: YES (groups with T17, T18, T19)
  - Message: `refactor(python): redesign plugin contract and SDK for sample/asset model`
  - Files: `crates/python/src/plugins/format.rs`
  - Pre-commit: `cargo check -p dman-python --features python`

- [x] 17. Python SDK Builder Rewrite

  **What to do**:
  - Rewrite `crates/python/src/sdk/builder.rs` (806 lines):
    - Replace `PendingImage` with `PendingSample` + `PendingAsset`:
      - `PendingSample`: name, metadata, list of `PendingAsset`
      - `PendingAsset`: asset_type, file_name, file_path, width, height, metadata
    - Replace `PendingAnnotation` with new version: category, bbox, segmentation, keypoints, metadata, target (sample or asset)
    - Update `DmanBuilder` (Python-exposed class):
      - `add_image(file_name, file_path, ...)` → `add_sample(name, ...)` + `add_asset(sample_name, asset_type, file_path, ...)` or convenience `add_image(name, file_path, ...)` that creates 1:1 Sample→Asset
      - `add_annotation(image_name, category, bbox)` → `add_annotation(sample_name, category, bbox=None, segmentation=None, keypoints=None, metadata=None, asset_name=None)`
      - `build()` → creates dataset with samples+assets+annotations
    - Fix BBox serialization to use `{x, y, width, height}` (currently uses `{x, y, w, h}`)
    - Keep a convenience `add_image()` method that internally calls `add_sample()` + `add_asset()` for backward-compatible Python usage

  **Must NOT do**:
  - Do NOT use `{x, y, w, h}` — standardize to `{x, y, width, height}`
  - Do NOT use `unwrap()` — use proper error handling

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: 806 lines of PyO3 builder pattern. Complex Python↔Rust interaction.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T18, T19, T20, T21)
  - **Parallel Group**: Wave 4
  - **Blocks**: T25
  - **Blocked By**: T16

  **References**:

  **Pattern References**:
  - `crates/python/src/sdk/builder.rs:1-806` — ENTIRE file. Key structures:
    - `PendingImage` → `PendingSample` + `PendingAsset`
    - `PendingAnnotation` → updated with segmentation, keypoints
    - `DmanBuilder` class: `add_image()`, `add_annotation()`, `build()` methods
  - `crates/python/src/plugins/format.rs` — Updated plugin contract (from T16) for consistent type patterns

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — `add_sample_with_single_asset()`, `add_annotation()` (from T7)
  - `crates/core/src/types/mod.rs` — `BBox { x, y, width, height }` — canonical serialization

  **WHY Each Reference Matters**:
  - `builder.rs`: This IS the file being rewritten. The current `PendingImage`/`PendingAnnotation` pattern must be understood to translate.
  - `format.rs`: The plugin contract (T16) establishes the new type names — builder should use consistent naming.

  **Acceptance Criteria**:
  - [ ] `PendingSample`, `PendingAsset` types replace `PendingImage`
  - [ ] `DmanBuilder` has `add_sample()`, `add_asset()`, convenience `add_image()`
  - [ ] BBox uses `{x, y, width, height}` (not `{x, y, w, h}`)
  - [ ] `cargo test -p dman-python` → all pass

  **QA Scenarios**:

  ```
  Scenario: Builder creates sample-asset hierarchy
    Tool: Bash (cargo test)
    Preconditions: Builder rewritten
    Steps:
      1. Run: cargo test -p dman-python builder
      2. Verify builder can create sample with multiple assets
    Expected Result: Multi-asset sample created successfully
    Failure Indicators: Type error, wrong field names
    Evidence: .sisyphus/evidence/task-17-builder-hierarchy.txt

  Scenario: BBox serialization consistency
    Tool: Bash (grep)
    Preconditions: Task complete
    Steps:
      1. Run: grep -n '"w"' crates/python/src/sdk/builder.rs
      2. Verify zero matches (no {x,y,w,h} pattern)
    Expected Result: No w/h shorthand — only width/height
    Failure Indicators: grep finds "w" as a JSON key
    Evidence: .sisyphus/evidence/task-17-bbox-consistency.txt
  ```

  **Commit**: YES (groups with T16, T18, T19)
  - Message: `refactor(python): redesign plugin contract and SDK for sample/asset model`
  - Files: `crates/python/src/sdk/builder.rs`
  - Pre-commit: `cargo test -p dman-python`

- [x] 18. Python SDK Loader Rewrite

  **What to do**:
  - Rewrite `crates/python/src/sdk/loader.rs` (493 lines):
    - Replace `ImageRow` with `SampleRow` + `AssetRow`:
      - `SampleRow`: id, name, metadata, assets (list of AssetRow)
      - `AssetRow`: id, asset_type, file_name, file_path, width, height, metadata
    - Replace `AnnotationRow` — add segmentation, keypoints, metadata fields + `asset_id` (optional)
    - Update `DmanDataset` (Python-exposed class):
      - `images()` → `samples()` returning list of SampleRow
      - `get_image(name)` → `get_sample(name)` returning SampleRow with its assets
      - `annotations(image_name)` → `annotations(sample_name, asset_name=None)` supporting both levels
      - Keep convenience `images()` alias if helpful for backward compat
    - Update iteration to go through samples→assets instead of images

  **Must NOT do**:
  - Do NOT leave `ImageRow` references
  - Do NOT use `unwrap()`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Loader is simpler than builder — mostly reading and exposing data. 493 lines.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T17, T19, T20, T21)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T16

  **References**:

  **Pattern References**:
  - `crates/python/src/sdk/loader.rs:1-493` — ENTIRE file. Key structures: `ImageRow`, `AnnotationRow`, `DmanDataset` class.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — `get_samples()`, `get_assets()`, `get_annotations_for_sample()`, `get_annotations_for_asset()`

  **WHY Each Reference Matters**:
  - `loader.rs`: File being rewritten. Shows current PyO3 class pattern.

  **Acceptance Criteria**:
  - [ ] `SampleRow`, `AssetRow` replace `ImageRow`
  - [ ] `DmanDataset` has `samples()`, `get_sample()` methods
  - [ ] Annotations queryable at both sample and asset level
  - [ ] `cargo test -p dman-python` → all pass

  **QA Scenarios**:

  ```
  Scenario: Loader reads sample-asset hierarchy
    Tool: Bash (cargo test)
    Preconditions: Loader rewritten, builder working (T17)
    Steps:
      1. Run: cargo test -p dman-python loader
      2. Verify samples() returns correct data with assets nested
    Expected Result: Full hierarchy accessible from Python
    Failure Indicators: Missing assets, wrong sample data
    Evidence: .sisyphus/evidence/task-18-loader-hierarchy.txt
  ```

  **Commit**: YES (groups with T16, T17, T19)
  - Message: `refactor(python): redesign plugin contract and SDK for sample/asset model`
  - Files: `crates/python/src/sdk/loader.rs`
  - Pre-commit: `cargo test -p dman-python`

- [x] 19. Python lib.rs + plugin_info Updates

  **What to do**:
  - In `crates/python/src/lib.rs`:
    - Update `PluginManager` if it references `Image` type
    - Update any PyO3 module registration that exposes old types
    - Ensure new types (`PySampleData`, `PyAssetData`, etc.) are properly registered in the Python module
  - In `crates/python/src/plugin_info.rs`:
    - Update `PluginInfo` if it references image-specific fields
  - Verify `__init__.py` in `python/dman/` doesn't need updates (it's mostly the CLI entrypoint)

  **Must NOT do**:
  - Do NOT change CLI entrypoint
  - Do NOT change plugin discovery mechanism

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small integration task — mostly ensuring module registration is correct.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T17, T18, T20, T21)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T16

  **References**:

  **Pattern References**:
  - `crates/python/src/lib.rs` — PluginManager, PyO3 module
  - `crates/python/src/plugin_info.rs` — PluginInfo struct

  **Acceptance Criteria**:
  - [ ] `cargo check -p dman-python --features python` compiles cleanly
  - [ ] No `Image`/`PyImgData`/`PyAnnData` references in lib.rs or plugin_info.rs

  **QA Scenarios**:

  ```
  Scenario: Python module compiles
    Tool: Bash (cargo check)
    Steps:
      1. Run: cargo check -p dman-python --features python
    Expected Result: Clean compilation
    Evidence: .sisyphus/evidence/task-19-python-module-check.txt
  ```

  **Commit**: YES (groups with T16, T17, T18)
  - Message: `refactor(python): redesign plugin contract and SDK for sample/asset model`
  - Files: `crates/python/src/lib.rs`, `crates/python/src/plugin_info.rs`
  - Pre-commit: `cargo check -p dman-python --features python`

- [x] 20. Server API Update

  **What to do**:
  - Update `crates/server/src/api.rs` (888 lines):
    - Replace image-centric endpoints with sample/asset endpoints:
      - `GET /datasets/{id}/images` → `GET /datasets/{id}/samples` + `GET /samples/{id}/assets`
      - `GET /images/{id}` → `GET /samples/{id}` + `GET /assets/{id}`
      - `POST /datasets/{id}/images` → `POST /datasets/{id}/samples` (with assets in body)
    - Update response types to use Sample/Asset instead of Image
    - Update SQL queries if the API has direct DB access
    - Keep the API RESTful and consistent with the new model

  **Must NOT do**:
  - Do NOT change authentication/CORS/middleware
  - Do NOT change server startup logic

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 888 lines of API code. Mostly mechanical but needs API design thought.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T16-T19)
  - **Parallel Group**: Wave 4
  - **Blocks**: T21
  - **Blocked By**: T7

  **References**:

  **Pattern References**:
  - `crates/server/src/api.rs:1-888` — ENTIRE API file. Search for `image`, `Image` to find all update points.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — DatasetService methods (from T7)
  - `crates/core/src/types/mod.rs` — Sample, Asset types

  **Acceptance Criteria**:
  - [ ] No `Image`/`image_id` references in api.rs (except in migration/compatibility context)
  - [ ] Sample/asset endpoints work
  - [ ] `cargo test -p dman-server` → all pass

  **QA Scenarios**:

  ```
  Scenario: Server compiles and tests pass
    Tool: Bash (cargo test)
    Steps:
      1. Run: cargo test -p dman-server
    Expected Result: All pass
    Failure Indicators: Type errors, SQL errors
    Evidence: .sisyphus/evidence/task-20-server-tests.txt
  ```

  **Commit**: YES (groups with T21)
  - Message: `refactor(server): update API and Label Studio for sample/asset model`
  - Files: `crates/server/src/api.rs`
  - Pre-commit: `cargo test -p dman-server`

- [x] 21. Label Studio Integration Update

  **What to do**:
  - Update `crates/server/src/label_studio/mod.rs` (743 lines):
    - Label Studio tasks are image-centric. For the integration:
      - When importing from Label Studio: create Sample+Asset for each task (1:1 mapping)
      - When exporting to Label Studio: convert Sample→Asset→Annotation to Label Studio task format
    - Update `create_tasks()`, `import_annotations()`, etc. to use sample/asset model
    - Label Studio's API uses `data.image` field — this maps to the asset's file_path

  **Must NOT do**:
  - Do NOT change Label Studio API interaction patterns
  - Do NOT add multi-asset Label Studio support (Label Studio is single-image per task)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 743 lines of integration code. Moderate complexity.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T16-T19)
  - **Parallel Group**: Wave 4
  - **Blocks**: None
  - **Blocked By**: T7, T20

  **References**:

  **Pattern References**:
  - `crates/server/src/label_studio/mod.rs:1-743` — Entire Label Studio integration. Search for `image`, `Image` references.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — DatasetService (T7)
  - `crates/core/src/types/mod.rs` — Sample, Asset, Annotation

  **Acceptance Criteria**:
  - [ ] Label Studio integration uses sample/asset model
  - [ ] No `Image` type references
  - [ ] `cargo test -p dman-server` → all pass

  **QA Scenarios**:

  ```
  Scenario: Label Studio module compiles
    Tool: Bash (cargo check)
    Steps:
      1. Run: cargo check -p dman-server
    Expected Result: Clean compilation
    Evidence: .sisyphus/evidence/task-21-label-studio-check.txt
  ```

  **Commit**: YES (groups with T20)
  - Message: `refactor(server): update API and Label Studio for sample/asset model`
  - Files: `crates/server/src/label_studio/mod.rs`
  - Pre-commit: `cargo test -p dman-server`

- [ ] 22. CLI Update — inspect, list, import, export

  **What to do**:
  - Update `crates/cli/src/main.rs` (532 lines):
    - `inspect` subcommand: Display `sample_count`, `asset_count`, `annotation_count` instead of `image_count`
    - `list` subcommand: If it shows image counts, update to sample/asset counts
    - `import` subcommand: Already delegates to format registry — should work if T10-T14 are done. Verify.
    - `export` subcommand: Same as import — verify it works with new model.
    - Update `build_registry()` if it references image types
    - Update any output formatting strings from "images" to "samples"/"assets"
  - Update CLI integration tests in `crates/cli/tests/integration/`:
    - `lifecycle.rs`: Update full lifecycle test to use new vocabulary
    - `schema.rs`: Update schema tests
    - `virtual_dataset.rs`: Update virtual dataset tests

  **Must NOT do**:
  - Do NOT change CLI argument structure unless necessary
  - Do NOT change the `clap` derive patterns

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 532 lines of CLI + integration tests. Moderate complexity.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T23, T24, T25)
  - **Parallel Group**: Wave 5
  - **Blocks**: T23
  - **Blocked By**: T7, T8, T9, T10

  **References**:

  **Pattern References**:
  - `crates/cli/src/main.rs:1-532` — CLI entry point. Key areas:
    - `inspect` command output formatting
    - `build_registry()` function
    - Import/export command handlers

  **Test References**:
  - `crates/cli/tests/integration/lifecycle.rs` — Full lifecycle: init → add → inspect → remove
  - `crates/cli/tests/integration/schema.rs` — Schema import tests
  - `crates/cli/tests/integration/virtual_dataset.rs` — Virtual dataset tests

  **WHY Each Reference Matters**:
  - `main.rs`: CLI output is user-facing — terminology must be correct.
  - Integration tests: These are the primary acceptance tests for CLI behavior.

  **Acceptance Criteria**:
  - [ ] `dman inspect` shows sample_count, asset_count (not image_count)
  - [ ] CLI output uses "samples"/"assets" vocabulary
  - [ ] `cargo test -p dman-cli --test integration` → all pass

  **QA Scenarios**:

  ```
  Scenario: CLI integration tests pass
    Tool: Bash (cargo test)
    Preconditions: All core and format tasks complete
    Steps:
      1. Run: cargo test -p dman-cli --test integration
      2. Verify lifecycle, schema, virtual_dataset tests all pass
    Expected Result: 0 failures
    Failure Indicators: Output mismatch, command failure
    Evidence: .sisyphus/evidence/task-22-cli-integration.txt

  Scenario: Inspect output shows correct vocabulary
    Tool: Bash (cargo test)
    Preconditions: CLI updated
    Steps:
      1. Run lifecycle integration test that includes inspect
      2. Verify output contains "samples:" and "assets:" not "images:"
    Expected Result: Correct terminology in output
    Failure Indicators: "images" appears in inspect output
    Evidence: .sisyphus/evidence/task-22-inspect-vocabulary.txt
  ```

  **Commit**: YES (groups with T23)
  - Message: `refactor(cli): update CLI and TUI for sample/asset model`
  - Files: `crates/cli/src/main.rs`, `crates/cli/tests/integration/`
  - Pre-commit: `cargo test -p dman-cli`

- [x] 23. TUI Update

  **What to do**:
  - Update `crates/cli/src/tui.rs`:
    - Replace any "Images" column/label with "Samples"/"Assets"
    - Update dataset detail view to show samples and assets instead of images
    - If TUI shows image thumbnails/paths, update to show asset paths
    - Update any SQL queries in TUI code to use samples/assets tables

  **Must NOT do**:
  - Do NOT change TUI layout/framework
  - Do NOT change key bindings

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: TUI code with ratatui. Moderate complexity.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T24, T25)
  - **Parallel Group**: Wave 5
  - **Blocks**: None
  - **Blocked By**: T7, T22

  **References**:

  **Pattern References**:
  - `crates/cli/src/tui.rs` — TUI module. Search for `image`, `Image` references.

  **API/Type References**:
  - `crates/core/src/dataset/mod.rs` — DatasetInfo (from T7) with sample_count, asset_count

  **Acceptance Criteria**:
  - [ ] TUI shows "Samples" and "Assets" not "Images"
  - [ ] No `Image` type references in tui.rs
  - [ ] `cargo check -p dman-cli` compiles

  **QA Scenarios**:

  ```
  Scenario: TUI compiles with new model
    Tool: Bash (cargo check)
    Steps:
      1. Run: cargo check -p dman-cli
    Expected Result: Clean compilation
    Evidence: .sisyphus/evidence/task-23-tui-check.txt
  ```

  **Commit**: YES (groups with T22)
  - Message: `refactor(cli): update CLI and TUI for sample/asset model`
  - Files: `crates/cli/src/tui.rs`
  - Pre-commit: `cargo check -p dman-cli`

- [x] 24. Documentation Update

  **What to do**:
  - Update `README.md`:
    - Feature overview should mention multi-view, multi-modal, segmentation support
    - Architecture description should mention sample/asset model
  - Update `quickstart.md`:
    - Any examples showing image operations should use sample/asset vocabulary
  - Update `docs/src/` mdBook pages:
    - `docs/src/features/catalog-and-cli.md` — inspect output examples
    - `docs/src/features/format-registry.md` — plugin contract description
    - `docs/src/features/python-sdk.md` — builder/loader examples with sample/asset API
    - `docs/src/features/label-studio.md` — integration description
    - `docs/src/features/tui.md` — TUI description
    - Any other pages referencing "images"
  - Verify `mdbook build` succeeds

  **Must NOT do**:
  - Do NOT change mdBook configuration
  - Do NOT add new doc pages (unless needed for multi-asset concepts)

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation-focused task.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T23, T25)
  - **Parallel Group**: Wave 5
  - **Blocks**: None
  - **Blocked By**: T22

  **References**:

  **Pattern References**:
  - `README.md` — Repo landing page
  - `quickstart.md` — Getting started guide
  - `docs/src/` — mdBook source tree

  **Acceptance Criteria**:
  - [ ] README mentions sample/asset model and multi-view/multi-modal support
  - [ ] No "image_count" or image-centric language in docs (except when describing format-specific behavior)
  - [ ] `mdbook build` succeeds

  **QA Scenarios**:

  ```
  Scenario: mdBook builds successfully
    Tool: Bash
    Steps:
      1. Run: mdbook build
    Expected Result: Build succeeds, no broken links
    Failure Indicators: Build error, missing pages
    Evidence: .sisyphus/evidence/task-24-mdbook-build.txt

  Scenario: No stale "image_count" in docs
    Tool: Bash (grep)
    Steps:
      1. Run: grep -rn "image_count" docs/ README.md quickstart.md
      2. Verify zero matches
    Expected Result: No stale references
    Evidence: .sisyphus/evidence/task-24-no-stale-refs.txt
  ```

  **Commit**: YES
  - Message: `docs: update all documentation for sample/asset model`
  - Files: `README.md`, `quickstart.md`, `docs/src/**`
  - Pre-commit: `mdbook build`

- [ ] 25. BBox Serialization Consistency Fix

  **What to do**:
  - Verify that ALL BBox serialization across the codebase uses `{x, y, width, height}` format:
    - `crates/core/src/types/mod.rs` — `BBox { x, y, width, height }` — already correct
    - `crates/python/src/plugins/format.rs` — was `{x, y, width, height}` — verify after T16
    - `crates/python/src/sdk/builder.rs` — was `{x, y, w, h}` — fixed in T17, verify
  - Search entire codebase for `"w"` and `"h"` as JSON keys that might be BBox shorthand
  - This is a verification + cleanup task to ensure T16/T17 didn't miss anything

  **Must NOT do**:
  - Do NOT change BBox struct fields (they're already `width`/`height`)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification pass — should be nearly zero changes after T16/T17.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with T22, T23, T24)
  - **Parallel Group**: Wave 5
  - **Blocks**: None
  - **Blocked By**: T16, T17

  **References**:

  **Pattern References**:
  - `crates/core/src/types/mod.rs:96-102` — Canonical `BBox` struct with `width`, `height` fields
  - `crates/python/src/plugins/format.rs` — Check for any `"w"` or `"h"` shorthand
  - `crates/python/src/sdk/builder.rs` — Check for any `"w"` or `"h"` shorthand

  **Acceptance Criteria**:
  - [ ] No `{x, y, w, h}` pattern anywhere in codebase
  - [ ] All BBox JSON uses `{x, y, width, height}`

  **QA Scenarios**:

  ```
  Scenario: No w/h shorthand in BBox serialization
    Tool: Bash (grep)
    Steps:
      1. Run: grep -rn '"w"' crates/python/src/ crates/core/src/
      2. Filter for BBox-related matches
      3. Verify zero BBox-related matches
    Expected Result: No w/h shorthand
    Evidence: .sisyphus/evidence/task-25-bbox-consistency.txt
  ```

  **Commit**: YES (groups with T26)
  - Message: `fix(python): standardize BBox serialization and final quality pass`
  - Files: (any remaining fixes)
  - Pre-commit: `cargo test`

- [x] 26. Final Clippy + Fmt + Full Test Pass

  **What to do**:
  - Run full quality pipeline:
    1. `cargo fmt --all` — fix any formatting issues
    2. `cargo clippy --workspace --all-targets --all-features -- -D warnings` — fix ALL clippy warnings
    3. `cargo test` — run ENTIRE test suite, fix any failures
  - This is the final quality gate before the verification wave.
  - Fix ANY remaining compilation errors, warnings, or test failures from the entire migration.

  **Must NOT do**:
  - Do NOT skip any clippy warnings — fix them all
  - Do NOT suppress warnings with `#[allow(...)]` unless genuinely justified

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical quality pass — run commands, fix issues.
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (must run after all other tasks)
  - **Parallel Group**: Wave 5 (final)
  - **Blocks**: F1, F2, F3, F4
  - **Blocked By**: T1-T25

  **References**:

  **Pattern References**:
  - `AGENTS.md` — Build and lint commands section

  **Acceptance Criteria**:
  - [ ] `cargo fmt --all` produces no changes
  - [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` → 0 warnings
  - [ ] `cargo test` → all pass (0 failures)

  **QA Scenarios**:

  ```
  Scenario: Full quality pipeline passes
    Tool: Bash
    Steps:
      1. Run: cargo fmt --all
      2. Run: cargo clippy --workspace --all-targets --all-features -- -D warnings
      3. Run: cargo test
    Expected Result: All three commands succeed with zero issues
    Failure Indicators: Any fmt change, clippy warning, or test failure
    Evidence: .sisyphus/evidence/task-26-quality-pipeline.txt
  ```

  **Commit**: YES (groups with T25)
  - Message: `fix(python): standardize BBox serialization and final quality pass`
  - Files: (any remaining fixes)
  - Pre-commit: `cargo test && cargo clippy --workspace --all-targets --all-features -- -D warnings`

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `cargo fmt --all`, `cargo clippy --workspace --all-targets --all-features -- -D warnings`, `cargo test`. Review all changed files for: `unwrap()` in lib crates, empty catches, stale comments, unused imports. Check AI slop: excessive comments, over-abstraction, generic variable names.
  Output: `Build [PASS/FAIL] | Clippy [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (YOLO import → inspect → export roundtrip with sample/asset model). Test edge cases: empty dataset, single-image dataset, multi-asset sample. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Wave | Commit | Message | Files | Pre-commit |
|------|--------|---------|-------|------------|
| 1 | T1+T2+T3 | `refactor(core): replace image model with sample/asset/annotation schema` | types/mod.rs, db/mod.rs, error.rs | `cargo test -p dman-core` |
| 1 | T4+T5 | `refactor(core): update storage layout and virtual dataset types for sample model` | storage/mod.rs, virtual_dataset/, types/mod.rs | `cargo check -p dman-core` |
| 2 | T6+T7 | `refactor(core): rewrite DatasetService for sample/asset CRUD` | dataset/mod.rs, catalog/mod.rs | `cargo test -p dman-core` |
| 2 | T8 | `refactor(core): update DatasetOps for sample/asset model` | ops/mod.rs, ops/transforms.rs | `cargo test -p dman-core` |
| 2 | T9+T10 | `refactor(core): update virtual dataset eval and format traits` | virtual_dataset/, formats/mod.rs | `cargo test -p dman-core` |
| 3 | T11 | `refactor(core): update schema module for asset-aware validation` | schema/mod.rs | `cargo test -p dman-core` |
| 3 | T12+T13+T14 | `refactor(core): rewrite YOLO, COCO, HuggingFace adapters for sample model` | formats/yolo/, formats/coco/, formats/huggingface/ | `cargo test -p dman-core` |
| 3 | T15 | `test(core): add multi-asset roundtrip test infrastructure` | tests/ | `cargo test -p dman-core` |
| 4 | T16+T17+T18+T19 | `refactor(python): redesign plugin contract and SDK for sample/asset model` | python/src/ | `cargo test -p dman-python` |
| 4 | T20+T21 | `refactor(server): update API and Label Studio for sample/asset model` | server/src/ | `cargo test -p dman-server` |
| 5 | T22+T23 | `refactor(cli): update CLI and TUI for sample/asset model` | cli/src/ | `cargo test -p dman-cli` |
| 5 | T24 | `docs: update all documentation for sample/asset model` | docs/, README.md, quickstart.md | `mdbook build` |
| 5 | T25+T26 | `fix(python): standardize BBox serialization and final quality pass` | python/src/, cargo fmt | `cargo test && cargo clippy ...` |

---

## Success Criteria

### Verification Commands
```bash
cargo fmt --all                                              # Expected: no changes
cargo clippy --workspace --all-targets --all-features -- -D warnings  # Expected: 0 warnings
cargo test                                                   # Expected: all pass
cargo test -p dman-core --test roundtrip                     # Expected: all pass (including new multi-asset)
cargo test -p dman-cli --test integration                    # Expected: all pass
```

### Final Checklist
- [ ] All "Must Have" items present and tested
- [ ] All "Must NOT Have" items verified absent
- [ ] No remaining references to `Image` struct or `image_id` in runtime code (test fixtures excepted)
- [ ] Multi-asset sample can be created via Python SDK and inspected via CLI
- [ ] Classic YOLO/COCO/HF import produces correct 1:1 Sample→Asset mapping
- [ ] BBox serialization uses `{x, y, width, height}` everywhere
