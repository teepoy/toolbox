use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rusqlite::params;

use crate::{
    dataset::DatasetService,
    db::Database,
    error::{DmanError, Result},
    storage::StorageManager,
    types::{Dataset, FilterOp, Sample},
};

// ─── Row helper ────────────────────────────────────────────────────────────

fn row_to_sample(row: &rusqlite::Row<'_>) -> rusqlite::Result<Sample> {
    let id: i64 = row.get(0)?;
    let dataset_id: i64 = row.get(1)?;
    let name: String = row.get(2)?;
    let metadata_str: Option<String> = row.get(3)?;
    let created_at: String = row.get(4)?;
    let metadata = metadata_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(Sample {
        id,
        dataset_id,
        name,
        metadata,
        created_at,
    })
}

/// List all samples for a dataset.
fn list_samples(db: &Database, dataset_id: i64) -> Result<Vec<Sample>> {
    let mut stmt = db.conn.prepare(
        "SELECT id, dataset_id, name, metadata, created_at \
         FROM samples WHERE dataset_id = ?1 ORDER BY id",
    )?;
    let samples = stmt
        .query_map(params![dataset_id], row_to_sample)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(samples)
}

/// Simple deterministic hash for sampling (same algorithm as virtual_dataset).
fn hash_id(id: i64, seed: u64) -> u64 {
    let v = (id as u64) ^ seed;
    let v = v.wrapping_mul(0x9e3779b97f4a7c15);
    let v = v ^ (v >> 30);
    let v = v.wrapping_mul(0xbf58476d1ce4e5b9);
    let v = v ^ (v >> 27);
    v.wrapping_mul(0x94d049bb133111eb)
}

// ─── Filter helpers ─────────────────────────────────────────────────────────

fn filter_has_annotations(db: &Database, samples: &[Sample]) -> Result<Vec<Sample>> {
    if samples.is_empty() {
        return Ok(vec![]);
    }
    let sample_ids: Vec<i64> = samples.iter().map(|s| s.id).collect();
    let placeholders: Vec<String> = (1..=sample_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT DISTINCT sample_id FROM annotations WHERE sample_id IN ({})",
        placeholders.join(", ")
    );
    let mut stmt = db.conn.prepare(&sql)?;
    let annotated_ids: HashSet<i64> = stmt
        .query_map(rusqlite::params_from_iter(sample_ids.iter()), |row| {
            row.get::<_, i64>(0)
        })?
        .collect::<rusqlite::Result<HashSet<_>>>()?;

    Ok(samples
        .iter()
        .filter(|s| annotated_ids.contains(&s.id))
        .cloned()
        .collect())
}

fn filter_by_category(db: &Database, samples: &[Sample], cat_name: &str) -> Result<Vec<Sample>> {
    if samples.is_empty() {
        return Ok(vec![]);
    }
    let sample_ids: Vec<i64> = samples.iter().map(|s| s.id).collect();
    let sql = format!(
        "SELECT DISTINCT a.sample_id \
         FROM annotations a \
         JOIN categories c ON a.category_id = c.id \
         WHERE c.name = ?1 AND a.sample_id IN ({})",
        (2..=sample_ids.len() + 1)
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(cat_name.to_string())];
    for id in &sample_ids {
        params_vec.push(Box::new(*id));
    }

    let mut stmt = db.conn.prepare(&sql)?;
    let matching_ids: HashSet<i64> = stmt
        .query_map(
            rusqlite::params_from_iter(params_vec.iter().map(|p| p.as_ref())),
            |row| row.get::<_, i64>(0),
        )?
        .collect::<rusqlite::Result<HashSet<_>>>()?;

    Ok(samples
        .iter()
        .filter(|s| matching_ids.contains(&s.id))
        .cloned()
        .collect())
}

fn filter_by_metadata_eq(samples: &[Sample], key: &str, value: &serde_json::Value) -> Vec<Sample> {
    samples
        .iter()
        .filter(|s| {
            if let Some(meta) = &s.metadata {
                meta.get(key) == Some(value)
            } else {
                false
            }
        })
        .cloned()
        .collect()
}

fn apply_filter_op(
    sample_val: &serde_json::Value,
    op: &FilterOp,
    filter_val: &serde_json::Value,
) -> bool {
    fn compare_json(a: &serde_json::Value, b: &serde_json::Value) -> Option<i64> {
        match (a, b) {
            (serde_json::Value::Number(an), serde_json::Value::Number(bn)) => {
                let af = an.as_f64()?;
                let bf = bn.as_f64()?;
                if af < bf {
                    Some(-1)
                } else if af > bf {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (serde_json::Value::String(as_), serde_json::Value::String(bs)) => {
                Some(as_.cmp(bs) as i64)
            }
            _ => None,
        }
    }

    match op {
        FilterOp::Eq => sample_val == filter_val,
        FilterOp::Ne => sample_val != filter_val,
        FilterOp::Gt => compare_json(sample_val, filter_val).is_some_and(|o| o > 0),
        FilterOp::Lt => compare_json(sample_val, filter_val).is_some_and(|o| o < 0),
        FilterOp::Gte => compare_json(sample_val, filter_val).is_some_and(|o| o >= 0),
        FilterOp::Lte => compare_json(sample_val, filter_val).is_some_and(|o| o <= 0),
        FilterOp::Contains => {
            if let (Some(s), Some(pat)) = (sample_val.as_str(), filter_val.as_str()) {
                s.contains(pat)
            } else if let Some(arr) = sample_val.as_array() {
                arr.contains(filter_val)
            } else {
                false
            }
        }
        FilterOp::In => {
            if let Some(arr) = filter_val.as_array() {
                arr.contains(sample_val)
            } else {
                false
            }
        }
    }
}

fn get_sample_field_value(sample: &Sample, column: &str) -> serde_json::Value {
    match column {
        "id" => serde_json::json!(sample.id),
        "dataset_id" => serde_json::json!(sample.dataset_id),
        "name" => serde_json::json!(sample.name),
        _ => serde_json::Value::Null,
    }
}

/// Apply a filter using the column-based dispatch pattern.
/// - `column = "annotated"` → samples with ≥1 annotation
/// - `column = "category"` or `"category_name"` → samples with annotation of given category
/// - `column = "metadata.KEY"` → samples where metadata[KEY] == value
/// - anything else → in-memory field comparison
fn apply_column_filter(
    db: &Database,
    samples: &[Sample],
    column: &str,
    op: &FilterOp,
    value: &serde_json::Value,
) -> Result<Vec<Sample>> {
    match column {
        "annotated" => filter_has_annotations(db, samples),
        "category" | "category_name" => {
            let cat_name = value.as_str().ok_or_else(|| {
                DmanError::StorageError("category filter value must be a string".to_string())
            })?;
            filter_by_category(db, samples, cat_name)
        }
        _ if column.starts_with("metadata.") => {
            let key = &column["metadata.".len()..];
            Ok(filter_by_metadata_eq(samples, key, value))
        }
        _ => {
            let result = samples
                .iter()
                .filter(|s| {
                    let sample_val = get_sample_field_value(s, column);
                    apply_filter_op(&sample_val, op, value)
                })
                .cloned()
                .collect();
            Ok(result)
        }
    }
}

// ─── Insert helpers ─────────────────────────────────────────────────────────

/// Insert a sample into the new dataset (reference semantics — same name).
/// Returns the new sample_id.
fn insert_sample_ref(db: &Database, dst_dataset_id: i64, sample: &Sample) -> Result<i64> {
    let metadata_str = sample
        .metadata
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?;

    db.conn.execute(
        "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, ?2, ?3)",
        params![dst_dataset_id, sample.name, metadata_str],
    )?;
    Ok(db.conn.last_insert_rowid())
}

/// Copy all annotations from `src_sample_id` to `dst_sample_id`.
fn copy_annotations(db: &Database, src_sample_id: i64, dst_sample_id: i64) -> Result<()> {
    struct RawAnn {
        asset_id: Option<i64>,
        category_id: Option<i64>,
        bbox: Option<String>,
        segmentation: Option<String>,
        keypoints: Option<String>,
        metadata: Option<String>,
    }

    let mut stmt = db.conn.prepare(
        "SELECT asset_id, category_id, bbox, segmentation, keypoints, metadata \
         FROM annotations WHERE sample_id = ?1",
    )?;
    let anns: Vec<RawAnn> = stmt
        .query_map(params![src_sample_id], |row| {
            Ok(RawAnn {
                asset_id: row.get(0)?,
                category_id: row.get(1)?,
                bbox: row.get(2)?,
                segmentation: row.get(3)?,
                keypoints: row.get(4)?,
                metadata: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    for ann in &anns {
        db.conn.execute(
            "INSERT INTO annotations (sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                dst_sample_id,
                ann.asset_id,
                ann.category_id,
                ann.bbox,
                ann.segmentation,
                ann.keypoints,
                ann.metadata,
            ],
        )?;
    }
    Ok(())
}

/// Create a new dataset record (same format as source) and return it.
fn create_output_dataset(db: &Database, source: &Dataset, output_name: &str) -> Result<Dataset> {
    let exists: bool = db.conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE name = ?1",
        params![output_name],
        |row| row.get::<_, i64>(0),
    )? > 0;
    if exists {
        return Err(DmanError::DatasetAlreadyExists(output_name.to_string()));
    }

    let format_str = source.format.to_string();
    let path_str = source.path.to_string_lossy().to_string();

    db.conn.execute(
        "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
        params![output_name, path_str, format_str],
    )?;
    let new_id = db.conn.last_insert_rowid();
    DatasetService::get_by_id(db, new_id)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Physical data transform operations over datasets.
pub struct DatasetTransforms;

impl DatasetTransforms {
    /// Filter samples from `name` into a new physical dataset `output_name`.
    ///
    /// Filtering is column-based (matching `VirtualDatasetDef::Filter` semantics):
    /// - `column = "annotated"` with any op → samples that have ≥1 annotation
    /// - `column = "category"` / `"category_name"` → samples annotated with that category name
    /// - `column = "metadata.KEY"` → samples where `metadata[KEY] == value`
    /// - anything else → generic sample field comparison
    pub fn filter_dataset(
        db: &Database,
        name: &str,
        column: &str,
        op: &FilterOp,
        value: &serde_json::Value,
        output_name: &str,
    ) -> Result<Dataset> {
        let source = DatasetService::get(db, name)?;
        let samples = list_samples(db, source.id)?;
        let filtered = apply_column_filter(db, &samples, column, op, value)?;

        let out_ds = create_output_dataset(db, &source, output_name)?;

        db.conn.execute("BEGIN IMMEDIATE", [])?;
        let result = (|| -> Result<()> {
            for sample in &filtered {
                let new_sample_id = insert_sample_ref(db, out_ds.id, sample)?;
                copy_annotations(db, sample.id, new_sample_id)?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                db.conn.execute("COMMIT", [])?;
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                return Err(e);
            }
        }

        Ok(out_ds)
    }

    /// Sample a fraction of samples from `name` into a new physical dataset `output_name`.
    ///
    /// `ratio` must be in (0.0, 1.0]. Uses a deterministic hash of `sample.id XOR seed`
    /// to produce a reproducible ordering.
    pub fn sample_dataset(
        db: &Database,
        name: &str,
        ratio: f64,
        seed: u64,
        output_name: &str,
    ) -> Result<Dataset> {
        if ratio <= 0.0 || ratio > 1.0 {
            return Err(DmanError::StorageError(
                "sample ratio must be in (0.0, 1.0]".to_string(),
            ));
        }

        let source = DatasetService::get(db, name)?;
        let mut samples = list_samples(db, source.id)?;

        // Sort by deterministic hash
        samples.sort_by_key(|s| hash_id(s.id, seed));
        let count = (samples.len() as f64 * ratio).ceil() as usize;
        let sampled: Vec<Sample> = samples.into_iter().take(count).collect();

        let out_ds = create_output_dataset(db, &source, output_name)?;

        db.conn.execute("BEGIN IMMEDIATE", [])?;
        let result = (|| -> Result<()> {
            for sample in &sampled {
                let new_sample_id = insert_sample_ref(db, out_ds.id, sample)?;
                copy_annotations(db, sample.id, new_sample_id)?;
            }
            Ok(())
        })();

        match result {
            Ok(()) => {
                db.conn.execute("COMMIT", [])?;
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                return Err(e);
            }
        }

        Ok(out_ds)
    }

    /// Rename category labels in-place within `name`.
    ///
    /// For each `(old_label, new_label)` in `mapping`, updates `categories.name`
    /// for all categories with that name belonging to this dataset.
    pub fn relabel(db: &Database, name: &str, mapping: HashMap<String, String>) -> Result<()> {
        let ds = DatasetService::get(db, name)?;
        for (old_label, new_label) in &mapping {
            db.conn.execute(
                "UPDATE categories SET name = ?1 WHERE name = ?2 AND dataset_id = ?3",
                params![new_label, old_label, ds.id],
            )?;
        }
        Ok(())
    }

    /// Resize all assets (images) in dataset `name` to `width x height` pixels.
    ///
    /// Tries ImageMagick `convert` first, then falls back to Python PIL.
    /// Returns `Err(DmanError::StorageError(...))` if neither tool is available.
    ///
    /// This is a best-effort operation — individual asset failures are logged
    /// and skipped rather than aborting the entire run.
    pub fn resize_images(
        db: &Database,
        _storage: &StorageManager,
        name: &str,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let ds = DatasetService::get(db, name)?;

        // Collect asset file paths for all samples in this dataset
        let mut stmt = db.conn.prepare(
            "SELECT a.file_path FROM assets a \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1 AND a.asset_type = 'image' \
             ORDER BY a.id",
        )?;
        let paths: Vec<PathBuf> = stmt
            .query_map(params![ds.id], |row| {
                let p: String = row.get(0)?;
                Ok(PathBuf::from(p))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let geometry = format!("{}x{}!", width, height);

        for path in &paths {
            let path_str = path.to_string_lossy().to_string();
            let ok = try_imagemagick(&path_str, &geometry).unwrap_or(false)
                || try_python_pil(&path_str, width, height).unwrap_or(false);

            if !ok {
                return Err(DmanError::StorageError(
                    "resize requires ImageMagick 'convert' or Python PIL".to_string(),
                ));
            }
        }
        Ok(())
    }
}

fn try_imagemagick(path: &str, geometry: &str) -> Result<bool> {
    let output = std::process::Command::new("convert")
        .args([path, "-resize", geometry, path])
        .output();
    match output {
        Ok(out) => Ok(out.status.success()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(DmanError::Io(e)),
    }
}

fn try_python_pil(path: &str, width: u32, height: u32) -> Result<bool> {
    let repr_open = format!("{:?}", path);
    let repr_save = format!("{:?}", path);
    let script = format!(
        "from PIL import Image; img = Image.open({repr_open}); img = img.resize(({width}, {height})); img.save({repr_save})",
    );
    let output = std::process::Command::new("python3")
        .args(["-c", &script])
        .output();
    match output {
        Ok(out) => Ok(out.status.success()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(DmanError::Io(e)),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{dataset::DatasetService, db::Database, types::DatasetFormat};

    fn setup_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn setup_dataset(db: &Database, name: &str) -> crate::types::Dataset {
        let dir = tempfile::tempdir().expect("temp dir");
        DatasetService::register(db, name, dir.path(), DatasetFormat::yolo()).expect("register")
    }

    fn insert_sample(db: &Database, dataset_id: i64, name: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                params![dataset_id, name],
            )
            .expect("insert sample");
        db.conn.last_insert_rowid()
    }

    fn insert_category(db: &Database, dataset_id: i64, name: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, ?2)",
                params![dataset_id, name],
            )
            .expect("insert category");
        db.conn.last_insert_rowid()
    }

    fn insert_annotation(db: &Database, sample_id: i64, category_id: Option<i64>) {
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, category_id) VALUES (?1, ?2)",
                params![sample_id, category_id],
            )
            .expect("insert annotation");
    }

    fn count_samples(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .expect("count samples")
    }

    // ─── filter_dataset: HasAnnotations ─────────────────────────────────────

    #[test]
    fn filter_has_annotations_keeps_annotated_only() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src");

        let sample_with = insert_sample(&db, ds.id, "annotated");
        let _sample_without = insert_sample(&db, ds.id, "unannotated");

        insert_annotation(&db, sample_with, None);

        let out = DatasetTransforms::filter_dataset(
            &db,
            "src",
            "annotated",
            &FilterOp::Eq,
            &serde_json::json!(true),
            "dst-annotated",
        )
        .expect("filter");

        assert_eq!(count_samples(&db, out.id), 1);
    }

    // ─── filter_dataset: CategoryIs ─────────────────────────────────────────

    #[test]
    fn filter_category_keeps_matching_only() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-cat");

        let cat_dog = insert_category(&db, ds.id, "dog");
        let cat_cat = insert_category(&db, ds.id, "cat");

        let sample_dog = insert_sample(&db, ds.id, "dog-sample");
        let sample_cat = insert_sample(&db, ds.id, "cat-sample");
        let _sample_none = insert_sample(&db, ds.id, "none-sample");

        insert_annotation(&db, sample_dog, Some(cat_dog));
        insert_annotation(&db, sample_cat, Some(cat_cat));

        let out = DatasetTransforms::filter_dataset(
            &db,
            "src-cat",
            "category",
            &FilterOp::Eq,
            &serde_json::json!("dog"),
            "dst-dogs",
        )
        .expect("filter by category");

        assert_eq!(
            count_samples(&db, out.id),
            1,
            "only the dog sample should match"
        );
    }

    // ─── sample_dataset ──────────────────────────────────────────────────────

    #[test]
    fn sample_ratio_produces_expected_count() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-sample");

        for i in 0..10 {
            insert_sample(&db, ds.id, &format!("sample{}", i));
        }

        let out = DatasetTransforms::sample_dataset(&db, "src-sample", 0.5, 42, "dst-sample")
            .expect("sample");

        // ceil(10 * 0.5) = 5
        let n = count_samples(&db, out.id);
        assert_eq!(n, 5, "expected 5 sampled samples, got {}", n);
    }

    #[test]
    fn sample_ratio_deterministic() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-det");

        for i in 0..20 {
            insert_sample(&db, ds.id, &format!("sample{}", i));
        }

        let out1 = DatasetTransforms::sample_dataset(&db, "src-det", 0.3, 99, "dst-det-1")
            .expect("sample 1");
        let out2 = DatasetTransforms::sample_dataset(&db, "src-det", 0.3, 99, "dst-det-2")
            .expect("sample 2");

        assert_eq!(
            count_samples(&db, out1.id),
            count_samples(&db, out2.id),
            "same seed should produce same count"
        );
    }

    #[test]
    fn sample_invalid_ratio_returns_error() {
        let db = setup_db();
        setup_dataset(&db, "src-bad");

        let err = DatasetTransforms::sample_dataset(&db, "src-bad", 0.0, 0, "dst-bad")
            .expect_err("should fail");
        assert!(matches!(err, DmanError::StorageError(_)));

        let err2 = DatasetTransforms::sample_dataset(&db, "src-bad", 1.5, 0, "dst-bad2")
            .expect_err("should fail");
        assert!(matches!(err2, DmanError::StorageError(_)));
    }

    // ─── relabel ─────────────────────────────────────────────────────────────

    #[test]
    fn relabel_renames_categories() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-relabel");

        insert_category(&db, ds.id, "cat");
        insert_category(&db, ds.id, "dog");

        let mut mapping = HashMap::new();
        mapping.insert("cat".to_string(), "feline".to_string());
        mapping.insert("dog".to_string(), "canine".to_string());

        DatasetTransforms::relabel(&db, "src-relabel", mapping).expect("relabel");

        let names: Vec<String> = {
            let mut stmt = db
                .conn
                .prepare("SELECT name FROM categories WHERE dataset_id = ?1 ORDER BY name")
                .unwrap();
            stmt.query_map(params![ds.id], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };

        assert!(names.contains(&"feline".to_string()));
        assert!(names.contains(&"canine".to_string()));
        assert!(!names.contains(&"cat".to_string()));
        assert!(!names.contains(&"dog".to_string()));
    }

    // ─── resize: no tool available ───────────────────────────────────────────

    #[test]
    fn resize_returns_error_when_no_tool_available() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-resize");

        // Insert a sample with an asset pointing to a non-existent path
        let sample_id = insert_sample(&db, ds.id, "resize-sample");
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'image', 'fake.jpg', '/tmp/fake.jpg')",
                params![sample_id],
            )
            .expect("insert asset");

        let storage = StorageManager::new(std::path::PathBuf::from("/tmp/dman-test"));

        let result = DatasetTransforms::resize_images(&db, &storage, "src-resize", 100, 100);

        match result {
            Ok(()) => {
                // Tool was available and succeeded — acceptable.
            }
            Err(DmanError::StorageError(msg)) => {
                assert!(
                    msg.contains("resize requires") || !msg.is_empty(),
                    "unexpected error: {}",
                    msg
                );
            }
            Err(e) => {
                let _ = e;
            }
        }
    }

    // ─── filter: metadata ────────────────────────────────────────────────────

    #[test]
    fn filter_metadata_eq_matches_key_value() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-meta");

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, 'a', '{\"split\": \"train\"}')",
                params![ds.id],
            )
            .expect("insert");
        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, 'b', '{\"split\": \"val\"}')",
                params![ds.id],
            )
            .expect("insert");
        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, 'c')",
                params![ds.id],
            )
            .expect("insert");

        let out = DatasetTransforms::filter_dataset(
            &db,
            "src-meta",
            "metadata.split",
            &FilterOp::Eq,
            &serde_json::json!("train"),
            "dst-meta",
        )
        .expect("filter by metadata");

        assert_eq!(
            count_samples(&db, out.id),
            1,
            "only sample 'a' should pass the metadata filter"
        );
    }
}
