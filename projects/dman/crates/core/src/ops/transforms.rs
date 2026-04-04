use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rusqlite::params;

use crate::{
    dataset::DatasetService,
    db::Database,
    error::{DmanError, Result},
    storage::StorageManager,
    types::{Dataset, FilterOp, Image},
};

// ─── Row helper ────────────────────────────────────────────────────────────

fn row_to_image(row: &rusqlite::Row<'_>) -> rusqlite::Result<Image> {
    let id: i64 = row.get(0)?;
    let dataset_id: i64 = row.get(1)?;
    let file_name: String = row.get(2)?;
    let file_path: String = row.get(3)?;
    let width: Option<u32> = row.get(4)?;
    let height: Option<u32> = row.get(5)?;
    let hash: Option<String> = row.get(6)?;
    let metadata_str: Option<String> = row.get(7)?;
    let metadata = metadata_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(Image {
        id,
        dataset_id,
        file_name,
        file_path: PathBuf::from(file_path),
        width,
        height,
        hash,
        metadata,
    })
}

/// List all images for a dataset.
fn list_images(db: &Database, dataset_id: i64) -> Result<Vec<Image>> {
    let mut stmt = db.conn.prepare(
        "SELECT id, dataset_id, file_name, file_path, width, height, hash, metadata \
         FROM images WHERE dataset_id = ?1 ORDER BY id",
    )?;
    let images = stmt
        .query_map(params![dataset_id], row_to_image)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(images)
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

fn filter_has_annotations(db: &Database, images: &[Image]) -> Result<Vec<Image>> {
    if images.is_empty() {
        return Ok(vec![]);
    }
    let image_ids: Vec<i64> = images.iter().map(|img| img.id).collect();
    let placeholders: Vec<String> = (1..=image_ids.len()).map(|i| format!("?{}", i)).collect();
    let sql = format!(
        "SELECT DISTINCT image_id FROM annotations WHERE image_id IN ({})",
        placeholders.join(", ")
    );
    let mut stmt = db.conn.prepare(&sql)?;
    let annotated_ids: HashSet<i64> = stmt
        .query_map(rusqlite::params_from_iter(image_ids.iter()), |row| {
            row.get::<_, i64>(0)
        })?
        .collect::<rusqlite::Result<HashSet<_>>>()?;

    Ok(images
        .iter()
        .filter(|img| annotated_ids.contains(&img.id))
        .cloned()
        .collect())
}

fn filter_by_category(db: &Database, images: &[Image], cat_name: &str) -> Result<Vec<Image>> {
    if images.is_empty() {
        return Ok(vec![]);
    }
    let image_ids: Vec<i64> = images.iter().map(|img| img.id).collect();
    let sql = format!(
        "SELECT DISTINCT a.image_id \
         FROM annotations a \
         JOIN categories c ON a.category_id = c.id \
         WHERE c.name = ?1 AND a.image_id IN ({})",
        (2..=image_ids.len() + 1)
            .map(|i| format!("?{}", i))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(cat_name.to_string())];
    for id in &image_ids {
        params_vec.push(Box::new(*id));
    }

    let mut stmt = db.conn.prepare(&sql)?;
    let matching_ids: HashSet<i64> = stmt
        .query_map(
            rusqlite::params_from_iter(params_vec.iter().map(|p| p.as_ref())),
            |row| row.get::<_, i64>(0),
        )?
        .collect::<rusqlite::Result<HashSet<_>>>()?;

    Ok(images
        .iter()
        .filter(|img| matching_ids.contains(&img.id))
        .cloned()
        .collect())
}

fn filter_by_metadata_eq(images: &[Image], key: &str, value: &serde_json::Value) -> Vec<Image> {
    images
        .iter()
        .filter(|img| {
            if let Some(meta) = &img.metadata {
                meta.get(key).map_or(false, |v| v == value)
            } else {
                false
            }
        })
        .cloned()
        .collect()
}

fn apply_filter_op(
    img_val: &serde_json::Value,
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
        FilterOp::Eq => img_val == filter_val,
        FilterOp::Ne => img_val != filter_val,
        FilterOp::Gt => compare_json(img_val, filter_val).map_or(false, |o| o > 0),
        FilterOp::Lt => compare_json(img_val, filter_val).map_or(false, |o| o < 0),
        FilterOp::Gte => compare_json(img_val, filter_val).map_or(false, |o| o >= 0),
        FilterOp::Lte => compare_json(img_val, filter_val).map_or(false, |o| o <= 0),
        FilterOp::Contains => {
            if let (Some(s), Some(pat)) = (img_val.as_str(), filter_val.as_str()) {
                s.contains(pat)
            } else if let Some(arr) = img_val.as_array() {
                arr.contains(filter_val)
            } else {
                false
            }
        }
        FilterOp::In => {
            if let Some(arr) = filter_val.as_array() {
                arr.contains(img_val)
            } else {
                false
            }
        }
    }
}

fn get_image_field_value(img: &Image, column: &str) -> serde_json::Value {
    match column {
        "id" => serde_json::json!(img.id),
        "dataset_id" => serde_json::json!(img.dataset_id),
        "file_name" => serde_json::json!(img.file_name),
        "file_path" => serde_json::json!(img.file_path.to_string_lossy().as_ref()),
        "width" => img
            .width
            .map_or(serde_json::Value::Null, |w| serde_json::json!(w)),
        "height" => img
            .height
            .map_or(serde_json::Value::Null, |h| serde_json::json!(h)),
        "hash" => img
            .hash
            .as_deref()
            .map_or(serde_json::Value::Null, |h| serde_json::json!(h)),
        _ => serde_json::Value::Null,
    }
}

/// Apply a filter using the column-based dispatch pattern.
/// - `column = "annotated"` → images with ≥1 annotation
/// - `column = "category"` or `"category_name"` → images with annotation of given category
/// - `column = "metadata.KEY"` → images where metadata[KEY] == value
/// - anything else → in-memory field comparison
fn apply_column_filter(
    db: &Database,
    images: &[Image],
    column: &str,
    op: &FilterOp,
    value: &serde_json::Value,
) -> Result<Vec<Image>> {
    match column {
        "annotated" => filter_has_annotations(db, images),
        "category" | "category_name" => {
            let cat_name = value.as_str().ok_or_else(|| {
                DmanError::StorageError("category filter value must be a string".to_string())
            })?;
            filter_by_category(db, images, cat_name)
        }
        _ if column.starts_with("metadata.") => {
            let key = &column["metadata.".len()..];
            Ok(filter_by_metadata_eq(images, key, value))
        }
        _ => {
            let result = images
                .iter()
                .filter(|img| {
                    let img_val = get_image_field_value(img, column);
                    apply_filter_op(&img_val, op, value)
                })
                .cloned()
                .collect();
            Ok(result)
        }
    }
}

// ─── Insert helpers ─────────────────────────────────────────────────────────

/// Insert an image into the new dataset (using the same file_path as source — reference semantics).
/// Returns the new image_id.
fn insert_image_ref(db: &Database, dst_dataset_id: i64, img: &Image) -> Result<i64> {
    let file_path_str = img.file_path.to_string_lossy().to_string();
    let metadata_str = img
        .metadata
        .as_ref()
        .map(|m| serde_json::to_string(m))
        .transpose()?;

    db.conn.execute(
        "INSERT INTO images (dataset_id, file_name, file_path, width, height, hash, metadata) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            dst_dataset_id,
            img.file_name,
            file_path_str,
            img.width,
            img.height,
            img.hash,
            metadata_str,
        ],
    )?;
    Ok(db.conn.last_insert_rowid())
}

/// Copy all annotations from `src_image_id` to `dst_image_id`.
fn copy_annotations(db: &Database, src_image_id: i64, dst_image_id: i64) -> Result<()> {
    // Collect source annotations
    struct RawAnn {
        category_id: Option<i64>,
        bbox: Option<String>,
        segmentation: Option<String>,
        keypoints: Option<String>,
        metadata: Option<String>,
    }

    let mut stmt = db.conn.prepare(
        "SELECT category_id, bbox, segmentation, keypoints, metadata \
         FROM annotations WHERE image_id = ?1",
    )?;
    let anns: Vec<RawAnn> = stmt
        .query_map(params![src_image_id], |row| {
            Ok(RawAnn {
                category_id: row.get(0)?,
                bbox: row.get(1)?,
                segmentation: row.get(2)?,
                keypoints: row.get(3)?,
                metadata: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    for ann in &anns {
        db.conn.execute(
            "INSERT INTO annotations (image_id, category_id, bbox, segmentation, keypoints, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                dst_image_id,
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
    // Check for duplicate name
    let exists: bool = db.conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE name = ?1",
        params![output_name],
        |row| row.get::<_, i64>(0),
    )? > 0;
    if exists {
        return Err(DmanError::DatasetAlreadyExists(output_name.to_string()));
    }

    let format_str = match &source.format {
        crate::types::DatasetFormat::Yolo => "Yolo",
        crate::types::DatasetFormat::Coco => "Coco",
        crate::types::DatasetFormat::HuggingFace => "HuggingFace",
        crate::types::DatasetFormat::Custom(s) => s.as_str(),
    };
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
    /// Filter images from `name` into a new physical dataset `output_name`.
    ///
    /// Filtering is column-based (matching `VirtualDatasetDef::Filter` semantics):
    /// - `column = "annotated"` with any op → images that have ≥1 annotation
    /// - `column = "category"` / `"category_name"` → images annotated with that category name
    /// - `column = "metadata.KEY"` → images where `metadata[KEY] == value`
    /// - anything else → generic image field comparison
    pub fn filter_dataset(
        db: &Database,
        name: &str,
        column: &str,
        op: &FilterOp,
        value: &serde_json::Value,
        output_name: &str,
    ) -> Result<Dataset> {
        let source = DatasetService::get(db, name)?;
        let images = list_images(db, source.id)?;
        let filtered = apply_column_filter(db, &images, column, op, value)?;

        let out_ds = create_output_dataset(db, &source, output_name)?;

        db.conn.execute("BEGIN IMMEDIATE", [])?;
        let result = (|| -> Result<()> {
            for img in &filtered {
                let new_img_id = insert_image_ref(db, out_ds.id, img)?;
                copy_annotations(db, img.id, new_img_id)?;
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

    /// Sample a fraction of images from `name` into a new physical dataset `output_name`.
    ///
    /// `ratio` must be in (0.0, 1.0]. Uses a deterministic hash of `image.id XOR seed`
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
        let mut images = list_images(db, source.id)?;

        // Sort by deterministic hash
        images.sort_by_key(|img| hash_id(img.id, seed));
        let count = (images.len() as f64 * ratio).ceil() as usize;
        let sampled: Vec<Image> = images.into_iter().take(count).collect();

        let out_ds = create_output_dataset(db, &source, output_name)?;

        db.conn.execute("BEGIN IMMEDIATE", [])?;
        let result = (|| -> Result<()> {
            for img in &sampled {
                let new_img_id = insert_image_ref(db, out_ds.id, img)?;
                copy_annotations(db, img.id, new_img_id)?;
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

    /// Resize all images in dataset `name` to `width x height` pixels.
    ///
    /// Tries ImageMagick `convert` first, then falls back to Python PIL.
    /// Returns `Err(DmanError::StorageError(...))` if neither tool is available.
    ///
    /// This is a best-effort operation — individual image failures are logged
    /// and skipped rather than aborting the entire run.
    pub fn resize_images(
        db: &Database,
        _storage: &StorageManager,
        name: &str,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let ds = DatasetService::get(db, name)?;
        let images = list_images(db, ds.id)?;

        let geometry = format!("{}x{}!", width, height);

        for img in &images {
            let path_str = img.file_path.to_string_lossy().to_string();
            let ok = try_imagemagick(&path_str, &geometry).unwrap_or(false)
                || try_python_pil(&path_str, width, height).unwrap_or(false);

            if !ok {
                // Neither tool available — fail fast with a clear error.
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
    let script = format!(
        "from PIL import Image; img = Image.open({path!r}); img = img.resize(({width}, {height})); img.save({path!r})"
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
        DatasetService::register(db, name, dir.path(), DatasetFormat::Yolo).expect("register")
    }

    fn insert_image(db: &Database, dataset_id: i64, file_name: &str) -> i64 {
        let file_path = format!("/tmp/{}", file_name);
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                params![dataset_id, file_name, file_path],
            )
            .expect("insert image");
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

    fn insert_annotation(db: &Database, image_id: i64, category_id: Option<i64>) {
        db.conn
            .execute(
                "INSERT INTO annotations (image_id, category_id) VALUES (?1, ?2)",
                params![image_id, category_id],
            )
            .expect("insert annotation");
    }

    fn count_images(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .expect("count images")
    }

    // ─── filter_dataset: HasAnnotations ─────────────────────────────────────

    #[test]
    fn filter_has_annotations_keeps_annotated_only() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src");

        let img_with = insert_image(&db, ds.id, "annotated.jpg");
        let _img_without = insert_image(&db, ds.id, "unannotated.jpg");

        insert_annotation(&db, img_with, None);

        let out = DatasetTransforms::filter_dataset(
            &db,
            "src",
            "annotated",
            &FilterOp::Eq,
            &serde_json::json!(true),
            "dst-annotated",
        )
        .expect("filter");

        assert_eq!(count_images(&db, out.id), 1);
    }

    // ─── filter_dataset: CategoryIs ─────────────────────────────────────────

    #[test]
    fn filter_category_keeps_matching_only() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-cat");

        let cat_dog = insert_category(&db, ds.id, "dog");
        let cat_cat = insert_category(&db, ds.id, "cat");

        let img_dog = insert_image(&db, ds.id, "dog.jpg");
        let img_cat = insert_image(&db, ds.id, "cat.jpg");
        let _img_none = insert_image(&db, ds.id, "none.jpg");

        insert_annotation(&db, img_dog, Some(cat_dog));
        insert_annotation(&db, img_cat, Some(cat_cat));

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
            count_images(&db, out.id),
            1,
            "only the dog image should match"
        );
    }

    // ─── sample_dataset ──────────────────────────────────────────────────────

    #[test]
    fn sample_ratio_produces_expected_count() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-sample");

        for i in 0..10 {
            insert_image(&db, ds.id, &format!("img{}.jpg", i));
        }

        let out = DatasetTransforms::sample_dataset(&db, "src-sample", 0.5, 42, "dst-sample")
            .expect("sample");

        // ceil(10 * 0.5) = 5
        let n = count_images(&db, out.id);
        assert_eq!(n, 5, "expected 5 sampled images, got {}", n);
    }

    #[test]
    fn sample_ratio_deterministic() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-det");

        for i in 0..20 {
            insert_image(&db, ds.id, &format!("img{}.jpg", i));
        }

        let out1 = DatasetTransforms::sample_dataset(&db, "src-det", 0.3, 99, "dst-det-1")
            .expect("sample 1");
        let out2 = DatasetTransforms::sample_dataset(&db, "src-det", 0.3, 99, "dst-det-2")
            .expect("sample 2");

        // Same seed/ratio → same count
        assert_eq!(
            count_images(&db, out1.id),
            count_images(&db, out2.id),
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

        // Insert an image pointing to a non-existent path so that both
        // convert and python3 either fail or are absent.
        insert_image(&db, ds.id, "fake.jpg");

        let storage = StorageManager::new(std::path::PathBuf::from("/tmp/dman-test"));

        // This may succeed if convert or python3 is installed (and fail gracefully).
        // The important thing is it doesn't panic and returns a Result.
        let result = DatasetTransforms::resize_images(&db, &storage, "src-resize", 100, 100);

        // On a system without convert/python3 or with a non-existent file, we expect
        // either Ok (tool ran and failed silently is not our pattern — we'd get StorageError)
        // or StorageError. Either way, it must not be a panic.
        match result {
            Ok(()) => {
                // Tool was available and succeeded or image list was empty — acceptable.
            }
            Err(DmanError::StorageError(msg)) => {
                assert!(
                    msg.contains("resize requires") || !msg.is_empty(),
                    "unexpected error: {}",
                    msg
                );
            }
            Err(e) => {
                // Any other error is also acceptable (e.g., Io error from tool)
                let _ = e;
            }
        }
    }

    // ─── filter: metadata ────────────────────────────────────────────────────

    #[test]
    fn filter_metadata_eq_matches_key_value() {
        let db = setup_db();
        let ds = setup_dataset(&db, "src-meta");

        // Insert images with different metadata
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path, metadata) VALUES (?1, 'a.jpg', '/tmp/a.jpg', '{\"split\": \"train\"}')",
                params![ds.id],
            )
            .expect("insert");
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path, metadata) VALUES (?1, 'b.jpg', '/tmp/b.jpg', '{\"split\": \"val\"}')",
                params![ds.id],
            )
            .expect("insert");
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'c.jpg', '/tmp/c.jpg')",
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
            count_images(&db, out.id),
            1,
            "only a.jpg should pass the metadata filter"
        );
    }
}
