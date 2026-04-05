use std::path::{Path, PathBuf};
use std::str::FromStr;

use rusqlite::params;

use crate::{
    db::Database,
    error::{DmanError, Result},
    types::{Annotation, Asset, AssetType, Category, Dataset, DatasetFormat, Sample},
};

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub dataset: Dataset,
    pub sample_count: u64,
    pub asset_count: u64,
    pub annotation_count: u64,
    pub category_count: u64,
    pub disk_size_bytes: u64,
}

fn row_to_dataset(row: &rusqlite::Row<'_>) -> rusqlite::Result<Dataset> {
    let id: i64 = row.get(0)?;
    let name: String = row.get(1)?;
    let path: String = row.get(2)?;
    let format_str: String = row.get(3)?;
    let schema_path: Option<String> = row.get(4)?;
    let created_at: String = row.get(5)?;
    let updated_at: Option<String> = row.get(6)?;
    let metadata_str: Option<String> = row.get(7)?;

    let format = DatasetFormat::from(format_str);
    let schema_path = schema_path.map(PathBuf::from);
    let metadata = metadata_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;

    Ok(Dataset {
        id,
        name,
        path: PathBuf::from(path),
        format,
        schema_path,
        created_at,
        updated_at,
        metadata,
    })
}

fn row_to_sample(row: &rusqlite::Row<'_>) -> rusqlite::Result<Sample> {
    let id: i64 = row.get(0)?;
    let dataset_id: i64 = row.get(1)?;
    let name: String = row.get(2)?;
    let metadata_str: Option<String> = row.get(3)?;
    let created_at: String = row.get(4)?;

    let metadata = metadata_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;

    Ok(Sample {
        id,
        dataset_id,
        name,
        metadata,
        created_at,
    })
}

fn row_to_asset(row: &rusqlite::Row<'_>) -> rusqlite::Result<Asset> {
    let id: i64 = row.get(0)?;
    let sample_id: i64 = row.get(1)?;
    let asset_type_str: String = row.get(2)?;
    let file_name: String = row.get(3)?;
    let file_path_str: String = row.get(4)?;
    let width: Option<i64> = row.get(5)?;
    let height: Option<i64> = row.get(6)?;
    let hash: Option<String> = row.get(7)?;
    let metadata_str: Option<String> = row.get(8)?;

    let asset_type =
        AssetType::from_str(&asset_type_str).unwrap_or(AssetType::Other(asset_type_str));
    let metadata = metadata_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    8,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;

    Ok(Asset {
        id,
        sample_id,
        asset_type,
        file_name,
        file_path: PathBuf::from(file_path_str),
        width: width.map(|w| w as u32),
        height: height.map(|h| h as u32),
        hash,
        metadata,
    })
}

fn row_to_annotation(row: &rusqlite::Row<'_>) -> rusqlite::Result<Annotation> {
    let id: i64 = row.get(0)?;
    let sample_id: i64 = row.get(1)?;
    let asset_id: Option<i64> = row.get(2)?;
    let category_id: Option<i64> = row.get(3)?;
    let bbox_str: Option<String> = row.get(4)?;
    let seg_str: Option<String> = row.get(5)?;
    let kp_str: Option<String> = row.get(6)?;
    let meta_str: Option<String> = row.get(7)?;

    let bbox = bbox_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    4,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;
    let segmentation = seg_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    5,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;
    let keypoints = kp_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    6,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;
    let metadata = meta_str
        .as_deref()
        .map(|s| {
            serde_json::from_str(s).map_err(|e| {
                rusqlite::Error::FromSqlConversionFailure(
                    7,
                    rusqlite::types::Type::Text,
                    Box::new(e),
                )
            })
        })
        .transpose()?;

    Ok(Annotation {
        id,
        sample_id,
        asset_id,
        category_id,
        bbox,
        segmentation,
        keypoints,
        metadata,
    })
}

pub struct DatasetService;

impl DatasetService {
    /// Register a new dataset. Returns `DmanError::StorageError` if `path` does not exist,
    /// or `DmanError::DatasetAlreadyExists` if `name` is already taken.
    pub fn register(
        db: &Database,
        name: &str,
        path: &Path,
        format: DatasetFormat,
    ) -> Result<Dataset> {
        if !path.exists() {
            return Err(DmanError::StorageError(format!(
                "path does not exist: {}",
                path.display()
            )));
        }

        let exists: bool = db.conn.query_row(
            "SELECT COUNT(*) FROM datasets WHERE name = ?1",
            params![name],
            |row| row.get::<_, i64>(0),
        )? > 0;

        if exists {
            return Err(DmanError::DatasetAlreadyExists(name.to_string()));
        }

        let format_str = format.to_string();
        let path_str = path.to_string_lossy().to_string();

        db.conn.execute(
            "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
            params![name, path_str, format_str],
        )?;

        let id = db.conn.last_insert_rowid();
        Self::get_by_id(db, id)
    }

    pub fn list(db: &Database) -> Result<Vec<Dataset>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets ORDER BY id",
        )?;

        let rows = stmt
            .query_map([], row_to_dataset)?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        Ok(rows)
    }

    pub fn get(db: &Database, name: &str) -> Result<Dataset> {
        let result = db.conn.query_row(
            "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets WHERE name = ?1",
            params![name],
            row_to_dataset,
        );

        match result {
            Ok(ds) => Ok(ds),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(DmanError::DatasetNotFound(name.to_string()))
            }
            Err(e) => Err(DmanError::Database(e)),
        }
    }

    pub fn get_by_id(db: &Database, id: i64) -> Result<Dataset> {
        let result = db.conn.query_row(
            "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets WHERE id = ?1",
            params![id],
            row_to_dataset,
        );

        match result {
            Ok(ds) => Ok(ds),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(DmanError::DatasetNotFound(format!("id={}", id)))
            }
            Err(e) => Err(DmanError::Database(e)),
        }
    }

    pub fn remove(db: &Database, name: &str) -> Result<()> {
        let ds = Self::get(db, name)?;
        let dataset_id = ds.id;

        // Delete in dependency order: patches → embeddings → predictions → annotations → assets → samples
        db.conn.execute(
            "DELETE FROM patches WHERE asset_id IN (SELECT id FROM assets WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1))",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM embeddings WHERE asset_id IN (SELECT id FROM assets WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1))",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM predictions WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM annotations WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM assets WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM samples WHERE dataset_id = ?1",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM categories WHERE dataset_id = ?1",
            params![dataset_id],
        )?;

        db.conn
            .execute("DELETE FROM datasets WHERE id = ?1", params![dataset_id])?;

        Ok(())
    }

    pub fn update_metadata(db: &Database, name: &str, metadata: serde_json::Value) -> Result<()> {
        let _ = Self::get(db, name)?;

        let metadata_str = serde_json::to_string(&metadata)?;
        db.conn.execute(
            "UPDATE datasets SET metadata = ?1, updated_at = CURRENT_TIMESTAMP WHERE name = ?2",
            params![metadata_str, name],
        )?;

        Ok(())
    }

    // ─── Sample CRUD ─────────────────────────────────────────────────────────

    /// Insert a new sample into the given dataset. Returns the new sample's ID.
    pub fn add_sample(
        db: &Database,
        dataset_id: i64,
        name: &str,
        metadata: Option<&serde_json::Value>,
    ) -> Result<i64> {
        let metadata_str = metadata.map(serde_json::to_string).transpose()?;
        db.conn.execute(
            "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, ?2, ?3)",
            params![dataset_id, name, metadata_str],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    /// Retrieve all samples for a dataset.
    pub fn get_samples(db: &Database, dataset_id: i64) -> Result<Vec<Sample>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, dataset_id, name, metadata, created_at FROM samples WHERE dataset_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![dataset_id], row_to_sample)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Count samples in a dataset.
    pub fn get_sample_count(db: &Database, dataset_id: i64) -> Result<i64> {
        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    /// Delete a sample (cascades to its assets, annotations, embeddings, patches, predictions).
    pub fn remove_sample(db: &Database, sample_id: i64) -> Result<()> {
        let rows_affected = db
            .conn
            .execute("DELETE FROM samples WHERE id = ?1", params![sample_id])?;
        if rows_affected == 0 {
            return Err(DmanError::SampleNotFound(format!("id={}", sample_id)));
        }
        Ok(())
    }

    // ─── Asset CRUD ───────────────────────────────────────────────────────────

    /// Insert a new asset for the given sample. Returns the new asset's ID.
    #[allow(clippy::too_many_arguments)]
    pub fn add_asset(
        db: &Database,
        sample_id: i64,
        asset_type: AssetType,
        file_name: &str,
        file_path: &Path,
        width: Option<u32>,
        height: Option<u32>,
        hash: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<i64> {
        let asset_type_str = asset_type.to_string();
        let file_path_str = file_path.to_string_lossy().to_string();
        let metadata_str = metadata.map(serde_json::to_string).transpose()?;
        db.conn.execute(
            "INSERT INTO assets (sample_id, asset_type, file_name, file_path, width, height, hash, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                sample_id,
                asset_type_str,
                file_name,
                file_path_str,
                width.map(|w| w as i64),
                height.map(|h| h as i64),
                hash,
                metadata_str,
            ],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    /// Convenience: create a sample and a single asset in one call.
    /// Returns `(sample_id, asset_id)`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_sample_with_single_asset(
        db: &Database,
        dataset_id: i64,
        name: &str,
        asset_type: AssetType,
        file_name: &str,
        file_path: &Path,
        width: Option<u32>,
        height: Option<u32>,
        hash: Option<&str>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<(i64, i64)> {
        let sample_id = Self::add_sample(db, dataset_id, name, None)?;
        let asset_id = Self::add_asset(
            db, sample_id, asset_type, file_name, file_path, width, height, hash, metadata,
        )?;
        Ok((sample_id, asset_id))
    }

    /// Retrieve all assets for a sample.
    pub fn get_assets(db: &Database, sample_id: i64) -> Result<Vec<Asset>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, sample_id, asset_type, file_name, file_path, width, height, hash, metadata \
             FROM assets WHERE sample_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![sample_id], row_to_asset)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Count all assets in a dataset (joins through samples).
    pub fn get_asset_count(db: &Database, dataset_id: i64) -> Result<i64> {
        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM assets a \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    // ─── Annotation CRUD ──────────────────────────────────────────────────────

    /// Add an annotation to a sample (optionally scoped to an asset).
    #[allow(clippy::too_many_arguments)]
    pub fn add_annotation(
        db: &Database,
        sample_id: i64,
        asset_id: Option<i64>,
        category_id: Option<i64>,
        bbox: Option<&crate::types::BBox>,
        segmentation: Option<&Vec<Vec<f64>>>,
        keypoints: Option<&Vec<f64>>,
        metadata: Option<&serde_json::Value>,
    ) -> Result<i64> {
        let bbox_str = bbox.map(serde_json::to_string).transpose()?;
        let seg_str = segmentation.map(serde_json::to_string).transpose()?;
        let kp_str = keypoints.map(serde_json::to_string).transpose()?;
        let meta_str = metadata.map(serde_json::to_string).transpose()?;

        db.conn.execute(
            "INSERT INTO annotations (sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![sample_id, asset_id, category_id, bbox_str, seg_str, kp_str, meta_str],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    /// Retrieve all annotations for a sample.
    pub fn get_annotations_for_sample(db: &Database, sample_id: i64) -> Result<Vec<Annotation>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
             FROM annotations WHERE sample_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![sample_id], row_to_annotation)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Retrieve all annotations scoped to a specific asset.
    pub fn get_annotations_for_asset(db: &Database, asset_id: i64) -> Result<Vec<Annotation>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
             FROM annotations WHERE asset_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![asset_id], row_to_annotation)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }

    /// Count annotations for a dataset (joins through samples).
    pub fn get_annotation_count(db: &Database, dataset_id: i64) -> Result<i64> {
        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM annotations \
             WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
            params![dataset_id],
            |row| row.get(0),
        )?;
        Ok(count)
    }

    // ─── Info / inspect ───────────────────────────────────────────────────────

    pub fn inspect(db: &Database, name: &str) -> Result<DatasetInfo> {
        let dataset = Self::get(db, name)?;
        let dataset_id = dataset.id;

        let sample_count = Self::get_sample_count(db, dataset_id)?;
        let asset_count = Self::get_asset_count(db, dataset_id)?;
        let annotation_count = Self::get_annotation_count(db, dataset_id)?;

        let category_count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )?;

        let disk_size_bytes = dir_size(&dataset.path);

        Ok(DatasetInfo {
            dataset,
            sample_count: sample_count as u64,
            asset_count: asset_count as u64,
            annotation_count: annotation_count as u64,
            category_count: category_count as u64,
            disk_size_bytes,
        })
    }

    pub fn get_info(db: &Database, dataset_id: i64) -> Result<DatasetInfo> {
        let dataset = Self::get_by_id(db, dataset_id)?;
        let sample_count = Self::get_sample_count(db, dataset_id)?;
        let asset_count = Self::get_asset_count(db, dataset_id)?;
        let annotation_count = Self::get_annotation_count(db, dataset_id)?;

        let category_count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )?;

        let disk_size_bytes = dir_size(&dataset.path);

        Ok(DatasetInfo {
            dataset,
            sample_count: sample_count as u64,
            asset_count: asset_count as u64,
            annotation_count: annotation_count as u64,
            category_count: category_count as u64,
            disk_size_bytes,
        })
    }

    // ─── Category helpers ─────────────────────────────────────────────────────

    pub fn get_categories(db: &Database, dataset_id: i64) -> Result<Vec<Category>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, dataset_id, name, supercategory FROM categories WHERE dataset_id = ?1 ORDER BY id",
        )?;
        let rows = stmt
            .query_map(params![dataset_id], |row| {
                Ok(Category {
                    id: row.get(0)?,
                    dataset_id: row.get(1)?,
                    name: row.get(2)?,
                    supercategory: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
}

fn dir_size(path: &Path) -> u64 {
    if !path.exists() {
        return 0;
    }
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Ok(meta) = p.metadata() {
                    total += meta.len();
                }
            } else if p.is_dir() {
                total += dir_size(&p);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp dir")
    }

    #[test]
    fn dataset_register_creates_record() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds = DatasetService::register(&db, "my-dataset", dir.path(), DatasetFormat::yolo())
            .expect("register");

        assert_eq!(ds.name, "my-dataset");
        assert_eq!(ds.format, DatasetFormat::yolo());
        assert!(ds.id > 0);
    }

    #[test]
    fn dataset_register_rejects_duplicate() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "dup", dir.path(), DatasetFormat::coco()).expect("first");

        let err = DatasetService::register(&db, "dup", dir.path(), DatasetFormat::coco())
            .expect_err("should fail on duplicate");

        assert!(
            matches!(err, DmanError::DatasetAlreadyExists(_)),
            "expected DatasetAlreadyExists, got {:?}",
            err
        );
    }

    #[test]
    fn dataset_register_rejects_nonexistent_path() {
        let db = in_memory_db();

        let err = DatasetService::register(
            &db,
            "no-path",
            Path::new("/nonexistent/path/that/does/not/exist"),
            DatasetFormat::yolo(),
        )
        .expect_err("should fail for missing path");

        assert!(
            matches!(err, DmanError::StorageError(_)),
            "expected StorageError, got {:?}",
            err
        );
    }

    #[test]
    fn dataset_list_empty() {
        let db = in_memory_db();
        let datasets = DatasetService::list(&db).expect("list");
        assert!(datasets.is_empty());
    }

    #[test]
    fn dataset_list_returns_all() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "ds1", dir.path(), DatasetFormat::yolo()).unwrap();
        DatasetService::register(&db, "ds2", dir.path(), DatasetFormat::coco()).unwrap();

        let datasets = DatasetService::list(&db).expect("list");
        assert_eq!(datasets.len(), 2);
        let names: Vec<_> = datasets.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"ds1"));
        assert!(names.contains(&"ds2"));
    }

    #[test]
    fn dataset_get_existing() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "find-me", dir.path(), DatasetFormat::huggingface()).unwrap();

        let ds = DatasetService::get(&db, "find-me").expect("get");
        assert_eq!(ds.name, "find-me");
        assert_eq!(ds.format, DatasetFormat::huggingface());
    }

    #[test]
    fn dataset_get_not_found() {
        let db = in_memory_db();
        let err = DatasetService::get(&db, "ghost").expect_err("should not find");
        assert!(
            matches!(err, DmanError::DatasetNotFound(_)),
            "got {:?}",
            err
        );
    }

    #[test]
    fn dataset_get_by_id_existing() {
        let db = in_memory_db();
        let dir = temp_dir();

        let created = DatasetService::register(&db, "by-id", dir.path(), DatasetFormat::yolo())
            .expect("register");
        let fetched = DatasetService::get_by_id(&db, created.id).expect("get_by_id");
        assert_eq!(fetched.name, "by-id");
    }

    #[test]
    fn dataset_get_by_id_not_found() {
        let db = in_memory_db();
        let err = DatasetService::get_by_id(&db, 9999).expect_err("missing");
        assert!(
            matches!(err, DmanError::DatasetNotFound(_)),
            "got {:?}",
            err
        );
    }

    #[test]
    fn dataset_remove_existing() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "to-remove", dir.path(), DatasetFormat::yolo()).unwrap();
        DatasetService::remove(&db, "to-remove").expect("remove");

        let err = DatasetService::get(&db, "to-remove").expect_err("should be gone");
        assert!(matches!(err, DmanError::DatasetNotFound(_)));
    }

    #[test]
    fn dataset_remove_not_found() {
        let db = in_memory_db();
        let err = DatasetService::remove(&db, "ghost").expect_err("not found");
        assert!(matches!(err, DmanError::DatasetNotFound(_)));
    }

    #[test]
    fn dataset_remove_cascade_deletes_samples_and_annotations() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds =
            DatasetService::register(&db, "cascade", dir.path(), DatasetFormat::yolo()).unwrap();
        let dataset_id = ds.id;

        let sample_id =
            DatasetService::add_sample(&db, dataset_id, "sample-1", None).expect("add sample");

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'cat')",
                params![dataset_id],
            )
            .unwrap();

        DatasetService::add_annotation(&db, sample_id, None, Some(1), None, None, None, None)
            .expect("add annotation");

        DatasetService::remove(&db, "cascade").expect("remove");

        let sample_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(sample_count, 0, "samples should be deleted");

        let cat_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cat_count, 0, "categories should be deleted");

        let ann_count: i64 = db
            .conn
            .query_row("SELECT COUNT(*) FROM annotations", [], |r| r.get(0))
            .unwrap();
        assert_eq!(ann_count, 0, "annotations should be deleted");
    }

    #[test]
    fn dataset_update_metadata() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "meta-ds", dir.path(), DatasetFormat::coco()).unwrap();

        let meta = serde_json::json!({"version": "1.0", "source": "custom"});
        DatasetService::update_metadata(&db, "meta-ds", meta.clone()).expect("update");

        let ds = DatasetService::get(&db, "meta-ds").unwrap();
        assert_eq!(ds.metadata.as_ref().unwrap()["version"], "1.0");
    }

    #[test]
    fn dataset_update_metadata_not_found() {
        let db = in_memory_db();
        let err =
            DatasetService::update_metadata(&db, "ghost", serde_json::json!({})).expect_err("nf");
        assert!(matches!(err, DmanError::DatasetNotFound(_)));
    }

    #[test]
    fn dataset_inspect_counts() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds =
            DatasetService::register(&db, "inspect-me", dir.path(), DatasetFormat::yolo()).unwrap();
        let dataset_id = ds.id;

        let s1 = DatasetService::add_sample(&db, dataset_id, "sample-a", None).unwrap();
        let s2 = DatasetService::add_sample(&db, dataset_id, "sample-b", None).unwrap();

        DatasetService::add_asset(
            &db,
            s1,
            AssetType::Image,
            "a.jpg",
            Path::new("/tmp/a.jpg"),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        DatasetService::add_asset(
            &db,
            s2,
            AssetType::Image,
            "b.jpg",
            Path::new("/tmp/b.jpg"),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'cat')",
                params![dataset_id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'dog')",
                params![dataset_id],
            )
            .unwrap();

        DatasetService::add_annotation(&db, s1, None, None, None, None, None, None).unwrap();
        DatasetService::add_annotation(&db, s1, None, None, None, None, None, None).unwrap();
        DatasetService::add_annotation(&db, s2, None, None, None, None, None, None).unwrap();

        let info = DatasetService::inspect(&db, "inspect-me").expect("inspect");
        assert_eq!(info.sample_count, 2);
        assert_eq!(info.asset_count, 2);
        assert_eq!(info.annotation_count, 3);
        assert_eq!(info.category_count, 2);
    }

    #[test]
    fn dataset_inspect_not_found() {
        let db = in_memory_db();
        let err = DatasetService::inspect(&db, "ghost").expect_err("not found");
        assert!(matches!(err, DmanError::DatasetNotFound(_)));
    }

    #[test]
    fn dataset_full_crud_cycle() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds = DatasetService::register(&db, "cycle-ds", dir.path(), DatasetFormat::yolo())
            .expect("register");
        assert_eq!(ds.name, "cycle-ds");

        let all = DatasetService::list(&db).expect("list");
        assert_eq!(all.len(), 1);

        let fetched = DatasetService::get(&db, "cycle-ds").expect("get");
        assert_eq!(fetched.id, ds.id);

        let by_id = DatasetService::get_by_id(&db, ds.id).expect("get_by_id");
        assert_eq!(by_id.name, "cycle-ds");

        DatasetService::update_metadata(&db, "cycle-ds", serde_json::json!({"key": "value"}))
            .expect("update_metadata");

        let info = DatasetService::inspect(&db, "cycle-ds").expect("inspect");
        assert_eq!(info.sample_count, 0);
        assert_eq!(info.annotation_count, 0);

        DatasetService::remove(&db, "cycle-ds").expect("remove");

        let list_after = DatasetService::list(&db).expect("list after remove");
        assert!(list_after.is_empty());
    }

    #[test]
    fn add_sample_and_asset_and_retrieve() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds = DatasetService::register(&db, "sample-test", dir.path(), DatasetFormat::yolo())
            .unwrap();

        let sample_id = DatasetService::add_sample(&db, ds.id, "frame-001", None).unwrap();
        assert!(sample_id > 0);

        let asset_id = DatasetService::add_asset(
            &db,
            sample_id,
            AssetType::Image,
            "frame-001.jpg",
            Path::new("/tmp/frame-001.jpg"),
            Some(640),
            Some(480),
            None,
            None,
        )
        .unwrap();
        assert!(asset_id > 0);

        let samples = DatasetService::get_samples(&db, ds.id).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].name, "frame-001");

        let assets = DatasetService::get_assets(&db, sample_id).unwrap();
        assert_eq!(assets.len(), 1);
        assert_eq!(assets[0].file_name, "frame-001.jpg");
        assert_eq!(assets[0].width, Some(640));
        assert_eq!(assets[0].height, Some(480));
    }

    #[test]
    fn add_sample_with_single_asset_convenience() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds =
            DatasetService::register(&db, "conv-test", dir.path(), DatasetFormat::coco()).unwrap();

        let (sample_id, asset_id) = DatasetService::add_sample_with_single_asset(
            &db,
            ds.id,
            "img-001",
            AssetType::Image,
            "img-001.jpg",
            Path::new("/tmp/img-001.jpg"),
            Some(1920),
            Some(1080),
            Some("abc123"),
            None,
        )
        .unwrap();

        assert!(sample_id > 0);
        assert!(asset_id > 0);

        assert_eq!(DatasetService::get_sample_count(&db, ds.id).unwrap(), 1);
        assert_eq!(DatasetService::get_asset_count(&db, ds.id).unwrap(), 1);
    }

    #[test]
    fn annotations_for_sample_and_asset() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds =
            DatasetService::register(&db, "ann-test", dir.path(), DatasetFormat::yolo()).unwrap();

        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        let asset_id = DatasetService::add_asset(
            &db,
            sample_id,
            AssetType::Image,
            "x.jpg",
            Path::new("/tmp/x.jpg"),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // annotation scoped to asset
        DatasetService::add_annotation(
            &db,
            sample_id,
            Some(asset_id),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        // annotation scoped to sample only
        DatasetService::add_annotation(&db, sample_id, None, None, None, None, None, None).unwrap();

        let sample_anns = DatasetService::get_annotations_for_sample(&db, sample_id).unwrap();
        assert_eq!(sample_anns.len(), 2);

        let asset_anns = DatasetService::get_annotations_for_asset(&db, asset_id).unwrap();
        assert_eq!(asset_anns.len(), 1);
        assert_eq!(asset_anns[0].asset_id, Some(asset_id));
    }

    #[test]
    fn remove_sample_cascades() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds =
            DatasetService::register(&db, "del-sample", dir.path(), DatasetFormat::yolo()).unwrap();

        let sample_id = DatasetService::add_sample(&db, ds.id, "to-del", None).unwrap();
        DatasetService::add_asset(
            &db,
            sample_id,
            AssetType::Image,
            "f.jpg",
            Path::new("/tmp/f.jpg"),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        DatasetService::remove_sample(&db, sample_id).unwrap();

        let samples = DatasetService::get_samples(&db, ds.id).unwrap();
        assert!(samples.is_empty());
    }

    #[test]
    fn remove_sample_not_found() {
        let db = in_memory_db();
        let err = DatasetService::remove_sample(&db, 9999).expect_err("should fail");
        assert!(matches!(err, DmanError::SampleNotFound(_)));
    }
}
