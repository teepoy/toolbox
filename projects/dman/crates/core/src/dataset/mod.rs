use std::path::{Path, PathBuf};

use rusqlite::params;

use crate::{
    db::Database,
    error::{DmanError, Result},
    types::{Category, Dataset, DatasetFormat},
};

#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub dataset: Dataset,
    pub image_count: u64,
    pub annotation_count: u64,
    pub categories: Vec<Category>,
    pub disk_size_bytes: u64,
}

fn format_from_str(s: &str) -> DatasetFormat {
    match s {
        "Yolo" => DatasetFormat::Yolo,
        "Coco" => DatasetFormat::Coco,
        "HuggingFace" => DatasetFormat::HuggingFace,
        other => DatasetFormat::Custom(other.to_string()),
    }
}

fn format_to_str(f: &DatasetFormat) -> String {
    match f {
        DatasetFormat::Yolo => "Yolo".to_string(),
        DatasetFormat::Coco => "Coco".to_string(),
        DatasetFormat::HuggingFace => "HuggingFace".to_string(),
        DatasetFormat::Custom(s) => s.clone(),
    }
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

    let format = format_from_str(&format_str);
    let schema_path = schema_path.map(PathBuf::from);
    let metadata = metadata_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

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

        let format_str = format_to_str(&format);
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

        db.conn.execute(
            "DELETE FROM patches WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM embeddings WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM predictions WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM annotations WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
            params![dataset_id],
        )?;
        db.conn.execute(
            "DELETE FROM images WHERE dataset_id = ?1",
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

    pub fn inspect(db: &Database, name: &str) -> Result<DatasetInfo> {
        let dataset = Self::get(db, name)?;
        let dataset_id = dataset.id;

        let image_count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )?;

        let annotation_count: i64 = db.conn.query_row(
            "SELECT COUNT(*) FROM annotations WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
            params![dataset_id],
            |row| row.get(0),
        )?;

        let mut stmt = db.conn.prepare(
            "SELECT id, dataset_id, name, supercategory FROM categories WHERE dataset_id = ?1 ORDER BY id",
        )?;
        let categories = stmt
            .query_map(params![dataset_id], |row| {
                Ok(Category {
                    id: row.get(0)?,
                    dataset_id: row.get(1)?,
                    name: row.get(2)?,
                    supercategory: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let disk_size_bytes = dir_size(&dataset.path);

        Ok(DatasetInfo {
            dataset,
            image_count: image_count as u64,
            annotation_count: annotation_count as u64,
            categories,
            disk_size_bytes,
        })
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

        let ds = DatasetService::register(&db, "my-dataset", dir.path(), DatasetFormat::Yolo)
            .expect("register");

        assert_eq!(ds.name, "my-dataset");
        assert_eq!(ds.format, DatasetFormat::Yolo);
        assert!(ds.id > 0);
    }

    #[test]
    fn dataset_register_rejects_duplicate() {
        let db = in_memory_db();
        let dir = temp_dir();

        DatasetService::register(&db, "dup", dir.path(), DatasetFormat::Coco).expect("first");

        let err = DatasetService::register(&db, "dup", dir.path(), DatasetFormat::Coco)
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
            DatasetFormat::Yolo,
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

        DatasetService::register(&db, "ds1", dir.path(), DatasetFormat::Yolo).unwrap();
        DatasetService::register(&db, "ds2", dir.path(), DatasetFormat::Coco).unwrap();

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

        DatasetService::register(&db, "find-me", dir.path(), DatasetFormat::HuggingFace).unwrap();

        let ds = DatasetService::get(&db, "find-me").expect("get");
        assert_eq!(ds.name, "find-me");
        assert_eq!(ds.format, DatasetFormat::HuggingFace);
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

        let created = DatasetService::register(&db, "by-id", dir.path(), DatasetFormat::Yolo)
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

        DatasetService::register(&db, "to-remove", dir.path(), DatasetFormat::Yolo).unwrap();
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
    fn dataset_remove_cascade_deletes_images_and_annotations() {
        let db = in_memory_db();
        let dir = temp_dir();

        let ds = DatasetService::register(&db, "cascade", dir.path(), DatasetFormat::Yolo).unwrap();
        let dataset_id = ds.id;

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'img.jpg', '/tmp/img.jpg')",
                params![dataset_id],
            )
            .unwrap();
        let image_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'cat')",
                params![dataset_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO annotations (image_id, category_id) VALUES (?1, 1)",
                params![image_id],
            )
            .unwrap();

        DatasetService::remove(&db, "cascade").expect("remove");

        let img_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(img_count, 0, "images should be deleted");

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

        DatasetService::register(&db, "meta-ds", dir.path(), DatasetFormat::Coco).unwrap();

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
            DatasetService::register(&db, "inspect-me", dir.path(), DatasetFormat::Yolo).unwrap();
        let dataset_id = ds.id;

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'a.jpg', '/tmp/a.jpg')",
                params![dataset_id],
            )
            .unwrap();
        let img1 = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'b.jpg', '/tmp/b.jpg')",
                params![dataset_id],
            )
            .unwrap();
        let img2 = db.conn.last_insert_rowid();

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

        db.conn
            .execute(
                "INSERT INTO annotations (image_id) VALUES (?1)",
                params![img1],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO annotations (image_id) VALUES (?1)",
                params![img1],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO annotations (image_id) VALUES (?1)",
                params![img2],
            )
            .unwrap();

        let info = DatasetService::inspect(&db, "inspect-me").expect("inspect");
        assert_eq!(info.image_count, 2);
        assert_eq!(info.annotation_count, 3);
        assert_eq!(info.categories.len(), 2);
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

        let ds = DatasetService::register(&db, "cycle-ds", dir.path(), DatasetFormat::Yolo)
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
        assert_eq!(info.image_count, 0);
        assert_eq!(info.annotation_count, 0);

        DatasetService::remove(&db, "cycle-ds").expect("remove");

        let list_after = DatasetService::list(&db).expect("list after remove");
        assert!(list_after.is_empty());
    }
}
