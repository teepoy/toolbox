use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use image::GenericImageView;

use crate::db::Database;
use crate::storage::StorageManager;
use crate::types::{BBox, Patch};
use crate::{DmanError, Result};

pub struct PatchService;

fn bbox_hash(bbox: &BBox) -> u64 {
    let mut hasher = DefaultHasher::new();
    bbox.x.to_bits().hash(&mut hasher);
    bbox.y.to_bits().hash(&mut hasher);
    bbox.width.to_bits().hash(&mut hasher);
    bbox.height.to_bits().hash(&mut hasher);
    hasher.finish()
}

impl PatchService {
    pub fn extract(
        db: &Database,
        _storage: &StorageManager,
        image_id: i64,
        bbox: &BBox,
        output_dir: &Path,
    ) -> Result<Patch> {
        let image_file_path: String = db
            .conn
            .query_row(
                "SELECT file_path FROM images WHERE id = ?1",
                rusqlite::params![image_id],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    DmanError::StorageError(format!("image {} not found", image_id))
                }
                other => DmanError::Database(other),
            })?;

        let img = image::open(&image_file_path)
            .map_err(|e| DmanError::StorageError(format!("failed to open image: {}", e)))?;

        let (img_w, img_h) = img.dimensions();
        let cx = (bbox.x as u32).min(img_w);
        let cy = (bbox.y as u32).min(img_h);
        let cw = (bbox.width as u32).min(img_w.saturating_sub(cx)).max(1);
        let ch = (bbox.height as u32).min(img_h.saturating_sub(cy)).max(1);

        let cropped = img.crop_imm(cx, cy, cw, ch);

        let filename = format!("{}_{:08x}.jpg", image_id, bbox_hash(bbox));
        std::fs::create_dir_all(output_dir)?;
        let output_path = output_dir.join(&filename);
        cropped
            .save(&output_path)
            .map_err(|e| DmanError::StorageError(format!("failed to save patch: {}", e)))?;

        let bbox_json = serde_json::to_string(bbox)?;
        let file_path_str = output_path
            .to_str()
            .ok_or_else(|| DmanError::StorageError("patch path is not valid UTF-8".to_string()))?;

        db.conn.execute(
            "INSERT INTO patches (image_id, bbox, file_path) VALUES (?1, ?2, ?3)",
            rusqlite::params![image_id, bbox_json, file_path_str],
        )?;
        let patch_id = db.conn.last_insert_rowid();

        Ok(Patch {
            id: patch_id,
            image_id,
            bbox: bbox.clone(),
            file_path: Some(output_path),
            metadata: None,
        })
    }

    pub fn extract_batch(
        db: &Database,
        storage: &StorageManager,
        dataset_id: i64,
        output_dir: &Path,
    ) -> Result<Vec<Patch>> {
        let mut stmt = db.conn.prepare(
            "SELECT a.image_id, a.bbox
             FROM annotations a
             JOIN images i ON a.image_id = i.id
             WHERE i.dataset_id = ?1
               AND a.bbox IS NOT NULL",
        )?;

        struct Row {
            image_id: i64,
            bbox_json: String,
        }

        let rows: Vec<Row> = stmt
            .query_map(rusqlite::params![dataset_id], |row| {
                Ok(Row {
                    image_id: row.get(0)?,
                    bbox_json: row.get(1)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut patches = Vec::new();
        for row in rows {
            let bbox: BBox = serde_json::from_str(&row.bbox_json)?;
            patches.push(Self::extract(db, storage, row.image_id, &bbox, output_dir)?);
        }
        Ok(patches)
    }

    pub fn get_by_image(db: &Database, image_id: i64) -> Result<Vec<Patch>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, image_id, bbox, file_path, metadata FROM patches WHERE image_id = ?1",
        )?;

        let rows = stmt
            .query_map(rusqlite::params![image_id], |row| {
                Ok(PatchRow {
                    id: row.get(0)?,
                    image_id: row.get(1)?,
                    bbox_json: row.get(2)?,
                    file_path: row.get::<_, Option<String>>(3)?,
                    metadata_json: row.get::<_, Option<String>>(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        rows.into_iter().map(row_to_patch).collect()
    }

    pub fn get_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Patch>> {
        let mut stmt = db.conn.prepare(
            "SELECT p.id, p.image_id, p.bbox, p.file_path, p.metadata
             FROM patches p
             JOIN images i ON p.image_id = i.id
             WHERE i.dataset_id = ?1",
        )?;

        let rows = stmt
            .query_map(rusqlite::params![dataset_id], |row| {
                Ok(PatchRow {
                    id: row.get(0)?,
                    image_id: row.get(1)?,
                    bbox_json: row.get(2)?,
                    file_path: row.get::<_, Option<String>>(3)?,
                    metadata_json: row.get::<_, Option<String>>(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        rows.into_iter().map(row_to_patch).collect()
    }

    pub fn delete_patch(db: &Database, _storage: &StorageManager, patch_id: i64) -> Result<()> {
        let file_path_opt: Option<String> = db
            .conn
            .query_row(
                "SELECT file_path FROM patches WHERE id = ?1",
                rusqlite::params![patch_id],
                |row| row.get(0),
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    DmanError::StorageError(format!("patch {} not found", patch_id))
                }
                other => DmanError::Database(other),
            })?;

        if let Some(fp) = file_path_opt {
            std::fs::remove_file(PathBuf::from(&fp)).ok();
        }

        db.conn.execute(
            "DELETE FROM patches WHERE id = ?1",
            rusqlite::params![patch_id],
        )?;

        Ok(())
    }
}

struct PatchRow {
    id: i64,
    image_id: i64,
    bbox_json: String,
    file_path: Option<String>,
    metadata_json: Option<String>,
}

fn row_to_patch(row: PatchRow) -> Result<Patch> {
    let bbox: BBox = serde_json::from_str(&row.bbox_json)?;
    let file_path = row.file_path.map(PathBuf::from);
    let metadata = row
        .metadata_json
        .map(|s| serde_json::from_str::<serde_json::Value>(&s))
        .transpose()?;
    Ok(Patch {
        id: row.id,
        image_id: row.image_id,
        bbox,
        file_path,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use crate::db::Database;
    use crate::storage::StorageManager;
    use crate::types::BBox;

    fn make_test_image(dir: &std::path::Path) -> PathBuf {
        let path = dir.join("test_img.jpg");
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([128u8, 64, 200]));
        image::DynamicImage::ImageRgb8(img)
            .save(&path)
            .expect("save test image");
        path
    }

    fn insert_image(db: &Database, src_path: &std::path::Path) -> (i64, i64) {
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                ["test-ds", "/tmp/test-ds"],
            )
            .expect("insert dataset");
        let dataset_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM datasets WHERE name = ?1",
                ["test-ds"],
                |r| r.get(0),
            )
            .expect("dataset id");

        let path_str = src_path.to_str().expect("path str");
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "test_img.jpg", path_str],
            )
            .expect("insert image");
        let image_id: i64 = db
            .conn
            .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
            .expect("image id");

        (dataset_id, image_id)
    }

    fn full_bbox() -> BBox {
        BBox {
            x: 0.0,
            y: 0.0,
            width: 4.0,
            height: 4.0,
        }
    }

    #[test]
    fn extract_creates_file() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (_, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        let patch = PatchService::extract(
            &db,
            &storage,
            image_id,
            &full_bbox(),
            &tmp.path().join("out"),
        )
        .unwrap();

        assert!(patch.file_path.is_some());
        assert!(patch.file_path.unwrap().exists());
    }

    #[test]
    fn extract_db_record() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (_, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        let bbox = full_bbox();
        let patch =
            PatchService::extract(&db, &storage, image_id, &bbox, &tmp.path().join("out")).unwrap();

        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM patches WHERE id = ?1 AND image_id = ?2",
                rusqlite::params![patch.id, image_id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(patch.image_id, image_id);
        assert_eq!(patch.bbox, bbox);
    }

    #[test]
    fn extract_output_dimensions() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (_, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        let patch = PatchService::extract(
            &db,
            &storage,
            image_id,
            &full_bbox(),
            &tmp.path().join("out"),
        )
        .unwrap();

        let saved = image::open(patch.file_path.unwrap()).expect("open saved patch");
        let (w, h) = saved.dimensions();
        assert_eq!(w, 4);
        assert_eq!(h, 4);
    }

    #[test]
    fn get_by_image_returns_patches() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (_, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        PatchService::extract(
            &db,
            &storage,
            image_id,
            &full_bbox(),
            &tmp.path().join("out"),
        )
        .unwrap();

        let patches = PatchService::get_by_image(&db, image_id).unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].image_id, image_id);
    }

    #[test]
    fn get_by_dataset_returns_patches() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (dataset_id, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        PatchService::extract(
            &db,
            &storage,
            image_id,
            &full_bbox(),
            &tmp.path().join("out"),
        )
        .unwrap();

        let patches = PatchService::get_by_dataset(&db, dataset_id).unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0].image_id, image_id);
    }

    #[test]
    fn delete_removes_file() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (_, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        let patch = PatchService::extract(
            &db,
            &storage,
            image_id,
            &full_bbox(),
            &tmp.path().join("out"),
        )
        .unwrap();

        let fp = patch.file_path.as_ref().unwrap().clone();
        assert!(fp.exists());

        PatchService::delete_patch(&db, &storage, patch.id).unwrap();

        assert!(!fp.exists());
        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM patches WHERE id = ?1",
                rusqlite::params![patch.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn extract_batch_creates_patches() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        let storage = StorageManager::new(tmp.path().to_path_buf());

        let (dataset_id, image_id) = insert_image(&db, &make_test_image(tmp.path()));
        let bbox_json = serde_json::to_string(&full_bbox()).unwrap();
        db.conn
            .execute(
                "INSERT INTO annotations (image_id, bbox) VALUES (?1, ?2)",
                rusqlite::params![image_id, bbox_json],
            )
            .unwrap();

        let patches =
            PatchService::extract_batch(&db, &storage, dataset_id, &tmp.path().join("out"))
                .unwrap();

        assert_eq!(patches.len(), 1);
        assert!(patches[0].file_path.as_ref().unwrap().exists());
    }

    #[test]
    fn get_by_image_empty_when_none() {
        let db = Database::open_in_memory().expect("db");
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                ["ds-empty", "/tmp"],
            )
            .unwrap();
        let dataset_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM datasets WHERE name = ?1",
                ["ds-empty"],
                |r| r.get(0),
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "ghost.jpg", "/tmp/ghost.jpg"],
            )
            .unwrap();
        let image_id: i64 = db
            .conn
            .query_row("SELECT last_insert_rowid()", [], |r| r.get(0))
            .unwrap();

        let patches = PatchService::get_by_image(&db, image_id).unwrap();
        assert!(patches.is_empty());
    }
}
