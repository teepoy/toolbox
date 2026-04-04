use std::path::PathBuf;

use rusqlite::params;

use crate::{
    dataset::DatasetService,
    db::Database,
    error::{DmanError, Result},
    storage::StorageManager,
    types::{Dataset, VirtualDataset, VirtualDatasetDef},
};

use super::{transforms::SchemaTransformer, VirtualDatasetService};

pub fn materialize(
    db: &Database,
    storage: &StorageManager,
    vds: &VirtualDataset,
    output_name: &str,
) -> Result<Dataset> {
    let mut images = VirtualDatasetService::evaluate(db, vds)?;

    let schema_ops = extract_schema_ops(&vds.definition);
    if !schema_ops.is_empty() {
        images = SchemaTransformer::apply_to_images(&images, &schema_ops)?;
    }

    let output_path = storage_path_for_output(storage, output_name);
    std::fs::create_dir_all(&output_path)?;

    db.conn.execute("BEGIN IMMEDIATE", [])?;

    let result = (|| -> Result<Dataset> {
        db.conn.execute(
            "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
            params![
                output_name,
                output_path.to_string_lossy().as_ref(),
                "Custom"
            ],
        )?;
        let new_dataset_id = db.conn.last_insert_rowid();

        for img in &images {
            let metadata_str = img
                .metadata
                .as_ref()
                .map(|m| serde_json::to_string(m))
                .transpose()?;

            db.conn.execute(
                "INSERT INTO images (dataset_id, file_name, file_path, width, height, hash, metadata) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    new_dataset_id,
                    img.file_name,
                    img.file_path.to_string_lossy().as_ref(),
                    img.width,
                    img.height,
                    img.hash,
                    metadata_str,
                ],
            )?;
            let new_image_id = db.conn.last_insert_rowid();
            copy_annotations(db, img.id, new_image_id)?;
        }

        DatasetService::get_by_id(db, new_dataset_id)
    })();

    match result {
        Ok(ds) => {
            db.conn.execute("COMMIT", [])?;
            Ok(ds)
        }
        Err(e) => {
            let _ = db.conn.execute("ROLLBACK", []);
            let _ = std::fs::remove_dir_all(&output_path);
            Err(e)
        }
    }
}

fn copy_annotations(db: &Database, old_image_id: i64, new_image_id: i64) -> Result<()> {
    #[derive(Debug)]
    struct RawAnnotation {
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

    let rows: Vec<RawAnnotation> = stmt
        .query_map(params![old_image_id], |row| {
            Ok(RawAnnotation {
                category_id: row.get(0)?,
                bbox: row.get(1)?,
                segmentation: row.get(2)?,
                keypoints: row.get(3)?,
                metadata: row.get(4)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DmanError::Database)?;

    for ann in rows {
        db.conn.execute(
            "INSERT INTO annotations (image_id, category_id, bbox, segmentation, keypoints, metadata) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                new_image_id,
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

fn storage_path_for_output(storage: &StorageManager, output_name: &str) -> PathBuf {
    // StorageManager doesn't expose base_path; derive it by probing get_image_path(0, "__probe__")
    // which returns base_path/0/images/__probe__, then navigate up 3 levels.
    let dummy = storage.get_image_path(0, "__probe__");
    let base = dummy
        .parent() // images/
        .and_then(|p| p.parent()) // 0/
        .and_then(|p| p.parent()) // base_path/
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("/tmp/dman"));
    base.join("datasets").join(output_name)
}

fn extract_schema_ops(def: &VirtualDatasetDef) -> Vec<crate::types::SchemaOp> {
    match def {
        VirtualDatasetDef::SchemaTransform { transforms } => transforms.clone(),
        VirtualDatasetDef::Chain(steps) => steps.iter().flat_map(extract_schema_ops).collect(),
        _ => vec![],
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        db::Database,
        storage::StorageManager,
        types::{FilterOp, SchemaOp, VirtualDatasetDef},
    };
    use tempfile::tempdir;

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn make_storage(dir: &std::path::Path) -> StorageManager {
        StorageManager::new(dir.to_path_buf())
    }

    fn insert_dataset(db: &Database, name: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO datasets (name, path, format) VALUES (?1, '/tmp', 'Custom')",
                params![name],
            )
            .expect("insert dataset");
        db.conn.last_insert_rowid()
    }

    fn insert_image(db: &Database, dataset_id: i64, file_name: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                params![dataset_id, file_name, format!("/tmp/{}", file_name)],
            )
            .expect("insert image");
        db.conn.last_insert_rowid()
    }

    fn insert_image_with_meta(
        db: &Database,
        dataset_id: i64,
        file_name: &str,
        meta: serde_json::Value,
    ) -> i64 {
        let meta_str = serde_json::to_string(&meta).unwrap();
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path, metadata) VALUES (?1, ?2, ?3, ?4)",
                params![dataset_id, file_name, format!("/tmp/{}", file_name), meta_str],
            )
            .expect("insert image with meta");
        db.conn.last_insert_rowid()
    }

    fn insert_annotation(db: &Database, image_id: i64, category_id: Option<i64>) -> i64 {
        db.conn
            .execute(
                "INSERT INTO annotations (image_id, category_id) VALUES (?1, ?2)",
                params![image_id, category_id],
            )
            .expect("insert annotation");
        db.conn.last_insert_rowid()
    }

    fn count_annotations(db: &Database, image_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id = ?1",
                params![image_id],
                |row| row.get(0),
            )
            .expect("count annotations")
    }

    fn count_images(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                params![dataset_id],
                |row| row.get(0),
            )
            .expect("count images")
    }

    #[test]
    fn test_materialize_simple() {
        let db = in_memory_db();
        let tmp = tempdir().unwrap();
        let storage = make_storage(tmp.path());

        let ds_id = insert_dataset(&db, "source-ds");
        insert_image(&db, ds_id, "a.jpg");
        insert_image(&db, ds_id, "b.jpg");
        insert_image(&db, ds_id, "c.jpg");

        let def = VirtualDatasetDef::Sample { ratio: 1.0 };
        let vds =
            VirtualDatasetService::create(&db, "all-vds", vec![ds_id], &def).expect("create vds");

        let result = materialize(&db, &storage, &vds, "materialized-all").expect("materialize");

        assert_eq!(result.name, "materialized-all");
        assert!(result.id > 0);
        assert_eq!(count_images(&db, result.id), 3);
    }

    #[test]
    fn test_materialize_filter() {
        let db = in_memory_db();
        let tmp = tempdir().unwrap();
        let storage = make_storage(tmp.path());

        let ds_id = insert_dataset(&db, "source-filter");
        let img1 = insert_image(&db, ds_id, "annotated1.jpg");
        let _img2 = insert_image(&db, ds_id, "unannotated.jpg");
        let img3 = insert_image(&db, ds_id, "annotated2.jpg");

        insert_annotation(&db, img1, None);
        insert_annotation(&db, img3, None);

        let def = VirtualDatasetDef::Filter {
            column: "annotated".to_string(),
            op: FilterOp::Eq,
            value: serde_json::json!(true),
        };
        let vds = VirtualDatasetService::create(&db, "filter-vds", vec![ds_id], &def)
            .expect("create vds");

        let result = materialize(&db, &storage, &vds, "materialized-filter").expect("materialize");

        assert_eq!(count_images(&db, result.id), 2);

        let new_image_ids: Vec<i64> = {
            let mut stmt = db
                .conn
                .prepare("SELECT id FROM images WHERE dataset_id = ?1")
                .unwrap();
            stmt.query_map(params![result.id], |row| row.get(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        let total_annotations: i64 = new_image_ids
            .iter()
            .map(|id| count_annotations(&db, *id))
            .sum();
        assert_eq!(
            total_annotations, 2,
            "each materialized image should have its annotation copied"
        );
    }

    #[test]
    fn test_materialize_schema_transform() {
        let db = in_memory_db();
        let tmp = tempdir().unwrap();
        let storage = make_storage(tmp.path());

        let ds_id = insert_dataset(&db, "source-schema");
        insert_image_with_meta(&db, ds_id, "img1.jpg", serde_json::json!({"label": "cat"}));
        insert_image_with_meta(&db, ds_id, "img2.jpg", serde_json::json!({"label": "dog"}));

        let def = VirtualDatasetDef::Chain(vec![
            VirtualDatasetDef::Sample { ratio: 1.0 },
            VirtualDatasetDef::SchemaTransform {
                transforms: vec![SchemaOp::RenameColumn {
                    from: "label".to_string(),
                    to: "class".to_string(),
                }],
            },
        ]);
        let vds = VirtualDatasetService::create(&db, "schema-vds", vec![ds_id], &def)
            .expect("create vds");

        let result = materialize(&db, &storage, &vds, "materialized-schema").expect("materialize");

        assert_eq!(count_images(&db, result.id), 2);

        let mut stmt = db
            .conn
            .prepare("SELECT metadata FROM images WHERE dataset_id = ?1 ORDER BY id")
            .unwrap();
        let metas: Vec<Option<String>> = stmt
            .query_map(params![result.id], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();

        for meta_opt in &metas {
            let meta_str = meta_opt.as_ref().expect("metadata should be present");
            let meta: serde_json::Value = serde_json::from_str(meta_str).unwrap();
            assert!(
                meta.get("label").is_none(),
                "old key 'label' should be absent"
            );
            assert!(
                meta.get("class").is_some(),
                "new key 'class' should be present"
            );
        }
    }

    #[test]
    fn test_rollback_on_failure() {
        let db = in_memory_db();
        let tmp = tempdir().unwrap();
        let storage = make_storage(tmp.path());

        let ds_id = insert_dataset(&db, "source-rollback");
        insert_image(&db, ds_id, "img1.jpg");
        insert_image(&db, ds_id, "img2.jpg");

        let def = VirtualDatasetDef::Sample { ratio: 1.0 };
        let vds = VirtualDatasetService::create(&db, "rollback-vds", vec![ds_id], &def)
            .expect("create vds");

        let first = materialize(&db, &storage, &vds, "output-once").expect("first materialize");
        assert_eq!(count_images(&db, first.id), 2);

        let err = materialize(&db, &storage, &vds, "output-once")
            .expect_err("should fail on duplicate name");

        assert!(
            matches!(err, DmanError::Database(_)),
            "expected Database error from UNIQUE constraint: {:?}",
            err
        );

        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM datasets WHERE name = 'output-once'",
                [],
                |row| row.get(0),
            )
            .expect("count");
        assert_eq!(
            count, 1,
            "only the first successful materialization should remain"
        );
        assert_eq!(count_images(&db, ds_id), 2);
    }
}
