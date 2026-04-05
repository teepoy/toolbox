use dman_core::{
    dataset::DatasetService,
    db::Database,
    error::DmanError,
    storage::StorageManager,
    types::{FilterOp, SchemaOp, VirtualDatasetDef},
    virtual_dataset::{VirtualDatasetService, materialize::materialize},
};
use rusqlite::params;
use tempfile::TempDir;

fn in_memory_db() -> Database {
    Database::open_in_memory().expect("in-memory DB")
}

fn make_storage(tmp: &TempDir) -> StorageManager {
    StorageManager::new(tmp.path().to_path_buf())
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

fn insert_sample(db: &Database, dataset_id: i64, name: &str) -> i64 {
    db.conn
        .execute(
            "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
            params![dataset_id, name],
        )
        .expect("insert sample");
    db.conn.last_insert_rowid()
}

fn insert_sample_with_meta(
    db: &Database,
    dataset_id: i64,
    name: &str,
    meta: serde_json::Value,
) -> i64 {
    let meta_str = serde_json::to_string(&meta).unwrap();
    db.conn
        .execute(
            "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, ?2, ?3)",
            params![dataset_id, name, meta_str],
        )
        .expect("insert sample with meta");
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

fn insert_annotation(db: &Database, sample_id: i64, category_id: Option<i64>) -> i64 {
    db.conn
        .execute(
            "INSERT INTO annotations (sample_id, category_id) VALUES (?1, ?2)",
            params![sample_id, category_id],
        )
        .expect("insert annotation");
    db.conn.last_insert_rowid()
}

fn count_samples(db: &Database, dataset_id: i64) -> i64 {
    db.conn
        .query_row(
            "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
            params![dataset_id],
            |row| row.get(0),
        )
        .expect("count samples")
}

fn count_annotations_for_dataset(db: &Database, dataset_id: i64) -> i64 {
    db.conn
        .query_row(
            "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
             (SELECT id FROM samples WHERE dataset_id = ?1)",
            params![dataset_id],
            |row| row.get(0),
        )
        .expect("count annotations")
}

#[test]
fn test_full_pipeline_create_evaluate_materialize() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-full");
    for i in 0..10 {
        insert_sample(&db, ds_id, &format!("img{:03}.jpg", i));
    }

    let def = VirtualDatasetDef::Sample { ratio: 1.0 };
    let vds = VirtualDatasetService::create(&db, "all-samples-vds", vec![ds_id], &def)
        .expect("create vds");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 10);

    let materialized = materialize(&db, &storage, &vds, "mat-full").expect("materialize");
    assert_eq!(count_samples(&db, materialized.id), 10);
    assert_eq!(materialized.name, "mat-full");

    let loaded = DatasetService::get_by_id(&db, materialized.id).expect("get_by_id");
    assert_eq!(loaded.name, "mat-full");
    assert_eq!(loaded.id, materialized.id);
}

#[test]
fn test_filter_by_category_pipeline() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-cats-dogs");
    let dog_cat_id = insert_category(&db, ds_id, "dog");
    let cat_cat_id = insert_category(&db, ds_id, "cat");

    for i in 0..5 {
        let s_id = insert_sample(&db, ds_id, &format!("dog{:02}.jpg", i));
        insert_annotation(&db, s_id, Some(dog_cat_id));
    }
    for i in 0..5 {
        let s_id = insert_sample(&db, ds_id, &format!("cat{:02}.jpg", i));
        insert_annotation(&db, s_id, Some(cat_cat_id));
    }

    let def = VirtualDatasetDef::Filter {
        column: "category".to_string(),
        op: FilterOp::Eq,
        value: serde_json::json!("dog"),
    };
    let vds =
        VirtualDatasetService::create(&db, "dog-only-vds", vec![ds_id], &def).expect("create vds");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 5);
    for s in &samples {
        assert!(s.name.starts_with("dog"), "unexpected: {}", s.name);
    }

    let mat = materialize(&db, &storage, &vds, "mat-dogs").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 5);
    assert_eq!(count_annotations_for_dataset(&db, mat.id), 5);
}

#[test]
fn test_filter_has_annotations() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-annotated");

    let annotated: Vec<i64> = (0..7)
        .map(|i| insert_sample(&db, ds_id, &format!("ann{:02}.jpg", i)))
        .collect();
    let _unannotated: Vec<i64> = (0..3)
        .map(|i| insert_sample(&db, ds_id, &format!("bare{:02}.jpg", i)))
        .collect();

    for &s_id in &annotated {
        insert_annotation(&db, s_id, None);
    }

    let def = VirtualDatasetDef::Filter {
        column: "annotated".to_string(),
        op: FilterOp::Eq,
        value: serde_json::json!(true),
    };
    let vds = VirtualDatasetService::create(&db, "annotated-only-vds", vec![ds_id], &def)
        .expect("create vds");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 7);

    let mat = materialize(&db, &storage, &vds, "mat-annotated").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 7);
}

#[test]
fn test_chain_filter_then_sample() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-chain");
    let dog_cat_id = insert_category(&db, ds_id, "dog");
    let cat_cat_id = insert_category(&db, ds_id, "cat");

    for i in 0..10 {
        let s_id = insert_sample(&db, ds_id, &format!("chain-dog{:02}.jpg", i));
        insert_annotation(&db, s_id, Some(dog_cat_id));
    }
    for i in 0..5 {
        let s_id = insert_sample(&db, ds_id, &format!("chain-cat{:02}.jpg", i));
        insert_annotation(&db, s_id, Some(cat_cat_id));
    }

    let def = VirtualDatasetDef::Chain(vec![
        VirtualDatasetDef::Filter {
            column: "category".to_string(),
            op: FilterOp::Eq,
            value: serde_json::json!("dog"),
        },
        VirtualDatasetDef::Sample { ratio: 0.5 },
    ]);
    let vds = VirtualDatasetService::create(&db, "chain-vds", vec![ds_id], &def).expect("create");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 5, "10 dogs * 50% = 5");
    for s in &samples {
        assert!(s.name.starts_with("chain-dog"), "wrong: {}", s.name);
    }

    let mat = materialize(&db, &storage, &vds, "mat-chain").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 5);
}

#[test]
fn test_merge_two_datasets_pipeline() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds1_id = insert_dataset(&db, "base-merge-a");
    let ds2_id = insert_dataset(&db, "base-merge-b");

    for i in 0..6 {
        insert_sample(&db, ds1_id, &format!("a{:02}.jpg", i));
    }
    for i in 0..4 {
        insert_sample(&db, ds2_id, &format!("b{:02}.jpg", i));
    }

    let def = VirtualDatasetDef::Merge {
        datasets: vec![ds2_id],
    };
    let vds =
        VirtualDatasetService::create(&db, "merged-ab-vds", vec![ds1_id], &def).expect("create");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 10);

    let mat = materialize(&db, &storage, &vds, "mat-merged").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 10);
}

#[test]
fn test_schema_transform_rename_via_materialize() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-schema");
    for i in 0..5 {
        insert_sample_with_meta(
            &db,
            ds_id,
            &format!("schema{:02}.jpg", i),
            serde_json::json!({"label": "cat", "score": 0.9}),
        );
    }

    let def = VirtualDatasetDef::Chain(vec![
        VirtualDatasetDef::Sample { ratio: 1.0 },
        VirtualDatasetDef::SchemaTransform {
            transforms: vec![SchemaOp::RenameColumn {
                from: "label".to_string(),
                to: "class".to_string(),
            }],
        },
    ]);
    let vds = VirtualDatasetService::create(&db, "schema-vds", vec![ds_id], &def).expect("create");

    let mat = materialize(&db, &storage, &vds, "mat-schema").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 5);

    let mut stmt = db
        .conn
        .prepare("SELECT metadata FROM samples WHERE dataset_id = ?1 ORDER BY id")
        .unwrap();
    let metas: Vec<Option<String>> = stmt
        .query_map(params![mat.id], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();

    assert_eq!(metas.len(), 5);
    for meta_opt in &metas {
        let meta_str = meta_opt.as_ref().expect("metadata should be present");
        let meta: serde_json::Value = serde_json::from_str(meta_str).unwrap();
        assert!(
            meta.get("label").is_none(),
            "old key 'label' should not exist"
        );
        assert!(meta.get("class").is_some(), "new key 'class' should exist");
    }
}

#[test]
fn test_vds_referencing_deleted_base_evaluates_empty() {
    let db = in_memory_db();

    let ds_id = insert_dataset(&db, "to-be-deleted");
    for i in 0..5 {
        insert_sample(&db, ds_id, &format!("img{:02}.jpg", i));
    }

    let def = VirtualDatasetDef::Sample { ratio: 1.0 };
    let vds =
        VirtualDatasetService::create(&db, "orphan-vds", vec![ds_id], &def).expect("create vds");

    let before = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate before delete");
    assert_eq!(before.len(), 5);

    db.conn
        .execute("DELETE FROM samples WHERE dataset_id = ?1", params![ds_id])
        .expect("delete samples");
    db.conn
        .execute("DELETE FROM datasets WHERE id = ?1", params![ds_id])
        .expect("delete dataset");

    let after = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate after delete");
    assert_eq!(after.len(), 0);
}

#[test]
fn test_preview_limits_results() {
    let db = in_memory_db();

    let ds_id = insert_dataset(&db, "base-preview");
    for i in 0..20 {
        insert_sample(&db, ds_id, &format!("prev{:03}.jpg", i));
    }

    let def = VirtualDatasetDef::Sample { ratio: 1.0 };
    let vds =
        VirtualDatasetService::create(&db, "preview-full-vds", vec![ds_id], &def).expect("create");

    let preview5 = VirtualDatasetService::preview(&db, &vds, 5).expect("preview 5");
    assert_eq!(preview5.len(), 5);

    let preview10 = VirtualDatasetService::preview(&db, &vds, 10).expect("preview 10");
    assert_eq!(preview10.len(), 10);

    let preview100 = VirtualDatasetService::preview(&db, &vds, 100).expect("preview 100");
    assert_eq!(preview100.len(), 20);
}

#[test]
fn test_vds_list_get_delete() {
    let db = in_memory_db();

    let ds_id = insert_dataset(&db, "base-crud");

    assert!(VirtualDatasetService::list(&db).unwrap().is_empty());

    let def = VirtualDatasetDef::Sample { ratio: 0.5 };
    VirtualDatasetService::create(&db, "vds-1", vec![ds_id], &def).expect("create 1");
    VirtualDatasetService::create(&db, "vds-2", vec![ds_id], &def).expect("create 2");
    VirtualDatasetService::create(&db, "vds-3", vec![ds_id], &def).expect("create 3");

    let all = VirtualDatasetService::list(&db).unwrap();
    assert_eq!(all.len(), 3);

    let fetched = VirtualDatasetService::get(&db, "vds-2").unwrap();
    assert_eq!(fetched.name, "vds-2");

    VirtualDatasetService::delete(&db, "vds-2").expect("delete");
    let after_delete = VirtualDatasetService::list(&db).unwrap();
    assert_eq!(after_delete.len(), 2);

    let names: Vec<&str> = after_delete.iter().map(|v| v.name.as_str()).collect();
    assert!(names.contains(&"vds-1"));
    assert!(names.contains(&"vds-3"));
    assert!(!names.contains(&"vds-2"));
}

#[test]
fn test_materialize_preserves_annotations() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-ann-copy");
    let cat_id = insert_category(&db, ds_id, "car");

    for i in 0..5 {
        let s_id = insert_sample(&db, ds_id, &format!("car{:02}.jpg", i));
        insert_annotation(&db, s_id, Some(cat_id));
        insert_annotation(&db, s_id, Some(cat_id));
    }

    let def = VirtualDatasetDef::Sample { ratio: 1.0 };
    let vds = VirtualDatasetService::create(&db, "car-vds", vec![ds_id], &def).expect("create vds");

    let mat = materialize(&db, &storage, &vds, "mat-car").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 5);
    assert_eq!(count_annotations_for_dataset(&db, mat.id), 10);
}

#[test]
fn test_delete_nonexistent_vds_returns_error() {
    let db = in_memory_db();
    let err = VirtualDatasetService::delete(&db, "ghost-vds").expect_err("should fail");
    assert!(
        matches!(err, DmanError::DatasetNotFound(_)),
        "expected DatasetNotFound, got {:?}",
        err
    );
}

#[test]
fn test_filter_by_metadata_field() {
    let tmp = TempDir::new().expect("tempdir");
    let db = in_memory_db();
    let storage = make_storage(&tmp);

    let ds_id = insert_dataset(&db, "base-meta-filter");

    for i in 0..8 {
        insert_sample_with_meta(
            &db,
            ds_id,
            &format!("train{:02}.jpg", i),
            serde_json::json!({"split": "train"}),
        );
    }
    for i in 0..4 {
        insert_sample_with_meta(
            &db,
            ds_id,
            &format!("val{:02}.jpg", i),
            serde_json::json!({"split": "val"}),
        );
    }

    let def = VirtualDatasetDef::Filter {
        column: "metadata.split".to_string(),
        op: FilterOp::Eq,
        value: serde_json::json!("train"),
    };
    let vds = VirtualDatasetService::create(&db, "train-split-vds", vec![ds_id], &def)
        .expect("create vds");

    let samples = VirtualDatasetService::evaluate(&db, &vds).expect("evaluate");
    assert_eq!(samples.len(), 8);

    let mat = materialize(&db, &storage, &vds, "mat-train").expect("materialize");
    assert_eq!(count_samples(&db, mat.id), 8);
}
