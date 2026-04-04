use std::fs;

use arrow::array::Array;
use dman_core::db::Database;
use dman_core::formats::coco::CocoImporter;
use dman_core::formats::huggingface::{HuggingFaceExporter, HuggingFaceImporter};
use dman_core::formats::{FormatExporter, FormatImporter};
use dman_core::storage::StorageManager;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tempfile::TempDir;

fn count_query(db: &Database, sql: &str, dataset_id: i64) -> i64 {
    db.conn
        .query_row(sql, rusqlite::params![dataset_id], |row| row.get(0))
        .expect("count query")
}

fn coco_fixture_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("coco")
}

#[test]
fn coco_import_then_hf_export_creates_parquet() {
    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let dataset = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-parquet")
        .expect("COCO import");

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("HuggingFace export");

    let parquet_path = out_tmp
        .path()
        .join("data")
        .join("train-00000-of-00001.parquet");
    assert!(
        parquet_path.exists(),
        "parquet file must be created at: {}",
        parquet_path.display()
    );
    assert!(
        parquet_path.metadata().expect("metadata").len() > 0,
        "parquet file must be non-empty"
    );
}

#[test]
fn coco_import_hf_export_row_count_matches_image_count() {
    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let dataset = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-row-count")
        .expect("COCO import");

    let original_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
        dataset.id,
    );

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("HuggingFace export");

    let parquet_path = out_tmp
        .path()
        .join("data")
        .join("train-00000-of-00001.parquet");
    let file = fs::File::open(&parquet_path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let mut reader = builder.build().expect("parquet reader");

    let mut total_rows = 0usize;
    for batch in &mut reader {
        total_rows += batch.expect("read batch").num_rows();
    }

    assert_eq!(
        total_rows as i64, original_img_count,
        "parquet row count ({total_rows}) must match original image count ({original_img_count})"
    );
}

#[test]
fn coco_import_hf_export_reimport_image_count_matches() {
    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let original_ds = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-reimport")
        .expect("COCO import");

    let original_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
        original_ds.id,
    );

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &original_ds, out_tmp.path())
        .expect("HuggingFace export");

    let hf_importer = HuggingFaceImporter;
    let reimported_ds = hf_importer
        .import(&db, &storage, out_tmp.path(), "coco-hf-reimported")
        .expect("HuggingFace re-import");

    let reimported_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
        reimported_ds.id,
    );

    assert_eq!(
        original_img_count, reimported_img_count,
        "COCO→HF→re-import image count must match: original={original_img_count}, reimported={reimported_img_count}"
    );
}

#[test]
fn coco_import_hf_export_parquet_has_file_name_column() {
    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let dataset = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-schema")
        .expect("COCO import");

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("HuggingFace export");

    let parquet_path = out_tmp
        .path()
        .join("data")
        .join("train-00000-of-00001.parquet");
    let file = fs::File::open(&parquet_path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let schema = builder.schema().clone();

    let has_file_name = schema.field_with_name("file_name").is_ok();
    assert!(
        has_file_name,
        "exported parquet must contain 'file_name' column; schema: {:?}",
        schema
    );
}

#[test]
fn coco_import_hf_export_file_names_are_non_empty() {
    use arrow::array::StringArray;

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let dataset = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-filenames")
        .expect("COCO import");

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("HuggingFace export");

    let parquet_path = out_tmp
        .path()
        .join("data")
        .join("train-00000-of-00001.parquet");
    let file = fs::File::open(&parquet_path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let mut reader = builder.build().expect("parquet reader");

    let mut all_names: Vec<String> = Vec::new();
    for batch_result in &mut reader {
        let batch = batch_result.expect("read batch");
        let col = batch
            .column_by_name("file_name")
            .expect("file_name column must exist");
        let string_arr = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("file_name must be StringArray");
        for i in 0..string_arr.len() {
            if !string_arr.is_null(i) {
                all_names.push(string_arr.value(i).to_string());
            }
        }
    }

    assert!(!all_names.is_empty(), "file_name column must have values");
    for name in &all_names {
        assert!(
            !name.is_empty(),
            "each file_name must be non-empty, found empty string"
        );
    }
}

#[test]
fn coco_hf_roundtrip_drop_annotations_gracefully() {
    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let coco_importer = CocoImporter;
    let ds = coco_importer
        .import(&db, &storage, &coco_fixture_dir(), "coco-hf-graceful")
        .expect("COCO import");

    let ann_count_before = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE image_id IN \
         (SELECT id FROM images WHERE dataset_id = ?1)",
        ds.id,
    );
    assert!(
        ann_count_before > 0,
        "COCO fixture must have annotations to test graceful drop"
    );

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &ds, out_tmp.path())
        .expect("HuggingFace export must not panic even with annotations present");

    let hf_importer = HuggingFaceImporter;
    let reimported_ds = hf_importer
        .import(&db, &storage, out_tmp.path(), "coco-hf-graceful-reimported")
        .expect("HuggingFace re-import must not panic");

    let ann_count_after = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE image_id IN \
         (SELECT id FROM images WHERE dataset_id = ?1)",
        reimported_ds.id,
    );
    assert_eq!(
        ann_count_after, 0,
        "HuggingFace format does not carry annotations, so reimported count must be 0"
    );
}
