use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use dman_core::db::Database;
use dman_core::formats::huggingface::{HuggingFaceExporter, HuggingFaceImporter};
use dman_core::formats::yolo::YoloExporter;
use dman_core::formats::{FormatExporter, FormatImporter};
use dman_core::storage::StorageManager;
use parquet::arrow::ArrowWriter;
use tempfile::TempDir;

fn write_png(path: &Path) {
    use image::RgbImage;
    let img = RgbImage::from_pixel(1, 1, image::Rgb([64u8, 128u8, 192u8]));
    img.save(path).expect("save 1×1 PNG");
}

fn build_parquet_fixture(tmp: &TempDir) -> std::path::PathBuf {
    let root = tmp.path().to_path_buf();
    let data_dir = root.join("data");
    fs::create_dir_all(&data_dir).expect("create data dir");

    let schema = Arc::new(Schema::new(vec![
        Field::new("file_name", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("metadata", DataType::Utf8, true),
    ]));

    let file_names = vec!["img001.png", "img002.png"];
    let file_paths: Vec<String> = file_names
        .iter()
        .map(|n| root.join(n).to_string_lossy().to_string())
        .collect();
    let metadata = vec!["{}", "{}"];

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(file_names.clone())) as ArrayRef,
            Arc::new(StringArray::from(file_paths.clone())) as ArrayRef,
            Arc::new(StringArray::from(metadata)) as ArrayRef,
        ],
    )
    .expect("create RecordBatch");

    let parquet_path = data_dir.join("train-00000-of-00001.parquet");
    let out_file = fs::File::create(&parquet_path).expect("create parquet file");
    let mut writer = ArrowWriter::try_new(out_file, schema, None).expect("create ArrowWriter");
    writer.write(&batch).expect("write batch");
    writer.close().expect("close writer");

    for name in &file_names {
        write_png(&root.join(name));
    }

    root
}

fn count_query(db: &Database, sql: &str, dataset_id: i64) -> i64 {
    db.conn
        .query_row(sql, rusqlite::params![dataset_id], |row| row.get(0))
        .expect("count query")
}

#[test]
fn parquet_import_then_yolo_export_creates_structure() {
    let fixture_tmp = TempDir::new().expect("fixture tempdir");
    let fixture_root = build_parquet_fixture(&fixture_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let hf_importer = HuggingFaceImporter;
    let dataset = hf_importer
        .import(&db, &storage, &fixture_root, "hf-yolo-struct")
        .expect("HuggingFace import");

    let out_tmp = TempDir::new().expect("yolo export tempdir");
    let yolo_exporter = YoloExporter;
    yolo_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("YOLO export should succeed");

    assert!(
        out_tmp.path().join("data.yaml").exists(),
        "data.yaml must exist after YOLO export"
    );
    assert!(
        out_tmp.path().join("images").join("train").is_dir(),
        "images/train dir must exist"
    );
    assert!(
        out_tmp.path().join("labels").join("train").is_dir(),
        "labels/train dir must exist"
    );
}

#[test]
fn parquet_import_yolo_export_data_yaml_is_valid() {
    let fixture_tmp = TempDir::new().expect("fixture tempdir");
    let fixture_root = build_parquet_fixture(&fixture_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let hf_importer = HuggingFaceImporter;
    let dataset = hf_importer
        .import(&db, &storage, &fixture_root, "hf-yolo-yaml")
        .expect("HuggingFace import");

    let out_tmp = TempDir::new().expect("yolo export tempdir");
    let yolo_exporter = YoloExporter;
    yolo_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("YOLO export");

    let yaml_content =
        fs::read_to_string(out_tmp.path().join("data.yaml")).expect("read data.yaml");
    assert!(
        yaml_content.contains("nc:"),
        "data.yaml must contain 'nc:' field"
    );
    assert!(
        yaml_content.contains("names:"),
        "data.yaml must contain 'names:' field"
    );
    assert!(
        yaml_content.contains("train:"),
        "data.yaml must contain 'train:' field"
    );
}

#[test]
fn parquet_import_yolo_export_reimport_image_count_matches() {
    let fixture_tmp = TempDir::new().expect("fixture tempdir");
    let fixture_root = build_parquet_fixture(&fixture_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let hf_importer = HuggingFaceImporter;
    let original_ds = hf_importer
        .import(&db, &storage, &fixture_root, "hf-yolo-reimport")
        .expect("HuggingFace import");

    let original_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
        original_ds.id,
    );

    let out_tmp = TempDir::new().expect("yolo export tempdir");
    let yolo_exporter = YoloExporter;
    yolo_exporter
        .export(&db, &storage, &original_ds, out_tmp.path())
        .expect("YOLO export");

    let exported_yaml =
        fs::read_to_string(out_tmp.path().join("data.yaml")).expect("read exported data.yaml");
    assert!(
        exported_yaml.contains("nc: 0"),
        "exported data.yaml should have nc: 0 for zero-category HF dataset"
    );

    let exported_label_dir = out_tmp.path().join("labels").join("train");
    let exported_label_count = fs::read_dir(&exported_label_dir)
        .expect("read labels/train")
        .filter_map(|e| e.ok())
        .count();
    assert_eq!(
        exported_label_count as i64, original_img_count,
        "YOLO export must write one label file per image: expected={original_img_count}, got={exported_label_count}"
    );
}

#[test]
fn parquet_import_hf_export_roundtrip_row_count() {
    let fixture_tmp = TempDir::new().expect("fixture tempdir");
    let fixture_root = build_parquet_fixture(&fixture_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let hf_importer = HuggingFaceImporter;
    let original_ds = hf_importer
        .import(&db, &storage, &fixture_root, "hf-hf-roundtrip")
        .expect("HuggingFace import");

    let original_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
        original_ds.id,
    );

    let out_tmp = TempDir::new().expect("hf export tempdir");
    let hf_exporter = HuggingFaceExporter;
    hf_exporter
        .export(&db, &storage, &original_ds, out_tmp.path())
        .expect("HuggingFace export");

    let parquet_path = out_tmp
        .path()
        .join("data")
        .join("train-00000-of-00001.parquet");
    let file = fs::File::open(&parquet_path).expect("open exported parquet");
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet builder");
    let mut reader = builder.build().expect("parquet reader");

    let mut total_rows = 0usize;
    for batch_result in &mut reader {
        total_rows += batch_result.expect("read batch").num_rows();
    }

    assert_eq!(
        total_rows as i64, original_img_count,
        "HuggingFace→HuggingFace roundtrip must preserve row count: original={original_img_count}, roundtrip={total_rows}"
    );
}

#[test]
fn parquet_import_yolo_export_no_annotations_graceful() {
    let fixture_tmp = TempDir::new().expect("fixture tempdir");
    let fixture_root = build_parquet_fixture(&fixture_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let hf_importer = HuggingFaceImporter;
    let dataset = hf_importer
        .import(&db, &storage, &fixture_root, "hf-yolo-no-ann")
        .expect("HuggingFace import");

    let ann_count = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
         (SELECT id FROM samples WHERE dataset_id = ?1)",
        dataset.id,
    );
    assert_eq!(
        ann_count, 0,
        "HuggingFace import must have 0 annotations (format does not carry bbox)"
    );

    let out_tmp = TempDir::new().expect("yolo export tempdir");
    let yolo_exporter = YoloExporter;
    yolo_exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("YOLO export must succeed even with no annotations");

    let labels_dir = out_tmp.path().join("labels").join("train");
    assert!(labels_dir.is_dir(), "labels/train dir must exist");

    let label_files: Vec<_> = fs::read_dir(&labels_dir)
        .expect("read labels dir")
        .filter_map(|e| e.ok())
        .collect();

    for entry in &label_files {
        let content = fs::read_to_string(entry.path()).expect("read label file");
        assert!(
            content.trim().is_empty(),
            "label file for image without annotations must be empty: {:?}",
            entry.path()
        );
    }
}
