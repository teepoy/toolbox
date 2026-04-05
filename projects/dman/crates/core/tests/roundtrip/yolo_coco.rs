use std::fs;
use std::path::Path;

use dman_core::db::Database;
use dman_core::formats::coco::{CocoExporter, CocoImporter};
use dman_core::formats::yolo::{YoloExporter, YoloImporter};
use dman_core::formats::{FormatExporter, FormatImporter};
use dman_core::storage::StorageManager;
use tempfile::TempDir;

fn write_png(path: &Path) {
    use image::RgbImage;
    let img = RgbImage::from_pixel(1, 1, image::Rgb([128u8, 128u8, 128u8]));
    img.save(path).expect("save 1×1 PNG");
}

fn build_yolo_fixture(tmp: &TempDir) -> std::path::PathBuf {
    let root = tmp.path().to_path_buf();

    let images_dir = root.join("images").join("train");
    let labels_dir = root.join("labels").join("train");
    fs::create_dir_all(&images_dir).expect("create images/train");
    fs::create_dir_all(&labels_dir).expect("create labels/train");

    write_png(&images_dir.join("img001.png"));

    fs::write(labels_dir.join("img001.txt"), "0 0.5 0.4 0.3 0.2\n").expect("write label file");

    fs::write(
        root.join("data.yaml"),
        "path: .\ntrain: images/train\nnc: 1\nnames:\n  0: cat\n",
    )
    .expect("write data.yaml");

    root
}

fn count_query(db: &Database, sql: &str, dataset_id: i64) -> i64 {
    db.conn
        .query_row(sql, rusqlite::params![dataset_id], |row| row.get(0))
        .expect("count query")
}

#[test]
fn yolo_import_then_coco_export_preserves_image_count() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let importer = YoloImporter;
    let dataset = importer
        .import(&db, &storage, &yolo_root, "yolo-coco-roundtrip")
        .expect("YOLO import should succeed");

    let img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
        dataset.id,
    );
    assert_eq!(img_count, 1, "imported 1 image from YOLO fixture");

    let out_tmp = TempDir::new().expect("coco export tempdir");
    let exporter = CocoExporter;
    exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("COCO export should succeed");

    let coco_json_path = out_tmp.path().join("annotations.json");
    assert!(coco_json_path.exists(), "annotations.json must be created");

    let content = fs::read_to_string(&coco_json_path).expect("read annotations.json");
    let coco: serde_json::Value = serde_json::from_str(&content).expect("parse annotations.json");

    let exported_img_count = coco["images"].as_array().expect("images array").len();
    assert_eq!(exported_img_count, 1, "COCO export should contain 1 image");
}

#[test]
fn yolo_import_then_coco_export_preserves_annotation_count() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let importer = YoloImporter;
    let dataset = importer
        .import(&db, &storage, &yolo_root, "yolo-coco-ann-count")
        .expect("YOLO import");

    let out_tmp = TempDir::new().expect("coco export tempdir");
    let exporter = CocoExporter;
    exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("COCO export");

    let content = fs::read_to_string(out_tmp.path().join("annotations.json")).expect("read json");
    let coco: serde_json::Value = serde_json::from_str(&content).expect("parse json");

    let exported_ann_count = coco["annotations"]
        .as_array()
        .expect("annotations array")
        .len();
    assert_eq!(
        exported_ann_count, 1,
        "COCO export should contain 1 annotation"
    );
}

#[test]
fn yolo_import_then_coco_export_preserves_category_labels() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let importer = YoloImporter;
    let dataset = importer
        .import(&db, &storage, &yolo_root, "yolo-coco-cats")
        .expect("YOLO import");

    let out_tmp = TempDir::new().expect("coco export tempdir");
    let exporter = CocoExporter;
    exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("COCO export");

    let content = fs::read_to_string(out_tmp.path().join("annotations.json")).expect("read json");
    let coco: serde_json::Value = serde_json::from_str(&content).expect("parse json");

    let categories = coco["categories"].as_array().expect("categories array");
    assert_eq!(categories.len(), 1, "should have 1 category");
    assert_eq!(
        categories[0]["name"].as_str().expect("name"),
        "cat",
        "category name should be 'cat'"
    );
}

#[test]
fn yolo_import_coco_export_reimport_image_count_matches() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let yolo_importer = YoloImporter;
    let yolo_ds = yolo_importer
        .import(&db, &storage, &yolo_root, "yolo-coco-reimport")
        .expect("YOLO import");

    let original_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
        yolo_ds.id,
    );

    let coco_tmp = TempDir::new().expect("coco export tempdir");
    let coco_exporter = CocoExporter;
    coco_exporter
        .export(&db, &storage, &yolo_ds, coco_tmp.path())
        .expect("COCO export");

    let coco_importer = CocoImporter;
    let reimported_ds = coco_importer
        .import(&db, &storage, coco_tmp.path(), "yolo-coco-reimported")
        .expect("COCO re-import");

    let reimported_img_count = count_query(
        &db,
        "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
        reimported_ds.id,
    );

    assert_eq!(
        original_img_count, reimported_img_count,
        "re-imported image count ({reimported_img_count}) must match original ({original_img_count})"
    );
}

#[test]
fn yolo_import_coco_export_reimport_annotation_count_matches() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let yolo_importer = YoloImporter;
    let yolo_ds = yolo_importer
        .import(&db, &storage, &yolo_root, "yolo-coco-ann-reimport")
        .expect("YOLO import");

    let original_ann_count = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
         (SELECT id FROM samples WHERE dataset_id = ?1)",
        yolo_ds.id,
    );

    let coco_tmp = TempDir::new().expect("coco export tempdir");
    let coco_exporter = CocoExporter;
    coco_exporter
        .export(&db, &storage, &yolo_ds, coco_tmp.path())
        .expect("COCO export");

    let coco_importer = CocoImporter;
    let reimported_ds = coco_importer
        .import(&db, &storage, coco_tmp.path(), "yolo-coco-ann-reimported")
        .expect("COCO re-import");

    let reimported_ann_count = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
         (SELECT id FROM samples WHERE dataset_id = ?1)",
        reimported_ds.id,
    );

    assert_eq!(
        original_ann_count, reimported_ann_count,
        "annotation count must survive YOLO→COCO→re-import round-trip"
    );
}

#[test]
fn yolo_import_coco_export_reimport_bbox_values_within_epsilon() {
    let yolo_tmp = TempDir::new().expect("yolo tempdir");
    let yolo_root = build_yolo_fixture(&yolo_tmp);

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let yolo_importer = YoloImporter;
    let yolo_ds = yolo_importer
        .import(&db, &storage, &yolo_root, "yolo-coco-bbox-eps")
        .expect("YOLO import");

    let original_bbox_str: String = db
        .conn
        .query_row(
            "SELECT bbox FROM annotations WHERE sample_id IN \
             (SELECT id FROM samples WHERE dataset_id = ?1) LIMIT 1",
            rusqlite::params![yolo_ds.id],
            |row| row.get(0),
        )
        .expect("fetch original bbox");
    let orig_bbox: serde_json::Value =
        serde_json::from_str(&original_bbox_str).expect("parse original bbox");

    let coco_tmp = TempDir::new().expect("coco export tempdir");
    let coco_exporter = CocoExporter;
    coco_exporter
        .export(&db, &storage, &yolo_ds, coco_tmp.path())
        .expect("COCO export");

    let coco_importer = CocoImporter;
    let reimported_ds = coco_importer
        .import(&db, &storage, coco_tmp.path(), "yolo-coco-bbox-reimported")
        .expect("COCO re-import");

    let reimported_bbox_str: String = db
        .conn
        .query_row(
            "SELECT bbox FROM annotations WHERE sample_id IN \
             (SELECT id FROM samples WHERE dataset_id = ?1) LIMIT 1",
            rusqlite::params![reimported_ds.id],
            |row| row.get(0),
        )
        .expect("fetch reimported bbox");
    let re_bbox: serde_json::Value =
        serde_json::from_str(&reimported_bbox_str).expect("parse reimported bbox");

    const EPSILON: f64 = 1e-4;
    for key in &["x", "y", "width", "height"] {
        let orig_v = orig_bbox[key].as_f64().expect("original bbox value");
        let re_v = re_bbox[key].as_f64().expect("reimported bbox value");
        assert!(
            (orig_v - re_v).abs() < EPSILON,
            "bbox[{key}]: original={orig_v} vs reimported={re_v}, delta exceeds ε={EPSILON}"
        );
    }
}

#[test]
fn yolo_import_yolo_export_roundtrip_annotation_count() {
    let fixture_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("yolo");

    let db = Database::open_in_memory().expect("in-memory DB");
    let storage_tmp = TempDir::new().expect("storage tempdir");
    let storage = StorageManager::new(storage_tmp.path().to_path_buf());

    let importer = YoloImporter;
    let dataset = importer
        .import(&db, &storage, &fixture_dir, "yolo-yolo-roundtrip")
        .expect("import YOLO fixture");

    let ann_count_before = count_query(
        &db,
        "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
         (SELECT id FROM samples WHERE dataset_id = ?1)",
        dataset.id,
    );

    let out_tmp = TempDir::new().expect("yolo export tempdir");
    let exporter = YoloExporter;
    exporter
        .export(&db, &storage, &dataset, out_tmp.path())
        .expect("YOLO export");

    let db2 = Database::open_in_memory().expect("second in-memory DB");
    let dataset2 = importer
        .import(&db2, &storage, out_tmp.path(), "yolo-yolo-reimported")
        .expect("re-import exported YOLO");

    let ann_count_after = count_query(
        &db2,
        "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
         (SELECT id FROM samples WHERE dataset_id = ?1)",
        dataset2.id,
    );

    assert_eq!(
        ann_count_before, ann_count_after,
        "YOLO→YOLO round-trip must preserve annotation count"
    );
}
