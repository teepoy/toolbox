use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::Deserialize;
use walkdir::WalkDir;

use crate::dataset::DatasetService;
use crate::db::Database;
use crate::formats::{FormatExporter, FormatImporter};
use crate::storage::StorageManager;
use crate::types::{Dataset, DatasetFormat};
use crate::{DmanError, Result};

#[derive(Debug, Deserialize)]
struct DataYaml {
    #[serde(default)]
    #[allow(dead_code)]
    path: Option<String>,
    #[serde(default)]
    train: Option<String>,
    names: NamesField,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum NamesField {
    List(Vec<String>),
    Map(HashMap<u64, String>),
}

impl NamesField {
    fn into_sorted_vec(self) -> Vec<String> {
        match self {
            NamesField::List(v) => v,
            NamesField::Map(m) => {
                let mut pairs: Vec<(u64, String)> = m.into_iter().collect();
                pairs.sort_by_key(|(k, _)| *k);
                pairs.into_iter().map(|(_, v)| v).collect()
            }
        }
    }
}

fn is_image_file(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .as_deref(),
        Some("jpg") | Some("jpeg") | Some("png")
    )
}

pub struct YoloImporter;

impl FormatImporter for YoloImporter {
    fn name(&self) -> &str {
        "yolo"
    }

    fn detect(&self, path: &Path) -> bool {
        path.is_dir() && path.join("data.yaml").exists()
    }

    fn import(
        &self,
        db: &Database,
        _storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset> {
        let yaml_path = path.join("data.yaml");
        let yaml_content = fs::read_to_string(&yaml_path).map_err(|e| DmanError::ImportFailed {
            path: yaml_path.clone(),
            reason: e.to_string(),
        })?;

        let data_yaml: DataYaml =
            serde_yaml::from_str(&yaml_content).map_err(DmanError::SerdeYaml)?;

        let class_names: Vec<String> = data_yaml.names.into_sorted_vec();

        let train_subdir = data_yaml
            .train
            .unwrap_or_else(|| "images/train".to_string());
        let images_dir = path.join(&train_subdir);

        // YOLO convention: images/train → labels/train (replace first path component)
        let labels_dir = {
            let rel = PathBuf::from(&train_subdir);
            let rest: PathBuf = rel.components().skip(1).collect();
            path.join("labels").join(rest)
        };

        let dataset = DatasetService::register(db, dataset_name, path, DatasetFormat::Yolo)?;
        let dataset_id = dataset.id;

        db.conn
            .execute("BEGIN IMMEDIATE", [])
            .map_err(DmanError::Database)?;

        let result = (|| -> Result<()> {
            for name in &class_names {
                db.conn.execute(
                    "INSERT INTO categories (dataset_id, name) VALUES (?1, ?2)",
                    params![dataset_id, name],
                )?;
            }

            let mut stmt = db
                .conn
                .prepare("SELECT id, name FROM categories WHERE dataset_id = ?1 ORDER BY id")?;
            let cat_rows: Vec<(i64, String)> = stmt
                .query_map(params![dataset_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(DmanError::Database)?;

            let class_to_cat_id: Vec<i64> = cat_rows.iter().map(|(id, _)| *id).collect();

            if images_dir.is_dir() {
                for entry in WalkDir::new(&images_dir)
                    .min_depth(1)
                    .max_depth(1)
                    .sort_by_file_name()
                {
                    let entry = entry.map_err(|e| DmanError::Io(e.into()))?;
                    let img_path = entry.path();

                    if !img_path.is_file() || !is_image_file(img_path) {
                        continue;
                    }

                    let file_name = img_path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();

                    let stem = img_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_string();

                    db.conn.execute(
                        "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                        params![dataset_id, file_name, img_path.to_string_lossy().as_ref()],
                    )?;
                    let image_id = db.conn.last_insert_rowid();

                    let label_path = labels_dir.join(format!("{stem}.txt"));

                    if label_path.exists() {
                        let content = fs::read_to_string(&label_path).map_err(DmanError::Io)?;

                        for line in content.lines() {
                            let line = line.trim();
                            if line.is_empty() {
                                continue;
                            }

                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() < 5 {
                                continue;
                            }

                            let class_id: usize =
                                parts[0].parse().map_err(|_| DmanError::ImportFailed {
                                    path: label_path.clone(),
                                    reason: format!("invalid class_id: {}", parts[0]),
                                })?;

                            let x: f64 = parts[1].parse().map_err(|_| DmanError::ImportFailed {
                                path: label_path.clone(),
                                reason: format!("invalid x_center: {}", parts[1]),
                            })?;
                            let y: f64 = parts[2].parse().map_err(|_| DmanError::ImportFailed {
                                path: label_path.clone(),
                                reason: format!("invalid y_center: {}", parts[2]),
                            })?;
                            let w: f64 = parts[3].parse().map_err(|_| DmanError::ImportFailed {
                                path: label_path.clone(),
                                reason: format!("invalid width: {}", parts[3]),
                            })?;
                            let h: f64 = parts[4].parse().map_err(|_| DmanError::ImportFailed {
                                path: label_path.clone(),
                                reason: format!("invalid height: {}", parts[4]),
                            })?;

                            // normalised YOLO centre-format: x,y = box centre; w,h = box size
                            let bbox_json = serde_json::json!({
                                "x": x,
                                "y": y,
                                "w": w,
                                "h": h,
                                "normalized": true
                            })
                            .to_string();

                            let cat_id: Option<i64> = class_to_cat_id.get(class_id).copied();

                            db.conn.execute(
                                "INSERT INTO annotations (image_id, category_id, bbox) VALUES (?1, ?2, ?3)",
                                params![image_id, cat_id, bbox_json],
                            )?;
                        }
                    }
                }
            }

            Ok(())
        })();

        match result {
            Ok(()) => {
                db.conn.execute("COMMIT", []).map_err(DmanError::Database)?;
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                return Err(e);
            }
        }

        Ok(dataset)
    }
}

// ---------------------------------------------------------------------------
// YoloExporter
// ---------------------------------------------------------------------------

/// Exports a dataset from `dman` to YOLO directory layout.
pub struct YoloExporter;

impl FormatExporter for YoloExporter {
    fn name(&self) -> &str {
        "yolo"
    }

    fn export(
        &self,
        db: &Database,
        _storage: &StorageManager,
        dataset: &Dataset,
        output_path: &Path,
    ) -> Result<()> {
        let dataset_id = dataset.id;

        // ── 1. Create directory structure ──────────────────────────────────
        let images_out = output_path.join("images").join("train");
        let labels_out = output_path.join("labels").join("train");
        fs::create_dir_all(&images_out).map_err(DmanError::Io)?;
        fs::create_dir_all(&labels_out).map_err(DmanError::Io)?;

        // ── 2. Load categories ────────────────────────────────────────────
        let mut cat_stmt = db
            .conn
            .prepare("SELECT id, name FROM categories WHERE dataset_id = ?1 ORDER BY id")?;
        let categories: Vec<(i64, String)> = cat_stmt
            .query_map(params![dataset_id], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DmanError::Database)?;

        // Map DB category id → YOLO class index
        let cat_id_to_idx: HashMap<i64, usize> = categories
            .iter()
            .enumerate()
            .map(|(idx, (id, _))| (*id, idx))
            .collect();

        // ── 3. Load images ────────────────────────────────────────────────
        let mut img_stmt = db.conn.prepare(
            "SELECT id, file_name, file_path FROM images WHERE dataset_id = ?1 ORDER BY id",
        )?;
        let images: Vec<(i64, String, String)> = img_stmt
            .query_map(params![dataset_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(DmanError::Database)?;

        // ── 4. For each image: copy file + write label txt ────────────────
        for (image_id, file_name, file_path) in &images {
            // Copy image file if it exists
            let src = Path::new(file_path);
            if src.exists() {
                let dst = images_out.join(file_name);
                fs::copy(src, &dst).map_err(DmanError::Io)?;
            }

            // Load annotations for this image
            let mut ann_stmt = db
                .conn
                .prepare("SELECT category_id, bbox FROM annotations WHERE image_id = ?1")?;
            let annotations: Vec<(Option<i64>, Option<String>)> = ann_stmt
                .query_map(params![image_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(DmanError::Database)?;

            // Write label file
            let stem = Path::new(file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(file_name.as_str());

            let label_file = labels_out.join(format!("{stem}.txt"));
            let mut lines = Vec::new();

            for (cat_id_opt, bbox_opt) in annotations {
                let Some(cat_id) = cat_id_opt else { continue };
                let Some(bbox_str) = bbox_opt else { continue };

                let Some(&class_idx) = cat_id_to_idx.get(&cat_id) else {
                    continue;
                };

                // Parse stored bbox JSON
                let bbox_val: serde_json::Value =
                    serde_json::from_str(&bbox_str).map_err(DmanError::SerdeJson)?;

                let x = bbox_val["x"].as_f64().unwrap_or(0.0);
                let y = bbox_val["y"].as_f64().unwrap_or(0.0);
                let w = bbox_val["w"].as_f64().unwrap_or(0.0);
                let h = bbox_val["h"].as_f64().unwrap_or(0.0);

                lines.push(format!("{class_idx} {x} {y} {w} {h}"));
            }

            fs::write(&label_file, lines.join("\n")).map_err(DmanError::Io)?;
        }

        // ── 5. Write data.yaml ────────────────────────────────────────────
        let mut yaml_lines = Vec::new();
        yaml_lines.push("path: .".to_string());
        yaml_lines.push("train: images/train".to_string());
        yaml_lines.push(format!("nc: {}", categories.len()));
        yaml_lines.push("names:".to_string());
        for (idx, (_, name)) in categories.iter().enumerate() {
            yaml_lines.push(format!("  {idx}: {name}"));
        }

        let yaml_content = yaml_lines.join("\n") + "\n";
        fs::write(output_path.join("data.yaml"), yaml_content).map_err(DmanError::Io)?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::tempdir;

    use super::{YoloExporter, YoloImporter};
    use crate::db::Database;
    use crate::formats::{FormatExporter, FormatImporter};
    use crate::storage::StorageManager;

    fn fixture_dir() -> std::path::PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest)
            .join("tests")
            .join("fixtures")
            .join("yolo")
    }

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn dummy_storage() -> StorageManager {
        StorageManager::new(std::path::PathBuf::from("/tmp/dman-test-storage"))
    }

    // ── detect ──────────────────────────────────────────────────────────────

    #[test]
    fn test_yolo_detects_fixture_dir() {
        let importer = YoloImporter;
        assert!(
            importer.detect(&fixture_dir()),
            "detect should return true for fixture dir containing data.yaml"
        );
    }

    #[test]
    fn test_yolo_detect_returns_false_for_non_yolo_dir() {
        let tmp = tempdir().unwrap();
        let importer = YoloImporter;
        assert!(
            !importer.detect(tmp.path()),
            "detect should return false when data.yaml is absent"
        );
    }

    // ── import ───────────────────────────────────────────────────────────────

    #[test]
    fn test_import_yolo_fixture() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-fixture")
            .expect("import should succeed");

        assert_eq!(dataset.name, "yolo-fixture");
        assert!(dataset.id > 0);

        // Verify exactly 3 images were inserted
        let img_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count images");

        assert_eq!(img_count, 3, "expected 3 images imported from fixture");
    }

    #[test]
    fn test_import_yolo_fixture_annotations() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-ann-check")
            .expect("import should succeed");

        // img001.txt has 1 annotation, img002.txt has 1, img003.txt has 2
        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id IN \
                 (SELECT id FROM images WHERE dataset_id = ?1)",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");

        assert_eq!(ann_count, 4, "expected 4 total annotations");
    }

    #[test]
    fn test_import_yolo_categories() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-cat-check")
            .expect("import should succeed");

        let cat_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count categories");

        // data.yaml has 2 classes: cat, dog
        assert_eq!(cat_count, 2, "expected 2 categories");
    }

    #[test]
    fn test_missing_label_file_ok() {
        // Build a temporary YOLO dataset with one image but NO label file.
        let tmp = tempdir().unwrap();
        let images_dir = tmp.path().join("images").join("train");
        std::fs::create_dir_all(&images_dir).unwrap();
        let labels_dir = tmp.path().join("labels").join("train");
        std::fs::create_dir_all(&labels_dir).unwrap();

        // Create a minimal 1x1 JPEG (not a real image, but the importer only
        // checks the file extension, not the content).
        std::fs::write(images_dir.join("img_no_label.jpg"), b"fake jpg").unwrap();

        // No corresponding label file is written.

        std::fs::write(
            tmp.path().join("data.yaml"),
            "path: .\ntrain: images/train\nnc: 1\nnames:\n  0: thing\n",
        )
        .unwrap();

        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, tmp.path(), "no-label-ds")
            .expect("import must succeed even without label file");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id IN \
                 (SELECT id FROM images WHERE dataset_id = ?1)",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");

        assert_eq!(
            ann_count, 0,
            "image with no label file should have 0 annotations"
        );
    }

    // ── export ───────────────────────────────────────────────────────────────

    #[test]
    fn test_export_creates_structure() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        // Import first so we have data to export
        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-export-test")
            .expect("import should succeed");

        let out_dir = tempdir().unwrap();
        let exporter = YoloExporter;
        exporter
            .export(&db, &storage, &dataset, out_dir.path())
            .expect("export should succeed");

        // data.yaml
        assert!(
            out_dir.path().join("data.yaml").exists(),
            "data.yaml must be created"
        );
        // images/ and labels/ dirs
        assert!(
            out_dir.path().join("images").join("train").is_dir(),
            "images/train dir must exist"
        );
        assert!(
            out_dir.path().join("labels").join("train").is_dir(),
            "labels/train dir must exist"
        );

        // Verify data.yaml content has nc and names
        let yaml_content = std::fs::read_to_string(out_dir.path().join("data.yaml")).unwrap();
        assert!(yaml_content.contains("nc:"), "data.yaml should contain nc");
        assert!(
            yaml_content.contains("names:"),
            "data.yaml should contain names"
        );
    }

    #[test]
    fn test_export_data_yaml_has_correct_classes() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-yaml-test")
            .expect("import");

        let out_dir = tempdir().unwrap();
        let exporter = YoloExporter;
        exporter
            .export(&db, &storage, &dataset, out_dir.path())
            .expect("export");

        let yaml_content = std::fs::read_to_string(out_dir.path().join("data.yaml")).unwrap();

        // The fixture has "cat" and "dog"
        assert!(
            yaml_content.contains("cat"),
            "data.yaml should mention 'cat'"
        );
        assert!(
            yaml_content.contains("dog"),
            "data.yaml should mention 'dog'"
        );
    }

    #[test]
    fn test_export_writes_label_files() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let importer = YoloImporter;

        let dataset = importer
            .import(&db, &storage, &fixture_dir(), "yolo-label-export-test")
            .expect("import");

        let out_dir = tempdir().unwrap();
        let exporter = YoloExporter;
        exporter
            .export(&db, &storage, &dataset, out_dir.path())
            .expect("export");

        let labels_dir = out_dir.path().join("labels").join("train");

        // img001 has 1 annotation → non-empty label file
        let label_001 = labels_dir.join("img001.txt");
        assert!(label_001.exists(), "img001.txt label should be created");
        let content = std::fs::read_to_string(&label_001).unwrap();
        assert!(
            !content.trim().is_empty(),
            "img001.txt should contain annotation lines"
        );

        // img003 has 2 annotations
        let label_003 = labels_dir.join("img003.txt");
        let content_003 = std::fs::read_to_string(&label_003).unwrap();
        let line_count = content_003.lines().filter(|l| !l.trim().is_empty()).count();
        assert_eq!(line_count, 2, "img003.txt should have 2 annotation lines");
    }

    // ── name ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_importer_name() {
        assert_eq!(YoloImporter.name(), "yolo");
    }

    #[test]
    fn test_exporter_name() {
        assert_eq!(YoloExporter.name(), "yolo");
    }
}
