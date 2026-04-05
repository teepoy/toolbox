use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::dataset::DatasetService;
use crate::db::Database;
use crate::formats::{FormatExporter, FormatImporter};
use crate::storage::StorageManager;
use crate::types::{AssetType, Dataset, DatasetFormat};
use crate::{DmanError, Result};

#[derive(Debug, Deserialize, Serialize)]
struct CocoImage {
    id: i64,
    file_name: String,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CocoAnnotation {
    id: i64,
    image_id: i64,
    category_id: i64,
    #[serde(default)]
    bbox: Vec<f64>,
    #[serde(default)]
    segmentation: serde_json::Value,
    #[serde(default)]
    area: f64,
    #[serde(default)]
    iscrowd: i64,
}

#[derive(Debug, Deserialize, Serialize)]
struct CocoCategory {
    id: i64,
    name: String,
    supercategory: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CocoJson {
    #[serde(default)]
    images: Vec<CocoImage>,
    #[serde(default)]
    annotations: Vec<CocoAnnotation>,
    #[serde(default)]
    categories: Vec<CocoCategory>,
    #[serde(skip_serializing_if = "Option::is_none")]
    info: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    licenses: Option<serde_json::Value>,
}

pub struct CocoImporter;

impl CocoImporter {
    fn find_annotations_file(path: &Path) -> Option<PathBuf> {
        let direct = path.join("annotations.json");
        if direct.is_file() {
            return Some(direct);
        }

        let ann_dir = path.join("annotations");
        if ann_dir.is_dir()
            && let Ok(entries) = fs::read_dir(&ann_dir)
        {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    return Some(p);
                }
            }
        }

        None
    }

    fn looks_like_coco(value: &serde_json::Value) -> bool {
        value.get("images").is_some()
            && value.get("annotations").is_some()
            && value.get("categories").is_some()
    }
}

impl FormatImporter for CocoImporter {
    fn name(&self) -> &str {
        "coco"
    }

    fn detect(&self, path: &Path) -> bool {
        if path.is_dir() {
            let ann_dir = path.join("annotations");
            if ann_dir.is_dir()
                && let Ok(entries) = fs::read_dir(&ann_dir)
            {
                let has_json = entries
                    .flatten()
                    .any(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"));
                if has_json {
                    return true;
                }
            }

            let direct = path.join("annotations.json");
            if direct.is_file()
                && let Ok(content) = fs::read_to_string(&direct)
                && let Ok(val) = serde_json::from_str::<serde_json::Value>(&content)
            {
                return Self::looks_like_coco(&val);
            }
        }

        if path.is_file()
            && path.extension().and_then(|e| e.to_str()) == Some("json")
            && let Ok(content) = fs::read_to_string(path)
            && let Ok(val) = serde_json::from_str::<serde_json::Value>(&content)
        {
            return Self::looks_like_coco(&val);
        }

        false
    }

    fn import(
        &self,
        db: &Database,
        _storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset> {
        let json_path = if path.is_file() {
            path.to_path_buf()
        } else {
            Self::find_annotations_file(path).ok_or_else(|| DmanError::ImportFailed {
                path: path.to_path_buf(),
                reason: "no COCO annotations JSON file found".to_string(),
            })?
        };

        let content = fs::read_to_string(&json_path)?;
        let coco: CocoJson = serde_json::from_str(&content)?;

        let dataset_path = if path.is_file() {
            path.parent().unwrap_or(path)
        } else {
            path
        };

        let dataset =
            DatasetService::register(db, dataset_name, dataset_path, DatasetFormat::coco())?;
        let dataset_id = dataset.id;

        db.conn.execute("BEGIN IMMEDIATE", [])?;

        let mut cat_id_map: HashMap<i64, i64> = HashMap::new();
        for cat in &coco.categories {
            db.conn.execute(
                "INSERT INTO categories (dataset_id, name, supercategory) VALUES (?1, ?2, ?3)",
                params![dataset_id, cat.name, cat.supercategory],
            )?;
            cat_id_map.insert(cat.id, db.conn.last_insert_rowid());
        }

        let mut img_id_map: HashMap<i64, (i64, i64)> = HashMap::new();
        for img in &coco.images {
            let file_path = dataset_path.join(&img.file_name);
            let stem = std::path::Path::new(&img.file_name)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&img.file_name)
                .to_string();
            let asset_type_str = AssetType::Image.to_string();
            db.conn.execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                params![dataset_id, stem],
            )?;
            let sample_id = db.conn.last_insert_rowid();
            db.conn.execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path, width, height) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    sample_id,
                    asset_type_str,
                    img.file_name,
                    file_path.to_string_lossy().as_ref(),
                    img.width,
                    img.height
                ],
            )?;
            let asset_id = db.conn.last_insert_rowid();
            img_id_map.insert(img.id, (sample_id, asset_id));
        }

        for ann in &coco.annotations {
            let (db_sample_id, db_asset_id) =
                img_id_map
                    .get(&ann.image_id)
                    .copied()
                    .ok_or_else(|| DmanError::ImportFailed {
                        path: json_path.clone(),
                        reason: format!(
                            "annotation {} references unknown image_id {}",
                            ann.id, ann.image_id
                        ),
                    })?;

            let db_cat_id = cat_id_map.get(&ann.category_id).copied();

            let bbox_json = if ann.bbox.len() == 4 {
                let bbox = serde_json::json!({
                    "x": ann.bbox[0],
                    "y": ann.bbox[1],
                    "width": ann.bbox[2],
                    "height": ann.bbox[3]
                });
                Some(serde_json::to_string(&bbox)?)
            } else {
                None
            };

            let seg_json = match &ann.segmentation {
                serde_json::Value::Array(segs) if !segs.is_empty() => {
                    Some(serde_json::to_string(&segs)?)
                }
                _ => None,
            };

            let meta = serde_json::json!({
                "coco_id": ann.id,
                "area": ann.area,
                "iscrowd": ann.iscrowd
            });
            let meta_str = serde_json::to_string(&meta)?;

            db.conn.execute(
                "INSERT INTO annotations (sample_id, asset_id, category_id, bbox, segmentation, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![db_sample_id, db_asset_id, db_cat_id, bbox_json, seg_json, meta_str],
            )?;
        }

        db.conn.execute("COMMIT", [])?;

        Ok(dataset)
    }
}

pub struct CocoExporter;

impl FormatExporter for CocoExporter {
    fn name(&self) -> &str {
        "coco"
    }

    fn export(
        &self,
        db: &Database,
        _storage: &StorageManager,
        dataset: &Dataset,
        output_path: &Path,
    ) -> Result<()> {
        let dataset_id = dataset.id;

        // Query categories
        let mut cat_stmt = db.conn.prepare(
            "SELECT id, name, supercategory FROM categories WHERE dataset_id = ?1 ORDER BY id",
        )?;
        let coco_categories: Vec<CocoCategory> = cat_stmt
            .query_map(params![dataset_id], |row| {
                Ok(CocoCategory {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    supercategory: row.get(2)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let cat_id_remap: HashMap<i64, i64> = coco_categories
            .iter()
            .enumerate()
            .map(|(i, c)| (c.id, (i as i64) + 1))
            .collect();

        let export_categories: Vec<CocoCategory> = coco_categories
            .iter()
            .enumerate()
            .map(|(i, c)| CocoCategory {
                id: (i as i64) + 1,
                name: c.name.clone(),
                supercategory: c.supercategory.clone(),
            })
            .collect();

        let mut img_stmt = db.conn.prepare(
            "SELECT a.id, a.file_name, a.width, a.height \
             FROM assets a \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1 ORDER BY a.id",
        )?;
        struct RawImg {
            db_id: i64,
            file_name: String,
            width: Option<u32>,
            height: Option<u32>,
        }
        let raw_images: Vec<RawImg> = img_stmt
            .query_map(params![dataset_id], |row| {
                Ok(RawImg {
                    db_id: row.get(0)?,
                    file_name: row.get(1)?,
                    width: row.get(2)?,
                    height: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let img_id_remap: HashMap<i64, i64> = raw_images
            .iter()
            .enumerate()
            .map(|(i, img)| (img.db_id, (i as i64) + 1))
            .collect();

        let export_images: Vec<CocoImage> = raw_images
            .iter()
            .enumerate()
            .map(|(i, img)| CocoImage {
                id: (i as i64) + 1,
                file_name: img.file_name.clone(),
                width: img.width,
                height: img.height,
            })
            .collect();

        let mut ann_stmt = db.conn.prepare(
            "SELECT a.id, a.asset_id, a.category_id, a.bbox, a.segmentation \
             FROM annotations a \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1 \
             ORDER BY a.id",
        )?;

        struct RawAnn {
            _db_id: i64,
            asset_id: Option<i64>,
            category_id: Option<i64>,
            bbox_str: Option<String>,
            seg_str: Option<String>,
        }

        let raw_anns: Vec<RawAnn> = ann_stmt
            .query_map(params![dataset_id], |row| {
                Ok(RawAnn {
                    _db_id: row.get(0)?,
                    asset_id: row.get(1)?,
                    category_id: row.get(2)?,
                    bbox_str: row.get(3)?,
                    seg_str: row.get(4)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut export_annotations: Vec<CocoAnnotation> = Vec::with_capacity(raw_anns.len());

        for (i, ann) in raw_anns.iter().enumerate() {
            let export_img_id = match ann.asset_id.and_then(|aid| img_id_remap.get(&aid)) {
                Some(id) => *id,
                None => continue,
            };

            let export_cat_id = ann
                .category_id
                .and_then(|cid| cat_id_remap.get(&cid).copied())
                .unwrap_or(0);

            let (bbox_vec, area) = if let Some(ref bs) = ann.bbox_str {
                if let Ok(obj) = serde_json::from_str::<serde_json::Value>(bs) {
                    let x = obj["x"].as_f64().unwrap_or(0.0);
                    let y = obj["y"].as_f64().unwrap_or(0.0);
                    let w = obj["width"].as_f64().unwrap_or(0.0);
                    let h = obj["height"].as_f64().unwrap_or(0.0);
                    (vec![x, y, w, h], w * h)
                } else {
                    (vec![], 0.0)
                }
            } else {
                (vec![], 0.0)
            };

            let seg_val = if let Some(ref ss) = ann.seg_str {
                serde_json::from_str(ss).unwrap_or(serde_json::Value::Array(vec![]))
            } else {
                serde_json::Value::Array(vec![])
            };

            export_annotations.push(CocoAnnotation {
                id: (i as i64) + 1,
                image_id: export_img_id,
                category_id: export_cat_id,
                bbox: bbox_vec,
                segmentation: seg_val,
                area,
                iscrowd: 0,
            });
        }

        let coco_out = CocoJson {
            images: export_images,
            annotations: export_annotations,
            categories: export_categories,
            info: Some(serde_json::json!({
                "description": dataset.name,
                "version": "1.0",
                "exported_by": "dman"
            })),
            licenses: None,
        };

        fs::create_dir_all(output_path)?;
        let out_file = output_path.join("annotations.json");
        let json_str = serde_json::to_string_pretty(&coco_out)?;
        fs::write(&out_file, json_str)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tempfile::tempdir;

    use super::*;
    use crate::db::Database;
    use crate::storage::StorageManager;

    fn fixture_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("coco")
    }

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn storage(tmp: &tempfile::TempDir) -> StorageManager {
        StorageManager::new(tmp.path().to_path_buf())
    }

    #[test]
    fn test_coco_detects_fixture_dir() {
        let imp = CocoImporter;
        assert!(
            imp.detect(&fixture_dir()),
            "detect() should return true for the fixture directory"
        );
    }

    #[test]
    fn test_coco_detect_returns_false_for_empty_dir() {
        let tmp = tempdir().unwrap();
        let imp = CocoImporter;
        assert!(!imp.detect(tmp.path()));
    }

    #[test]
    fn test_coco_detect_annotations_subdir() {
        let tmp = tempdir().unwrap();
        let ann_dir = tmp.path().join("annotations");
        fs::create_dir_all(&ann_dir).unwrap();
        fs::write(ann_dir.join("instances_train.json"), b"{}").unwrap();
        let imp = CocoImporter;
        assert!(imp.detect(tmp.path()));
    }

    // ── import ─────────────────────────────────────────────────────────────

    #[test]
    fn test_import_coco_fixture() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);
        let imp = CocoImporter;

        let ds = imp
            .import(&db, &storage, &fixture_dir(), "test-coco")
            .expect("import should succeed");

        assert_eq!(ds.name, "test-coco");
        assert_eq!(ds.format, DatasetFormat::coco());

        let img_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![ds.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(img_count, 3, "should have 3 samples");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
                params![ds.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ann_count, 5, "should have 5 annotations");

        let cat_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
                params![ds.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cat_count, 2, "should have 2 categories");
    }

    #[test]
    fn test_import_bbox_stored_correctly() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);
        let imp = CocoImporter;

        let ds = imp
            .import(&db, &storage, &fixture_dir(), "bbox-test")
            .expect("import");

        let bbox_str: String = db
            .conn
            .query_row(
                "SELECT bbox FROM annotations WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1) ORDER BY id LIMIT 1",
                params![ds.id],
                |r| r.get(0),
            )
            .unwrap();

        let bbox: serde_json::Value = serde_json::from_str(&bbox_str).unwrap();
        assert_eq!(bbox["x"], 100.0);
        assert_eq!(bbox["y"], 120.0);
        assert_eq!(bbox["width"], 200.0);
        assert_eq!(bbox["height"], 150.0);
    }

    #[test]
    fn test_import_categories_have_supercategory() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);
        let imp = CocoImporter;

        let ds = imp
            .import(&db, &storage, &fixture_dir(), "cat-test")
            .expect("import");

        let supercats: Vec<Option<String>> = {
            let mut stmt = db
                .conn
                .prepare("SELECT supercategory FROM categories WHERE dataset_id = ?1 ORDER BY id")
                .unwrap();
            stmt.query_map(params![ds.id], |r| r.get(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap()
        };
        assert_eq!(supercats.len(), 2);
        for sc in &supercats {
            assert_eq!(sc.as_deref(), Some("animal"));
        }
    }

    #[test]
    fn test_export_creates_valid_json() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);

        let imp = CocoImporter;
        let ds = imp
            .import(&db, &storage, &fixture_dir(), "export-test")
            .expect("import");

        let out_dir = tmp.path().join("output");
        let exp = CocoExporter;
        exp.export(&db, &storage, &ds, &out_dir)
            .expect("export should succeed");

        let out_file = out_dir.join("annotations.json");
        assert!(out_file.exists(), "annotations.json should be created");

        let content = fs::read_to_string(&out_file).unwrap();
        let coco: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(coco.get("images").is_some());
        assert!(coco.get("annotations").is_some());
        assert!(coco.get("categories").is_some());
        assert!(coco.get("info").is_some());
    }

    #[test]
    fn test_export_correct_counts() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);

        let imp = CocoImporter;
        let ds = imp
            .import(&db, &storage, &fixture_dir(), "count-test")
            .expect("import");

        let out_dir = tmp.path().join("output");
        let exp = CocoExporter;
        exp.export(&db, &storage, &ds, &out_dir).expect("export");

        let content = fs::read_to_string(out_dir.join("annotations.json")).unwrap();
        let coco: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(coco["images"].as_array().unwrap().len(), 3);
        assert_eq!(coco["annotations"].as_array().unwrap().len(), 5);
        assert_eq!(coco["categories"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_roundtrip() {
        let tmp = tempdir().unwrap();
        let db = in_memory_db();
        let storage = storage(&tmp);

        let imp = CocoImporter;
        let ds = imp
            .import(&db, &storage, &fixture_dir(), "roundtrip")
            .expect("import");

        let out_dir = tmp.path().join("roundtrip_output");
        let exp = CocoExporter;
        exp.export(&db, &storage, &ds, &out_dir).expect("export");

        let content = fs::read_to_string(out_dir.join("annotations.json")).unwrap();
        let exported: serde_json::Value = serde_json::from_str(&content).unwrap();

        assert_eq!(
            exported["images"].as_array().unwrap().len(),
            3,
            "roundtrip: image count"
        );
        assert_eq!(
            exported["annotations"].as_array().unwrap().len(),
            5,
            "roundtrip: annotation count"
        );
        assert_eq!(
            exported["categories"].as_array().unwrap().len(),
            2,
            "roundtrip: category count"
        );

        let first_ann = &exported["annotations"][0];
        let bbox = first_ann["bbox"].as_array().unwrap();
        assert_eq!(bbox.len(), 4, "bbox should have 4 elements");
    }

    #[test]
    fn test_importer_name() {
        assert_eq!(CocoImporter.name(), "coco");
    }

    #[test]
    fn test_exporter_name() {
        assert_eq!(CocoExporter.name(), "coco");
    }
}
