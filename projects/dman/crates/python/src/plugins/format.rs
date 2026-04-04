#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
use dman_core::error::{DmanError, Result};
#[cfg(feature = "python")]
use dman_core::types::Dataset;

use crate::PluginInfo;

pub struct PythonFormatImporter {
    #[allow(dead_code)]
    info: PluginInfo,
}

impl PythonFormatImporter {
    pub fn new(info: PluginInfo) -> Self {
        Self { info }
    }
}

pub struct PythonFormatExporter {
    #[allow(dead_code)]
    info: PluginInfo,
}

impl PythonFormatExporter {
    pub fn new(info: PluginInfo) -> Self {
        Self { info }
    }
}

#[cfg(feature = "python")]
mod python_impl {
    use std::ffi::CString;
    use std::path::Path;

    use dman_core::{
        db::Database,
        error::{DmanError, Result},
        storage::StorageManager,
        types::{Dataset, DatasetFormat},
    };
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};

    use dman_core::formats::{FormatExporter, FormatImporter};

    use super::{PythonFormatExporter, PythonFormatImporter};

    // CString values must be constructed before Python::attach to avoid
    // temporaries being dropped while the closure borrows them.
    fn with_plugin_module<T, F>(plugin_path: &Path, f: F) -> Result<T>
    where
        F: for<'py> FnOnce(Python<'py>, &Bound<'py, pyo3::types::PyModule>) -> Result<T>,
    {
        let code = std::fs::read_to_string(plugin_path)
            .map_err(|e| DmanError::StorageError(e.to_string()))?;
        let path_str = plugin_path
            .to_str()
            .ok_or_else(|| DmanError::PluginError("non-UTF8 plugin path".to_string()))?;

        let c_code =
            CString::new(code.as_str()).map_err(|e| DmanError::PluginError(e.to_string()))?;
        let c_path = CString::new(path_str).map_err(|e| DmanError::PluginError(e.to_string()))?;
        let c_mod = CString::new("plugin").map_err(|e| DmanError::PluginError(e.to_string()))?;

        Python::attach(|py| {
            let module = pyo3::types::PyModule::from_code(py, &c_code, &c_path, &c_mod)
                .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
            f(py, &module)
        })
    }

    // ── FormatImporter ────────────────────────────────────────────────────────

    impl FormatImporter for PythonFormatImporter {
        fn name(&self) -> &str {
            &self.info.name
        }

        fn detect(&self, path: &Path) -> bool {
            let path_str = match path.to_str() {
                Some(s) => s.to_string(),
                None => return false,
            };

            let result = with_plugin_module(&self.info.path, |_py, module| {
                match module.getattr("detect") {
                    Err(_) => Ok(false),
                    Ok(fn_detect) => {
                        let result = fn_detect
                            .call1((path_str.as_str(),))
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let detected: bool = result
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        Ok(detected)
                    }
                }
            });

            result.unwrap_or(false)
        }

        fn import(
            &self,
            db: &Database,
            _storage: &StorageManager,
            path: &Path,
            dataset_name: &str,
        ) -> Result<Dataset> {
            let path_str = path
                .to_str()
                .ok_or_else(|| DmanError::PluginError("non-UTF8 import path".to_string()))?
                .to_string();
            let dataset_name = dataset_name.to_string();
            let plugin_path = self.info.path.clone();

            let (images_data, annotations_data): (Vec<PyImgData>, Vec<PyAnnData>) =
                with_plugin_module(&plugin_path, |_py, module| {
                    let result = module
                        .call_method1("import_dataset", (path_str.as_str(),))
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                    let dict = result
                        .cast::<PyDict>()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                    let images_obj = dict
                        .get_item("images")
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                        .ok_or_else(|| {
                            DmanError::PluginError("missing 'images' key in result".to_string())
                        })?;
                    let images_list = images_obj
                        .cast::<PyList>()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                    let anns_obj = dict
                        .get_item("annotations")
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                        .ok_or_else(|| {
                            DmanError::PluginError(
                                "missing 'annotations' key in result".to_string(),
                            )
                        })?;
                    let anns_list = anns_obj
                        .cast::<PyList>()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                    let mut imgs = Vec::new();
                    for item in images_list.iter() {
                        let d = item
                            .cast::<PyDict>()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let file_name: String = d
                            .get_item("file_name")
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                            .ok_or_else(|| {
                                DmanError::PluginError("image dict missing 'file_name'".to_string())
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        imgs.push(PyImgData { file_name });
                    }

                    let mut anns = Vec::new();
                    for item in anns_list.iter() {
                        let d = item
                            .cast::<PyDict>()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let image_file_name: String = d
                            .get_item("image_file_name")
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                            .ok_or_else(|| {
                                DmanError::PluginError(
                                    "annotation dict missing 'image_file_name'".to_string(),
                                )
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let category: String = d
                            .get_item("category")
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                            .ok_or_else(|| {
                                DmanError::PluginError(
                                    "annotation dict missing 'category'".to_string(),
                                )
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let bbox_list = d
                            .get_item("bbox")
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
                            .ok_or_else(|| {
                                DmanError::PluginError("annotation dict missing 'bbox'".to_string())
                            })?;
                        let bbox_vec: Vec<f64> = bbox_list
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        if bbox_vec.len() < 4 {
                            return Err(DmanError::PluginError(
                                "bbox must have 4 elements [x, y, w, h]".to_string(),
                            ));
                        }
                        anns.push(PyAnnData {
                            image_file_name,
                            category,
                            bbox: [bbox_vec[0], bbox_vec[1], bbox_vec[2], bbox_vec[3]],
                        });
                    }

                    Ok((imgs, anns))
                })?;

            db.conn
                .execute("BEGIN IMMEDIATE", [])
                .map_err(DmanError::Database)?;

            let result = (|| -> Result<Dataset> {
                let dataset = dman_core::dataset::DatasetService::register(
                    db,
                    &dataset_name,
                    path,
                    DatasetFormat::Custom(self.info.name.clone()),
                )?;

                let mut file_to_id: std::collections::HashMap<String, i64> =
                    std::collections::HashMap::new();

                for img in &images_data {
                    let file_path = path.join(&img.file_name);
                    let file_path_str = file_path.to_string_lossy().to_string();
                    db.conn
                        .execute(
                            "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                            rusqlite::params![dataset.id, img.file_name, file_path_str],
                        )
                        .map_err(DmanError::Database)?;
                    let image_id = db.conn.last_insert_rowid();
                    file_to_id.insert(img.file_name.clone(), image_id);
                }

                for ann in &annotations_data {
                    let image_id =
                        file_to_id
                            .get(&ann.image_file_name)
                            .copied()
                            .ok_or_else(|| {
                                DmanError::PluginError(format!(
                                    "annotation references unknown image '{}'",
                                    ann.image_file_name
                                ))
                            })?;

                    db.conn
                        .execute(
                            "INSERT OR IGNORE INTO categories (dataset_id, name) VALUES (?1, ?2)",
                            rusqlite::params![dataset.id, ann.category],
                        )
                        .map_err(DmanError::Database)?;
                    let category_id: i64 = db
                        .conn
                        .query_row(
                            "SELECT id FROM categories WHERE dataset_id = ?1 AND name = ?2",
                            rusqlite::params![dataset.id, ann.category],
                            |row| row.get(0),
                        )
                        .map_err(DmanError::Database)?;

                    let bbox_json = serde_json::json!({
                        "x": ann.bbox[0],
                        "y": ann.bbox[1],
                        "width": ann.bbox[2],
                        "height": ann.bbox[3],
                    })
                    .to_string();

                    db.conn
                        .execute(
                            "INSERT INTO annotations (image_id, category_id, bbox) VALUES (?1, ?2, ?3)",
                            rusqlite::params![image_id, category_id, bbox_json],
                        )
                        .map_err(DmanError::Database)?;
                }

                Ok(dataset)
            })();

            match result {
                Ok(ds) => {
                    db.conn.execute("COMMIT", []).map_err(DmanError::Database)?;
                    Ok(ds)
                }
                Err(e) => {
                    let _ = db.conn.execute("ROLLBACK", []);
                    Err(e)
                }
            }
        }
    }

    impl FormatExporter for PythonFormatExporter {
        fn name(&self) -> &str {
            &self.info.name
        }

        fn export(
            &self,
            db: &Database,
            _storage: &StorageManager,
            dataset: &Dataset,
            output_path: &Path,
        ) -> Result<()> {
            let output_path_str = output_path
                .to_str()
                .ok_or_else(|| DmanError::PluginError("non-UTF8 output path".to_string()))?
                .to_string();
            let dataset_id = dataset.id;
            let plugin_path = self.info.path.clone();

            let images: Vec<ExportImg> = {
                let mut stmt = db
                    .conn
                    .prepare(
                        "SELECT id, file_name, file_path, width, height FROM images WHERE dataset_id = ?1",
                    )
                    .map_err(DmanError::Database)?;
                stmt.query_map(rusqlite::params![dataset_id], |row| {
                    Ok(ExportImg {
                        id: row.get(0)?,
                        file_name: row.get(1)?,
                        file_path: row.get(2)?,
                        width: row.get(3)?,
                        height: row.get(4)?,
                    })
                })
                .map_err(DmanError::Database)?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(DmanError::Database)?
            };

            let annotations: Vec<ExportAnn> = {
                let mut stmt = db
                    .conn
                    .prepare(
                        "SELECT a.image_id, c.name, a.bbox \
                         FROM annotations a \
                         LEFT JOIN categories c ON c.id = a.category_id \
                         WHERE a.image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
                    )
                    .map_err(DmanError::Database)?;
                stmt.query_map(rusqlite::params![dataset_id], |row| {
                    Ok(ExportAnn {
                        image_id: row.get(0)?,
                        category: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                        bbox_json: row.get::<_, Option<String>>(2)?,
                    })
                })
                .map_err(DmanError::Database)?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(DmanError::Database)?
            };

            let id_to_name: std::collections::HashMap<i64, String> =
                images.iter().map(|i| (i.id, i.file_name.clone())).collect();

            with_plugin_module(&plugin_path, |py, module| {
                let py_images = PyList::empty(py);
                for img in &images {
                    let d = PyDict::new(py);
                    d.set_item("file_name", &img.file_name)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    d.set_item("file_path", &img.file_path)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    if let Some(w) = img.width {
                        d.set_item("width", w)
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    }
                    if let Some(h) = img.height {
                        d.set_item("height", h)
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    }
                    py_images
                        .append(d)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                }

                let py_annotations = PyList::empty(py);
                for ann in &annotations {
                    let image_file_name =
                        id_to_name.get(&ann.image_id).cloned().unwrap_or_default();

                    let bbox_vec: Vec<f64> = if let Some(ref j) = ann.bbox_json {
                        let v: serde_json::Value =
                            serde_json::from_str(j).unwrap_or(serde_json::Value::Null);
                        if let serde_json::Value::Object(ref m) = v {
                            vec![
                                m.get("x").and_then(|x| x.as_f64()).unwrap_or(0.0),
                                m.get("y").and_then(|x| x.as_f64()).unwrap_or(0.0),
                                m.get("width").and_then(|x| x.as_f64()).unwrap_or(0.0),
                                m.get("height").and_then(|x| x.as_f64()).unwrap_or(0.0),
                            ]
                        } else {
                            vec![0.0, 0.0, 0.0, 0.0]
                        }
                    } else {
                        vec![0.0, 0.0, 0.0, 0.0]
                    };

                    let d = PyDict::new(py);
                    d.set_item("image_file_name", &image_file_name)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    d.set_item("category", &ann.category)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    d.set_item("bbox", bbox_vec)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                    py_annotations
                        .append(d)
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                }

                let data_dict = PyDict::new(py);
                data_dict
                    .set_item("images", py_images)
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                data_dict
                    .set_item("annotations", py_annotations)
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                module
                    .call_method1("export_dataset", (data_dict, output_path_str.as_str()))
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                Ok(())
            })
        }
    }

    struct PyImgData {
        file_name: String,
    }

    struct PyAnnData {
        image_file_name: String,
        category: String,
        bbox: [f64; 4],
    }

    struct ExportImg {
        id: i64,
        file_name: String,
        file_path: String,
        width: Option<u32>,
        height: Option<u32>,
    }

    struct ExportAnn {
        image_id: i64,
        category: String,
        bbox_json: Option<String>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn make_info(name: &str, plugin_type: &str) -> PluginInfo {
        PluginInfo::new(
            name,
            plugin_type,
            "1.0.0",
            PathBuf::from("/tmp/fake_plugin.py"),
        )
    }

    #[test]
    fn test_python_format_importer_name() {
        let importer = PythonFormatImporter::new(make_info("my_format", "format"));
        assert_eq!(importer.info.name, "my_format");
    }

    #[test]
    fn test_python_format_exporter_name() {
        let exporter = PythonFormatExporter::new(make_info("my_exporter", "format"));
        assert_eq!(exporter.info.name, "my_exporter");
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_detect_returns_false_when_no_detect_fn() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "nodect", "type": "format", "version": "1.0.0"}}

def import_dataset(path):
    return {{"images": [], "annotations": []}}

def export_dataset(data, path):
    pass
"#
        )
        .expect("write plugin");

        let info = PluginInfo::new("nodect", "format", "1.0.0", tmp.path().to_path_buf());
        let importer = PythonFormatImporter::new(info);
        assert!(!importer.detect(std::path::Path::new("/tmp/some_dataset")));
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_detect_calls_plugin_detect_fn() {
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "withdetect", "type": "format", "version": "1.0.0"}}

def detect(path):
    return True

def import_dataset(path):
    return {{"images": [], "annotations": []}}

def export_dataset(data, path):
    pass
"#
        )
        .expect("write plugin");

        let info = PluginInfo::new("withdetect", "format", "1.0.0", tmp.path().to_path_buf());
        let importer = PythonFormatImporter::new(info);
        assert!(importer.detect(std::path::Path::new("/tmp/some_dataset")));
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_import_inserts_images_and_annotations() {
        use tempfile::TempDir;

        let tmp_dir = TempDir::new().expect("tmpdir");
        let plugin_file = tmp_dir.path().join("fmt_plugin.py");
        std::fs::write(
            &plugin_file,
            r#"
dman_plugin = {"name": "test_fmt", "type": "format", "version": "1.0.0"}

def import_dataset(path):
    return {
        "images": [
            {"file_name": "img001.jpg"},
            {"file_name": "img002.jpg"},
        ],
        "annotations": [
            {"image_file_name": "img001.jpg", "category": "cat", "bbox": [10.0, 20.0, 30.0, 40.0]},
            {"image_file_name": "img002.jpg", "category": "dog", "bbox": [5.0, 5.0, 50.0, 60.0]},
        ],
    }

def export_dataset(data, path):
    pass
"#,
        )
        .expect("write plugin");

        let info = PluginInfo::new("test_fmt", "format", "1.0.0", plugin_file);
        let importer = PythonFormatImporter::new(info);

        let data_dir = tmp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        use dman_core::formats::FormatImporter;
        let dataset = importer
            .import(&db, &storage, &data_dir, "test_import_ds")
            .expect("import");
        assert_eq!(dataset.name, "test_import_ds");

        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count images");
        assert_eq!(count, 2, "expected 2 images");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id IN \
                 (SELECT id FROM images WHERE dataset_id = ?1)",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");
        assert_eq!(ann_count, 2, "expected 2 annotations");
    }
}
