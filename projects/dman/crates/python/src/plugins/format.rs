use crate::PluginInfo;

use dman_core::formats::{FormatExporter, FormatImporter};

type ImporterExporterParts = (Vec<Box<dyn FormatImporter>>, Vec<Box<dyn FormatExporter>>);

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

pub struct PythonFormatRegistry {
    importers: Vec<Box<dyn FormatImporter>>,
    exporters: Vec<Box<dyn FormatExporter>>,
}

impl PythonFormatRegistry {
    pub fn into_importers(self) -> Vec<Box<dyn FormatImporter>> {
        self.importers
    }

    pub fn into_exporters(self) -> Vec<Box<dyn FormatExporter>> {
        self.exporters
    }

    pub fn into_parts(self) -> ImporterExporterParts {
        (self.importers, self.exporters)
    }
}

#[cfg(feature = "python")]
pub fn load_python_format_registry(
    plugin_dirs: Vec<std::path::PathBuf>,
) -> dman_core::Result<PythonFormatRegistry> {
    let manager = crate::PluginManager::new(plugin_dirs);
    let plugins = manager.discover()?;
    let mut importers: Vec<Box<dyn FormatImporter>> = Vec::new();
    let mut exporters: Vec<Box<dyn FormatExporter>> = Vec::new();

    for info in plugins {
        if info.plugin_type == "format" {
            importers.push(Box::new(PythonFormatImporter::new(info.clone())));
            exporters.push(Box::new(PythonFormatExporter::new(info)));
        }
    }

    Ok(PythonFormatRegistry {
        importers,
        exporters,
    })
}

#[cfg(feature = "python")]
mod python_impl {
    use std::ffi::CString;
    use std::path::Path;
    use std::str::FromStr;

    use dman_core::{
        db::Database,
        error::{DmanError, Result},
        storage::StorageManager,
        types::{AssetType, BBox, Dataset, DatasetFormat},
    };
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};

    use dman_core::formats::{FormatExporter, FormatImporter};

    use super::{PythonFormatExporter, PythonFormatImporter};

    #[pyclass]
    pub struct PyAnnotationData {
        #[pyo3(get, set)]
        pub category: String,
        #[pyo3(get, set)]
        pub bbox: Option<Py<PyAny>>,
        #[pyo3(get, set)]
        pub segmentation: Option<Py<PyAny>>,
        #[pyo3(get, set)]
        pub keypoints: Option<Py<PyAny>>,
        #[pyo3(get, set)]
        pub metadata: Option<Py<PyAny>>,
    }

    #[pymethods]
    impl PyAnnotationData {
        #[new]
        #[pyo3(signature = (category, bbox=None, segmentation=None, keypoints=None, metadata=None))]
        pub fn new(
            category: String,
            bbox: Option<Py<PyAny>>,
            segmentation: Option<Py<PyAny>>,
            keypoints: Option<Py<PyAny>>,
            metadata: Option<Py<PyAny>>,
        ) -> Self {
            Self {
                category,
                bbox,
                segmentation,
                keypoints,
                metadata,
            }
        }
    }

    #[pyclass]
    pub struct PyAssetData {
        #[pyo3(get, set)]
        pub asset_type: String,
        #[pyo3(get, set)]
        pub file_name: String,
        #[pyo3(get, set)]
        pub file_path: String,
        #[pyo3(get, set)]
        pub width: Option<i64>,
        #[pyo3(get, set)]
        pub height: Option<i64>,
        #[pyo3(get, set)]
        pub metadata: Option<Py<PyAny>>,
        #[pyo3(get, set)]
        pub annotations: Vec<Py<PyAnnotationData>>,
    }

    #[pymethods]
    impl PyAssetData {
        #[new]
        #[pyo3(signature = (asset_type, file_name, file_path, width=None, height=None, metadata=None, annotations=None))]
        pub fn new(
            asset_type: String,
            file_name: String,
            file_path: String,
            width: Option<i64>,
            height: Option<i64>,
            metadata: Option<Py<PyAny>>,
            annotations: Option<Vec<Py<PyAnnotationData>>>,
        ) -> Self {
            Self {
                asset_type,
                file_name,
                file_path,
                width,
                height,
                metadata,
                annotations: annotations.unwrap_or_default(),
            }
        }
    }

    #[pyclass]
    pub struct PySampleData {
        #[pyo3(get, set)]
        pub name: String,
        #[pyo3(get, set)]
        pub metadata: Option<Py<PyAny>>,
        #[pyo3(get, set)]
        pub assets: Vec<Py<PyAssetData>>,
        #[pyo3(get, set)]
        pub annotations: Vec<Py<PyAnnotationData>>,
    }

    #[pymethods]
    impl PySampleData {
        #[new]
        #[pyo3(signature = (name, metadata=None, assets=None, annotations=None))]
        pub fn new(
            name: String,
            metadata: Option<Py<PyAny>>,
            assets: Option<Vec<Py<PyAssetData>>>,
            annotations: Option<Vec<Py<PyAnnotationData>>>,
        ) -> Self {
            Self {
                name,
                metadata,
                assets: assets.unwrap_or_default(),
                annotations: annotations.unwrap_or_default(),
            }
        }
    }

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
        let c_mod = CString::new(path_str).map_err(|e| DmanError::PluginError(e.to_string()))?;

        Python::attach(|py| {
            let module = pyo3::types::PyModule::from_code(py, &c_code, &c_path, &c_mod)
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            f(py, &module)
        })
    }

    fn get_or_create_category(db: &Database, dataset_id: i64, name: &str) -> Result<i64> {
        db.conn
            .execute(
                "INSERT OR IGNORE INTO categories (dataset_id, name) VALUES (?1, ?2)",
                rusqlite::params![dataset_id, name],
            )
            .map_err(DmanError::Database)?;
        let id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM categories WHERE dataset_id = ?1 AND name = ?2",
                rusqlite::params![dataset_id, name],
                |row| row.get(0),
            )
            .map_err(DmanError::Database)?;
        Ok(id)
    }

    fn category_name_by_id(db: &Database, category_id: i64) -> Result<String> {
        let name: String = db
            .conn
            .query_row(
                "SELECT name FROM categories WHERE id = ?1",
                rusqlite::params![category_id],
                |row| row.get(0),
            )
            .map_err(DmanError::Database)?;
        Ok(name)
    }

    struct ParsedAnnotation {
        category: String,
        bbox: Option<BBox>,
        segmentation: Option<Vec<Vec<f64>>>,
        keypoints: Option<Vec<f64>>,
        metadata_json: Option<String>,
    }

    fn extract_annotation(ann_obj: &Bound<'_, PyAny>) -> Result<ParsedAnnotation> {
        let category: String = ann_obj
            .getattr("category")
            .map_err(|e| DmanError::PluginError(format!("annotation missing 'category': {e}")))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(format!("'category' not a string: {e}")))?;

        let bbox = {
            let bbox_obj = ann_obj
                .getattr("bbox")
                .map_err(|e| DmanError::PluginError(format!("annotation missing 'bbox': {e}")))?;
            if bbox_obj.is_none() {
                None
            } else {
                let d = bbox_obj
                    .cast::<PyDict>()
                    .map_err(|e| DmanError::PluginError(format!("'bbox' must be a dict: {e}")))?;
                let x: f64 = d
                    .get_item("x")
                    .map_err(|e| DmanError::PluginError(e.to_string()))?
                    .ok_or_else(|| DmanError::PluginError("bbox dict missing 'x'".to_string()))?
                    .extract()
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                let y: f64 = d
                    .get_item("y")
                    .map_err(|e| DmanError::PluginError(e.to_string()))?
                    .ok_or_else(|| DmanError::PluginError("bbox dict missing 'y'".to_string()))?
                    .extract()
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                let width: f64 = d
                    .get_item("width")
                    .map_err(|e| DmanError::PluginError(e.to_string()))?
                    .ok_or_else(|| DmanError::PluginError("bbox dict missing 'width'".to_string()))?
                    .extract()
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                let height: f64 = d
                    .get_item("height")
                    .map_err(|e| DmanError::PluginError(e.to_string()))?
                    .ok_or_else(|| {
                        DmanError::PluginError("bbox dict missing 'height'".to_string())
                    })?
                    .extract()
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                Some(BBox {
                    x,
                    y,
                    width,
                    height,
                })
            }
        };

        let segmentation: Option<Vec<Vec<f64>>> = {
            let seg_obj = ann_obj.getattr("segmentation").map_err(|e| {
                DmanError::PluginError(format!("annotation missing 'segmentation': {e}"))
            })?;
            if seg_obj.is_none() {
                None
            } else {
                Some(
                    seg_obj
                        .extract()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?,
                )
            }
        };

        let keypoints: Option<Vec<f64>> = {
            let kp_obj = ann_obj.getattr("keypoints").map_err(|e| {
                DmanError::PluginError(format!("annotation missing 'keypoints': {e}"))
            })?;
            if kp_obj.is_none() {
                None
            } else {
                Some(
                    kp_obj
                        .extract()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?,
                )
            }
        };

        let metadata_json: Option<String> = {
            let meta_obj = ann_obj.getattr("metadata").map_err(|e| {
                DmanError::PluginError(format!("annotation missing 'metadata': {e}"))
            })?;
            if meta_obj.is_none() {
                None
            } else {
                let py = ann_obj.py();
                let json_str: String = py
                    .import("json")
                    .and_then(|json_mod| json_mod.call_method1("dumps", (&meta_obj,)))
                    .and_then(|r| r.extract::<String>())
                    .unwrap_or_else(|_| "null".to_string());
                if json_str == "null" {
                    None
                } else {
                    Some(json_str)
                }
            }
        };

        Ok(ParsedAnnotation {
            category,
            bbox,
            segmentation,
            keypoints,
            metadata_json,
        })
    }

    fn build_py_annotation<'py>(
        py: Python<'py>,
        db: &Database,
        ann: &dman_core::types::Annotation,
    ) -> Result<Bound<'py, PyAny>> {
        let category_name = match ann.category_id {
            Some(cid) => category_name_by_id(db, cid)?,
            None => String::new(),
        };

        let bbox_py: Option<Py<PyAny>> = if let Some(ref b) = ann.bbox {
            let d = PyDict::new(py);
            d.set_item("x", b.x)
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            d.set_item("y", b.y)
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            d.set_item("width", b.width)
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            d.set_item("height", b.height)
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            Some(d.into_any().unbind())
        } else {
            None
        };

        let seg_py: Option<Py<PyAny>> = ann.segmentation.as_ref().map(|s| {
            s.into_pyobject(py)
                .map(|obj| obj.into_any().unbind())
                .unwrap_or_else(|_| py.None())
        });

        let kp_py: Option<Py<PyAny>> = ann.keypoints.as_ref().map(|k| {
            k.into_pyobject(py)
                .map(|obj| obj.into_any().unbind())
                .unwrap_or_else(|_| py.None())
        });

        let py_ann = PyAnnotationData {
            category: category_name,
            bbox: bbox_py,
            segmentation: seg_py,
            keypoints: kp_py,
            metadata: None,
        };

        py_ann
            .into_pyobject(py)
            .map(|bound| bound.into_any())
            .map_err(|e| DmanError::PluginError(e.to_string()))
    }

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
                            .map_err(|e| DmanError::PluginError(e.to_string()))?;
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

            struct ParsedSample {
                name: String,
                assets: Vec<ParsedAsset>,
                annotations: Vec<ParsedAnnotation>,
            }

            struct ParsedAsset {
                asset_type: AssetType,
                file_name: String,
                file_path: String,
                width: Option<u32>,
                height: Option<u32>,
                annotations: Vec<ParsedAnnotation>,
            }

            let plugin_name = self.info.name.clone();
            let samples: Vec<ParsedSample> = with_plugin_module(&plugin_path, |_py, module| {
                module.getattr("import_dataset").map_err(|_| {
                    DmanError::PluginError(format!(
                        "plugin '{}' is missing required function 'import_dataset'",
                        plugin_name
                    ))
                })?;
                let result = module
                    .call_method1("import_dataset", (path_str.as_str(),))
                    .map_err(|e| DmanError::PluginError(e.to_string()))?;

                let samples_list = result.cast::<PyList>().map_err(|e| {
                    DmanError::PluginError(format!("import_dataset must return a list: {e}"))
                })?;

                let mut parsed = Vec::new();
                for sample_obj in samples_list.iter() {
                    let name: String = sample_obj
                        .getattr("name")
                        .map_err(|e| {
                            DmanError::PluginError(format!("PySampleData missing 'name': {e}"))
                        })?
                        .extract()
                        .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                    let ann_attr = sample_obj.getattr("annotations").map_err(|e| {
                        DmanError::PluginError(format!("PySampleData missing 'annotations': {e}"))
                    })?;
                    let ann_list = ann_attr.cast::<PyList>().map_err(|e| {
                        DmanError::PluginError(format!("'annotations' must be a list: {e}"))
                    })?;

                    let mut sample_anns = Vec::new();
                    for ann_obj in ann_list.iter() {
                        sample_anns.push(extract_annotation(&ann_obj)?);
                    }

                    let assets_attr = sample_obj.getattr("assets").map_err(|e| {
                        DmanError::PluginError(format!("PySampleData missing 'assets': {e}"))
                    })?;
                    let assets_list = assets_attr.cast::<PyList>().map_err(|e| {
                        DmanError::PluginError(format!("'assets' must be a list: {e}"))
                    })?;

                    let mut parsed_assets = Vec::new();
                    for asset_obj in assets_list.iter() {
                        let asset_type_str: String = asset_obj
                            .getattr("asset_type")
                            .map_err(|e| {
                                DmanError::PluginError(format!(
                                    "PyAssetData missing 'asset_type': {e}"
                                ))
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                        let asset_type = AssetType::from_str(&asset_type_str)
                            .unwrap_or(AssetType::Other(asset_type_str));

                        let file_name: String = asset_obj
                            .getattr("file_name")
                            .map_err(|e| {
                                DmanError::PluginError(format!(
                                    "PyAssetData missing 'file_name': {e}"
                                ))
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                        let file_path: String = asset_obj
                            .getattr("file_path")
                            .map_err(|e| {
                                DmanError::PluginError(format!(
                                    "PyAssetData missing 'file_path': {e}"
                                ))
                            })?
                            .extract()
                            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

                        let width: Option<u32> = {
                            let w = asset_obj
                                .getattr("width")
                                .map_err(|e| DmanError::PluginError(e.to_string()))?;
                            if w.is_none() {
                                None
                            } else {
                                let v: i64 = w
                                    .extract()
                                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                                Some(v as u32)
                            }
                        };

                        let height: Option<u32> = {
                            let h = asset_obj
                                .getattr("height")
                                .map_err(|e| DmanError::PluginError(e.to_string()))?;
                            if h.is_none() {
                                None
                            } else {
                                let v: i64 = h
                                    .extract()
                                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                                Some(v as u32)
                            }
                        };

                        let asset_ann_attr = asset_obj.getattr("annotations").map_err(|e| {
                            DmanError::PluginError(format!(
                                "PyAssetData missing 'annotations': {e}"
                            ))
                        })?;
                        let asset_ann_list = asset_ann_attr.cast::<PyList>().map_err(|e| {
                            DmanError::PluginError(format!(
                                "asset 'annotations' must be a list: {e}"
                            ))
                        })?;

                        let mut asset_anns = Vec::new();
                        for ann_obj in asset_ann_list.iter() {
                            asset_anns.push(extract_annotation(&ann_obj)?);
                        }

                        parsed_assets.push(ParsedAsset {
                            asset_type,
                            file_name,
                            file_path,
                            width,
                            height,
                            annotations: asset_anns,
                        });
                    }

                    parsed.push(ParsedSample {
                        name,
                        assets: parsed_assets,
                        annotations: sample_anns,
                    });
                }

                Ok(parsed)
            })?;

            db.conn
                .execute("BEGIN IMMEDIATE", [])
                .map_err(DmanError::Database)?;

            let result = (|| -> Result<Dataset> {
                let dataset = dman_core::dataset::DatasetService::register(
                    db,
                    &dataset_name,
                    path,
                    DatasetFormat::new(self.info.name.clone()),
                )?;
                let dataset_id = dataset.id;

                for sample in &samples {
                    let sample_id = dman_core::dataset::DatasetService::add_sample(
                        db,
                        dataset_id,
                        &sample.name,
                        None,
                    )?;

                    for ann in &sample.annotations {
                        let category_id = get_or_create_category(db, dataset_id, &ann.category)?;
                        let meta: Option<serde_json::Value> = ann
                            .metadata_json
                            .as_deref()
                            .map(|s| {
                                serde_json::from_str(s).map_err(|e| {
                                    DmanError::PluginError(format!(
                                        "invalid metadata JSON in annotation: {e}"
                                    ))
                                })
                            })
                            .transpose()?;
                        dman_core::dataset::DatasetService::add_annotation(
                            db,
                            sample_id,
                            None,
                            Some(category_id),
                            ann.bbox.as_ref(),
                            ann.segmentation.as_ref(),
                            ann.keypoints.as_ref(),
                            meta.as_ref(),
                        )?;
                    }

                    for asset in &sample.assets {
                        let asset_id = dman_core::dataset::DatasetService::add_asset(
                            db,
                            sample_id,
                            asset.asset_type.clone(),
                            &asset.file_name,
                            std::path::Path::new(&asset.file_path),
                            asset.width,
                            asset.height,
                            None,
                            None,
                        )?;

                        for ann in &asset.annotations {
                            let category_id =
                                get_or_create_category(db, dataset_id, &ann.category)?;
                            let meta: Option<serde_json::Value> = ann
                                .metadata_json
                                .as_deref()
                                .map(|s| {
                                    serde_json::from_str(s).map_err(|e| {
                                        DmanError::PluginError(format!(
                                            "invalid metadata JSON in asset annotation: {e}"
                                        ))
                                    })
                                })
                                .transpose()?;
                            dman_core::dataset::DatasetService::add_annotation(
                                db,
                                sample_id,
                                Some(asset_id),
                                Some(category_id),
                                ann.bbox.as_ref(),
                                ann.segmentation.as_ref(),
                                ann.keypoints.as_ref(),
                                meta.as_ref(),
                            )?;
                        }
                    }
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
            let plugin_name = self.info.name.clone();

            let samples = dman_core::dataset::DatasetService::get_samples(db, dataset_id)?;

            with_plugin_module(&plugin_path, |py, module| {
                let py_samples = PyList::empty(py);

                for sample in &samples {
                    let sample_id = sample.id;

                    let all_sample_anns =
                        dman_core::dataset::DatasetService::get_annotations_for_sample(
                            db, sample_id,
                        )?;
                    let sample_anns: Vec<_> = all_sample_anns
                        .iter()
                        .filter(|a| a.asset_id.is_none())
                        .collect();

                    let py_sample_anns = PyList::empty(py);
                    for ann in sample_anns {
                        let py_ann = build_py_annotation(py, db, ann)?;
                        py_sample_anns
                            .append(py_ann)
                            .map_err(|e| DmanError::PluginError(e.to_string()))?;
                    }

                    let assets = dman_core::dataset::DatasetService::get_assets(db, sample_id)?;
                    let py_assets = PyList::empty(py);

                    for asset in &assets {
                        let asset_id = asset.id;
                        let asset_anns =
                            dman_core::dataset::DatasetService::get_annotations_for_asset(
                                db, asset_id,
                            )?;

                        let py_asset_anns = PyList::empty(py);
                        for ann in &asset_anns {
                            let py_ann = build_py_annotation(py, db, ann)?;
                            py_asset_anns
                                .append(py_ann)
                                .map_err(|e| DmanError::PluginError(e.to_string()))?;
                        }

                        let py_asset = PyAssetData {
                            asset_type: asset.asset_type.to_string(),
                            file_name: asset.file_name.clone(),
                            file_path: asset.file_path.to_string_lossy().to_string(),
                            width: asset.width.map(|w| w as i64),
                            height: asset.height.map(|h| h as i64),
                            metadata: None,
                            annotations: py_asset_anns
                                .iter()
                                .map(|obj| {
                                    obj.extract::<Py<PyAnnotationData>>()
                                        .map_err(|e| DmanError::PluginError(e.to_string()))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        };
                        py_assets
                            .append(
                                py_asset
                                    .into_pyobject(py)
                                    .map_err(|e| DmanError::PluginError(e.to_string()))?,
                            )
                            .map_err(|e| DmanError::PluginError(e.to_string()))?;
                    }

                    let py_sample = PySampleData {
                        name: sample.name.clone(),
                        metadata: None,
                        assets: py_assets
                            .iter()
                            .map(|obj| {
                                obj.extract::<Py<PyAssetData>>()
                                    .map_err(|e| DmanError::PluginError(e.to_string()))
                            })
                            .collect::<Result<Vec<_>>>()?,
                        annotations: py_sample_anns
                            .iter()
                            .map(|obj| {
                                obj.extract::<Py<PyAnnotationData>>()
                                    .map_err(|e| DmanError::PluginError(e.to_string()))
                            })
                            .collect::<Result<Vec<_>>>()?,
                    };

                    py_samples
                        .append(
                            py_sample
                                .into_pyobject(py)
                                .map_err(|e| DmanError::PluginError(e.to_string()))?,
                        )
                        .map_err(|e| DmanError::PluginError(e.to_string()))?;
                }

                module
                    .getattr("export_dataset")
                    .map_err(|_| {
                        DmanError::PluginError(format!(
                            "plugin '{}' is missing required function 'export_dataset'",
                            plugin_name
                        ))
                    })
                    .map(|_| ())?;
                module
                    .call_method1("export_dataset", (py_samples, output_path_str.as_str()))
                    .map_err(|e| DmanError::PluginError(e.to_string()))?;

                Ok(())
            })
        }
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
        use dman_core::formats::FormatImporter;
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "nodect", "type": "format", "version": "1.0.0"}}

def import_dataset(path):
    return []

def export_dataset(samples, path):
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
        use dman_core::formats::FormatImporter;
        use std::io::Write;
        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "withdetect", "type": "format", "version": "1.0.0"}}

def detect(path):
    return True

def import_dataset(path):
    return []

def export_dataset(samples, path):
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
    fn test_import_inserts_samples_and_annotations() {
        use tempfile::TempDir;

        let tmp_dir = TempDir::new().expect("tmpdir");
        let plugin_file = tmp_dir.path().join("fmt_plugin.py");
        std::fs::write(
            &plugin_file,
            r#"
dman_plugin = {"name": "test_fmt", "type": "format", "version": "1.0.0"}

class PyAnnotationData:
    def __init__(self, category, bbox=None, segmentation=None, keypoints=None, metadata=None):
        self.category = category
        self.bbox = bbox
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.metadata = metadata

class PyAssetData:
    def __init__(self, asset_type, file_name, file_path, width=None, height=None, metadata=None, annotations=None):
        self.asset_type = asset_type
        self.file_name = file_name
        self.file_path = file_path
        self.width = width
        self.height = height
        self.metadata = metadata
        self.annotations = annotations or []

class PySampleData:
    def __init__(self, name, metadata=None, assets=None, annotations=None):
        self.name = name
        self.metadata = metadata
        self.assets = assets or []
        self.annotations = annotations or []

def import_dataset(path):
    ann1 = PyAnnotationData("cat", bbox={"x": 10.0, "y": 20.0, "width": 30.0, "height": 40.0})
    ann2 = PyAnnotationData("dog", bbox={"x": 5.0, "y": 5.0, "width": 50.0, "height": 60.0})
    asset1 = PyAssetData("image", "img001.jpg", path + "/img001.jpg", 640, 480, annotations=[ann1])
    asset2 = PyAssetData("image", "img002.jpg", path + "/img002.jpg", 1280, 720, annotations=[ann2])
    sample1 = PySampleData("sample-001", assets=[asset1])
    sample2 = PySampleData("sample-002", assets=[asset2])
    return [sample1, sample2]

def export_dataset(samples, path):
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

        let sample_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count samples");
        assert_eq!(sample_count, 2, "expected 2 samples");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
                 (SELECT id FROM samples WHERE dataset_id = ?1)",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");
        assert_eq!(ann_count, 2, "expected 2 annotations");
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_import_fails_when_no_import_dataset_fn() {
        use dman_core::formats::FormatImporter;
        use std::io::Write;

        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "no_import_fn", "type": "format", "version": "1.0.0"}}

def export_dataset(samples, path):
    pass
"#
        )
        .expect("write plugin");

        let info = PluginInfo::new("no_import_fn", "format", "1.0.0", tmp.path().to_path_buf());
        let importer = PythonFormatImporter::new(info);

        let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
        let data_dir = tmp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        let result = importer.import(&db, &storage, &data_dir, "test_ds");
        assert!(result.is_err(), "expected error for missing import_dataset");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("import_dataset"),
            "error should mention 'import_dataset', got: {err_msg}"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_export_fails_when_no_export_dataset_fn() {
        use dman_core::formats::FormatExporter;
        use std::io::Write;

        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "no_export_fn", "type": "format", "version": "1.0.0"}}

def import_dataset(path):
    return []
"#
        )
        .expect("write plugin");

        let info = PluginInfo::new("no_export_fn", "format", "1.0.0", tmp.path().to_path_buf());
        let exporter = PythonFormatExporter::new(info);

        let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        let dataset = dman_core::dataset::DatasetService::register(
            &db,
            "test_export_ds",
            std::path::Path::new("/tmp"),
            dman_core::types::DatasetFormat::new("no_export_fn"),
        )
        .expect("register dataset");

        let output_dir = tmp_dir.path().join("output");
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        let result = exporter.export(&db, &storage, &dataset, &output_dir);
        assert!(result.is_err(), "expected error for missing export_dataset");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("export_dataset"),
            "error should mention 'export_dataset', got: {err_msg}"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_import_fails_bad_bbox_format() {
        use dman_core::formats::FormatImporter;

        let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
        let plugin_file = tmp_dir.path().join("bad_bbox_plugin.py");
        std::fs::write(
            &plugin_file,
            r#"
dman_plugin = {"name": "bad_bbox", "type": "format", "version": "1.0.0"}

class PyAnnotationData:
    def __init__(self, category, bbox=None, segmentation=None, keypoints=None, metadata=None):
        self.category = category
        self.bbox = bbox
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.metadata = metadata

class PyAssetData:
    def __init__(self, asset_type, file_name, file_path, width=None, height=None, metadata=None, annotations=None):
        self.asset_type = asset_type
        self.file_name = file_name
        self.file_path = file_path
        self.width = width
        self.height = height
        self.metadata = metadata
        self.annotations = annotations or []

class PySampleData:
    def __init__(self, name, metadata=None, assets=None, annotations=None):
        self.name = name
        self.metadata = metadata
        self.assets = assets or []
        self.annotations = annotations or []

def import_dataset(path):
    ann = PyAnnotationData("cat", bbox=[10.0, 20.0, 30.0, 40.0])
    asset = PyAssetData("image", "img.jpg", path + "/img.jpg", annotations=[ann])
    sample = PySampleData("sample-001", assets=[asset])
    return [sample]

def export_dataset(samples, path):
    pass
"#,
        )
        .expect("write plugin");

        let info = PluginInfo::new("bad_bbox", "format", "1.0.0", plugin_file);
        let importer = PythonFormatImporter::new(info);

        let data_dir = tmp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        let result = importer.import(&db, &storage, &data_dir, "bad_bbox_ds");
        assert!(result.is_err(), "expected error for bad bbox format");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("bbox"),
            "error should mention 'bbox', got: {err_msg}"
        );
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_import_with_sample_level_annotations() {
        use dman_core::formats::FormatImporter;

        let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
        let plugin_file = tmp_dir.path().join("sample_ann_plugin.py");
        std::fs::write(
            &plugin_file,
            r#"
dman_plugin = {"name": "sample_ann_fmt", "type": "format", "version": "1.0.0"}

class PyAnnotationData:
    def __init__(self, category, bbox=None, segmentation=None, keypoints=None, metadata=None):
        self.category = category
        self.bbox = bbox
        self.segmentation = segmentation
        self.keypoints = keypoints
        self.metadata = metadata

class PySampleData:
    def __init__(self, name, metadata=None, assets=None, annotations=None):
        self.name = name
        self.metadata = metadata
        self.assets = assets or []
        self.annotations = annotations or []

def import_dataset(path):
    ann = PyAnnotationData("cat", bbox={"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0})
    sample = PySampleData("sample-001", annotations=[ann], assets=[])
    return [sample]

def export_dataset(samples, path):
    pass
"#,
        )
        .expect("write plugin");

        let info = PluginInfo::new("sample_ann_fmt", "format", "1.0.0", plugin_file);
        let importer = PythonFormatImporter::new(info);

        let data_dir = tmp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        let dataset = importer
            .import(&db, &storage, &data_dir, "sample_ann_ds")
            .expect("import should succeed");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id IN \
                 (SELECT id FROM samples WHERE dataset_id = ?1)",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");
        assert_eq!(ann_count, 1, "expected 1 sample-level annotation");
    }

    #[cfg(feature = "python")]
    #[test]
    fn test_import_empty_samples_list() {
        use dman_core::formats::FormatImporter;
        use std::io::Write;

        let tmp = tempfile::NamedTempFile::with_suffix(".py").expect("tempfile");
        writeln!(
            tmp.as_file(),
            r#"
dman_plugin = {{"name": "empty_fmt", "type": "format", "version": "1.0.0"}}

def import_dataset(path):
    return []

def export_dataset(samples, path):
    pass
"#
        )
        .expect("write plugin");

        let info = PluginInfo::new("empty_fmt", "format", "1.0.0", tmp.path().to_path_buf());
        let importer = PythonFormatImporter::new(info);

        let tmp_dir = tempfile::TempDir::new().expect("tmpdir");
        let data_dir = tmp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).expect("create data dir");

        let db = dman_core::db::Database::open_in_memory().expect("db");
        let storage = dman_core::storage::StorageManager::new(tmp_dir.path().to_path_buf());

        let dataset = importer
            .import(&db, &storage, &data_dir, "empty_ds")
            .expect("import should succeed with empty list");

        let sample_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                rusqlite::params![dataset.id],
                |row| row.get(0),
            )
            .expect("count samples");
        assert_eq!(sample_count, 0, "expected 0 samples");
    }
}
