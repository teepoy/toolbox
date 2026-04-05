#[cfg(feature = "python")]
pub use python_impl::{DmanDatasetBuilder, DmanDatasetUpdater, create_dataset, update_dataset};

#[cfg(feature = "python")]
pub mod python_impl {
    use std::str::FromStr;

    use dman_core::{
        catalog::Catalog,
        dataset::DatasetService,
        db::Database,
        types::{AssetType, BBox, DatasetFormat},
    };
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rusqlite::params;

    use crate::sdk::loader::DmanDataset;

    #[derive(Debug, Clone)]
    pub(crate) struct PendingAsset {
        pub asset_type: String,
        pub file_name: String,
        pub file_path: String,
        pub width: Option<i64>,
        pub height: Option<i64>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PendingSample {
        pub name: String,
        pub assets: Vec<PendingAsset>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PendingAnnotation {
        pub sample_name: String,
        pub asset_name: Option<String>,
        pub category: String,
        pub bbox: Option<Vec<f64>>,
        pub segmentation: Option<Vec<Vec<f64>>>,
        pub keypoints: Option<Vec<f64>>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PendingCategory {
        pub name: String,
        pub supercategory: Option<String>,
    }

    #[derive(Debug)]
    pub(crate) enum UpdateOp {
        AddSample {
            name: String,
        },
        AddAsset {
            sample_name: String,
            asset_type: String,
            file_path: String,
            width: Option<i64>,
            height: Option<i64>,
        },
        AddAnnotation {
            sample_db_id: i64,
            asset_db_id: Option<i64>,
            category: String,
            bbox: Option<Vec<f64>>,
            metadata: Option<String>,
        },
        RemoveSample {
            sample_db_id: i64,
        },
    }

    #[pyclass(name = "DmanDatasetBuilder", unsendable)]
    pub struct DmanDatasetBuilder {
        pub(crate) db: Database,
        pub(crate) name: String,
        pub(crate) pending_samples: Vec<PendingSample>,
        pub(crate) pending_annotations: Vec<PendingAnnotation>,
        pub(crate) pending_categories: Vec<PendingCategory>,
    }

    impl DmanDatasetBuilder {
        pub fn new(name: &str, schema_path: Option<String>) -> PyResult<Self> {
            let db = open_catalog_db()?;
            let _ = schema_path;
            Ok(Self {
                db,
                name: name.to_string(),
                pending_samples: Vec::new(),
                pending_annotations: Vec::new(),
                pending_categories: Vec::new(),
            })
        }

        pub fn new_with_db(db: Database, name: &str, schema_path: Option<String>) -> Self {
            let _ = schema_path;
            Self {
                db,
                name: name.to_string(),
                pending_samples: Vec::new(),
                pending_annotations: Vec::new(),
                pending_categories: Vec::new(),
            }
        }

        pub fn add_sample_internal(
            &mut self,
            name: &str,
            _metadata: Option<String>,
        ) -> PyResult<usize> {
            let idx = self.pending_samples.len();
            self.pending_samples.push(PendingSample {
                name: name.to_string(),
                assets: Vec::new(),
            });
            Ok(idx)
        }

        pub fn add_asset_internal(
            &mut self,
            sample_name: &str,
            asset_type: &str,
            file_path: &str,
            width: Option<i64>,
            height: Option<i64>,
            _metadata: Option<String>,
        ) -> PyResult<()> {
            let sample = self
                .pending_samples
                .iter_mut()
                .find(|s| s.name == sample_name)
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "no pending sample named '{sample_name}'"
                    ))
                })?;

            let file_name = std::path::Path::new(file_path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| file_path.to_string());

            sample.assets.push(PendingAsset {
                asset_type: asset_type.to_string(),
                file_name,
                file_path: file_path.to_string(),
                width,
                height,
            });
            Ok(())
        }

        pub fn add_image_internal(
            &mut self,
            path: &str,
            metadata: Option<String>,
        ) -> PyResult<usize> {
            let file_name = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());
            let sample_name = file_name.clone();
            let idx = self.add_sample_internal(&sample_name, None)?;
            self.add_asset_internal(&sample_name, "image", path, None, None, metadata)?;
            Ok(idx)
        }

        pub fn add_annotation_internal(
            &mut self,
            sample_name: &str,
            category: &str,
            bbox: Option<Vec<f64>>,
            segmentation: Option<Vec<Vec<f64>>>,
            keypoints: Option<Vec<f64>>,
            asset_name: Option<String>,
        ) -> PyResult<()> {
            if !self.pending_samples.iter().any(|s| s.name == sample_name) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "no pending sample named '{sample_name}'"
                )));
            }
            if let Some(ref b) = bbox
                && b.len() != 4
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bbox must have 4 elements [x, y, width, height], got {}",
                    b.len()
                )));
            }
            self.pending_annotations.push(PendingAnnotation {
                sample_name: sample_name.to_string(),
                asset_name,
                category: category.to_string(),
                bbox,
                segmentation,
                keypoints,
            });
            Ok(())
        }

        pub fn build_internal(&mut self) -> PyResult<DmanDataset> {
            let dataset_path = std::env::temp_dir().join(format!("dman-builder-{}", &self.name));
            std::fs::create_dir_all(&dataset_path).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("create dataset dir: {e}"))
            })?;

            self.db.conn.execute("BEGIN IMMEDIATE", []).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("BEGIN failed: {e}"))
            })?;

            let result = self.build_inner(&dataset_path);

            match result {
                Ok(ds) => {
                    self.db.conn.execute("COMMIT", []).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("COMMIT failed: {e}"))
                    })?;
                    Ok(ds)
                }
                Err(e) => {
                    let _ = self.db.conn.execute("ROLLBACK", []);
                    Err(e)
                }
            }
        }

        fn build_inner(&self, dataset_path: &std::path::Path) -> PyResult<DmanDataset> {
            let ds = DatasetService::register(
                &self.db,
                &self.name,
                dataset_path,
                DatasetFormat::builder(),
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("register dataset: {e}"))
            })?;
            let dataset_id = ds.id;

            let mut category_map: std::collections::HashMap<String, i64> =
                std::collections::HashMap::new();

            for pc in &self.pending_categories {
                self.db
                    .conn
                    .execute(
                        "INSERT OR IGNORE INTO categories (dataset_id, name, supercategory) \
                         VALUES (?1, ?2, ?3)",
                        params![dataset_id, pc.name, pc.supercategory],
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("insert category: {e}"))
                    })?;
                let cat_id: i64 = self
                    .db
                    .conn
                    .query_row(
                        "SELECT id FROM categories WHERE dataset_id = ?1 AND name = ?2",
                        params![dataset_id, pc.name],
                        |r| r.get(0),
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("get category id: {e}"))
                    })?;
                category_map.insert(pc.name.clone(), cat_id);
            }

            for ann in &self.pending_annotations {
                if !category_map.contains_key(&ann.category) {
                    self.db
                        .conn
                        .execute(
                            "INSERT OR IGNORE INTO categories (dataset_id, name) VALUES (?1, ?2)",
                            params![dataset_id, ann.category],
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "auto-insert category: {e}"
                            ))
                        })?;
                    let cat_id: i64 = self
                        .db
                        .conn
                        .query_row(
                            "SELECT id FROM categories WHERE dataset_id = ?1 AND name = ?2",
                            params![dataset_id, ann.category],
                            |r| r.get(0),
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "get auto category id: {e}"
                            ))
                        })?;
                    category_map.insert(ann.category.clone(), cat_id);
                }
            }

            let mut sample_id_map: std::collections::HashMap<String, i64> =
                std::collections::HashMap::new();
            let mut asset_id_map: std::collections::HashMap<(String, String), i64> =
                std::collections::HashMap::new();

            for ps in &self.pending_samples {
                let sample_id = DatasetService::add_sample(&self.db, dataset_id, &ps.name, None)
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("insert sample: {e}"))
                    })?;
                sample_id_map.insert(ps.name.clone(), sample_id);

                for pa in &ps.assets {
                    let asset_type =
                        AssetType::from_str(&pa.asset_type).unwrap_or(AssetType::Image);
                    let asset_id = DatasetService::add_asset(
                        &self.db,
                        sample_id,
                        asset_type,
                        &pa.file_name,
                        std::path::Path::new(&pa.file_path),
                        pa.width.map(|w| w as u32),
                        pa.height.map(|h| h as u32),
                        None,
                        None,
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("insert asset: {e}"))
                    })?;
                    asset_id_map.insert((ps.name.clone(), pa.file_name.clone()), asset_id);
                }
            }

            for ann in &self.pending_annotations {
                let sample_id = *sample_id_map.get(&ann.sample_name).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "annotation references unknown sample '{}'",
                        ann.sample_name
                    ))
                })?;

                let asset_id: Option<i64> = if let Some(ref asset_name) = ann.asset_name {
                    asset_id_map
                        .get(&(ann.sample_name.clone(), asset_name.clone()))
                        .copied()
                } else {
                    let sample = self
                        .pending_samples
                        .iter()
                        .find(|s| s.name == ann.sample_name);
                    sample.and_then(|s| {
                        s.assets.first().and_then(|a| {
                            asset_id_map
                                .get(&(ann.sample_name.clone(), a.file_name.clone()))
                                .copied()
                        })
                    })
                };

                let category_id = category_map.get(&ann.category).copied();

                let bbox: Option<BBox> = ann.bbox.as_ref().map(|b| BBox {
                    x: b[0],
                    y: b[1],
                    width: b[2],
                    height: b[3],
                });

                DatasetService::add_annotation(
                    &self.db,
                    sample_id,
                    asset_id,
                    category_id,
                    bbox.as_ref(),
                    ann.segmentation.as_ref(),
                    ann.keypoints.as_ref(),
                    None,
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("insert annotation: {e}"))
                })?;
            }

            DmanDataset::from_name_with_db(&self.db, &self.name)
        }
    }

    #[pymethods]
    impl DmanDatasetBuilder {
        #[pyo3(signature = (name, metadata=None))]
        pub fn add_sample(
            &mut self,
            name: String,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<usize> {
            let meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.add_sample_internal(&name, meta_json)
        }

        #[pyo3(signature = (sample_name, asset_type, file_path, width=None, height=None, metadata=None))]
        pub fn add_asset(
            &mut self,
            sample_name: String,
            asset_type: String,
            file_path: String,
            width: Option<i64>,
            height: Option<i64>,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            let meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.add_asset_internal(
                &sample_name,
                &asset_type,
                &file_path,
                width,
                height,
                meta_json,
            )
        }

        #[pyo3(signature = (path, metadata=None))]
        pub fn add_image(
            &mut self,
            path: String,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<i64> {
            let meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            let idx = self.add_image_internal(&path, meta_json)?;
            Ok(idx as i64)
        }

        #[pyo3(signature = (sample_name, category, bbox=None, segmentation=None, keypoints=None, metadata=None, asset_name=None))]
        #[allow(clippy::too_many_arguments)]
        pub fn add_annotation(
            &mut self,
            sample_name: String,
            category: String,
            bbox: Option<Vec<f64>>,
            segmentation: Option<Vec<Vec<f64>>>,
            keypoints: Option<Vec<f64>>,
            metadata: Option<Bound<'_, PyDict>>,
            asset_name: Option<String>,
        ) -> PyResult<()> {
            let _meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.add_annotation_internal(
                &sample_name,
                &category,
                bbox,
                segmentation,
                keypoints,
                asset_name,
            )
        }

        #[pyo3(signature = (name, supercategory=None))]
        pub fn set_category(
            &mut self,
            name: String,
            supercategory: Option<String>,
        ) -> PyResult<()> {
            self.pending_categories.push(PendingCategory {
                name,
                supercategory,
            });
            Ok(())
        }

        pub fn build(&mut self) -> PyResult<DmanDataset> {
            self.build_internal()
        }
    }

    #[pyclass(name = "DmanDatasetUpdater", unsendable)]
    pub struct DmanDatasetUpdater {
        pub(crate) db: Database,
        pub(crate) dataset_id: i64,
        pub(crate) pending_ops: Vec<UpdateOp>,
    }

    impl DmanDatasetUpdater {
        pub fn new(name: &str) -> PyResult<Self> {
            let db = open_catalog_db()?;
            Self::new_with_db(db, name)
        }

        pub fn new_with_db(db: Database, name: &str) -> PyResult<Self> {
            let ds = DatasetService::get(&db, name).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("Dataset not found: {e}"))
            })?;
            Ok(Self {
                db,
                dataset_id: ds.id,
                pending_ops: Vec::new(),
            })
        }

        pub fn add_sample_internal(
            &mut self,
            name: &str,
            _metadata: Option<String>,
        ) -> PyResult<i64> {
            DatasetService::add_sample(&self.db, self.dataset_id, name, None).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("add_sample insert: {e}"))
            })
        }

        pub fn add_image_internal(
            &mut self,
            sample_id: i64,
            path: &str,
            _metadata: Option<String>,
        ) -> PyResult<i64> {
            let file_name = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());

            DatasetService::add_asset(
                &self.db,
                sample_id,
                AssetType::Image,
                &file_name,
                std::path::Path::new(path),
                None,
                None,
                None,
                None,
            )
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("add_image insert: {e}"))
            })
        }

        pub fn apply_internal(&mut self) -> PyResult<()> {
            self.db.conn.execute("BEGIN IMMEDIATE", []).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("BEGIN failed: {e}"))
            })?;

            let result = self.apply_inner();
            match result {
                Ok(()) => {
                    self.db.conn.execute("COMMIT", []).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("COMMIT failed: {e}"))
                    })?;
                    self.pending_ops.clear();
                    Ok(())
                }
                Err(e) => {
                    let _ = self.db.conn.execute("ROLLBACK", []);
                    Err(e)
                }
            }
        }

        fn apply_inner(&self) -> PyResult<()> {
            let mut category_map: std::collections::HashMap<String, i64> =
                std::collections::HashMap::new();

            for op in &self.pending_ops {
                if let UpdateOp::AddAnnotation { category, .. } = op
                    && !category_map.contains_key(category)
                {
                    self.db
                        .conn
                        .execute(
                            "INSERT OR IGNORE INTO categories (dataset_id, name) \
                             VALUES (?1, ?2)",
                            params![self.dataset_id, category],
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "insert category: {e}"
                            ))
                        })?;
                    let cat_id: i64 = self
                        .db
                        .conn
                        .query_row(
                            "SELECT id FROM categories WHERE dataset_id = ?1 AND name = ?2",
                            params![self.dataset_id, category],
                            |r| r.get(0),
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "get category id: {e}"
                            ))
                        })?;
                    category_map.insert(category.clone(), cat_id);
                }
            }

            for op in &self.pending_ops {
                match op {
                    UpdateOp::AddSample { name } => {
                        DatasetService::add_sample(&self.db, self.dataset_id, name, None).map_err(
                            |e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "add sample: {e}"
                                ))
                            },
                        )?;
                    }
                    UpdateOp::AddAsset {
                        sample_name,
                        asset_type,
                        file_path,
                        width,
                        height,
                    } => {
                        let sample_id: i64 = self
                            .db
                            .conn
                            .query_row(
                                "SELECT id FROM samples WHERE dataset_id = ?1 AND name = ?2",
                                params![self.dataset_id, sample_name],
                                |r| r.get(0),
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "resolve sample '{sample_name}': {e}"
                                ))
                            })?;

                        let file_name = std::path::Path::new(file_path.as_str())
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| file_path.clone());

                        let asset_type_val =
                            AssetType::from_str(asset_type).unwrap_or(AssetType::Image);

                        DatasetService::add_asset(
                            &self.db,
                            sample_id,
                            asset_type_val,
                            &file_name,
                            std::path::Path::new(file_path.as_str()),
                            width.map(|w| w as u32),
                            height.map(|h| h as u32),
                            None,
                            None,
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!("add asset: {e}"))
                        })?;
                    }
                    UpdateOp::AddAnnotation {
                        sample_db_id,
                        asset_db_id,
                        category,
                        bbox,
                        metadata,
                    } => {
                        let category_id = category_map.get(category).copied();

                        let bbox_val: Option<BBox> = bbox.as_ref().map(|b| BBox {
                            x: b[0],
                            y: b[1],
                            width: b[2],
                            height: b[3],
                        });

                        let meta_val: Option<serde_json::Value> = metadata
                            .as_deref()
                            .and_then(|s| serde_json::from_str(s).ok());

                        DatasetService::add_annotation(
                            &self.db,
                            *sample_db_id,
                            *asset_db_id,
                            category_id,
                            bbox_val.as_ref(),
                            None,
                            None,
                            meta_val.as_ref(),
                        )
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "add annotation: {e}"
                            ))
                        })?;
                    }
                    UpdateOp::RemoveSample { sample_db_id } => {
                        DatasetService::remove_sample(&self.db, *sample_db_id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!("remove sample: {e}"))
                        })?;
                    }
                }
            }
            Ok(())
        }
    }

    #[pymethods]
    impl DmanDatasetUpdater {
        #[pyo3(signature = (name, metadata=None))]
        pub fn add_sample(
            &mut self,
            name: String,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            let _meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.pending_ops.push(UpdateOp::AddSample { name });
            Ok(())
        }

        #[pyo3(signature = (sample_name, asset_type, file_path, width=None, height=None, metadata=None))]
        pub fn add_asset(
            &mut self,
            sample_name: String,
            asset_type: String,
            file_path: String,
            width: Option<i64>,
            height: Option<i64>,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            let _meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.pending_ops.push(UpdateOp::AddAsset {
                sample_name,
                asset_type,
                file_path,
                width,
                height,
            });
            Ok(())
        }

        #[pyo3(signature = (sample_id, category, bbox=None, asset_id=None, metadata=None))]
        pub fn add_annotation(
            &mut self,
            sample_id: i64,
            category: String,
            bbox: Option<Vec<f64>>,
            asset_id: Option<i64>,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            if let Some(ref b) = bbox
                && b.len() != 4
            {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bbox must have 4 elements [x, y, width, height], got {}",
                    b.len()
                )));
            }
            let meta_json = metadata
                .map(|d| -> PyResult<String> {
                    d.py()
                        .import("json")
                        .and_then(|j| j.call_method1("dumps", (d.as_any(),)))
                        .and_then(|r| r.extract::<String>())
                })
                .transpose()?;
            self.pending_ops.push(UpdateOp::AddAnnotation {
                sample_db_id: sample_id,
                asset_db_id: asset_id,
                category,
                bbox,
                metadata: meta_json,
            });
            Ok(())
        }

        pub fn remove_sample(&mut self, sample_id: i64) -> PyResult<()> {
            self.pending_ops.push(UpdateOp::RemoveSample {
                sample_db_id: sample_id,
            });
            Ok(())
        }

        pub fn apply(&mut self) -> PyResult<()> {
            self.apply_internal()
        }
    }

    fn open_catalog_db() -> PyResult<Database> {
        let home = Catalog::home_path();
        let db_path = home.join("catalog.db");
        if !db_path.exists() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "dman catalog not initialized. Run `dman init` first.",
            ));
        }
        Database::open(&db_path).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Cannot open catalog: {e}"))
        })
    }

    #[pyfunction]
    #[pyo3(signature = (name, schema_path=None))]
    pub fn create_dataset(
        name: String,
        schema_path: Option<String>,
    ) -> PyResult<DmanDatasetBuilder> {
        DmanDatasetBuilder::new(&name, schema_path)
    }

    #[pyfunction]
    pub fn update_dataset(name: String) -> PyResult<DmanDatasetUpdater> {
        DmanDatasetUpdater::new(&name)
    }
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::python_impl::{DmanDatasetBuilder, DmanDatasetUpdater, PendingCategory, UpdateOp};
    use dman_core::{dataset::DatasetService, db::Database, types::DatasetFormat};
    use rusqlite::params;
    use tempfile::TempDir;

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory db")
    }

    #[test]
    fn builder_creates_dataset_and_inserts_samples() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "test-builder", None);

        builder
            .add_image_internal(tmp.path().join("img.jpg").to_str().unwrap(), None)
            .expect("add image");
        builder
            .add_annotation_internal(
                "img.jpg",
                "cat",
                Some(vec![10.0, 20.0, 50.0, 50.0]),
                None,
                None,
                None,
            )
            .expect("add annotation");

        let ds = builder.build_internal().expect("build");
        assert_eq!(ds.name, "test-builder");

        let sample_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
            .expect("count samples");
        assert_eq!(sample_count, 1);

        let asset_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM assets", [], |r| r.get(0))
            .expect("count assets");
        assert_eq!(asset_count, 1);
    }

    #[test]
    fn builder_registers_explicit_category() {
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "cat-ds", None);

        builder.pending_categories.push(PendingCategory {
            name: "dog".to_string(),
            supercategory: Some("animal".to_string()),
        });

        let ds = builder.build_internal().expect("build");

        let cat_count: i64 = builder
            .db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
                params![ds.dataset_id],
                |r| r.get(0),
            )
            .expect("count categories");
        assert_eq!(cat_count, 1);
    }

    #[test]
    fn builder_rollback_on_duplicate_name() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        DatasetService::register(&db, "dup-ds", tmp.path(), DatasetFormat::new("x"))
            .expect("pre-register");

        let mut builder = DmanDatasetBuilder::new_with_db(db, "dup-ds", None);
        builder
            .add_image_internal(tmp.path().join("z.jpg").to_str().unwrap(), None)
            .expect("add");

        let result = builder.build_internal();
        assert!(result.is_err(), "expected error on duplicate dataset name");

        let sample_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
            .expect("count");
        assert_eq!(sample_count, 0, "rollback must leave no samples");
    }

    #[test]
    fn builder_multiple_samples_and_annotations() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "multi-ds", None);

        for i in 0..3 {
            let img_path = tmp.path().join(format!("img{}.jpg", i));
            let img_str = img_path.to_str().unwrap();
            builder.add_image_internal(img_str, None).expect("add");
            let img_name = format!("img{}.jpg", i);
            builder
                .add_annotation_internal(
                    &img_name,
                    "obj",
                    Some(vec![0.0, 0.0, 10.0, 10.0]),
                    None,
                    None,
                    None,
                )
                .expect("ann");
        }

        builder.build_internal().expect("build");

        let sample_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
            .expect("count samples");
        assert_eq!(sample_count, 3);
    }

    #[test]
    fn builder_empty_dataset_works() {
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "empty-builder-ds", None);
        let ds = builder.build_internal().expect("build empty");
        assert_eq!(ds.name, "empty-builder-ds");

        let sample_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM samples", [], |r| r.get(0))
            .expect("count");
        assert_eq!(sample_count, 0);
    }

    #[test]
    fn updater_new_with_db_errors_for_missing_dataset() {
        let db = in_memory_db();
        let result = DmanDatasetUpdater::new_with_db(db, "does-not-exist");
        assert!(result.is_err(), "expected error for missing dataset");
    }

    #[test]
    fn updater_add_sample_internal_returns_positive_id() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        DatasetService::register(&db, "updater-ds", tmp.path(), DatasetFormat::coco())
            .expect("register");

        let mut updater = DmanDatasetUpdater::new_with_db(db, "updater-ds").expect("updater");
        let sample_id = updater
            .add_sample_internal("new-sample", None)
            .expect("add sample");
        assert!(sample_id > 0);
    }

    #[test]
    fn updater_apply_removes_sample_and_adds_new() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        let ds = DatasetService::register(&db, "apply-ds", tmp.path(), DatasetFormat::coco())
            .expect("register");

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                params![ds.id, "old-sample"],
            )
            .expect("insert old sample");
        let old_sample_id = db.conn.last_insert_rowid();

        let mut updater = DmanDatasetUpdater::new_with_db(db, "apply-ds").expect("updater");
        updater.pending_ops.push(UpdateOp::RemoveSample {
            sample_db_id: old_sample_id,
        });
        updater.pending_ops.push(UpdateOp::AddSample {
            name: "new-sample".to_string(),
        });

        updater.apply_internal().expect("apply");

        let sample_count: i64 = updater
            .db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![updater.dataset_id],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(sample_count, 1);

        let new_name: String = updater
            .db
            .conn
            .query_row(
                "SELECT name FROM samples WHERE dataset_id = ?1",
                params![updater.dataset_id],
                |r| r.get(0),
            )
            .expect("name");
        assert_eq!(new_name, "new-sample");
    }

    #[test]
    fn updater_apply_adds_annotation_with_category() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        let ds = DatasetService::register(&db, "ann-ds", tmp.path(), DatasetFormat::coco())
            .expect("register");

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                params![ds.id, "sample-001"],
            )
            .expect("insert sample");
        let sample_id = db.conn.last_insert_rowid();

        let mut updater = DmanDatasetUpdater::new_with_db(db, "ann-ds").expect("updater");
        updater.pending_ops.push(UpdateOp::AddAnnotation {
            sample_db_id: sample_id,
            asset_db_id: None,
            category: "cat".to_string(),
            bbox: Some(vec![1.0, 2.0, 3.0, 4.0]),
            metadata: None,
        });

        updater.apply_internal().expect("apply");

        let ann_count: i64 = updater
            .db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id = ?1",
                params![sample_id],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(ann_count, 1);

        let bbox_json: String = updater
            .db
            .conn
            .query_row(
                "SELECT bbox FROM annotations WHERE sample_id = ?1",
                params![sample_id],
                |r| r.get(0),
            )
            .expect("bbox");
        assert!(
            bbox_json.contains("\"width\""),
            "expected 'width' key in bbox JSON, got: {bbox_json}"
        );
        assert!(
            bbox_json.contains("\"height\""),
            "expected 'height' key in bbox JSON, got: {bbox_json}"
        );
        assert!(
            !bbox_json.contains("\"w\""),
            "bbox JSON must not use 'w' key, got: {bbox_json}"
        );
        assert!(
            !bbox_json.contains("\"h\""),
            "bbox JSON must not use 'h' key, got: {bbox_json}"
        );
    }

    #[test]
    fn builder_bbox_uses_width_height_keys() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "bbox-test-ds", None);

        builder
            .add_image_internal(tmp.path().join("a.jpg").to_str().unwrap(), None)
            .expect("add image");
        builder
            .add_annotation_internal(
                "a.jpg",
                "bird",
                Some(vec![5.0, 10.0, 100.0, 80.0]),
                None,
                None,
                None,
            )
            .expect("add annotation");

        builder.build_internal().expect("build");

        let bbox_json: String = builder
            .db
            .conn
            .query_row("SELECT bbox FROM annotations LIMIT 1", [], |r| r.get(0))
            .expect("bbox");

        assert!(
            bbox_json.contains("\"width\""),
            "expected 'width' key in bbox JSON, got: {bbox_json}"
        );
        assert!(
            bbox_json.contains("\"height\""),
            "expected 'height' key in bbox JSON, got: {bbox_json}"
        );
        assert!(
            !bbox_json.contains("\"w\""),
            "bbox JSON must not use short 'w' key, got: {bbox_json}"
        );
        assert!(
            !bbox_json.contains("\"h\""),
            "bbox JSON must not use short 'h' key, got: {bbox_json}"
        );
    }

    #[test]
    fn add_sample_and_asset_separate_methods() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "sep-ds", None);

        builder
            .add_sample_internal("sample-a", None)
            .expect("add sample");
        builder
            .add_asset_internal(
                "sample-a",
                "image",
                tmp.path().join("a.jpg").to_str().unwrap(),
                Some(640),
                Some(480),
                None,
            )
            .expect("add asset");

        builder.build_internal().expect("build");

        let asset_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM assets", [], |r| r.get(0))
            .expect("count assets");
        assert_eq!(asset_count, 1);

        let (w, h): (Option<i64>, Option<i64>) = builder
            .db
            .conn
            .query_row("SELECT width, height FROM assets LIMIT 1", [], |r| {
                Ok((r.get(0)?, r.get(1)?))
            })
            .expect("asset dims");
        assert_eq!(w, Some(640));
        assert_eq!(h, Some(480));
    }
}
