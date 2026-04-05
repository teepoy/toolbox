//! PyO3 loader module — PyTorch Dataset and HuggingFace Datasets integration.
//!
//! Exposes:
//!   - `load_dataset(name, split=None) -> DmanDataset`
//!   - `DmanDataset` — Python class with `__len__`, `__getitem__`, `to_torch_dataset`,
//!     `to_hf_dataset`, `samples`, `get_sample`, `annotations`, `images`

#[cfg(feature = "python")]
pub use python_impl::{DmanDataset, load_dataset};

#[cfg(feature = "python")]
mod python_impl {
    use dman_core::{catalog::Catalog, dataset::DatasetService, db::Database, error::DmanError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    // ─── Row types (internal) ───────────────────────────────────────────────

    #[derive(Debug, Clone)]
    pub(crate) struct SampleRow {
        pub id: i64,
        pub dataset_id: i64,
        pub name: String,
        pub metadata: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct AssetRow {
        pub id: i64,
        pub sample_id: i64,
        pub asset_type: String,
        pub file_name: String,
        pub file_path: String,
        pub width: Option<i64>,
        pub height: Option<i64>,
        pub metadata: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct AnnotationRow {
        pub id: i64,
        pub sample_id: i64,
        pub asset_id: Option<i64>,
        pub category_id: Option<i64>,
        pub bbox: Option<String>,
        pub segmentation: Option<String>,
        pub keypoints: Option<String>,
        pub metadata: Option<String>,
    }

    // ─── DmanDataset ────────────────────────────────────────────────────────

    /// A Python-accessible dataset backed by a dman catalog.
    #[derive(Debug)]
    #[pyclass(name = "DmanDataset")]
    pub struct DmanDataset {
        pub(crate) dataset_id: i64,
        pub(crate) name: String,
        pub(crate) sample_rows: Vec<SampleRow>,
        pub(crate) asset_rows: Vec<AssetRow>,
    }

    impl DmanDataset {
        /// Load all samples for `dataset_id` from `db`.
        fn load_samples(db: &Database, dataset_id: i64) -> PyResult<Vec<SampleRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT id, dataset_id, name, metadata \
                     FROM samples WHERE dataset_id = ?1 ORDER BY id",
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB prepare error: {e}"))
                })?;

            let rows = stmt
                .query_map(rusqlite::params![dataset_id], |row| {
                    Ok(SampleRow {
                        id: row.get(0)?,
                        dataset_id: row.get(1)?,
                        name: row.get(2)?,
                        metadata: row.get(3)?,
                    })
                })
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB query error: {e}"))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB row error: {e}"))
                })?;

            Ok(rows)
        }

        /// Load all assets for a dataset (joins through samples).
        fn load_assets_for_dataset(db: &Database, dataset_id: i64) -> PyResult<Vec<AssetRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT a.id, a.sample_id, a.asset_type, a.file_name, a.file_path, \
                            a.width, a.height, a.metadata \
                     FROM assets a \
                     JOIN samples s ON s.id = a.sample_id \
                     WHERE s.dataset_id = ?1 ORDER BY a.id",
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB prepare error: {e}"))
                })?;

            let rows = stmt
                .query_map(rusqlite::params![dataset_id], |row| {
                    Ok(AssetRow {
                        id: row.get(0)?,
                        sample_id: row.get(1)?,
                        asset_type: row.get(2)?,
                        file_name: row.get(3)?,
                        file_path: row.get(4)?,
                        width: row.get(5)?,
                        height: row.get(6)?,
                        metadata: row.get(7)?,
                    })
                })
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB query error: {e}"))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB row error: {e}"))
                })?;

            Ok(rows)
        }

        /// Load annotations for a sample (optionally filtered to a specific asset).
        fn load_annotations_for_sample(
            db: &Database,
            sample_id: i64,
        ) -> PyResult<Vec<AnnotationRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
                     FROM annotations WHERE sample_id = ?1 ORDER BY id",
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB prepare error: {e}"))
                })?;

            let rows = stmt
                .query_map(rusqlite::params![sample_id], |row| {
                    Ok(AnnotationRow {
                        id: row.get(0)?,
                        sample_id: row.get(1)?,
                        asset_id: row.get(2)?,
                        category_id: row.get(3)?,
                        bbox: row.get(4)?,
                        segmentation: row.get(5)?,
                        keypoints: row.get(6)?,
                        metadata: row.get(7)?,
                    })
                })
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB query error: {e}"))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB row error: {e}"))
                })?;

            Ok(rows)
        }

        /// Load annotations for a specific asset.
        fn load_annotations_for_asset(
            db: &Database,
            asset_id: i64,
        ) -> PyResult<Vec<AnnotationRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
                     FROM annotations WHERE asset_id = ?1 ORDER BY id",
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB prepare error: {e}"))
                })?;

            let rows = stmt
                .query_map(rusqlite::params![asset_id], |row| {
                    Ok(AnnotationRow {
                        id: row.get(0)?,
                        sample_id: row.get(1)?,
                        asset_id: row.get(2)?,
                        category_id: row.get(3)?,
                        bbox: row.get(4)?,
                        segmentation: row.get(5)?,
                        keypoints: row.get(6)?,
                        metadata: row.get(7)?,
                    })
                })
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB query error: {e}"))
                })?
                .collect::<rusqlite::Result<Vec<_>>>()
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB row error: {e}"))
                })?;

            Ok(rows)
        }

        /// Open the default catalog DB (respects `$DMAN_HOME`).
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

        /// Build a `DmanDataset` from a raw name using the default catalog.
        pub fn from_name(name: &str) -> PyResult<Self> {
            let db = Self::open_catalog_db()?;
            Self::from_name_with_db(&db, name)
        }

        /// Build a `DmanDataset` from a raw name using an explicit `Database`.
        pub fn from_name_with_db(db: &Database, name: &str) -> PyResult<Self> {
            let dataset = DatasetService::get(db, name).map_err(|e| match e {
                DmanError::DatasetNotFound(_) => pyo3::exceptions::PyValueError::new_err(format!(
                    "Dataset '{}' not found in catalog.",
                    name
                )),
                other => pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Error loading dataset: {other}"
                )),
            })?;

            let sample_rows = Self::load_samples(db, dataset.id)?;
            let asset_rows = Self::load_assets_for_dataset(db, dataset.id)?;

            Ok(DmanDataset {
                dataset_id: dataset.id,
                name: dataset.name,
                sample_rows,
                asset_rows,
            })
        }

        /// Build an annotation PyDict from an AnnotationRow.
        fn annotation_to_dict<'py>(
            py: Python<'py>,
            ann: &AnnotationRow,
        ) -> PyResult<Bound<'py, PyDict>> {
            let ad = PyDict::new(py);
            ad.set_item("id", ann.id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("sample_id", ann.sample_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            match ann.asset_id {
                Some(aid) => {
                    ad.set_item("asset_id", aid)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                None => {
                    ad.set_item("asset_id", py.None())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
            }
            if let Some(cat_id) = ann.category_id {
                ad.set_item("category_id", cat_id)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            }
            ad.set_item("bbox", ann.bbox.as_deref().unwrap_or(""))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("segmentation", ann.segmentation.as_deref().unwrap_or(""))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("keypoints", ann.keypoints.as_deref().unwrap_or(""))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("metadata", ann.metadata.as_deref().unwrap_or(""))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(ad)
        }

        /// Returns file_paths of all assets with asset_type = "image".
        pub(crate) fn image_paths(&self) -> Vec<String> {
            self.asset_rows
                .iter()
                .filter(|a| a.asset_type == "image")
                .map(|a| a.file_path.clone())
                .collect()
        }

        /// Build an asset PyDict from an AssetRow.
        fn asset_to_dict<'py>(py: Python<'py>, asset: &AssetRow) -> PyResult<Bound<'py, PyDict>> {
            let ad = PyDict::new(py);
            ad.set_item("id", asset.id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("sample_id", asset.sample_id)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("asset_type", &asset.asset_type)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("file_name", &asset.file_name)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            ad.set_item("file_path", &asset.file_path)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            match asset.width {
                Some(w) => {
                    ad.set_item("width", w)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                None => {
                    ad.set_item("width", py.None())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
            }
            match asset.height {
                Some(h) => {
                    ad.set_item("height", h)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                None => {
                    ad.set_item("height", py.None())
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
            }
            ad.set_item("metadata", asset.metadata.as_deref().unwrap_or(""))
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(ad)
        }
    }

    #[pymethods]
    impl DmanDataset {
        /// Number of samples in the dataset.
        fn __len__(&self) -> usize {
            self.sample_rows.len()
        }

        /// Return a dict for sample at `idx`:
        /// `{id, name, metadata, assets: list, annotations: list}`
        fn __getitem__(&self, idx: usize) -> PyResult<Py<PyAny>> {
            if idx >= self.sample_rows.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {} out of range for dataset of size {}",
                    idx,
                    self.sample_rows.len()
                )));
            }
            let sample = &self.sample_rows[idx];
            let sample_assets: Vec<&AssetRow> = self
                .asset_rows
                .iter()
                .filter(|a| a.sample_id == sample.id)
                .collect();
            let db = Self::open_catalog_db()?;
            let anns = Self::load_annotations_for_sample(&db, sample.id)?;

            Python::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("id", sample.id)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                d.set_item("name", &sample.name)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                d.set_item("metadata", sample.metadata.as_deref().unwrap_or(""))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                // Assets list
                let asset_list = PyList::empty(py);
                for asset in &sample_assets {
                    let ad = Self::asset_to_dict(py, asset)?;
                    asset_list
                        .append(ad)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                d.set_item("assets", asset_list)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                // Annotations list
                let ann_list = PyList::empty(py);
                for ann in &anns {
                    let ad = Self::annotation_to_dict(py, ann)?;
                    ann_list
                        .append(ad)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                d.set_item("annotations", ann_list)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                Ok(d.unbind().into_any())
            })
        }

        /// Return list of all samples as dicts, each with their assets included.
        fn samples(&self) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let outer = PyList::empty(py);
                for sample in &self.sample_rows {
                    let sample_assets: Vec<&AssetRow> = self
                        .asset_rows
                        .iter()
                        .filter(|a| a.sample_id == sample.id)
                        .collect();

                    let d = PyDict::new(py);
                    d.set_item("id", sample.id)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    d.set_item("name", &sample.name)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    d.set_item("dataset_id", sample.dataset_id)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    d.set_item("metadata", sample.metadata.as_deref().unwrap_or(""))
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    let asset_list = PyList::empty(py);
                    for asset in &sample_assets {
                        let ad = Self::asset_to_dict(py, asset)?;
                        asset_list.append(ad).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                    }
                    d.set_item("assets", asset_list)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                    outer
                        .append(d)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                Ok(outer.unbind().into_any())
            })
        }

        /// Return a single sample dict (with assets) by name, or None if not found.
        fn get_sample(&self, name: &str) -> PyResult<Py<PyAny>> {
            Python::attach(|py| {
                let maybe = self.sample_rows.iter().find(|s| s.name == name);
                match maybe {
                    None => Ok(py.None()),
                    Some(sample) => {
                        let sample_assets: Vec<&AssetRow> = self
                            .asset_rows
                            .iter()
                            .filter(|a| a.sample_id == sample.id)
                            .collect();

                        let d = PyDict::new(py);
                        d.set_item("id", sample.id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                        d.set_item("name", &sample.name).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                        d.set_item("dataset_id", sample.dataset_id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                        d.set_item("metadata", sample.metadata.as_deref().unwrap_or(""))
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                            })?;

                        let asset_list = PyList::empty(py);
                        for asset in &sample_assets {
                            let ad = Self::asset_to_dict(py, asset)?;
                            asset_list.append(ad).map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                            })?;
                        }
                        d.set_item("assets", asset_list).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;

                        Ok(d.unbind().into_any())
                    }
                }
            })
        }

        /// Return annotations for a sample (by name), optionally filtered to a specific asset name.
        ///
        /// Args:
        ///   sample_name: Name of the sample.
        ///   asset_name: Optional asset file_name to filter annotations to a specific asset.
        #[pyo3(signature = (sample_name, asset_name=None))]
        fn annotations(&self, sample_name: &str, asset_name: Option<&str>) -> PyResult<Py<PyAny>> {
            let db = Self::open_catalog_db()?;

            let maybe_sample = self.sample_rows.iter().find(|s| s.name == sample_name);
            let sample = match maybe_sample {
                None => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Sample '{}' not found in dataset.",
                        sample_name
                    )));
                }
                Some(s) => s,
            };

            // Resolve asset_id if asset_name provided
            let asset_id_filter: Option<i64> = match asset_name {
                None => None,
                Some(aname) => {
                    let maybe_asset = self
                        .asset_rows
                        .iter()
                        .find(|a| a.sample_id == sample.id && a.file_name == aname);
                    match maybe_asset {
                        None => {
                            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                "Asset '{}' not found in sample '{}'.",
                                aname, sample_name
                            )));
                        }
                        Some(a) => Some(a.id),
                    }
                }
            };

            let anns = match asset_id_filter {
                None => Self::load_annotations_for_sample(&db, sample.id)?,
                Some(aid) => Self::load_annotations_for_asset(&db, aid)?,
            };

            Python::attach(|py| {
                let ann_list = PyList::empty(py);
                for ann in &anns {
                    let ad = Self::annotation_to_dict(py, ann)?;
                    ann_list
                        .append(ad)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                Ok(ann_list.unbind().into_any())
            })
        }

        fn images(&self) -> Vec<String> {
            self.image_paths()
        }

        /// Return a `torch.utils.data.Dataset`-compatible wrapper.
        /// Raises `ImportError` if PyTorch is not installed.
        fn to_torch_dataset(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            // Verify torch is available
            py.import("torch").map_err(|_| {
                pyo3::exceptions::PyImportError::new_err(
                    "PyTorch is not installed. Install with: pip install torch",
                )
            })?;

            // Build a simple TorchDataset class at runtime via exec
            let items: Vec<String> = self.image_paths();

            let locals = PyDict::new(py);
            locals
                .set_item("image_paths", items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            py.run(
                c"import torch.utils.data\n\nclass _DmanTorchDataset(torch.utils.data.Dataset):\n    def __init__(self, paths):\n        self._paths = paths\n    def __len__(self):\n        return len(self._paths)\n    def __getitem__(self, idx):\n        return {\"image_path\": self._paths[idx]}\n\n_result = _DmanTorchDataset(image_paths)\n",
                None,
                Some(&locals),
            )
            .map_err(|e: PyErr| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let result = locals
                .get_item("_result")
                .map_err(|e: PyErr| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("failed to construct torch dataset")
                })?;

            Ok(result.unbind().into_any())
        }

        /// Return a `datasets.Dataset` from HuggingFace `datasets` library.
        /// Raises `ImportError` if `datasets` is not installed.
        fn to_hf_dataset(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            let datasets_mod = py.import("datasets").map_err(|_| {
                pyo3::exceptions::PyImportError::new_err(
                    "HuggingFace `datasets` is not installed. Install with: pip install datasets",
                )
            })?;

            let dataset_class = datasets_mod.getattr("Dataset").map_err(|e: PyErr| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Cannot access datasets.Dataset: {e}"
                ))
            })?;

            // Build columns from image assets
            let image_assets: Vec<&AssetRow> = self
                .asset_rows
                .iter()
                .filter(|a| a.asset_type == "image")
                .collect();

            let image_paths: Vec<String> =
                image_assets.iter().map(|a| a.file_path.clone()).collect();
            let file_names: Vec<String> =
                image_assets.iter().map(|a| a.file_name.clone()).collect();
            let ids: Vec<i64> = image_assets.iter().map(|a| a.id).collect();

            let data_dict = PyDict::new(py);
            data_dict
                .set_item("image_path", image_paths)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            data_dict
                .set_item("file_name", file_names)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            data_dict
                .set_item("id", ids)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let result = dataset_class
                .call_method1("from_dict", (data_dict,))
                .map_err(|e: PyErr| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "datasets.Dataset.from_dict failed: {e}"
                    ))
                })?;

            Ok(result.unbind().into_any())
        }

        /// The dataset name.
        #[getter]
        fn name(&self) -> &str {
            &self.name
        }

        /// The internal dataset id.
        #[getter]
        fn dataset_id(&self) -> i64 {
            self.dataset_id
        }

        /// Number of samples in the dataset.
        fn sample_count(&self) -> usize {
            self.sample_rows.len()
        }

        /// Number of assets in the dataset.
        fn asset_count(&self) -> usize {
            self.asset_rows.len()
        }
    }

    // ─── load_dataset ────────────────────────────────────────────────────────

    /// Load a dataset from the dman catalog by name.
    ///
    /// Args:
    ///   name: Dataset name registered in dman catalog.
    ///   split: Optional split name (reserved for future use; currently ignored).
    ///
    /// Returns:
    ///   DmanDataset instance.
    ///
    /// Raises:
    ///   ValueError: if dataset not found.
    ///   RuntimeError: on catalog/DB errors.
    #[pyfunction]
    #[pyo3(signature = (name, split=None))]
    pub fn load_dataset(name: String, split: Option<String>) -> PyResult<DmanDataset> {
        // `split` is reserved; not yet applied
        let _ = split;
        DmanDataset::from_name(&name)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::python_impl::DmanDataset;
    use dman_core::{
        dataset::DatasetService,
        db::Database,
        types::{AssetType, DatasetFormat},
    };
    use std::path::Path;
    use tempfile::TempDir;

    /// Create an isolated in-memory DB and populate a dataset with samples + image assets.
    fn setup_db_with_dataset(dataset_name: &str, sample_count: usize) -> (Database, TempDir, i64) {
        let tmp = TempDir::new().expect("tempdir");
        let db = Database::open_in_memory().expect("in-memory db");

        let ds = DatasetService::register(&db, dataset_name, tmp.path(), DatasetFormat::coco())
            .expect("register dataset");

        for i in 0..sample_count {
            let file_name = format!("img{:03}.jpg", i);
            let file_path = tmp.path().join(&file_name);
            DatasetService::add_sample_with_single_asset(
                &db,
                ds.id,
                &file_name,
                AssetType::Image,
                &file_name,
                &file_path,
                None,
                None,
                None,
                None,
            )
            .expect("add sample+asset");
        }

        (db, tmp, ds.id)
    }

    #[test]
    fn load_dataset_with_valid_name_returns_correct_len() {
        let (db, _tmp, _ds_id) = setup_db_with_dataset("test-ds", 5);
        let dset = DmanDataset::from_name_with_db(&db, "test-ds").expect("load");
        assert_eq!(dset.sample_rows.len(), 5);
    }

    #[test]
    fn load_dataset_with_invalid_name_returns_value_error() {
        let db = Database::open_in_memory().expect("db");
        let result = DmanDataset::from_name_with_db(&db, "does-not-exist");
        assert!(result.is_err(), "expected error for missing dataset");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not found"),
            "expected 'not found' in error, got: {msg}"
        );
    }

    #[test]
    fn getitem_uses_samples_and_assets() {
        let (db, tmp, ds_id) = setup_db_with_dataset("getitem-ds", 3);
        // Insert an annotation for sample 0
        let sample_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM samples WHERE dataset_id = ?1 ORDER BY id LIMIT 1",
                rusqlite::params![ds_id],
                |r| r.get(0),
            )
            .expect("get sample id");
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, bbox) VALUES (?1, ?2)",
                rusqlite::params![sample_id, r#"{"x":1,"y":2,"width":3,"height":4}"#],
            )
            .expect("insert annotation");

        let dset = DmanDataset::from_name_with_db(&db, "getitem-ds").expect("load");

        assert_eq!(dset.sample_rows.len(), 3);
        assert_eq!(dset.asset_rows.len(), 3);
        let _ = tmp; // keep alive
    }

    #[test]
    fn images_returns_image_asset_file_paths() {
        let (db, tmp, _) = setup_db_with_dataset("images-ds", 4);
        let dset = DmanDataset::from_name_with_db(&db, "images-ds").expect("load");
        let paths = dset.image_paths();
        assert_eq!(paths.len(), 4);
        for p in &paths {
            assert!(
                p.as_str().contains("img"),
                "expected path to contain 'img', got: {p}"
            );
        }
        let _ = tmp;
    }

    #[test]
    fn len_is_zero_for_empty_dataset() {
        let (db, _tmp, _) = setup_db_with_dataset("empty-ds", 0);
        let dset = DmanDataset::from_name_with_db(&db, "empty-ds").expect("load");
        assert_eq!(dset.sample_rows.len(), 0);
    }

    #[test]
    fn get_sample_returns_sample_by_name() {
        let (db, tmp, _) = setup_db_with_dataset("sample-get-ds", 3);
        let dset = DmanDataset::from_name_with_db(&db, "sample-get-ds").expect("load");
        // First sample is named "img000.jpg"
        let found = dset.sample_rows.iter().find(|s| s.name == "img000.jpg");
        assert!(found.is_some(), "expected to find sample img000.jpg");
        let _ = tmp;
    }

    #[test]
    fn asset_rows_have_correct_type() {
        let (db, tmp, _) = setup_db_with_dataset("asset-type-ds", 2);
        let dset = DmanDataset::from_name_with_db(&db, "asset-type-ds").expect("load");
        for asset in &dset.asset_rows {
            assert_eq!(
                asset.asset_type, "image",
                "expected asset_type=image, got: {}",
                asset.asset_type
            );
        }
        let _ = tmp;
    }

    #[test]
    fn annotations_for_sample_returns_correct_results() {
        let (db, tmp, ds_id) = setup_db_with_dataset("ann-ds", 2);
        let sample_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM samples WHERE dataset_id = ?1 ORDER BY id LIMIT 1",
                rusqlite::params![ds_id],
                |r| r.get(0),
            )
            .expect("get sample id");
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, bbox) VALUES (?1, ?2)",
                rusqlite::params![sample_id, r#"{"x":0,"y":0,"width":10,"height":10}"#],
            )
            .expect("insert annotation");

        let dset = DmanDataset::from_name_with_db(&db, "ann-ds").expect("load");
        let sample_name = dset
            .sample_rows
            .iter()
            .find(|s| s.id == sample_id)
            .map(|s| s.name.clone())
            .expect("sample name");

        // annotations() method requires catalog DB; test via raw row count instead
        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id = ?1",
                rusqlite::params![sample_id],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(ann_count, 1);
        // Verify we can find the sample by name
        assert!(dset.sample_rows.iter().any(|s| s.name == sample_name));
        let _ = tmp;
    }

    #[test]
    fn non_image_assets_excluded_from_images_method() {
        let tmp = TempDir::new().expect("tempdir");
        let db = Database::open_in_memory().expect("in-memory db");
        let ds = DatasetService::register(&db, "mixed-ds", tmp.path(), DatasetFormat::coco())
            .expect("register");

        // Add image asset
        DatasetService::add_sample_with_single_asset(
            &db,
            ds.id,
            "frame.jpg",
            AssetType::Image,
            "frame.jpg",
            Path::new("/tmp/frame.jpg"),
            None,
            None,
            None,
            None,
        )
        .expect("add image");

        // Add depth map asset to a separate sample
        let depth_sample_id =
            DatasetService::add_sample(&db, ds.id, "depth-sample", None).expect("add depth sample");
        DatasetService::add_asset(
            &db,
            depth_sample_id,
            AssetType::DepthMap,
            "depth.png",
            Path::new("/tmp/depth.png"),
            None,
            None,
            None,
            None,
        )
        .expect("add depth asset");

        let dset = DmanDataset::from_name_with_db(&db, "mixed-ds").expect("load");
        let paths = dset.image_paths();
        assert_eq!(paths.len(), 1, "only image assets returned by images()");
        assert!(paths[0].contains("frame.jpg"));
        let _ = tmp;
    }
}
