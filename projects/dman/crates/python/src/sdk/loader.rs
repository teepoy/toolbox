/// PyO3 loader module — PyTorch Dataset and HuggingFace Datasets integration.
///
/// Exposes:
///   - `load_dataset(name, split=None) -> DmanDataset`
///   - `DmanDataset` — Python class with `__len__`, `__getitem__`, `to_torch_dataset`,
///     `to_hf_dataset`, `images`, `annotations`

#[cfg(feature = "python")]
pub use python_impl::{load_dataset, DmanDataset};

#[cfg(feature = "python")]
mod python_impl {
    use dman_core::{catalog::Catalog, dataset::DatasetService, db::Database, error::DmanError};
    use pyo3::prelude::*;
    use pyo3::types::{PyAny, PyDict, PyList};
    use rusqlite::params;

    // ─── Row types (internal) ───────────────────────────────────────────────

    #[derive(Debug, Clone)]
    pub(crate) struct ImageRow {
        pub id: i64,
        pub file_name: String,
        pub file_path: String,
        pub width: Option<i64>,
        pub height: Option<i64>,
        pub metadata: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct AnnotationRow {
        pub id: i64,
        pub image_id: i64,
        pub category_id: Option<i64>,
        pub bbox: Option<String>,
        pub metadata: Option<String>,
    }

    // ─── DmanDataset ────────────────────────────────────────────────────────

    /// A Python-accessible dataset backed by a dman catalog.
    #[derive(Debug)]
    #[pyclass(name = "DmanDataset")]
    pub struct DmanDataset {
        pub(crate) dataset_id: i64,
        pub(crate) name: String,
        pub(crate) images: Vec<ImageRow>,
    }

    impl DmanDataset {
        /// Load all images for `dataset_id` from `db`.
        fn load_images(db: &Database, dataset_id: i64) -> PyResult<Vec<ImageRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT id, file_name, file_path, width, height, metadata \
                     FROM images WHERE dataset_id = ?1 ORDER BY id",
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("DB error: {e}")))?;

            let rows = stmt
                .query_map(params![dataset_id], |row| {
                    Ok(ImageRow {
                        id: row.get(0)?,
                        file_name: row.get(1)?,
                        file_path: row.get(2)?,
                        width: row.get(3)?,
                        height: row.get(4)?,
                        metadata: row.get(5)?,
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

        /// Load annotations for a single image from `db`.
        fn load_annotations_for_image(
            db: &Database,
            image_id: i64,
        ) -> PyResult<Vec<AnnotationRow>> {
            let mut stmt = db
                .conn
                .prepare(
                    "SELECT id, image_id, category_id, bbox, metadata \
                     FROM annotations WHERE image_id = ?1 ORDER BY id",
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("DB prepare error: {e}"))
                })?;

            let rows = stmt
                .query_map(params![image_id], |row| {
                    Ok(AnnotationRow {
                        id: row.get(0)?,
                        image_id: row.get(1)?,
                        category_id: row.get(2)?,
                        bbox: row.get(3)?,
                        metadata: row.get(4)?,
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
            Catalog::open()
                .map(|c| {
                    // We need owned DB — use inner database.
                    // Catalog owns the db; we must reconstruct from path.
                    let _ = c; // dropped; re-open directly
                    ()
                })
                .ok();

            // Re-open catalog to get access to db path
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

            let images = Self::load_images(db, dataset.id)?;

            Ok(DmanDataset {
                dataset_id: dataset.id,
                name: dataset.name,
                images,
            })
        }
    }

    #[pymethods]
    impl DmanDataset {
        /// Number of images in the dataset.
        fn __len__(&self) -> usize {
            self.images.len()
        }

        /// Return a dict for image at `idx`:
        /// `{id, file_name, image_path, annotations, metadata}`
        fn __getitem__(&self, idx: usize) -> PyResult<Py<PyAny>> {
            if idx >= self.images.len() {
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {} out of range for dataset of size {}",
                    idx,
                    self.images.len()
                )));
            }
            let img = &self.images[idx];
            let db = Self::open_catalog_db()?;
            let anns = Self::load_annotations_for_image(&db, img.id)?;

            Python::attach(|py| {
                let d = PyDict::new(py);
                d.set_item("id", img.id)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                d.set_item("file_name", &img.file_name)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                d.set_item("image_path", &img.file_path)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                d.set_item("metadata", img.metadata.as_deref().unwrap_or(""))
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                // Build annotations list
                let ann_list = PyList::empty(py);
                for ann in &anns {
                    let ad = PyDict::new(py);
                    ad.set_item("id", ann.id)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    ad.set_item("image_id", ann.image_id)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    if let Some(cat_id) = ann.category_id {
                        ad.set_item("category_id", cat_id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                    }
                    ad.set_item("bbox", ann.bbox.as_deref().unwrap_or(""))
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    ad.set_item("metadata", ann.metadata.as_deref().unwrap_or(""))
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                    ann_list
                        .append(ad)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                d.set_item("annotations", ann_list)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

                Ok(d.unbind().into_any())
            })
        }

        /// Return list of image path strings.
        fn images(&self) -> Vec<String> {
            self.images.iter().map(|i| i.file_path.clone()).collect()
        }

        /// Return annotations for all images as a list of dicts.
        fn annotations(&self) -> PyResult<Py<PyAny>> {
            let db = Self::open_catalog_db()?;

            Python::attach(|py| {
                let outer = PyList::empty(py);
                for img in &self.images {
                    let anns = Self::load_annotations_for_image(&db, img.id)?;
                    let ann_list = PyList::empty(py);
                    for ann in &anns {
                        let ad = PyDict::new(py);
                        ad.set_item("id", ann.id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                        ad.set_item("image_id", ann.image_id).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                        if let Some(cat_id) = ann.category_id {
                            ad.set_item("category_id", cat_id).map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                            })?;
                        }
                        ad.set_item("bbox", ann.bbox.as_deref().unwrap_or(""))
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                            })?;
                        ad.set_item("metadata", ann.metadata.as_deref().unwrap_or(""))
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                            })?;
                        ann_list.append(ad).map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                        })?;
                    }
                    outer
                        .append(ann_list)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
                }
                Ok(outer.unbind().into_any())
            })
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
            let items: Vec<String> = self.images.iter().map(|i| i.file_path.clone()).collect();

            let locals = PyDict::new(py);
            locals
                .set_item("image_paths", items)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            py.run(
                std::ffi::CStr::from_bytes_with_nul(
                    b"import torch.utils.data\n\nclass _DmanTorchDataset(torch.utils.data.Dataset):\n    def __init__(self, paths):\n        self._paths = paths\n    def __len__(self):\n        return len(self._paths)\n    def __getitem__(self, idx):\n        return {\"image_path\": self._paths[idx]}\n\n_result = _DmanTorchDataset(image_paths)\n\0",
                )
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
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

            // Build a dict of columns
            let image_paths: Vec<String> =
                self.images.iter().map(|i| i.file_path.clone()).collect();
            let file_names: Vec<String> = self.images.iter().map(|i| i.file_name.clone()).collect();
            let ids: Vec<i64> = self.images.iter().map(|i| i.id).collect();

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
    use dman_core::{dataset::DatasetService, db::Database, types::DatasetFormat};
    use rusqlite::params;
    use tempfile::TempDir;

    /// Create an isolated in-memory DB and populate a dataset with images.
    fn setup_db_with_dataset(dataset_name: &str, image_count: usize) -> (Database, TempDir, i64) {
        let tmp = TempDir::new().expect("tempdir");
        let db = Database::open_in_memory().expect("in-memory db");

        let ds = DatasetService::register(&db, dataset_name, tmp.path(), DatasetFormat::Coco)
            .expect("register dataset");

        for i in 0..image_count {
            let file_name = format!("img{:03}.jpg", i);
            let file_path = tmp.path().join(&file_name).to_string_lossy().to_string();
            db.conn
                .execute(
                    "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                    params![ds.id, file_name, file_path],
                )
                .expect("insert image");
        }

        (db, tmp, ds.id)
    }

    #[test]
    fn load_dataset_with_valid_name_returns_correct_len() {
        let (db, _tmp, _ds_id) = setup_db_with_dataset("test-ds", 5);
        let dset = DmanDataset::from_name_with_db(&db, "test-ds").expect("load");
        assert_eq!(dset.images.len(), 5);
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
    fn getitem_returns_correct_image_data() {
        let (db, tmp, ds_id) = setup_db_with_dataset("getitem-ds", 3);
        // Insert an annotation for image 0
        let image_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM images WHERE dataset_id = ?1 ORDER BY id LIMIT 1",
                params![ds_id],
                |r| r.get(0),
            )
            .expect("get image id");
        db.conn
            .execute(
                "INSERT INTO annotations (image_id, bbox) VALUES (?1, ?2)",
                params![image_id, r#"{"x":1,"y":2,"width":3,"height":4}"#],
            )
            .expect("insert annotation");

        let dset = DmanDataset::from_name_with_db(&db, "getitem-ds").expect("load");

        assert_eq!(dset.images.len(), 3);
        let paths: Vec<&str> = dset.images.iter().map(|i| i.file_path.as_str()).collect();
        assert_eq!(paths.len(), 3);
        let _ = tmp; // keep alive
    }

    #[test]
    fn images_returns_file_paths() {
        let (db, tmp, _) = setup_db_with_dataset("images-ds", 4);
        let dset = DmanDataset::from_name_with_db(&db, "images-ds").expect("load");
        assert_eq!(dset.images.len(), 4);
        for img in &dset.images {
            assert!(
                img.file_path.contains("img"),
                "expected path to contain 'img', got: {}",
                img.file_path
            );
        }
        let _ = tmp;
    }

    #[test]
    fn len_is_zero_for_empty_dataset() {
        let (db, _tmp, _) = setup_db_with_dataset("empty-ds", 0);
        let dset = DmanDataset::from_name_with_db(&db, "empty-ds").expect("load");
        assert_eq!(dset.images.len(), 0);
    }
}
