#[cfg(feature = "python")]
pub use python_impl::{create_dataset, update_dataset, DmanDatasetBuilder, DmanDatasetUpdater};

#[cfg(feature = "python")]
pub mod python_impl {
    use dman_core::{
        catalog::Catalog, dataset::DatasetService, db::Database, types::DatasetFormat,
    };
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use rusqlite::params;
    use serde_json::json;

    use crate::sdk::loader::DmanDataset;

    #[derive(Debug, Clone)]
    pub(crate) struct PendingImage {
        pub path: String,
        pub metadata: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PendingAnnotation {
        pub image_index: usize,
        pub category: String,
        pub bbox: Vec<f64>,
        pub metadata: Option<String>,
    }

    #[derive(Debug, Clone)]
    pub(crate) struct PendingCategory {
        pub name: String,
        pub supercategory: Option<String>,
    }

    #[derive(Debug)]
    pub(crate) enum UpdateOp {
        AddImage {
            path: String,
            metadata: Option<String>,
        },
        AddAnnotation {
            image_db_id: i64,
            category: String,
            bbox: Vec<f64>,
            metadata: Option<String>,
        },
        RemoveImage {
            image_db_id: i64,
        },
    }

    #[pyclass(name = "DmanDatasetBuilder", unsendable)]
    pub struct DmanDatasetBuilder {
        pub(crate) db: Database,
        pub(crate) name: String,
        pub(crate) schema_path: Option<String>,
        pub(crate) pending_images: Vec<PendingImage>,
        pub(crate) pending_annotations: Vec<PendingAnnotation>,
        pub(crate) pending_categories: Vec<PendingCategory>,
    }

    impl DmanDatasetBuilder {
        pub fn new(name: &str, schema_path: Option<String>) -> PyResult<Self> {
            let db = open_catalog_db()?;
            Ok(Self {
                db,
                name: name.to_string(),
                schema_path,
                pending_images: Vec::new(),
                pending_annotations: Vec::new(),
                pending_categories: Vec::new(),
            })
        }

        pub fn new_with_db(db: Database, name: &str, schema_path: Option<String>) -> Self {
            Self {
                db,
                name: name.to_string(),
                schema_path,
                pending_images: Vec::new(),
                pending_annotations: Vec::new(),
                pending_categories: Vec::new(),
            }
        }

        pub fn add_image_internal(
            &mut self,
            path: &str,
            metadata: Option<String>,
        ) -> PyResult<usize> {
            let idx = self.pending_images.len();
            self.pending_images.push(PendingImage {
                path: path.to_string(),
                metadata,
            });
            Ok(idx)
        }

        pub fn add_annotation_internal(
            &mut self,
            image_index: usize,
            category: &str,
            bbox: Vec<f64>,
            metadata: Option<String>,
        ) -> PyResult<()> {
            if image_index >= self.pending_images.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "image_index {} out of range (have {} pending images)",
                    image_index,
                    self.pending_images.len()
                )));
            }
            if bbox.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bbox must have 4 elements [x, y, w, h], got {}",
                    bbox.len()
                )));
            }
            self.pending_annotations.push(PendingAnnotation {
                image_index,
                category: category.to_string(),
                bbox,
                metadata,
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
                DatasetFormat::Custom("builder".to_string()),
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

            let mut image_db_ids: Vec<i64> = Vec::with_capacity(self.pending_images.len());
            for pi in &self.pending_images {
                let file_name = std::path::Path::new(&pi.path)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| pi.path.clone());

                self.db
                    .conn
                    .execute(
                        "INSERT INTO images (dataset_id, file_name, file_path, metadata) \
                         VALUES (?1, ?2, ?3, ?4)",
                        params![dataset_id, file_name, pi.path, pi.metadata],
                    )
                    .map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("insert image: {e}"))
                    })?;
                image_db_ids.push(self.db.conn.last_insert_rowid());
            }

            for ann in &self.pending_annotations {
                let image_db_id = image_db_ids[ann.image_index];
                let category_id = category_map.get(&ann.category).copied();
                let bbox_json = serde_json::to_string(&json!({
                    "x": ann.bbox[0],
                    "y": ann.bbox[1],
                    "w": ann.bbox[2],
                    "h": ann.bbox[3]
                }))
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("serialize bbox: {e}"))
                })?;

                self.db
                    .conn
                    .execute(
                        "INSERT INTO annotations (image_id, category_id, bbox, metadata) \
                         VALUES (?1, ?2, ?3, ?4)",
                        params![image_db_id, category_id, bbox_json, ann.metadata],
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
        #[pyo3(signature = (path, metadata=None))]
        pub fn add_image(
            &mut self,
            path: String,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<i64> {
            let meta_json = metadata
                .map(|d| Ok::<String, PyErr>(d.to_string()))
                .transpose()?;
            let idx = self.add_image_internal(&path, meta_json)?;
            Ok(idx as i64)
        }

        #[pyo3(signature = (image_id, category, bbox, metadata=None))]
        pub fn add_annotation(
            &mut self,
            image_id: i64,
            category: String,
            bbox: Vec<f64>,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            let meta_json = metadata
                .map(|d| Ok::<String, PyErr>(d.to_string()))
                .transpose()?;
            self.add_annotation_internal(image_id as usize, &category, bbox, meta_json)
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
        pub(crate) dataset_name: String,
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
                dataset_name: name.to_string(),
                pending_ops: Vec::new(),
            })
        }

        pub fn add_image_internal(
            &mut self,
            path: &str,
            metadata: Option<String>,
        ) -> PyResult<i64> {
            let file_name = std::path::Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());

            self.db
                .conn
                .execute(
                    "INSERT INTO images (dataset_id, file_name, file_path, metadata) \
                     VALUES (?1, ?2, ?3, ?4)",
                    params![self.dataset_id, file_name, path, metadata],
                )
                .map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!("add_image insert: {e}"))
                })?;
            Ok(self.db.conn.last_insert_rowid())
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
                if let UpdateOp::AddAnnotation { category, .. } = op {
                    if !category_map.contains_key(category) {
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
            }

            for op in &self.pending_ops {
                match op {
                    UpdateOp::AddImage { path, metadata } => {
                        let file_name = std::path::Path::new(path)
                            .file_name()
                            .map(|n| n.to_string_lossy().to_string())
                            .unwrap_or_else(|| path.clone());
                        self.db
                            .conn
                            .execute(
                                "INSERT INTO images (dataset_id, file_name, file_path, metadata) \
                                 VALUES (?1, ?2, ?3, ?4)",
                                params![self.dataset_id, file_name, path, metadata],
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!("add image: {e}"))
                            })?;
                    }
                    UpdateOp::AddAnnotation {
                        image_db_id,
                        category,
                        bbox,
                        metadata,
                    } => {
                        let category_id = category_map.get(category).copied();
                        let bbox_json = serde_json::to_string(&json!({
                            "x": bbox[0],
                            "y": bbox[1],
                            "w": bbox[2],
                            "h": bbox[3]
                        }))
                        .map_err(|e| {
                            pyo3::exceptions::PyRuntimeError::new_err(format!(
                                "serialize bbox: {e}"
                            ))
                        })?;
                        self.db
                            .conn
                            .execute(
                                "INSERT INTO annotations (image_id, category_id, bbox, metadata) \
                                 VALUES (?1, ?2, ?3, ?4)",
                                params![image_db_id, category_id, bbox_json, metadata],
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "add annotation: {e}"
                                ))
                            })?;
                    }
                    UpdateOp::RemoveImage { image_db_id } => {
                        self.db
                            .conn
                            .execute(
                                "DELETE FROM annotations WHERE image_id = ?1",
                                params![image_db_id],
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "delete annotations: {e}"
                                ))
                            })?;
                        self.db
                            .conn
                            .execute(
                                "DELETE FROM images WHERE id = ?1 AND dataset_id = ?2",
                                params![image_db_id, self.dataset_id],
                            )
                            .map_err(|e| {
                                pyo3::exceptions::PyRuntimeError::new_err(format!(
                                    "delete image: {e}"
                                ))
                            })?;
                    }
                }
            }
            Ok(())
        }
    }

    #[pymethods]
    impl DmanDatasetUpdater {
        #[pyo3(signature = (path, metadata=None))]
        pub fn add_image(
            &mut self,
            path: String,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<i64> {
            let meta_json = metadata
                .map(|d| Ok::<String, PyErr>(d.to_string()))
                .transpose()?;
            self.pending_ops.push(UpdateOp::AddImage {
                path,
                metadata: meta_json,
            });
            Ok(-1)
        }

        #[pyo3(signature = (image_id, category, bbox, metadata=None))]
        pub fn add_annotation(
            &mut self,
            image_id: i64,
            category: String,
            bbox: Vec<f64>,
            metadata: Option<Bound<'_, PyDict>>,
        ) -> PyResult<()> {
            if bbox.len() != 4 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "bbox must have 4 elements [x, y, w, h], got {}",
                    bbox.len()
                )));
            }
            let meta_json = metadata
                .map(|d| Ok::<String, PyErr>(d.to_string()))
                .transpose()?;
            self.pending_ops.push(UpdateOp::AddAnnotation {
                image_db_id: image_id,
                category,
                bbox,
                metadata: meta_json,
            });
            Ok(())
        }

        pub fn remove_image(&mut self, image_id: i64) -> PyResult<()> {
            self.pending_ops.push(UpdateOp::RemoveImage {
                image_db_id: image_id,
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
    fn builder_creates_dataset_and_returns_dman_dataset() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "test-builder", None);

        let img_id = builder
            .add_image_internal(tmp.path().join("img.jpg").to_str().unwrap(), None)
            .expect("add image");
        builder
            .add_annotation_internal(img_id, "cat", vec![10.0, 20.0, 50.0, 50.0], None)
            .expect("add annotation");

        let ds = builder.build_internal().expect("build");
        assert_eq!(ds.images.len(), 1);
        assert_eq!(ds.name, "test-builder");
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

        DatasetService::register(
            &db,
            "dup-ds",
            tmp.path(),
            DatasetFormat::Custom("x".to_string()),
        )
        .expect("pre-register");

        let mut builder = DmanDatasetBuilder::new_with_db(db, "dup-ds", None);
        let _ = builder
            .add_image_internal(tmp.path().join("z.jpg").to_str().unwrap(), None)
            .expect("add");

        let result = builder.build_internal();
        assert!(result.is_err(), "expected error on duplicate dataset name");

        let img_count: i64 = builder
            .db
            .conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .expect("count");
        assert_eq!(img_count, 0, "rollback must leave no images");
    }

    #[test]
    fn builder_multiple_images_and_annotations() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "multi-ds", None);

        for i in 0..3 {
            let img_path = tmp.path().join(format!("img{}.jpg", i));
            let idx = builder
                .add_image_internal(img_path.to_str().unwrap(), None)
                .expect("add");
            builder
                .add_annotation_internal(idx, "obj", vec![0.0, 0.0, 10.0, 10.0], None)
                .expect("ann");
        }

        let ds = builder.build_internal().expect("build");
        assert_eq!(ds.images.len(), 3);
    }

    #[test]
    fn builder_empty_dataset_works() {
        let db = in_memory_db();
        let mut builder = DmanDatasetBuilder::new_with_db(db, "empty-builder-ds", None);
        let ds = builder.build_internal().expect("build empty");
        assert_eq!(ds.images.len(), 0);
        assert_eq!(ds.name, "empty-builder-ds");
    }

    #[test]
    fn updater_new_with_db_errors_for_missing_dataset() {
        let db = in_memory_db();
        let result = DmanDatasetUpdater::new_with_db(db, "does-not-exist");
        assert!(result.is_err(), "expected error for missing dataset");
    }

    #[test]
    fn updater_add_image_internal_returns_positive_id() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        DatasetService::register(&db, "updater-ds", tmp.path(), DatasetFormat::Coco)
            .expect("register");

        let mut updater = DmanDatasetUpdater::new_with_db(db, "updater-ds").expect("updater");
        let img_id = updater
            .add_image_internal(tmp.path().join("new.jpg").to_str().unwrap(), None)
            .expect("add image");
        assert!(img_id > 0);
    }

    #[test]
    fn updater_apply_removes_image_and_adds_new() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        let ds = DatasetService::register(&db, "apply-ds", tmp.path(), DatasetFormat::Coco)
            .expect("register");

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                params![ds.id, "old.jpg", "/tmp/old.jpg"],
            )
            .expect("insert old image");
        let old_img_id = db.conn.last_insert_rowid();

        let mut updater = DmanDatasetUpdater::new_with_db(db, "apply-ds").expect("updater");
        updater.pending_ops.push(UpdateOp::RemoveImage {
            image_db_id: old_img_id,
        });
        updater.pending_ops.push(UpdateOp::AddImage {
            path: tmp.path().join("new.jpg").to_str().unwrap().to_string(),
            metadata: None,
        });

        updater.apply_internal().expect("apply");

        let img_count: i64 = updater
            .db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                params![updater.dataset_id],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(img_count, 1);

        let new_name: String = updater
            .db
            .conn
            .query_row(
                "SELECT file_name FROM images WHERE dataset_id = ?1",
                params![updater.dataset_id],
                |r| r.get(0),
            )
            .expect("name");
        assert_eq!(new_name, "new.jpg");
    }

    #[test]
    fn updater_apply_adds_annotation_with_category() {
        let tmp = TempDir::new().expect("tempdir");
        let db = in_memory_db();

        let ds = DatasetService::register(&db, "ann-ds", tmp.path(), DatasetFormat::Coco)
            .expect("register");

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                params![ds.id, "img.jpg", "/tmp/img.jpg"],
            )
            .expect("insert image");
        let img_id = db.conn.last_insert_rowid();

        let mut updater = DmanDatasetUpdater::new_with_db(db, "ann-ds").expect("updater");
        updater.pending_ops.push(UpdateOp::AddAnnotation {
            image_db_id: img_id,
            category: "cat".to_string(),
            bbox: vec![1.0, 2.0, 3.0, 4.0],
            metadata: None,
        });

        updater.apply_internal().expect("apply");

        let ann_count: i64 = updater
            .db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id = ?1",
                params![img_id],
                |r| r.get(0),
            )
            .expect("count");
        assert_eq!(ann_count, 1);
    }
}
