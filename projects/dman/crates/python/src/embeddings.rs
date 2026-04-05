#[cfg(feature = "python")]
use std::path::Path;

#[cfg(feature = "python")]
use dman_core::{
    db::Database,
    embeddings::EmbeddingService,
    error::{DmanError, Result},
};

#[cfg(feature = "python")]
pub fn compute_embeddings(
    db: &Database,
    dataset_id: i64,
    model_script: &Path,
    batch_size: usize,
) -> Result<u64> {
    use std::ffi::CString;

    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyDictMethods, PyList, PyListMethods};

    struct AssetRow {
        id: i64,
        file_path: String,
    }

    let code = std::fs::read_to_string(model_script)
        .map_err(|e| DmanError::StorageError(e.to_string()))?;
    let path_str = model_script
        .to_str()
        .ok_or_else(|| DmanError::PluginError("non-UTF8 plugin path".to_string()))?;

    let code_c = CString::new(code).map_err(|e| DmanError::PluginError(e.to_string()))?;
    let file_c = CString::new(path_str).map_err(|e| DmanError::PluginError(e.to_string()))?;
    let mod_c = CString::new(path_str).map_err(|e| DmanError::PluginError(e.to_string()))?;

    let images: Vec<AssetRow> = {
        let mut stmt = db
            .conn
            .prepare(
                "SELECT a.id, a.file_path \
                 FROM assets a \
                 JOIN samples s ON a.sample_id = s.id \
                 WHERE s.dataset_id = ?1 AND a.asset_type = 'image' \
                 ORDER BY a.id",
            )
            .map_err(DmanError::Database)?;

        stmt.query_map(rusqlite::params![dataset_id], |row| {
            Ok(AssetRow {
                id: row.get(0)?,
                file_path: row.get(1)?,
            })
        })
        .map_err(DmanError::Database)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DmanError::Database)?
    };

    if images.is_empty() {
        return Ok(0);
    }

    let batch_size = batch_size.max(1);

    Python::attach(|py| {
        let module = pyo3::types::PyModule::from_code(py, &code_c, &file_c, &mod_c)
            .map_err(|e| DmanError::PluginError(e.to_string()))?;

        let marker = module
            .getattr("dman_plugin")
            .map_err(|_| DmanError::PluginError("missing dman_plugin".to_string()))?;
        let plugin = marker
            .cast::<PyDict>()
            .map_err(|e| DmanError::PluginError(e.to_string()))?;

        let plugin_type: String = plugin
            .get_item("type")
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
            .ok_or_else(|| DmanError::PluginError("missing 'type' in dman_plugin".to_string()))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        if plugin_type != "embeddings" {
            return Err(DmanError::PluginError(format!(
                "plugin type must be 'embeddings', got '{plugin_type}'"
            )));
        }

        let model_name: String = plugin
            .get_item("name")
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
            .ok_or_else(|| DmanError::PluginError("missing 'name' in dman_plugin".to_string()))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        let compute = module
            .getattr("compute")
            .map_err(|_| DmanError::PluginError("missing compute()".to_string()))?;

        let mut total_stored = 0_u64;
        for chunk in images.chunks(batch_size) {
            let batch_paths: Vec<&str> =
                chunk.iter().map(|image| image.file_path.as_str()).collect();
            let py_paths =
                PyList::new(py, batch_paths).map_err(|e| DmanError::PluginError(e.to_string()))?;

            let result = compute
                .call1((py_paths,))
                .map_err(|e| DmanError::PluginError(e.to_string()))?;
            let vectors = result
                .cast::<PyList>()
                .map_err(|e| DmanError::PluginError(e.to_string()))?;

            if vectors.len() != chunk.len() {
                return Err(DmanError::PluginError(format!(
                    "plugin returned {} vectors for {} images",
                    vectors.len(),
                    chunk.len()
                )));
            }

            for (asset, item) in chunk.iter().zip(vectors.iter()) {
                let vector: Vec<f32> = item
                    .extract()
                    .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;
                EmbeddingService::store(db, asset.id, &model_name, &vector)?;
                total_stored += 1;
            }
        }

        Ok(total_stored)
    })
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    use std::fs;

    use dman_core::embeddings::EmbeddingService;
    use tempfile::TempDir;

    fn setup_db_with_images(image_count: usize) -> (Database, i64) {
        let db = Database::open_in_memory().expect("open db");

        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                rusqlite::params!["test-embeddings", "/tmp/test-embeddings"],
            )
            .expect("insert dataset");
        let dataset_id = db.conn.last_insert_rowid();

        for idx in 0..image_count {
            db.conn
                .execute(
                    "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                    rusqlite::params![dataset_id, format!("img-{idx:03}")],
                )
                .expect("insert sample");
            let sample_id = db.conn.last_insert_rowid();
            db.conn
                .execute(
                    "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'image', ?2, ?3)",
                    rusqlite::params![
                        sample_id,
                        format!("img-{idx:03}.jpg"),
                        format!("/tmp/img-{idx:03}.jpg")
                    ],
                )
                .expect("insert asset");
        }

        (db, dataset_id)
    }

    #[test]
    fn embeddings_compute_embeddings_processes_all_images() {
        let (db, dataset_id) = setup_db_with_images(10);
        let temp_dir = TempDir::new().expect("temp dir");
        let plugin_path = temp_dir.path().join("embed_plugin.py");

        fs::write(
            &plugin_path,
            r#"
import random

dman_plugin = {"name": "test-embed-model", "type": "embeddings", "model": "dummy", "dimension": 4}

def compute(image_paths):
    return [[random.random() for _ in range(4)] for _ in image_paths]
"#,
        )
        .expect("write plugin");

        let stored =
            compute_embeddings(&db, dataset_id, &plugin_path, 3).expect("compute embeddings");
        assert_eq!(stored, 10);

        let embeddings =
            EmbeddingService::list_by_dataset(&db, dataset_id).expect("list embeddings");
        assert_eq!(embeddings.len(), 10);
        assert!(
            embeddings
                .iter()
                .all(|embedding| embedding.vector.len() == 4)
        );
        assert!(
            embeddings
                .iter()
                .all(|embedding| embedding.model_name == "test-embed-model")
        );
    }
}
