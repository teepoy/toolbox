use std::fs;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rusqlite::params;
use walkdir::WalkDir;

use crate::dataset::DatasetService;
use crate::db::Database;
use crate::formats::{FormatExporter, FormatImporter};
use crate::storage::StorageManager;
use crate::types::{AssetType, Dataset, DatasetFormat};
use crate::{DmanError, Result};

const IMAGE_COLUMN_NAMES: &[&str] = &["file_name", "image", "path", "image_path"];

fn find_parquet_files(root: &Path) -> Vec<std::path::PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("parquet"))
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect()
}

pub struct HuggingFaceImporter;

impl FormatImporter for HuggingFaceImporter {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn detect(&self, path: &Path) -> bool {
        !find_parquet_files(path).is_empty()
    }

    fn import(
        &self,
        db: &Database,
        _storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset> {
        let parquet_files = find_parquet_files(path);
        if parquet_files.is_empty() {
            return Err(DmanError::ImportFailed {
                path: path.to_path_buf(),
                reason: "no .parquet files found in directory".to_string(),
            });
        }

        let dataset =
            DatasetService::register(db, dataset_name, path, DatasetFormat::huggingface())?;

        let mut records: Vec<(String, String, String)> = Vec::new();

        for pq_path in &parquet_files {
            let file = fs::File::open(pq_path).map_err(DmanError::Io)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                DmanError::ImportFailed {
                    path: pq_path.clone(),
                    reason: e.to_string(),
                }
            })?;
            let mut reader = builder.build().map_err(|e| DmanError::ImportFailed {
                path: pq_path.clone(),
                reason: e.to_string(),
            })?;

            for batch_result in &mut reader {
                let batch = batch_result.map_err(|e| DmanError::ImportFailed {
                    path: pq_path.clone(),
                    reason: e.to_string(),
                })?;

                let col_name = IMAGE_COLUMN_NAMES
                    .iter()
                    .find(|&&name| batch.schema().field_with_name(name).is_ok())
                    .copied();

                let num_rows = batch.num_rows();
                for row_idx in 0..num_rows {
                    let file_name = if let Some(col) = col_name {
                        let arr =
                            batch
                                .column_by_name(col)
                                .ok_or_else(|| DmanError::ImportFailed {
                                    path: pq_path.clone(),
                                    reason: format!("column '{}' not found", col),
                                })?;
                        let string_arr =
                            arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                                DmanError::ImportFailed {
                                    path: pq_path.clone(),
                                    reason: format!("column '{}' is not a string array", col),
                                }
                            })?;
                        if string_arr.is_null(row_idx) {
                            format!("row_{}", row_idx)
                        } else {
                            string_arr.value(row_idx).to_string()
                        }
                    } else {
                        format!("row_{}", row_idx)
                    };

                    let mut meta_map = serde_json::Map::new();
                    for col_field in batch.schema().fields() {
                        let col_arr = batch.column_by_name(col_field.name()).ok_or_else(|| {
                            DmanError::ImportFailed {
                                path: pq_path.clone(),
                                reason: format!("column '{}' missing", col_field.name()),
                            }
                        })?;
                        if let Some(str_arr) = col_arr.as_any().downcast_ref::<StringArray>() {
                            let val = if str_arr.is_null(row_idx) {
                                serde_json::Value::Null
                            } else {
                                serde_json::Value::String(str_arr.value(row_idx).to_string())
                            };
                            meta_map.insert(col_field.name().clone(), val);
                        }
                    }
                    let metadata_json = serde_json::to_string(&serde_json::Value::Object(meta_map))
                        .map_err(DmanError::SerdeJson)?;

                    let file_path = pq_path
                        .parent()
                        .unwrap_or(path)
                        .join(&file_name)
                        .to_string_lossy()
                        .to_string();

                    records.push((file_name, file_path, metadata_json));
                }
            }
        }

        db.conn
            .execute("BEGIN IMMEDIATE", [])
            .map_err(DmanError::Database)?;

        let insert_result = (|| -> Result<()> {
            for (file_name, file_path, _metadata_json) in &records {
                let stem = std::path::Path::new(file_name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(file_name.as_str())
                    .to_string();
                let asset_type_str = AssetType::Image.to_string();
                db.conn
                    .execute(
                        "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                        params![dataset.id, stem],
                    )
                    .map_err(DmanError::Database)?;
                let sample_id = db.conn.last_insert_rowid();
                db.conn
                    .execute(
                        "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, ?2, ?3, ?4)",
                        params![sample_id, asset_type_str, file_name, file_path],
                    )
                    .map_err(DmanError::Database)?;
            }
            Ok(())
        })();

        match insert_result {
            Ok(()) => {
                db.conn.execute("COMMIT", []).map_err(DmanError::Database)?;
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                let _ = DatasetService::remove(db, dataset_name);
                return Err(e);
            }
        }

        Ok(dataset)
    }
}

pub struct HuggingFaceExporter;

impl FormatExporter for HuggingFaceExporter {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn export(
        &self,
        db: &Database,
        _storage: &StorageManager,
        dataset: &Dataset,
        output_path: &Path,
    ) -> Result<()> {
        let mut stmt = db
            .conn
            .prepare(
                "SELECT a.file_name, a.file_path, a.metadata \
                 FROM assets a \
                 JOIN samples s ON a.sample_id = s.id \
                 WHERE s.dataset_id = ?1 \
                 ORDER BY a.id",
            )
            .map_err(DmanError::Database)?;

        let mut filenames: Vec<String> = Vec::new();
        let mut filepaths: Vec<String> = Vec::new();
        let mut metadatas: Vec<String> = Vec::new();

        let rows = stmt
            .query_map(params![dataset.id], |row| {
                let file_name: String = row.get(0)?;
                let file_path: String = row.get(1)?;
                let metadata: Option<String> = row.get(2)?;
                Ok((file_name, file_path, metadata))
            })
            .map_err(DmanError::Database)?;

        for row_result in rows {
            let (file_name, file_path, metadata) = row_result.map_err(DmanError::Database)?;
            filenames.push(file_name);
            filepaths.push(file_path);
            metadatas.push(metadata.unwrap_or_default());
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("file_name", DataType::Utf8, false),
            Field::new("file_path", DataType::Utf8, false),
            Field::new("metadata", DataType::Utf8, true),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(filenames)) as ArrayRef,
                Arc::new(StringArray::from(filepaths)) as ArrayRef,
                Arc::new(StringArray::from(metadatas)) as ArrayRef,
            ],
        )
        .map_err(|e| DmanError::ExportFailed {
            path: output_path.to_path_buf(),
            reason: e.to_string(),
        })?;

        let data_dir = output_path.join("data");
        fs::create_dir_all(&data_dir).map_err(DmanError::Io)?;
        let out_file_path = data_dir.join("train-00000-of-00001.parquet");
        let out_file = fs::File::create(&out_file_path).map_err(DmanError::Io)?;

        let mut writer =
            ArrowWriter::try_new(out_file, schema, None).map_err(|e| DmanError::ExportFailed {
                path: out_file_path.clone(),
                reason: e.to_string(),
            })?;

        writer.write(&batch).map_err(|e| DmanError::ExportFailed {
            path: out_file_path.clone(),
            reason: e.to_string(),
        })?;

        writer.close().map_err(|e| DmanError::ExportFailed {
            path: out_file_path,
            reason: e.to_string(),
        })?;

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
            .join("huggingface")
    }

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    fn dummy_storage() -> StorageManager {
        StorageManager::new(PathBuf::from("/tmp"))
    }

    #[test]
    fn test_importer_detects_hf_dir() {
        let imp = HuggingFaceImporter;
        let fixture = fixture_dir();
        assert!(
            fixture.exists(),
            "fixture dir should exist: {}",
            fixture.display()
        );
        assert!(
            imp.detect(&fixture),
            "detect should return true for HuggingFace fixture dir"
        );
    }

    #[test]
    fn test_importer_does_not_detect_empty_dir() {
        let tmp = tempdir().expect("tempdir");
        let imp = HuggingFaceImporter;
        assert!(
            !imp.detect(tmp.path()),
            "detect should return false for empty dir"
        );
    }

    #[test]
    fn test_import_fixture() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let imp = HuggingFaceImporter;
        let fixture = fixture_dir();

        let dataset = imp
            .import(&db, &storage, &fixture, "hf-test")
            .expect("import should succeed");

        assert_eq!(dataset.name, "hf-test");
        assert_eq!(dataset.format, DatasetFormat::huggingface());

        let count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![dataset.id],
                |row| row.get(0),
            )
            .expect("count query");
        assert_eq!(count, 3, "fixture has 3 rows -> 3 samples expected");
    }

    #[test]
    fn test_import_populates_file_name() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let imp = HuggingFaceImporter;
        let fixture = fixture_dir();

        let dataset = imp
            .import(&db, &storage, &fixture, "hf-names")
            .expect("import");

        let mut stmt = db
            .conn
            .prepare(
                "SELECT a.file_name FROM assets a \
                 JOIN samples s ON a.sample_id = s.id \
                 WHERE s.dataset_id = ?1 ORDER BY a.id",
            )
            .expect("prepare");
        let names: Vec<String> = stmt
            .query_map(params![dataset.id], |row| row.get(0))
            .expect("query_map")
            .collect::<rusqlite::Result<Vec<_>>>()
            .expect("collect");

        assert!(!names.is_empty(), "should have at least one image name");
        for name in &names {
            assert!(!name.is_empty(), "file_name should not be empty");
        }
    }

    #[test]
    fn test_export_creates_parquet() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let imp = HuggingFaceImporter;
        let fixture = fixture_dir();

        let dataset = imp
            .import(&db, &storage, &fixture, "hf-export")
            .expect("import for export test");

        let out_dir = tempdir().expect("output tempdir");
        let exp = HuggingFaceExporter;
        exp.export(&db, &storage, &dataset, out_dir.path())
            .expect("export should succeed");

        let parquet_path = out_dir
            .path()
            .join("data")
            .join("train-00000-of-00001.parquet");
        assert!(
            parquet_path.exists(),
            "exported parquet file should exist: {}",
            parquet_path.display()
        );
        assert!(
            parquet_path.metadata().expect("metadata").len() > 0,
            "exported parquet file should be non-empty"
        );
    }

    #[test]
    fn test_export_roundtrip_row_count() {
        let db = in_memory_db();
        let storage = dummy_storage();
        let imp = HuggingFaceImporter;
        let exp = HuggingFaceExporter;
        let fixture = fixture_dir();

        let dataset = imp
            .import(&db, &storage, &fixture, "hf-roundtrip")
            .expect("import");

        let out_dir = tempdir().expect("output tempdir");
        exp.export(&db, &storage, &dataset, out_dir.path())
            .expect("export");

        let parquet_path = out_dir
            .path()
            .join("data")
            .join("train-00000-of-00001.parquet");
        let file = std::fs::File::open(&parquet_path).expect("open exported parquet");
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader builder");
        let mut reader = builder.build().expect("parquet reader");
        let mut total_rows = 0usize;
        for batch in &mut reader {
            total_rows += batch.expect("batch").num_rows();
        }
        assert_eq!(total_rows, 3, "roundtrip should preserve 3 rows");
    }
}
