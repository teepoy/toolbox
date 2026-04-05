use std::path::PathBuf;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DmanError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    SerdeJson(#[from] serde_json::Error),

    #[error("YAML error: {0}")]
    SerdeYaml(#[from] serde_yaml::Error),

    #[error("TOML parse error: {0}")]
    Toml(#[from] toml::de::Error),

    #[error("Schema validation error: {0}")]
    SchemaValidation(String),

    #[error("Dataset not found: {0}")]
    DatasetNotFound(String),

    #[error("sample not found: {0}")]
    SampleNotFound(String),

    #[error("asset not found: {0}")]
    AssetNotFound(String),

    #[error("invalid asset type: {0}")]
    InvalidAssetType(String),

    #[error("dataset already exists: {0}")]
    DatasetAlreadyExists(String),

    #[error("Format not supported: {0}")]
    FormatUnsupported(String),

    #[error("Import failed at {path}: {reason}")]
    ImportFailed { path: PathBuf, reason: String },

    #[error("Export failed at {path}: {reason}")]
    ExportFailed { path: PathBuf, reason: String },

    #[error("Plugin error: {0}")]
    PluginError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Migration error: {0}")]
    MigrationError(String),

    #[error("Circular reference detected: {0}")]
    CircularReference(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, DmanError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_schema_validation() {
        let e = DmanError::SchemaValidation("field 'label' missing".to_string());
        assert!(!e.to_string().is_empty());
        assert!(e.to_string().contains("field 'label' missing"));
    }

    #[test]
    fn test_display_dataset_not_found() {
        let e = DmanError::DatasetNotFound("my-dataset".to_string());
        assert!(e.to_string().contains("my-dataset"));
    }

    #[test]
    fn sample_not_found_displays_correctly() {
        let e = DmanError::SampleNotFound("sample-123".to_string());
        assert!(e.to_string().contains("sample-123"));
    }

    #[test]
    fn asset_not_found_displays_correctly() {
        let e = DmanError::AssetNotFound("asset-123".to_string());
        assert!(e.to_string().contains("asset-123"));
    }

    #[test]
    fn invalid_asset_type_displays_correctly() {
        let e = DmanError::InvalidAssetType("gif".to_string());
        assert!(e.to_string().contains("gif"));
    }

    #[test]
    fn test_display_import_failed() {
        let e = DmanError::ImportFailed {
            path: PathBuf::from("/tmp/test"),
            reason: "file corrupt".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("/tmp/test") || s.contains("tmp"));
        assert!(s.contains("file corrupt"));
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let dman_err: DmanError = io_err.into();
        assert!(matches!(dman_err, DmanError::Io(_)));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err: Result<serde_json::Value> =
            serde_json::from_str("{invalid}").map_err(DmanError::from);
        match json_err {
            Ok(_) => panic!("expected error"),
            Err(err) => assert!(matches!(err, DmanError::SerdeJson(_))),
        }
    }

    #[test]
    fn test_result_type_alias() {
        let ok: Result<i32> = Ok(42);
        let err: Result<i32> = Err(DmanError::StorageError("test".to_string()));
        assert!(matches!(ok, Ok(42)));
        assert!(err.is_err());
    }
}
