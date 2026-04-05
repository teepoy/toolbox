pub mod coco;
pub mod huggingface;
pub mod yolo;
use std::path::Path;

use crate::Result;
use crate::db::Database;
use crate::storage::StorageManager;
use crate::types::Dataset;

/// A plugin that can import a dataset from an on-disk format into `dman`.
///
/// Implementors must be `Send + Sync` so they can be stored in a shared
/// registry and used from multiple threads.
pub trait FormatImporter: Send + Sync {
    /// Human-readable name that identifies this format (e.g. `"yolo"`).
    fn name(&self) -> &str;

    /// Returns `true` when this importer recognises the layout at `path`.
    ///
    /// `path` may point to a file *or* a directory – implementations should
    /// handle both gracefully.
    fn detect(&self, path: &Path) -> bool;

    /// Import the dataset at `path` into the database, returning a [`Dataset`]
    /// record that has already been persisted.
    fn import(
        &self,
        db: &Database,
        storage: &StorageManager,
        path: &Path,
        dataset_name: &str,
    ) -> Result<Dataset>;
}

/// A plugin that can export a dataset from `dman` to an on-disk format.
///
/// Implementors must be `Send + Sync` so they can be stored in a shared
/// registry and used from multiple threads.
pub trait FormatExporter: Send + Sync {
    /// Human-readable name that identifies this format (e.g. `"yolo"`).
    fn name(&self) -> &str;

    /// Write the dataset to `output_path` in this format.
    fn export(
        &self,
        db: &Database,
        storage: &StorageManager,
        dataset: &Dataset,
        output_path: &Path,
    ) -> Result<()>;
}

/// Central registry for format importers and exporters.
pub struct FormatRegistry {
    importers: Vec<Box<dyn FormatImporter>>,
    exporters: Vec<Box<dyn FormatExporter>>,
}

impl FormatRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            importers: Vec::new(),
            exporters: Vec::new(),
        }
    }

    /// Register an importer plugin.
    pub fn register_importer(&mut self, importer: Box<dyn FormatImporter>) {
        self.importers.push(importer);
    }

    /// Register an exporter plugin.
    pub fn register_exporter(&mut self, exporter: Box<dyn FormatExporter>) {
        self.exporters.push(exporter);
    }

    /// Ask every registered importer whether it recognises `path`.
    ///
    /// Returns the *name* of the first importer that claims the path, or
    /// `None` when no importer recognises it.
    pub fn detect_format(&self, path: &Path) -> Option<&str> {
        self.importers
            .iter()
            .find(|imp| imp.detect(path))
            .map(|imp| imp.name())
    }

    /// Look up an importer by name.
    pub fn get_importer(&self, name: &str) -> Option<&dyn FormatImporter> {
        self.importers
            .iter()
            .find(|imp| imp.name() == name)
            .map(|imp| imp.as_ref())
    }

    /// Look up an exporter by name.
    pub fn get_exporter(&self, name: &str) -> Option<&dyn FormatExporter> {
        self.exporters
            .iter()
            .find(|exp| exp.name() == name)
            .map(|exp| exp.as_ref())
    }

    pub fn default_registry() -> Self {
        let mut registry = Self::new();

        registry.register_importer(Box::new(yolo::YoloImporter));
        registry.register_exporter(Box::new(yolo::YoloExporter));

        registry.register_importer(Box::new(coco::CocoImporter));
        registry.register_exporter(Box::new(coco::CocoExporter));

        registry.register_importer(Box::new(huggingface::HuggingFaceImporter));
        registry.register_exporter(Box::new(huggingface::HuggingFaceExporter));

        registry
    }
}

impl Default for FormatRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::FormatRegistry;

    #[test]
    fn new_registry_is_empty() {
        let reg = FormatRegistry::new();
        assert!(reg.get_importer("yolo").is_none());
        assert!(reg.get_exporter("yolo").is_none());
        assert!(reg.detect_format(Path::new("/tmp")).is_none());
    }

    #[test]
    fn default_impl_delegates_to_new() {
        let reg = FormatRegistry::default();
        assert!(reg.get_importer("yolo").is_none());
    }

    #[test]
    fn default_registry_has_yolo_importer() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_importer("yolo").is_some(),
            "yolo importer should be registered"
        );
    }

    #[test]
    fn default_registry_has_yolo_exporter() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_exporter("yolo").is_some(),
            "yolo exporter should be registered"
        );
    }

    #[test]
    fn default_registry_has_coco_importer() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_importer("coco").is_some(),
            "coco importer should be registered"
        );
    }

    #[test]
    fn default_registry_has_coco_exporter() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_exporter("coco").is_some(),
            "coco exporter should be registered"
        );
    }

    #[test]
    fn default_registry_has_huggingface_importer() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_importer("huggingface").is_some(),
            "huggingface importer should be registered"
        );
    }

    #[test]
    fn default_registry_has_huggingface_exporter() {
        let reg = FormatRegistry::default_registry();
        assert!(
            reg.get_exporter("huggingface").is_some(),
            "huggingface exporter should be registered"
        );
    }

    #[test]
    fn unknown_importer_returns_none() {
        let reg = FormatRegistry::default_registry();
        assert!(reg.get_importer("does_not_exist").is_none());
    }

    #[test]
    fn unknown_exporter_returns_none() {
        let reg = FormatRegistry::default_registry();
        assert!(reg.get_exporter("does_not_exist").is_none());
    }

    #[test]
    fn detect_format_returns_none_for_unknown_path() {
        let reg = FormatRegistry::default_registry();
        let arbitrary = Path::new("/tmp/some_dataset_dir");
        assert!(
            reg.detect_format(arbitrary).is_none(),
            "detect_format should return None when no built-in provider matches"
        );
    }

    #[test]
    fn built_in_importer_names_match() {
        let reg = FormatRegistry::default_registry();
        for name in &["yolo", "coco", "huggingface"] {
            let imp = reg.get_importer(name).expect("importer should exist");
            assert_eq!(imp.name(), *name);
        }
    }

    #[test]
    fn exporter_names_match() {
        let reg = FormatRegistry::default_registry();
        for name in &["yolo", "coco", "huggingface"] {
            let exp = reg.get_exporter(name).expect("should exist");
            assert_eq!(exp.name(), *name);
        }
    }

    #[test]
    fn can_register_custom_importer() {
        use super::FormatImporter;
        use crate::Result;
        use crate::db::Database;
        use crate::storage::StorageManager;
        use crate::types::Dataset;
        use std::path::PathBuf;

        struct CustomImporter;
        impl FormatImporter for CustomImporter {
            fn name(&self) -> &str {
                "custom"
            }
            fn detect(&self, _path: &Path) -> bool {
                false
            }
            fn import(
                &self,
                _db: &Database,
                _storage: &StorageManager,
                _path: &Path,
                _dataset_name: &str,
            ) -> Result<Dataset> {
                Ok(Dataset {
                    id: 0,
                    name: "custom".to_string(),
                    path: PathBuf::from("/tmp"),
                    format: crate::types::DatasetFormat::new("custom"),
                    schema_path: None,
                    created_at: "2026-01-01".to_string(),
                    updated_at: None,
                    metadata: None,
                })
            }
        }

        let mut reg = FormatRegistry::new();
        reg.register_importer(Box::new(CustomImporter));
        assert!(reg.get_importer("custom").is_some());
        assert_eq!(reg.get_importer("custom").unwrap().name(), "custom");
    }
}
