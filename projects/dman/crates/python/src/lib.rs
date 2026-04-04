pub mod plugin_info;
pub mod plugins;

pub use plugin_info::PluginInfo;

use std::path::{Path, PathBuf};

use dman_core::error::{DmanError, Result};
use walkdir::WalkDir;

pub struct PluginManager {
    plugin_dirs: Vec<PathBuf>,
}

impl PluginManager {
    pub fn new(plugin_dirs: Vec<PathBuf>) -> Self {
        Self { plugin_dirs }
    }

    pub fn discover(&self) -> Result<Vec<PluginInfo>> {
        let mut plugins = Vec::new();
        for dir in &self.plugin_dirs {
            discover_plugins_from_dir(dir, &mut plugins)?;
        }
        Ok(plugins)
    }
}

fn discover_plugins_from_dir(dir: &Path, out: &mut Vec<PluginInfo>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in WalkDir::new(dir)
        .min_depth(1)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("py") {
            let content = std::fs::read_to_string(path)
                .map_err(|e| DmanError::StorageError(e.to_string()))?;
            if let Some(info) = parse_plugin_marker(&content, path.to_path_buf()) {
                out.push(info);
            }
        }
    }
    Ok(())
}

fn parse_plugin_marker(content: &str, path: PathBuf) -> Option<PluginInfo> {
    if !content.contains("dman_plugin") {
        return None;
    }

    let name = extract_str_field(content, "name")?;
    let plugin_type = extract_str_field(content, "type")?;
    let version = extract_str_field(content, "version")?;

    Some(PluginInfo::new(name, plugin_type, version, path))
}

fn extract_str_field<'a>(content: &'a str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\"");
    let needle2 = format!("'{field}'");

    let pos = content
        .find(needle.as_str())
        .or_else(|| content.find(needle2.as_str()))?;

    let rest = &content[pos..];
    let colon = rest.find(':')?;
    let after_colon = rest[colon + 1..].trim_start();

    if let Some(stripped) = after_colon.strip_prefix('"') {
        let end = stripped.find('"')?;
        return Some(stripped[..end].to_string());
    }
    if let Some(stripped) = after_colon.strip_prefix('\'') {
        let end = stripped.find('\'')?;
        return Some(stripped[..end].to_string());
    }
    None
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
pub fn load_plugin_with_python(path: &Path) -> Result<PluginInfo> {
    use pyo3::types::PyDictMethods;
    use std::ffi::CString;

    let code = std::fs::read_to_string(path).map_err(|e| DmanError::StorageError(e.to_string()))?;
    let path_str = path
        .to_str()
        .ok_or_else(|| DmanError::PluginError("non-UTF8 path".to_string()))?;

    let c_code = CString::new(code)
        .map_err(|e: std::ffi::NulError| DmanError::PluginError(e.to_string()))?;
    let c_path = CString::new(path_str)
        .map_err(|e: std::ffi::NulError| DmanError::PluginError(e.to_string()))?;
    let c_mod = CString::new("plugin")
        .map_err(|e: std::ffi::NulError| DmanError::PluginError(e.to_string()))?;

    Python::attach(|py| {
        let module = PyModule::from_code(py, &c_code, &c_path, &c_mod)
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        let marker = module
            .getattr("dman_plugin")
            .map_err(|_: PyErr| DmanError::PluginError("missing dman_plugin".to_string()))?;
        let dict = marker
            .cast::<pyo3::types::PyDict>()
            .map_err(|e| DmanError::PluginError(e.to_string()))?;

        let name: String = dict
            .get_item("name")
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
            .ok_or_else(|| DmanError::PluginError("missing 'name'".to_string()))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        let plugin_type: String = dict
            .get_item("type")
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
            .ok_or_else(|| DmanError::PluginError("missing 'type'".to_string()))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        let version: String = dict
            .get_item("version")
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?
            .ok_or_else(|| DmanError::PluginError("missing 'version'".to_string()))?
            .extract()
            .map_err(|e: PyErr| DmanError::PluginError(e.to_string()))?;

        Ok(PluginInfo::new(
            name,
            plugin_type,
            version,
            path.to_path_buf(),
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_plugin(dir: &Path, filename: &str, content: &str) -> PathBuf {
        let p = dir.join(filename);
        fs::write(&p, content).unwrap();
        p
    }

    #[test]
    fn discover_finds_plugin_with_double_quotes() {
        let tmp = TempDir::new().unwrap();
        write_plugin(
            tmp.path(),
            "my_format.py",
            r#"
dman_plugin = {
    "name": "my_format",
    "type": "format",
    "version": "1.0.0",
}
"#,
        );
        let mgr = PluginManager::new(vec![tmp.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "my_format");
        assert_eq!(found[0].plugin_type, "format");
        assert_eq!(found[0].version, "1.0.0");
    }

    #[test]
    fn discover_finds_plugin_with_single_quotes() {
        let tmp = TempDir::new().unwrap();
        write_plugin(
            tmp.path(),
            "single_quote.py",
            r#"
dman_plugin = {
    'name': 'sq_plugin',
    'type': 'exporter',
    'version': '2.1.0',
}
"#,
        );
        let mgr = PluginManager::new(vec![tmp.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].name, "sq_plugin");
        assert_eq!(found[0].plugin_type, "exporter");
        assert_eq!(found[0].version, "2.1.0");
    }

    #[test]
    fn discover_skips_py_without_marker() {
        let tmp = TempDir::new().unwrap();
        write_plugin(tmp.path(), "no_marker.py", "def hello(): pass\n");
        let mgr = PluginManager::new(vec![tmp.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn discover_skips_non_py_files() {
        let tmp = TempDir::new().unwrap();
        write_plugin(
            tmp.path(),
            "not_a_plugin.txt",
            r#"dman_plugin = {"name": "x", "type": "y", "version": "1.0"}"#,
        );
        let mgr = PluginManager::new(vec![tmp.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn discover_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let mgr = PluginManager::new(vec![tmp.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn discover_nonexistent_dir_returns_empty() {
        let mgr = PluginManager::new(vec![PathBuf::from("/nonexistent/no/such/dir")]);
        let found = mgr.discover().unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn discover_multiple_dirs() {
        let tmp1 = TempDir::new().unwrap();
        let tmp2 = TempDir::new().unwrap();
        write_plugin(
            tmp1.path(),
            "a.py",
            r#"dman_plugin = {"name": "A", "type": "format", "version": "1.0"}"#,
        );
        write_plugin(
            tmp2.path(),
            "b.py",
            r#"dman_plugin = {"name": "B", "type": "importer", "version": "0.5"}"#,
        );
        let mgr = PluginManager::new(vec![tmp1.path().to_path_buf(), tmp2.path().to_path_buf()]);
        let found = mgr.discover().unwrap();
        assert_eq!(found.len(), 2);
        let names: Vec<&str> = found.iter().map(|p| p.name.as_str()).collect();
        assert!(names.contains(&"A"));
        assert!(names.contains(&"B"));
    }

    #[test]
    fn parse_plugin_marker_extracts_fields() {
        let content = r#"dman_plugin = {"name": "test", "type": "format", "version": "1.0"}"#;
        let info = parse_plugin_marker(content, PathBuf::from("/tmp/test.py")).unwrap();
        assert_eq!(info.name, "test");
        assert_eq!(info.plugin_type, "format");
        assert_eq!(info.version, "1.0");
    }

    #[test]
    fn parse_plugin_marker_returns_none_without_marker() {
        let result = parse_plugin_marker("def foo(): pass", PathBuf::from("/x.py"));
        assert!(result.is_none());
    }
}
