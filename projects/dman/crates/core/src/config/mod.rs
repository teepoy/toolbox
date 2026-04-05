use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

use crate::error::{DmanError, Result};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DmanConfig {
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub python: PythonConfig,
    #[serde(default)]
    pub ui: UiConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StorageConfig {
    #[serde(default = "default_storage_strategy")]
    pub strategy: String,
    #[serde(default = "default_storage_base_path")]
    pub base_path: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    #[serde(default = "default_server_host")]
    pub host: String,
    #[serde(default = "default_server_port")]
    pub port: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PythonConfig {
    #[serde(default = "default_python_interpreter")]
    pub interpreter: String,
    #[serde(default = "default_python_plugin_paths")]
    pub plugin_paths: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    #[serde(default = "default_ui_default_page_size")]
    pub default_page_size: u32,
    #[serde(default = "default_ui_thumbnail_size")]
    pub thumbnail_size: u32,
}

impl DmanConfig {
    pub fn load(path: &Path) -> Result<DmanConfig> {
        if !path.exists() {
            return Ok(DmanConfig::default());
        }

        let contents = fs::read_to_string(path)?;
        toml::from_str(&contents).map_err(DmanError::from)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents =
            toml::to_string_pretty(self).map_err(|err| DmanError::ConfigError(err.to_string()))?;
        fs::write(path, contents)?;
        Ok(())
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            strategy: default_storage_strategy(),
            base_path: default_storage_base_path(),
        }
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_server_host(),
            port: default_server_port(),
        }
    }
}

impl Default for PythonConfig {
    fn default() -> Self {
        Self {
            interpreter: default_python_interpreter(),
            plugin_paths: default_python_plugin_paths(),
        }
    }
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            default_page_size: default_ui_default_page_size(),
            thumbnail_size: default_ui_thumbnail_size(),
        }
    }
}

fn default_storage_strategy() -> String {
    "copy".to_string()
}

fn default_storage_base_path() -> String {
    "~/.dman/data".to_string()
}

fn default_server_host() -> String {
    "127.0.0.1".to_string()
}

fn default_server_port() -> u16 {
    8080
}

fn default_python_interpreter() -> String {
    "python3".to_string()
}

fn default_python_plugin_paths() -> Vec<String> {
    vec!["~/.dman/plugins".to_string()]
}

fn default_ui_default_page_size() -> u32 {
    50
}

fn default_ui_thumbnail_size() -> u32 {
    128
}

#[cfg(test)]
mod tests {
    use super::DmanConfig;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn default_config_has_sensible_values() {
        let config = DmanConfig::default();

        assert_eq!(config.storage.strategy, "copy");
        assert!(!config.storage.base_path.is_empty());
        assert!(!config.server.host.is_empty());
        assert!(config.server.port > 0);
        assert!(!config.python.interpreter.is_empty());
        assert!(!config.python.plugin_paths.is_empty());
        assert!(config.ui.default_page_size > 0);
        assert!(config.ui.thumbnail_size > 0);
    }

    #[test]
    fn partial_toml_merges_with_defaults() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        fs::write(&path, "[server]\nhost = \"0.0.0.0\"\n").unwrap();

        let config = DmanConfig::load(&path).unwrap();

        assert_eq!(config.server.host, "0.0.0.0");
        assert_eq!(config.server.port, 8080);
        assert_eq!(config.storage.strategy, "copy");
        assert!(!config.python.plugin_paths.is_empty());
    }

    #[test]
    fn save_and_load_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nested/config.toml");

        let original = DmanConfig::default();
        original.save(&path).unwrap();

        let loaded = DmanConfig::load(&path).unwrap();

        assert_eq!(loaded.storage.strategy, original.storage.strategy);
        assert_eq!(loaded.storage.base_path, original.storage.base_path);
        assert_eq!(loaded.server.host, original.server.host);
        assert_eq!(loaded.server.port, original.server.port);
        assert_eq!(loaded.python.interpreter, original.python.interpreter);
        assert_eq!(loaded.python.plugin_paths, original.python.plugin_paths);
        assert_eq!(loaded.ui.default_page_size, original.ui.default_page_size);
        assert_eq!(loaded.ui.thumbnail_size, original.ui.thumbnail_size);
    }

    #[test]
    fn missing_file_returns_default_config() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("missing.toml");

        let config = DmanConfig::load(&path).unwrap();

        assert_eq!(config, DmanConfig::default());
    }
}
