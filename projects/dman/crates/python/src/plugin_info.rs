use std::path::PathBuf;

/// Metadata extracted from a discovered Python plugin file.
#[derive(Debug, Clone, PartialEq)]
pub struct PluginInfo {
    /// Plugin name (from `dman_plugin["name"]`)
    pub name: String,
    /// Plugin type (from `dman_plugin["type"]`)
    pub plugin_type: String,
    /// Plugin version (from `dman_plugin["version"]`)
    pub version: String,
    /// Absolute path to the `.py` file
    pub path: PathBuf,
}

impl PluginInfo {
    /// Construct a new `PluginInfo`.
    pub fn new(
        name: impl Into<String>,
        plugin_type: impl Into<String>,
        version: impl Into<String>,
        path: PathBuf,
    ) -> Self {
        Self {
            name: name.into(),
            plugin_type: plugin_type.into(),
            version: version.into(),
            path,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_info_fields() {
        let p = PluginInfo::new("my_plugin", "format", "1.2.3", PathBuf::from("/tmp/my.py"));
        assert_eq!(p.name, "my_plugin");
        assert_eq!(p.plugin_type, "format");
        assert_eq!(p.version, "1.2.3");
        assert_eq!(p.path, PathBuf::from("/tmp/my.py"));
    }

    #[test]
    fn test_plugin_info_clone() {
        let p = PluginInfo::new("a", "b", "c", PathBuf::from("/x.py"));
        let q = p.clone();
        assert_eq!(p, q);
    }
}
