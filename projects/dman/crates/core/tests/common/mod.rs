use std::path::{Path, PathBuf};
use tempfile::TempDir;

pub fn fixture_path(name: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .join("tests")
        .join("fixtures")
        .join(name)
}

pub fn create_temp_dir() -> TempDir {
    tempfile::tempdir().expect("should create temp dir")
}
