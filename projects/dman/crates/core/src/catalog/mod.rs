use std::path::PathBuf;

use crate::{DmanError, Result, db::Database};

pub struct Catalog {
    db: Database,
    home: PathBuf,
}

impl Catalog {
    /// Returns the catalog home directory: `$DMAN_HOME` if set, otherwise `~/.dman/`.
    pub fn home_path() -> PathBuf {
        if let Ok(val) = std::env::var("DMAN_HOME") {
            PathBuf::from(val)
        } else {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("~"))
                .join(".dman")
        }
    }

    /// Initialize catalog at the given path: create dirs, open/create DB, run migrations.
    fn init_at(home: PathBuf) -> Result<Catalog> {
        std::fs::create_dir_all(&home)?;
        std::fs::create_dir_all(home.join("plugins"))?;
        std::fs::create_dir_all(home.join("data"))?;

        let db_path = home.join("catalog.db");
        let db = Database::open(&db_path)?;

        Ok(Catalog { db, home })
    }

    /// Create `~/.dman/` (or `$DMAN_HOME`) if not exists, create subdirs, open/create DB.
    pub fn init() -> Result<Catalog> {
        Self::init_at(Self::home_path())
    }

    /// Open existing catalog. Returns error if `catalog.db` doesn't exist.
    pub fn open() -> Result<Catalog> {
        Self::open_at(Self::home_path())
    }

    /// Open catalog at a given path. Returns error if `catalog.db` doesn't exist.
    fn open_at(home: PathBuf) -> Result<Catalog> {
        let db_path = home.join("catalog.db");
        if !db_path.exists() {
            return Err(DmanError::StorageError(
                "catalog not initialized - run `dman init`".to_string(),
            ));
        }
        let db = Database::open(&db_path)?;
        Ok(Catalog { db, home })
    }

    /// Expose underlying database.
    pub fn db(&self) -> &Database {
        &self.db
    }

    /// Returns `{home}/config.toml`.
    pub fn config_path(&self) -> PathBuf {
        self.home.join("config.toml")
    }

    /// Returns `{home}/plugins/`.
    pub fn plugins_path(&self) -> PathBuf {
        self.home.join("plugins")
    }

    /// Returns `{home}/data/`.
    pub fn data_path(&self) -> PathBuf {
        self.home.join("data")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn init_creates_catalog_db_and_subdirs() {
        let dir = tempdir().expect("tempdir");
        let home = dir.path().join("dman_home");

        let catalog = Catalog::init_at(home.clone()).expect("init_at should succeed");

        assert!(home.join("catalog.db").exists(), "catalog.db should exist");
        assert!(home.join("plugins").is_dir(), "plugins/ should be a dir");
        assert!(home.join("data").is_dir(), "data/ should be a dir");
        let _ = catalog.db();
    }

    #[test]
    fn open_returns_error_when_not_initialized() {
        let dir = tempdir().expect("tempdir");
        let home = dir.path().join("nonexistent_dman");

        let result = Catalog::open_at(home);
        assert!(result.is_err(), "open_at should fail if catalog.db absent");
        match result.err().expect("already checked is_err") {
            DmanError::StorageError(msg) => {
                assert!(
                    msg.contains("catalog not initialized"),
                    "unexpected msg: {msg}"
                );
            }
            other => panic!("expected StorageError, got: {other:?}"),
        }
    }

    #[test]
    fn open_succeeds_after_init() {
        let dir = tempdir().expect("tempdir");
        let home = dir.path().join("dman_home");

        Catalog::init_at(home.clone()).expect("init should succeed");
        Catalog::open_at(home).expect("open_at should succeed after init");
    }

    #[test]
    fn path_helpers_return_correct_paths() {
        let dir = tempdir().expect("tempdir");
        let home = dir.path().join("dman_home");

        let catalog = Catalog::init_at(home.clone()).expect("init");

        assert_eq!(catalog.config_path(), home.join("config.toml"));
        assert_eq!(catalog.plugins_path(), home.join("plugins"));
        assert_eq!(catalog.data_path(), home.join("data"));
    }

    #[test]
    fn home_path_uses_dman_home_env_var() {
        let dir = tempdir().expect("tempdir");
        let expected = dir.path().to_path_buf();

        let original = std::env::var("DMAN_HOME").ok();

        // SAFETY: single-threaded test context; value is restored immediately after assertion
        unsafe {
            std::env::set_var("DMAN_HOME", &expected);
        }
        let result = Catalog::home_path();

        match original {
            Some(v) => unsafe { std::env::set_var("DMAN_HOME", v) },
            None => unsafe { std::env::remove_var("DMAN_HOME") },
        }

        assert_eq!(result, expected);
    }

    #[test]
    fn home_path_falls_back_when_no_env_var() {
        let original = std::env::var("DMAN_HOME").ok();
        unsafe {
            std::env::remove_var("DMAN_HOME");
        }

        let result = Catalog::home_path();

        if let Some(v) = original {
            unsafe { std::env::set_var("DMAN_HOME", v) };
        }

        assert!(
            result.ends_with(".dman"),
            "expected path ending with .dman, got: {result:?}"
        );
    }
}
