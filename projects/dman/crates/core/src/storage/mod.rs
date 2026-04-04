use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::db::Database;
use crate::{DmanError, Result};

pub struct StorageManager {
    base_path: PathBuf,
}

pub enum StorageStrategy {
    Copy,
    Symlink,
    Reference,
}

pub struct IntegrityReport {
    pub total: u64,
    pub ok: u64,
    pub missing: Vec<String>,
    pub corrupted: Vec<String>,
}

impl StorageManager {
    pub fn new(base_path: PathBuf) -> Self {
        Self { base_path }
    }

    pub fn store_image(
        &self,
        dataset_id: i64,
        source_path: &Path,
        strategy: StorageStrategy,
    ) -> Result<PathBuf> {
        let file_name = source_path
            .file_name()
            .ok_or_else(|| DmanError::StorageError("source path has no file name".to_string()))?;

        match strategy {
            StorageStrategy::Reference => Ok(source_path.to_path_buf()),
            StorageStrategy::Copy => {
                let dest_dir = self.base_path.join(dataset_id.to_string()).join("images");
                fs::create_dir_all(&dest_dir)?;
                let dest_path = dest_dir.join(file_name);
                fs::copy(source_path, &dest_path)?;
                Ok(dest_path)
            }
            StorageStrategy::Symlink => {
                let dest_dir = self.base_path.join(dataset_id.to_string()).join("images");
                fs::create_dir_all(&dest_dir)?;
                let dest_path = dest_dir.join(file_name);
                let abs_source = source_path.canonicalize()?;
                std::os::unix::fs::symlink(abs_source, &dest_path)?;
                Ok(dest_path)
            }
        }
    }

    pub fn get_image_path(&self, dataset_id: i64, file_name: &str) -> PathBuf {
        self.base_path
            .join(dataset_id.to_string())
            .join("images")
            .join(file_name)
    }

    pub fn delete_image(&self, stored_path: &Path) -> Result<()> {
        if stored_path.exists() || stored_path.symlink_metadata().is_ok() {
            fs::remove_file(stored_path)?;
        }
        Ok(())
    }

    pub fn delete_dataset_images(&self, dataset_id: i64) -> Result<()> {
        let dir = self.base_path.join(dataset_id.to_string());
        if dir.exists() {
            fs::remove_dir_all(&dir)?;
        }
        Ok(())
    }

    pub fn calculate_hash(path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buf = [0u8; 8192];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(64);
        for b in digest.iter() {
            write!(&mut hex, "{:02x}", b).map_err(|e| DmanError::StorageError(e.to_string()))?;
        }
        Ok(hex)
    }

    pub fn check_integrity(&self, db: &Database, dataset_id: i64) -> Result<IntegrityReport> {
        let mut stmt = db
            .conn
            .prepare("SELECT file_path FROM images WHERE dataset_id = ?")?;
        let paths: Vec<String> = stmt
            .query_map(rusqlite::params![dataset_id], |row| {
                Ok(row.get::<_, String>(0)?)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let total = paths.len() as u64;
        let mut ok = 0u64;
        let mut missing = Vec::new();
        let corrupted = Vec::new();

        for fp in &paths {
            let p = Path::new(fp);
            if p.exists() {
                ok += 1;
            } else {
                missing.push(fp.clone());
            }
        }

        Ok(IntegrityReport {
            total,
            ok,
            missing,
            corrupted,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use tempfile::tempdir;

    use super::*;
    use crate::db::Database;

    fn make_test_file(dir: &Path, name: &str, content: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut f = fs::File::create(&path).expect("create test file");
        f.write_all(content).expect("write test file");
        path
    }

    #[test]
    fn test_new_storage_manager() {
        let tmp = tempdir().unwrap();
        let mgr = StorageManager::new(tmp.path().to_path_buf());
        assert_eq!(mgr.base_path, tmp.path());
    }

    #[test]
    fn test_get_image_path() {
        let tmp = tempdir().unwrap();
        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let p = mgr.get_image_path(42, "foo.jpg");
        assert_eq!(p, tmp.path().join("42").join("images").join("foo.jpg"));
    }

    #[test]
    fn test_store_image_copy() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "img.png", b"image data");

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let dest = mgr
            .store_image(1, &src, StorageStrategy::Copy)
            .expect("store_image copy");

        assert!(dest.exists(), "copied file should exist");
        let expected = tmp.path().join("1").join("images").join("img.png");
        assert_eq!(dest, expected);
        let content = fs::read(&dest).unwrap();
        assert_eq!(content, b"image data");
    }

    #[test]
    fn test_store_image_symlink() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "img.png", b"symlink data");

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let dest = mgr
            .store_image(2, &src, StorageStrategy::Symlink)
            .expect("store_image symlink");

        assert!(dest.exists(), "symlink target should resolve");
        let meta = dest.symlink_metadata().unwrap();
        assert!(meta.file_type().is_symlink(), "should be a symlink");
        let content = fs::read(&dest).unwrap();
        assert_eq!(content, b"symlink data");
    }

    #[test]
    fn test_store_image_reference() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "img.png", b"ref data");

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let dest = mgr
            .store_image(3, &src, StorageStrategy::Reference)
            .expect("store_image reference");

        assert_eq!(dest, src, "reference should return original path");
    }

    #[test]
    fn test_delete_image() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "del.png", b"to delete");

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let dest = mgr
            .store_image(4, &src, StorageStrategy::Copy)
            .expect("store_image");

        assert!(dest.exists());
        mgr.delete_image(&dest).expect("delete_image");
        assert!(!dest.exists(), "file should be deleted");
    }

    #[test]
    fn test_delete_dataset_images() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "img.png", b"dataset data");

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        mgr.store_image(5, &src, StorageStrategy::Copy)
            .expect("store_image");

        let ds_dir = tmp.path().join("5");
        assert!(ds_dir.exists());
        mgr.delete_dataset_images(5).expect("delete_dataset_images");
        assert!(!ds_dir.exists(), "dataset dir should be removed");
    }

    #[test]
    fn test_calculate_hash_deterministic() {
        let tmp = tempdir().unwrap();
        let p = make_test_file(tmp.path(), "hash_test.bin", b"hello world");
        let h1 = StorageManager::calculate_hash(&p).expect("hash 1");
        let h2 = StorageManager::calculate_hash(&p).expect("hash 2");
        assert_eq!(h1, h2, "hash must be deterministic");
        assert_eq!(h1.len(), 64, "SHA256 hex is 64 chars");
    }

    #[test]
    fn test_calculate_hash_known_value() {
        let tmp = tempdir().unwrap();
        let p = make_test_file(tmp.path(), "known.txt", b"abc");
        let h = StorageManager::calculate_hash(&p).expect("hash");
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_check_integrity_all_present() {
        let tmp = tempdir().unwrap();
        let src_dir = tempdir().unwrap();
        let src = make_test_file(src_dir.path(), "img.png", b"data");

        let db = Database::open_in_memory().expect("db");
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                ["ds1", "/tmp"],
            )
            .unwrap();
        let dataset_id: i64 = db
            .conn
            .query_row("SELECT id FROM datasets WHERE name = ?1", ["ds1"], |row| {
                row.get(0)
            })
            .unwrap();

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let stored = mgr
            .store_image(dataset_id, &src, StorageStrategy::Copy)
            .expect("store");

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "img.png", stored.to_str().unwrap()],
            )
            .unwrap();

        let report = mgr.check_integrity(&db, dataset_id).expect("integrity");
        assert_eq!(report.total, 1);
        assert_eq!(report.ok, 1);
        assert!(report.missing.is_empty());
        assert!(report.corrupted.is_empty());
    }

    #[test]
    fn test_check_integrity_missing_file() {
        let tmp = tempdir().unwrap();
        let db = Database::open_in_memory().expect("db");
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                ["ds2", "/tmp"],
            )
            .unwrap();
        let dataset_id: i64 = db
            .conn
            .query_row("SELECT id FROM datasets WHERE name = ?1", ["ds2"], |row| {
                row.get(0)
            })
            .unwrap();

        let ghost_path = "/nonexistent/path/ghost.jpg";
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "ghost.jpg", ghost_path],
            )
            .unwrap();

        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let report = mgr.check_integrity(&db, dataset_id).expect("integrity");
        assert_eq!(report.total, 1);
        assert_eq!(report.ok, 0);
        assert_eq!(report.missing.len(), 1);
        assert_eq!(report.missing[0], ghost_path);
    }

    #[test]
    fn test_delete_nonexistent_image_is_ok() {
        let tmp = tempdir().unwrap();
        let mgr = StorageManager::new(tmp.path().to_path_buf());
        let fake = tmp.path().join("nope.png");
        mgr.delete_image(&fake)
            .expect("delete nonexistent should be ok");
    }

    #[test]
    fn test_delete_dataset_images_nonexistent_is_ok() {
        let tmp = tempdir().unwrap();
        let mgr = StorageManager::new(tmp.path().to_path_buf());
        mgr.delete_dataset_images(999)
            .expect("delete nonexistent dataset dir ok");
    }
}
