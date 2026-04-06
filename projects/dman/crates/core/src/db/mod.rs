use std::path::Path;

use rusqlite::Connection;
use rusqlite_migration::{M, Migrations};

fn migrations() -> Migrations<'static> {
    Migrations::new(vec![M::up(
        "
        CREATE TABLE IF NOT EXISTS datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            format TEXT,
            schema_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            asset_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            width INTEGER,
            height INTEGER,
            hash TEXT,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            asset_id INTEGER REFERENCES assets(id) ON DELETE SET NULL,
            category_id INTEGER,
            bbox TEXT,
            segmentation TEXT,
            keypoints TEXT,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            supercategory TEXT,
            UNIQUE (dataset_id, name)
        );
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            model_name TEXT NOT NULL,
            vector BLOB NOT NULL,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
            asset_id INTEGER REFERENCES assets(id) ON DELETE SET NULL,
            model_version TEXT NOT NULL,
            result TEXT NOT NULL,
            score REAL
        );
        CREATE TABLE IF NOT EXISTS patches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id INTEGER NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
            bbox TEXT NOT NULL,
            file_path TEXT,
            metadata TEXT
        );
        CREATE TABLE IF NOT EXISTS virtual_datasets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            source_datasets TEXT NOT NULL,
            definition TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_samples_dataset_id ON samples(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_assets_sample_id ON assets(sample_id);
        CREATE INDEX IF NOT EXISTS idx_assets_hash ON assets(hash);
        CREATE INDEX IF NOT EXISTS idx_annotations_sample_id ON annotations(sample_id);
        CREATE INDEX IF NOT EXISTS idx_annotations_asset_id ON annotations(asset_id);
        CREATE INDEX IF NOT EXISTS idx_categories_dataset_id ON categories(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_embeddings_asset_id ON embeddings(asset_id);
        ",
        ),
        M::up(
            "
        ALTER TABLE patches ADD COLUMN annotation_id INTEGER REFERENCES annotations(id) ON DELETE SET NULL;
        CREATE INDEX IF NOT EXISTS idx_patches_annotation_id ON patches(annotation_id);
        ",
        ),
    ])
}

pub struct Database {
    pub conn: Connection,
}

impl Database {
    pub fn open<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let conn = Connection::open(path)?;
        let mut db = Self { conn };
        db.enable_wal()?;
        db.migrate()?;
        Ok(db)
    }

    pub fn open_in_memory() -> crate::Result<Self> {
        let conn = Connection::open_in_memory()?;
        let mut db = Self { conn };
        db.enable_wal()?;
        db.migrate()?;
        Ok(db)
    }

    fn enable_wal(&self) -> crate::Result<()> {
        self.conn
            .execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        Ok(())
    }

    pub fn migrate(&mut self) -> crate::Result<()> {
        migrations()
            .to_latest(&mut self.conn)
            .map_err(|e| crate::DmanError::MigrationError(e.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Database;

    #[test]
    fn open_in_memory_works() {
        Database::open_in_memory().expect("should open in-memory DB");
    }

    #[test]
    fn creates_expected_tables() {
        let db = Database::open_in_memory().expect("should open in-memory DB");
        for table in [
            "datasets",
            "samples",
            "assets",
            "annotations",
            "categories",
            "embeddings",
            "predictions",
            "patches",
            "virtual_datasets",
        ] {
            let count: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?1",
                    [table],
                    |row| row.get(0),
                )
                .expect("should query sqlite_master");
            assert_eq!(count, 1, "table '{}' should exist", table);
        }
    }

    #[test]
    fn creates_expected_indices() {
        let db = Database::open_in_memory().expect("should open in-memory DB");
        for index in [
            "idx_samples_dataset_id",
            "idx_assets_sample_id",
            "idx_assets_hash",
            "idx_annotations_sample_id",
            "idx_annotations_asset_id",
            "idx_categories_dataset_id",
            "idx_embeddings_asset_id",
            "idx_patches_annotation_id",
        ] {
            let count: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
                    [index],
                    |row| row.get(0),
                )
                .expect("should query sqlite_master");
            assert_eq!(count, 1, "index '{}' should exist", index);
        }
    }

    #[test]
    fn can_insert_and_read_dataset() {
        let db = Database::open_in_memory().expect("should open in-memory DB");
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                ["test-dataset", "/tmp/test"],
            )
            .expect("should insert dataset");

        let name: String = db
            .conn
            .query_row(
                "SELECT name FROM datasets WHERE name=?1",
                ["test-dataset"],
                |row| row.get(0),
            )
            .expect("should read dataset");
        assert_eq!(name, "test-dataset");

        let dataset_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM datasets WHERE name=?1",
                ["test-dataset"],
                |row| row.get(0),
            )
            .expect("should read dataset id");

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                rusqlite::params![dataset_id, "sample-001"],
            )
            .expect("should insert sample");

        let sample_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM samples WHERE dataset_id=?1",
                [dataset_id],
                |row| row.get(0),
            )
            .expect("should read sample id");

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![sample_id, "image", "frame.jpg", "/tmp/frame.jpg"],
            )
            .expect("should insert asset");

        let asset_id: i64 = db
            .conn
            .query_row(
                "SELECT id FROM assets WHERE sample_id=?1",
                [sample_id],
                |row| row.get(0),
            )
            .expect("should read asset id");

        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, asset_id) VALUES (?1, ?2)",
                rusqlite::params![sample_id, asset_id],
            )
            .expect("should insert annotation with asset_id");

        db.conn
            .execute(
                "INSERT INTO annotations (sample_id) VALUES (?1)",
                rusqlite::params![sample_id],
            )
            .expect("should insert annotation with null asset_id");

        let annotation_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id=?1",
                [sample_id],
                |row| row.get(0),
            )
            .expect("should count annotations");
        assert_eq!(annotation_count, 2, "should have 2 annotations for sample");
    }

    #[test]
    fn wal_pragmas_execute() {
        let db = Database::open_in_memory().expect("should open in-memory DB");
        let mode: String = db
            .conn
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .expect("should query journal mode");
        assert!(!mode.is_empty());
    }
}
