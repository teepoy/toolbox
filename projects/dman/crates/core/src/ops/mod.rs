pub mod transforms;

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use rusqlite::params;

use crate::{
    Dataset, DatasetFormat,
    db::Database,
    error::{DmanError, Result},
};

fn row_to_dataset(row: &rusqlite::Row<'_>) -> rusqlite::Result<Dataset> {
    let id: i64 = row.get(0)?;
    let name: String = row.get(1)?;
    let path: String = row.get(2)?;
    let format_str: String = row.get(3)?;
    let schema_path: Option<String> = row.get(4)?;
    let created_at: String = row.get(5)?;
    let updated_at: Option<String> = row.get(6)?;
    let metadata_str: Option<String> = row.get(7)?;

    let format = DatasetFormat::from(format_str);
    let schema_path = schema_path.map(PathBuf::from);
    let metadata = metadata_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());

    Ok(Dataset {
        id,
        name,
        path: PathBuf::from(path),
        format,
        schema_path,
        created_at,
        updated_at,
        metadata,
    })
}

fn get_dataset_by_name(db: &Database, name: &str) -> Result<Dataset> {
    let result = db.conn.query_row(
        "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets WHERE name = ?1",
        params![name],
        row_to_dataset,
    );
    match result {
        Ok(ds) => Ok(ds),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            Err(DmanError::DatasetNotFound(name.to_string()))
        }
        Err(e) => Err(DmanError::Database(e)),
    }
}

fn get_dataset_by_id(db: &Database, id: i64) -> Result<Dataset> {
    let result = db.conn.query_row(
        "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets WHERE id = ?1",
        params![id],
        row_to_dataset,
    );
    match result {
        Ok(ds) => Ok(ds),
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            Err(DmanError::DatasetNotFound(format!("id={}", id)))
        }
        Err(e) => Err(DmanError::Database(e)),
    }
}

fn dataset_exists(db: &Database, name: &str) -> Result<bool> {
    let count: i64 = db.conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE name = ?1",
        params![name],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

struct RawSample {
    id: i64,
    name: String,
    metadata: Option<String>,
}

struct RawAsset {
    id: i64,
    asset_type: String,
    file_name: String,
    file_path: String,
    width: Option<i64>,
    height: Option<i64>,
    hash: Option<String>,
    metadata: Option<String>,
}

struct RawAnnotation {
    asset_id: Option<i64>,
    category_id: Option<i64>,
    bbox: Option<String>,
    segmentation: Option<String>,
    keypoints: Option<String>,
    metadata: Option<String>,
}

fn fetch_samples(db: &Database, dataset_id: i64) -> Result<Vec<RawSample>> {
    let mut stmt = db
        .conn
        .prepare("SELECT id, name, metadata FROM samples WHERE dataset_id = ?1 ORDER BY id")?;
    let samples = stmt
        .query_map(params![dataset_id], |row| {
            Ok(RawSample {
                id: row.get(0)?,
                name: row.get(1)?,
                metadata: row.get(2)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DmanError::Database)?;
    Ok(samples)
}

fn fetch_assets(db: &Database, sample_id: i64) -> Result<Vec<RawAsset>> {
    let mut stmt = db.conn.prepare(
        "SELECT id, asset_type, file_name, file_path, width, height, hash, metadata FROM assets WHERE sample_id = ?1 ORDER BY id",
    )?;
    let assets = stmt
        .query_map(params![sample_id], |row| {
            Ok(RawAsset {
                id: row.get(0)?,
                asset_type: row.get(1)?,
                file_name: row.get(2)?,
                file_path: row.get(3)?,
                width: row.get(4)?,
                height: row.get(5)?,
                hash: row.get(6)?,
                metadata: row.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DmanError::Database)?;
    Ok(assets)
}

fn fetch_annotations(db: &Database, sample_id: i64) -> Result<Vec<RawAnnotation>> {
    let mut stmt = db.conn.prepare(
        "SELECT asset_id, category_id, bbox, segmentation, keypoints, metadata FROM annotations WHERE sample_id = ?1 ORDER BY id",
    )?;
    let anns = stmt
        .query_map(params![sample_id], |row| {
            Ok(RawAnnotation {
                asset_id: row.get(0)?,
                category_id: row.get(1)?,
                bbox: row.get(2)?,
                segmentation: row.get(3)?,
                keypoints: row.get(4)?,
                metadata: row.get(5)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(DmanError::Database)?;
    Ok(anns)
}

/// Copy a sample (with its assets and annotations) into a new dataset.
/// Returns the new sample_id.
/// `asset_id_map` maps old asset_id -> new asset_id for FK maintenance.
fn copy_sample(db: &Database, sample: &RawSample, new_dataset_id: i64) -> Result<i64> {
    db.conn.execute(
        "INSERT INTO samples (dataset_id, name, metadata) VALUES (?1, ?2, ?3)",
        params![new_dataset_id, sample.name, sample.metadata],
    )?;
    let new_sample_id = db.conn.last_insert_rowid();

    let assets = fetch_assets(db, sample.id)?;
    let mut asset_id_map: HashMap<i64, i64> = HashMap::new();

    for asset in &assets {
        db.conn.execute(
            "INSERT INTO assets (sample_id, asset_type, file_name, file_path, width, height, hash, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                new_sample_id,
                asset.asset_type,
                asset.file_name,
                asset.file_path,
                asset.width,
                asset.height,
                asset.hash,
                asset.metadata
            ],
        )?;
        let new_asset_id = db.conn.last_insert_rowid();
        asset_id_map.insert(asset.id, new_asset_id);
    }

    let anns = fetch_annotations(db, sample.id)?;
    for ann in &anns {
        let new_asset_id = ann.asset_id.and_then(|aid| asset_id_map.get(&aid).copied());
        db.conn.execute(
            "INSERT INTO annotations (sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                new_sample_id,
                new_asset_id,
                ann.category_id,
                ann.bbox,
                ann.segmentation,
                ann.keypoints,
                ann.metadata
            ],
        )?;
    }

    Ok(new_sample_id)
}

/// Physical dataset operations: rename, duplicate, merge, split.
pub struct DatasetOps;

impl DatasetOps {
    /// Rename a dataset.
    ///
    /// Returns `DmanError::DatasetNotFound` if `old_name` does not exist.
    /// Returns `DmanError::DatasetAlreadyExists` if `new_name` already exists.
    pub fn rename(db: &Database, old_name: &str, new_name: &str) -> Result<()> {
        // Check old exists
        if !dataset_exists(db, old_name)? {
            return Err(DmanError::DatasetNotFound(old_name.to_string()));
        }
        // Check new doesn't exist
        if dataset_exists(db, new_name)? {
            return Err(DmanError::DatasetAlreadyExists(new_name.to_string()));
        }

        db.conn.execute("BEGIN IMMEDIATE", [])?;

        let result = (|| -> Result<()> {
            db.conn.execute(
                "UPDATE datasets SET name = ?1, updated_at = CURRENT_TIMESTAMP WHERE name = ?2",
                params![new_name, old_name],
            )?;
            Ok(())
        })();

        match result {
            Ok(()) => {
                db.conn.execute("COMMIT", [])?;
                Ok(())
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    /// Duplicate a dataset.
    ///
    /// Creates a new dataset record with `new_name` (same format/path/metadata).
    /// Copies all samples + assets (reference paths) and their annotations.
    /// Returns the new `Dataset`.
    pub fn duplicate(db: &Database, name: &str, new_name: &str) -> Result<Dataset> {
        let src = get_dataset_by_name(db, name)?;

        if dataset_exists(db, new_name)? {
            return Err(DmanError::DatasetAlreadyExists(new_name.to_string()));
        }

        db.conn.execute("BEGIN IMMEDIATE", [])?;

        let result = (|| -> Result<i64> {
            let format_str = src.format.to_string();
            let path_str = src.path.to_string_lossy().to_string();
            let schema_str = src
                .schema_path
                .as_ref()
                .map(|p| p.to_string_lossy().to_string());
            let metadata_str = src
                .metadata
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;

            db.conn.execute(
                "INSERT INTO datasets (name, path, format, schema_path, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![new_name, path_str, format_str, schema_str, metadata_str],
            )?;
            let new_dataset_id = db.conn.last_insert_rowid();

            let samples = fetch_samples(db, src.id)?;
            for sample in &samples {
                copy_sample(db, sample, new_dataset_id)?;
            }

            Ok(new_dataset_id)
        })();

        match result {
            Ok(new_dataset_id) => {
                db.conn.execute("COMMIT", [])?;
                get_dataset_by_id(db, new_dataset_id)
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    /// Merge multiple datasets into a new output dataset.
    ///
    /// Creates a new dataset record with `output_name`.
    /// For each source dataset, copies all samples (reference paths) and their annotations.
    /// Returns the new `Dataset`.
    pub fn merge(db: &Database, names: &[&str], output_name: &str) -> Result<Dataset> {
        if names.is_empty() {
            return Err(DmanError::InvalidInput(
                "merge requires at least one source dataset".to_string(),
            ));
        }

        // Validate all source datasets exist and check output doesn't
        let mut sources = Vec::with_capacity(names.len());
        for &name in names {
            sources.push(get_dataset_by_name(db, name)?);
        }

        if dataset_exists(db, output_name)? {
            return Err(DmanError::DatasetAlreadyExists(output_name.to_string()));
        }

        // Pick format from first source
        let first = &sources[0];
        let format_str = first.format.to_string();
        let path_str = first.path.to_string_lossy().to_string();

        db.conn.execute("BEGIN IMMEDIATE", [])?;

        let result = (|| -> Result<i64> {
            db.conn.execute(
                "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
                params![output_name, path_str, format_str],
            )?;
            let new_dataset_id = db.conn.last_insert_rowid();

            for src in &sources {
                let samples = fetch_samples(db, src.id)?;
                for sample in &samples {
                    copy_sample(db, sample, new_dataset_id)?;
                }
            }

            Ok(new_dataset_id)
        })();

        match result {
            Ok(new_dataset_id) => {
                db.conn.execute("COMMIT", [])?;
                get_dataset_by_id(db, new_dataset_id)
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }

    /// Split a dataset into multiple sub-datasets based on named ratios.
    ///
    /// `ratios` must sum to approximately 1.0 (within 0.01 tolerance).
    /// Samples are distributed deterministically using `hash(sample_id XOR seed)`.
    /// New datasets are named `"{name}_{split_name}"`.
    /// All assets of a sample stay with that sample — never splits a sample's assets.
    /// Returns vec of new Datasets in ratio key-sorted order.
    pub fn split(
        db: &Database,
        name: &str,
        ratios: HashMap<String, f64>,
        seed: u64,
    ) -> Result<Vec<Dataset>> {
        if ratios.is_empty() {
            return Err(DmanError::InvalidInput(
                "split ratios map must not be empty".to_string(),
            ));
        }

        // Validate ratios sum
        let total: f64 = ratios.values().sum();
        if (total - 1.0_f64).abs() > 0.01 {
            return Err(DmanError::InvalidInput(format!(
                "split ratios must sum to 1.0 (got {:.4})",
                total
            )));
        }

        let src = get_dataset_by_name(db, name)?;

        // Sort ratio_vec by key for deterministic ordering
        let mut ratio_vec: Vec<(String, f64)> = ratios.into_iter().collect();
        ratio_vec.sort_by(|a, b| a.0.cmp(&b.0));

        // Check none of the output names exist
        for (split_name, _) in &ratio_vec {
            let out_name = format!("{}_{}", name, split_name);
            if dataset_exists(db, &out_name)? {
                return Err(DmanError::DatasetAlreadyExists(out_name));
            }
        }

        // Fetch all samples
        let mut samples = fetch_samples(db, src.id)?;

        // Sort samples by deterministic hash
        samples.sort_by_key(|s| {
            let mut hasher = DefaultHasher::new();
            (s.id as u64 ^ seed).hash(&mut hasher);
            hasher.finish()
        });

        let n = samples.len();

        // Build cumulative thresholds
        let mut cumulative = Vec::with_capacity(ratio_vec.len());
        let mut cum = 0.0_f64;
        for (_, ratio) in &ratio_vec {
            cum += ratio;
            cumulative.push(cum);
        }

        // Assign each sample to a bucket
        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); ratio_vec.len()];
        for (i, _) in samples.iter().enumerate() {
            let frac = if n > 0 {
                (i as f64 + 0.5) / n as f64
            } else {
                0.0
            };
            let mut assigned = ratio_vec.len() - 1; // default to last bucket
            for (bucket_idx, &threshold) in cumulative.iter().enumerate() {
                if frac < threshold {
                    assigned = bucket_idx;
                    break;
                }
            }
            buckets[assigned].push(i);
        }

        let format_str = src.format.to_string();
        let path_str = src.path.to_string_lossy().to_string();

        db.conn.execute("BEGIN IMMEDIATE", [])?;

        let result = (|| -> Result<Vec<i64>> {
            let mut new_ids = Vec::with_capacity(ratio_vec.len());

            for (bucket_idx, (split_name, _)) in ratio_vec.iter().enumerate() {
                let out_name = format!("{}_{}", name, split_name);

                db.conn.execute(
                    "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
                    params![out_name, path_str, format_str],
                )?;
                let new_dataset_id = db.conn.last_insert_rowid();
                new_ids.push(new_dataset_id);

                for &sample_idx in &buckets[bucket_idx] {
                    let sample = &samples[sample_idx];
                    copy_sample(db, sample, new_dataset_id)?;
                }
            }

            Ok(new_ids)
        })();

        match result {
            Ok(new_ids) => {
                db.conn.execute("COMMIT", [])?;
                let mut datasets = Vec::with_capacity(new_ids.len());
                for id in new_ids {
                    datasets.push(get_dataset_by_id(db, id)?);
                }
                Ok(datasets)
            }
            Err(e) => {
                let _ = db.conn.execute("ROLLBACK", []);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    /// Insert a dataset record directly (bypasses path existence check)
    fn insert_dataset(db: &Database, name: &str, path: &str, format: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
                params![name, path, format],
            )
            .expect("insert dataset");
        db.conn.last_insert_rowid()
    }

    /// Insert a sample and one default image asset. Returns the sample_id.
    fn insert_sample(db: &Database, dataset_id: i64, file_name: &str, file_path: &str) -> i64 {
        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                params![dataset_id, file_name],
            )
            .expect("insert sample");
        let sample_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, ?2, ?3, ?4)",
                params![sample_id, "image", file_name, file_path],
            )
            .expect("insert asset");
        sample_id
    }

    fn insert_annotation(db: &Database, sample_id: i64, category_id: Option<i64>) -> i64 {
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, category_id) VALUES (?1, ?2)",
                params![sample_id, category_id],
            )
            .expect("insert annotation");
        db.conn.last_insert_rowid()
    }

    fn count_samples(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM samples WHERE dataset_id = ?1",
                params![dataset_id],
                |r| r.get(0),
            )
            .unwrap()
    }

    fn count_annotations_for_dataset(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
                params![dataset_id],
                |r| r.get(0),
            )
            .unwrap()
    }

    // ─────────────────────────────────────────────────────────────────────────
    // rename tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn rename_basic() {
        let db = in_memory_db();
        insert_dataset(&db, "alpha", "/tmp/alpha", "Yolo");

        DatasetOps::rename(&db, "alpha", "beta").expect("rename should succeed");

        // old name gone
        let err = get_dataset_by_name(&db, "alpha").unwrap_err();
        assert!(matches!(err, DmanError::DatasetNotFound(_)));

        // new name present
        let ds = get_dataset_by_name(&db, "beta").expect("beta should exist");
        assert_eq!(ds.name, "beta");
    }

    #[test]
    fn rename_not_found() {
        let db = in_memory_db();
        let err = DatasetOps::rename(&db, "ghost", "new-name").unwrap_err();
        assert!(
            matches!(err, DmanError::DatasetNotFound(_)),
            "expected DatasetNotFound, got {:?}",
            err
        );
    }

    #[test]
    fn rename_target_already_exists() {
        let db = in_memory_db();
        insert_dataset(&db, "src", "/tmp/src", "Yolo");
        insert_dataset(&db, "dst", "/tmp/dst", "Coco");

        let err = DatasetOps::rename(&db, "src", "dst").unwrap_err();
        assert!(
            matches!(err, DmanError::DatasetAlreadyExists(_)),
            "expected DatasetAlreadyExists, got {:?}",
            err
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // duplicate tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn duplicate_basic() {
        let db = in_memory_db();
        let src_id = insert_dataset(&db, "original", "/tmp/orig", "Coco");

        // Add 3 samples with annotations
        let s1 = insert_sample(&db, src_id, "a.jpg", "/data/a.jpg");
        let s2 = insert_sample(&db, src_id, "b.jpg", "/data/b.jpg");
        let s3 = insert_sample(&db, src_id, "c.jpg", "/data/c.jpg");
        insert_annotation(&db, s1, Some(1));
        insert_annotation(&db, s1, Some(2));
        insert_annotation(&db, s2, Some(1));
        insert_annotation(&db, s3, None);

        let new_ds = DatasetOps::duplicate(&db, "original", "copy").expect("duplicate");

        assert_eq!(new_ds.name, "copy");
        assert_ne!(new_ds.id, src_id);

        // Both exist with same sample count
        assert_eq!(count_samples(&db, src_id), 3);
        assert_eq!(count_samples(&db, new_ds.id), 3);

        // Annotations copied
        assert_eq!(count_annotations_for_dataset(&db, src_id), 4);
        assert_eq!(count_annotations_for_dataset(&db, new_ds.id), 4);
    }

    #[test]
    fn duplicate_not_found() {
        let db = in_memory_db();
        let err = DatasetOps::duplicate(&db, "ghost", "new-copy").unwrap_err();
        assert!(matches!(err, DmanError::DatasetNotFound(_)));
    }

    #[test]
    fn duplicate_target_exists() {
        let db = in_memory_db();
        insert_dataset(&db, "src", "/tmp/src", "Yolo");
        insert_dataset(&db, "dst", "/tmp/dst", "Yolo");

        let err = DatasetOps::duplicate(&db, "src", "dst").unwrap_err();
        assert!(matches!(err, DmanError::DatasetAlreadyExists(_)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // merge tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn merge_two() {
        let db = in_memory_db();
        let ds1 = insert_dataset(&db, "ds1", "/tmp/ds1", "Yolo");
        let ds2 = insert_dataset(&db, "ds2", "/tmp/ds2", "Yolo");

        insert_sample(&db, ds1, "a.jpg", "/data/a.jpg");
        insert_sample(&db, ds1, "b.jpg", "/data/b.jpg");
        let s_c = insert_sample(&db, ds2, "c.jpg", "/data/c.jpg");
        insert_annotation(&db, s_c, Some(1));

        let merged = DatasetOps::merge(&db, &["ds1", "ds2"], "merged").expect("merge");

        assert_eq!(merged.name, "merged");
        // Union of all samples: 2 + 1 = 3
        assert_eq!(count_samples(&db, merged.id), 3);
        // Union of annotations: 0 + 1 = 1
        assert_eq!(count_annotations_for_dataset(&db, merged.id), 1);

        // Source datasets untouched
        assert_eq!(count_samples(&db, ds1), 2);
        assert_eq!(count_samples(&db, ds2), 1);
    }

    #[test]
    fn merge_output_already_exists() {
        let db = in_memory_db();
        insert_dataset(&db, "a", "/tmp/a", "Yolo");
        insert_dataset(&db, "b", "/tmp/b", "Yolo");
        insert_dataset(&db, "merged", "/tmp/merged", "Yolo");

        let err = DatasetOps::merge(&db, &["a", "b"], "merged").unwrap_err();
        assert!(matches!(err, DmanError::DatasetAlreadyExists(_)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // split tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn split_two_way() {
        let db = in_memory_db();
        let src_id = insert_dataset(&db, "full", "/tmp/full", "Yolo");

        // Insert 100 samples
        for i in 0..100_i64 {
            insert_sample(
                &db,
                src_id,
                &format!("img_{}.jpg", i),
                &format!("/data/img_{}.jpg", i),
            );
        }

        let mut ratios = HashMap::new();
        ratios.insert("train".to_string(), 0.8);
        ratios.insert("val".to_string(), 0.2);

        let splits = DatasetOps::split(&db, "full", ratios, 42).expect("split");
        assert_eq!(splits.len(), 2);

        let total_split: i64 = splits.iter().map(|ds| count_samples(&db, ds.id)).sum();
        assert_eq!(total_split, 100, "all samples should be assigned");

        // Check approximate ratios (within 5%)
        for ds in &splits {
            let count = count_samples(&db, ds.id);
            let ratio = count as f64 / 100.0;
            let (split_name, expected_ratio) = if ds.name.ends_with("_train") {
                ("train", 0.8)
            } else {
                ("val", 0.2)
            };
            assert!(
                (ratio - expected_ratio).abs() < 0.05,
                "split '{}' ratio {:.2} not within 5% of expected {:.2}",
                split_name,
                ratio,
                expected_ratio
            );
        }

        // Names follow convention
        let names: Vec<_> = splits.iter().map(|d| d.name.as_str()).collect();
        assert!(names.contains(&"full_train"), "should have full_train");
        assert!(names.contains(&"full_val"), "should have full_val");
    }

    #[test]
    fn split_invalid_ratios() {
        let db = in_memory_db();
        insert_dataset(&db, "ds", "/tmp/ds", "Yolo");

        let mut ratios = HashMap::new();
        ratios.insert("a".to_string(), 0.5);
        ratios.insert("b".to_string(), 0.3); // sum = 0.8, not 1.0

        let err = DatasetOps::split(&db, "ds", ratios, 0).unwrap_err();
        assert!(
            matches!(err, DmanError::InvalidInput(_)),
            "expected InvalidInput for bad ratios, got {:?}",
            err
        );
    }

    #[test]
    fn split_three_way() {
        let db = in_memory_db();
        let src_id = insert_dataset(&db, "big", "/tmp/big", "Coco");

        for i in 0..100_i64 {
            insert_sample(&db, src_id, &format!("{}.jpg", i), &format!("/d/{}.jpg", i));
        }

        let mut ratios = HashMap::new();
        ratios.insert("train".to_string(), 0.7);
        ratios.insert("val".to_string(), 0.2);
        ratios.insert("test".to_string(), 0.1);

        let splits = DatasetOps::split(&db, "big", ratios, 123).expect("split3");
        assert_eq!(splits.len(), 3);

        let total: i64 = splits.iter().map(|ds| count_samples(&db, ds.id)).sum();
        assert_eq!(total, 100);
    }
}
