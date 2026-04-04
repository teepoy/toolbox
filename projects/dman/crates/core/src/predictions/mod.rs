use std::collections::HashMap;

use crate::{db::Database, types::Prediction, Result};

/// Stateless service for storing and retrieving ML model predictions.
///
/// All methods take a `&Database` reference; no instance state is held.
/// The predictions table schema:
///   `predictions(id INTEGER PK, image_id INTEGER FK, model_version TEXT, result TEXT, score REAL)`
pub struct PredictionService;

/// A comparison of predictions across multiple models for a single dataset.
pub struct ComparisonReport {
    /// The model versions included in this comparison.
    pub models: Vec<String>,
    /// Per-image comparisons: each entry maps model versions to their prediction result.
    pub per_image: Vec<ImageComparison>,
}

/// Predictions from multiple models for a single image.
pub struct ImageComparison {
    /// The image's database ID.
    pub image_id: i64,
    /// Maps model_version → prediction result JSON.
    pub predictions: HashMap<String, serde_json::Value>,
}

impl PredictionService {
    /// Store a prediction for the given image and model version.
    ///
    /// `result` is serialized to a JSON string for storage.
    /// Returns the newly inserted row ID.
    pub fn store(
        db: &Database,
        image_id: i64,
        model_version: &str,
        result: serde_json::Value,
        score: Option<f64>,
    ) -> Result<i64> {
        let result_str = serde_json::to_string(&result)?;
        db.conn.execute(
            "INSERT INTO predictions (image_id, model_version, result, score) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![image_id, model_version, result_str, score],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    /// Retrieve all predictions stored for the given image.
    pub fn get_by_image(db: &Database, image_id: i64) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, image_id, model_version, result, score \
             FROM predictions \
             WHERE image_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![image_id], |row| {
            let id: i64 = row.get(0)?;
            let img_id: i64 = row.get(1)?;
            let model_version: String = row.get(2)?;
            let result_str: String = row.get(3)?;
            let score: Option<f64> = row.get(4)?;
            Ok((id, img_id, model_version, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, img_id, model_version, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                image_id: img_id,
                model_version,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    /// Retrieve all predictions for images belonging to the given dataset.
    pub fn get_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT p.id, p.image_id, p.model_version, p.result, p.score \
             FROM predictions p \
             WHERE p.image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id], |row| {
            let id: i64 = row.get(0)?;
            let image_id: i64 = row.get(1)?;
            let model_version: String = row.get(2)?;
            let result_str: String = row.get(3)?;
            let score: Option<f64> = row.get(4)?;
            Ok((id, image_id, model_version, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, image_id, model_version, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                image_id,
                model_version,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    /// Retrieve all predictions for images in the given dataset filtered by model version.
    pub fn get_by_model(
        db: &Database,
        dataset_id: i64,
        model_version: &str,
    ) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT p.id, p.image_id, p.model_version, p.result, p.score \
             FROM predictions p \
             JOIN images i ON p.image_id = i.id \
             WHERE i.dataset_id = ?1 AND p.model_version = ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id, model_version], |row| {
            let id: i64 = row.get(0)?;
            let image_id: i64 = row.get(1)?;
            let mv: String = row.get(2)?;
            let result_str: String = row.get(3)?;
            let score: Option<f64> = row.get(4)?;
            Ok((id, image_id, mv, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, image_id, mv, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                image_id,
                model_version: mv,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    /// Delete all predictions for the given model version within a dataset.
    pub fn delete_by_model(db: &Database, dataset_id: i64, model_version: &str) -> Result<()> {
        db.conn.execute(
            "DELETE FROM predictions \
             WHERE model_version = ?1 \
             AND image_id IN (SELECT id FROM images WHERE dataset_id = ?2)",
            rusqlite::params![model_version, dataset_id],
        )?;
        Ok(())
    }

    /// Compare predictions across multiple models for all images in a dataset.
    ///
    /// Returns a [`ComparisonReport`] with per-image maps of `model_version → result`.
    pub fn compare_models(
        db: &Database,
        dataset_id: i64,
        models: &[&str],
    ) -> Result<ComparisonReport> {
        let mut per_image_map: HashMap<i64, HashMap<String, serde_json::Value>> = HashMap::new();

        for &model in models {
            let preds = Self::get_by_model(db, dataset_id, model)?;
            for pred in preds {
                per_image_map
                    .entry(pred.image_id)
                    .or_default()
                    .insert(pred.model_version, pred.result);
            }
        }

        let mut per_image: Vec<ImageComparison> = per_image_map
            .into_iter()
            .map(|(image_id, predictions)| ImageComparison {
                image_id,
                predictions,
            })
            .collect();

        per_image.sort_by_key(|ic| ic.image_id);

        Ok(ComparisonReport {
            models: models.iter().map(|s| s.to_string()).collect(),
            per_image,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn seed_image(db: &Database, dataset_name: &str) -> (i64, i64) {
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                rusqlite::params![dataset_name, "/tmp/test"],
            )
            .expect("insert dataset");
        let dataset_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "img001.jpg", "/tmp/img001.jpg"],
            )
            .expect("insert image");
        let image_id = db.conn.last_insert_rowid();

        (dataset_id, image_id)
    }

    fn seed_second_image(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, ?2, ?3)",
                rusqlite::params![dataset_id, "img002.jpg", "/tmp/img002.jpg"],
            )
            .expect("insert second image");
        db.conn.last_insert_rowid()
    }

    #[test]
    fn test_store_and_get_by_image() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, image_id) = seed_image(&db, "ds-store");

        let result = serde_json::json!({"labels": ["cat", "dog"], "confidence": 0.95});
        let pred_id = PredictionService::store(&db, image_id, "v1.0", result.clone(), Some(0.95))
            .expect("store");
        assert!(pred_id > 0);

        let preds = PredictionService::get_by_image(&db, image_id).expect("get_by_image");
        assert_eq!(preds.len(), 1);
        let pred = &preds[0];
        assert_eq!(pred.image_id, image_id);
        assert_eq!(pred.model_version, "v1.0");
        assert_eq!(pred.result, result, "result JSON must round-trip exactly");
        assert_eq!(pred.score, Some(0.95));
    }

    #[test]
    fn test_store_round_trip_preserves_json() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, image_id) = seed_image(&db, "ds-roundtrip");

        let complex_result = serde_json::json!({
            "detections": [
                {"bbox": [10, 20, 100, 80], "class": "car", "score": 0.87},
                {"bbox": [200, 150, 50, 60], "class": "person", "score": 0.92}
            ],
            "metadata": {"model": "yolov8", "version": "1.2.3"}
        });

        PredictionService::store(&db, image_id, "yolov8-1.2.3", complex_result.clone(), None)
            .expect("store complex");

        let preds = PredictionService::get_by_image(&db, image_id).expect("get");
        assert_eq!(preds.len(), 1);
        assert_eq!(
            preds[0].result, complex_result,
            "complex JSON must round-trip without data loss"
        );
        assert_eq!(preds[0].score, None);
    }

    #[test]
    fn test_get_by_dataset() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id1) = seed_image(&db, "ds-dataset");
        let image_id2 = seed_second_image(&db, dataset_id);

        PredictionService::store(&db, image_id1, "v1.0", serde_json::json!({"a": 1}), None)
            .expect("store img1");
        PredictionService::store(&db, image_id2, "v1.0", serde_json::json!({"b": 2}), None)
            .expect("store img2");

        let preds = PredictionService::get_by_dataset(&db, dataset_id).expect("get_by_dataset");
        assert_eq!(preds.len(), 2);

        let image_ids: Vec<i64> = preds.iter().map(|p| p.image_id).collect();
        assert!(image_ids.contains(&image_id1));
        assert!(image_ids.contains(&image_id2));
    }

    #[test]
    fn test_get_by_model_filters_correctly() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-model-filter");

        PredictionService::store(
            &db,
            image_id,
            "v1.0",
            serde_json::json!({"model": "v1"}),
            Some(0.8),
        )
        .expect("store v1");
        PredictionService::store(
            &db,
            image_id,
            "v2.0",
            serde_json::json!({"model": "v2"}),
            Some(0.9),
        )
        .expect("store v2");

        let v1_preds = PredictionService::get_by_model(&db, dataset_id, "v1.0").expect("get v1");
        assert_eq!(v1_preds.len(), 1);
        assert_eq!(v1_preds[0].model_version, "v1.0");

        let v2_preds = PredictionService::get_by_model(&db, dataset_id, "v2.0").expect("get v2");
        assert_eq!(v2_preds.len(), 1);
        assert_eq!(v2_preds[0].model_version, "v2.0");

        let none_preds =
            PredictionService::get_by_model(&db, dataset_id, "v99.0").expect("get v99");
        assert!(none_preds.is_empty());
    }

    #[test]
    fn test_delete_by_model() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-delete-model");

        PredictionService::store(&db, image_id, "v1.0", serde_json::json!({"x": 1}), None)
            .expect("store v1");
        PredictionService::store(&db, image_id, "v2.0", serde_json::json!({"x": 2}), None)
            .expect("store v2");

        assert_eq!(
            PredictionService::get_by_dataset(&db, dataset_id)
                .expect("before delete")
                .len(),
            2
        );

        PredictionService::delete_by_model(&db, dataset_id, "v1.0").expect("delete v1");

        let remaining = PredictionService::get_by_dataset(&db, dataset_id).expect("after delete");
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].model_version, "v2.0");
    }

    #[test]
    fn test_compare_models() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id1) = seed_image(&db, "ds-compare");
        let image_id2 = seed_second_image(&db, dataset_id);

        let v1_result1 = serde_json::json!({"pred": "cat"});
        let v1_result2 = serde_json::json!({"pred": "dog"});
        let v2_result1 = serde_json::json!({"pred": "kitten"});
        let v2_result2 = serde_json::json!({"pred": "puppy"});

        PredictionService::store(&db, image_id1, "v1.0", v1_result1.clone(), None)
            .expect("store v1 img1");
        PredictionService::store(&db, image_id2, "v1.0", v1_result2.clone(), None)
            .expect("store v1 img2");
        PredictionService::store(&db, image_id1, "v2.0", v2_result1.clone(), None)
            .expect("store v2 img1");
        PredictionService::store(&db, image_id2, "v2.0", v2_result2.clone(), None)
            .expect("store v2 img2");

        let report =
            PredictionService::compare_models(&db, dataset_id, &["v1.0", "v2.0"]).expect("compare");

        assert_eq!(report.models, vec!["v1.0", "v2.0"]);
        assert_eq!(report.per_image.len(), 2);

        let ic1 = report
            .per_image
            .iter()
            .find(|ic| ic.image_id == image_id1)
            .expect("img1");
        assert_eq!(ic1.predictions.get("v1.0"), Some(&v1_result1));
        assert_eq!(ic1.predictions.get("v2.0"), Some(&v2_result1));

        let ic2 = report
            .per_image
            .iter()
            .find(|ic| ic.image_id == image_id2)
            .expect("img2");
        assert_eq!(ic2.predictions.get("v1.0"), Some(&v1_result2));
        assert_eq!(ic2.predictions.get("v2.0"), Some(&v2_result2));
    }

    #[test]
    fn test_get_by_image_empty() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, image_id) = seed_image(&db, "ds-empty-img");

        let preds = PredictionService::get_by_image(&db, image_id).expect("get empty");
        assert!(preds.is_empty());
    }
}
