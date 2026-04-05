use std::collections::HashMap;

use crate::{Result, db::Database, types::Prediction};

pub struct PredictionService;

pub struct ComparisonReport {
    pub models: Vec<String>,
    pub per_sample: Vec<SampleComparison>,
}

pub struct SampleComparison {
    pub sample_id: i64,
    pub predictions: HashMap<String, serde_json::Value>,
}

impl PredictionService {
    pub fn store(
        db: &Database,
        sample_id: i64,
        model_version: &str,
        result: serde_json::Value,
        score: Option<f64>,
    ) -> Result<i64> {
        let result_str = serde_json::to_string(&result)?;
        db.conn.execute(
            "INSERT INTO predictions (sample_id, model_version, result, score) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![sample_id, model_version, result_str, score],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    pub fn get_by_sample(db: &Database, sample_id: i64) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, sample_id, asset_id, model_version, result, score \
             FROM predictions \
             WHERE sample_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![sample_id], |row| {
            let id: i64 = row.get(0)?;
            let sid: i64 = row.get(1)?;
            let asset_id: Option<i64> = row.get(2)?;
            let model_version: String = row.get(3)?;
            let result_str: String = row.get(4)?;
            let score: Option<f64> = row.get(5)?;
            Ok((id, sid, asset_id, model_version, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, sid, asset_id, model_version, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                sample_id: sid,
                asset_id,
                model_version,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    pub fn get_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT p.id, p.sample_id, p.asset_id, p.model_version, p.result, p.score \
             FROM predictions p \
             WHERE p.sample_id IN (SELECT id FROM samples WHERE dataset_id = ?1)",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id], |row| {
            let id: i64 = row.get(0)?;
            let sample_id: i64 = row.get(1)?;
            let asset_id: Option<i64> = row.get(2)?;
            let model_version: String = row.get(3)?;
            let result_str: String = row.get(4)?;
            let score: Option<f64> = row.get(5)?;
            Ok((id, sample_id, asset_id, model_version, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, sample_id, asset_id, model_version, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                sample_id,
                asset_id,
                model_version,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    pub fn get_by_model(
        db: &Database,
        dataset_id: i64,
        model_version: &str,
    ) -> Result<Vec<Prediction>> {
        let mut stmt = db.conn.prepare(
            "SELECT p.id, p.sample_id, p.asset_id, p.model_version, p.result, p.score \
             FROM predictions p \
             JOIN samples s ON p.sample_id = s.id \
             WHERE s.dataset_id = ?1 AND p.model_version = ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id, model_version], |row| {
            let id: i64 = row.get(0)?;
            let sample_id: i64 = row.get(1)?;
            let asset_id: Option<i64> = row.get(2)?;
            let mv: String = row.get(3)?;
            let result_str: String = row.get(4)?;
            let score: Option<f64> = row.get(5)?;
            Ok((id, sample_id, asset_id, mv, result_str, score))
        })?;

        let mut predictions = Vec::new();
        for row in rows {
            let (id, sample_id, asset_id, mv, result_str, score) = row?;
            let result: serde_json::Value = serde_json::from_str(&result_str)?;
            predictions.push(Prediction {
                id,
                sample_id,
                asset_id,
                model_version: mv,
                result,
                score,
            });
        }
        Ok(predictions)
    }

    pub fn delete_by_model(db: &Database, dataset_id: i64, model_version: &str) -> Result<()> {
        db.conn.execute(
            "DELETE FROM predictions \
             WHERE model_version = ?1 \
             AND sample_id IN (SELECT id FROM samples WHERE dataset_id = ?2)",
            rusqlite::params![model_version, dataset_id],
        )?;
        Ok(())
    }

    pub fn compare_models(
        db: &Database,
        dataset_id: i64,
        models: &[&str],
    ) -> Result<ComparisonReport> {
        let mut per_sample_map: HashMap<i64, HashMap<String, serde_json::Value>> = HashMap::new();

        for &model in models {
            let preds = Self::get_by_model(db, dataset_id, model)?;
            for pred in preds {
                per_sample_map
                    .entry(pred.sample_id)
                    .or_default()
                    .insert(pred.model_version, pred.result);
            }
        }

        let mut per_sample: Vec<SampleComparison> = per_sample_map
            .into_iter()
            .map(|(sample_id, predictions)| SampleComparison {
                sample_id,
                predictions,
            })
            .collect();

        per_sample.sort_by_key(|sc| sc.sample_id);

        Ok(ComparisonReport {
            models: models.iter().map(|s| s.to_string()).collect(),
            per_sample,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn seed_sample(db: &Database, dataset_name: &str) -> (i64, i64) {
        db.conn
            .execute(
                "INSERT INTO datasets (name, path) VALUES (?1, ?2)",
                rusqlite::params![dataset_name, "/tmp/test"],
            )
            .expect("insert dataset");
        let dataset_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                rusqlite::params![dataset_id, "sample-001"],
            )
            .expect("insert sample");
        let sample_id = db.conn.last_insert_rowid();

        (dataset_id, sample_id)
    }

    fn seed_second_sample(db: &Database, dataset_id: i64) -> i64 {
        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                rusqlite::params![dataset_id, "sample-002"],
            )
            .expect("insert second sample");
        db.conn.last_insert_rowid()
    }

    #[test]
    fn test_store_and_get_by_sample() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, sample_id) = seed_sample(&db, "ds-store");

        let result = serde_json::json!({"labels": ["cat", "dog"], "confidence": 0.95});
        let pred_id = PredictionService::store(&db, sample_id, "v1.0", result.clone(), Some(0.95))
            .expect("store");
        assert!(pred_id > 0);

        let preds = PredictionService::get_by_sample(&db, sample_id).expect("get_by_sample");
        assert_eq!(preds.len(), 1);
        let pred = &preds[0];
        assert_eq!(pred.sample_id, sample_id);
        assert_eq!(pred.model_version, "v1.0");
        assert_eq!(pred.result, result, "result JSON must round-trip exactly");
        assert_eq!(pred.score, Some(0.95));
    }

    #[test]
    fn test_store_round_trip_preserves_json() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, sample_id) = seed_sample(&db, "ds-roundtrip");

        let complex_result = serde_json::json!({
            "detections": [
                {"bbox": [10, 20, 100, 80], "class": "car", "score": 0.87},
                {"bbox": [200, 150, 50, 60], "class": "person", "score": 0.92}
            ],
            "metadata": {"model": "yolov8", "version": "1.2.3"}
        });

        PredictionService::store(&db, sample_id, "yolov8-1.2.3", complex_result.clone(), None)
            .expect("store complex");

        let preds = PredictionService::get_by_sample(&db, sample_id).expect("get");
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
        let (dataset_id, sample_id1) = seed_sample(&db, "ds-dataset");
        let sample_id2 = seed_second_sample(&db, dataset_id);

        PredictionService::store(&db, sample_id1, "v1.0", serde_json::json!({"a": 1}), None)
            .expect("store s1");
        PredictionService::store(&db, sample_id2, "v1.0", serde_json::json!({"b": 2}), None)
            .expect("store s2");

        let preds = PredictionService::get_by_dataset(&db, dataset_id).expect("get_by_dataset");
        assert_eq!(preds.len(), 2);

        let sample_ids: Vec<i64> = preds.iter().map(|p| p.sample_id).collect();
        assert!(sample_ids.contains(&sample_id1));
        assert!(sample_ids.contains(&sample_id2));
    }

    #[test]
    fn test_get_by_model_filters_correctly() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, sample_id) = seed_sample(&db, "ds-model-filter");

        PredictionService::store(
            &db,
            sample_id,
            "v1.0",
            serde_json::json!({"model": "v1"}),
            Some(0.8),
        )
        .expect("store v1");
        PredictionService::store(
            &db,
            sample_id,
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
        let (dataset_id, sample_id) = seed_sample(&db, "ds-delete-model");

        PredictionService::store(&db, sample_id, "v1.0", serde_json::json!({"x": 1}), None)
            .expect("store v1");
        PredictionService::store(&db, sample_id, "v2.0", serde_json::json!({"x": 2}), None)
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
        let (dataset_id, sample_id1) = seed_sample(&db, "ds-compare");
        let sample_id2 = seed_second_sample(&db, dataset_id);

        let v1_result1 = serde_json::json!({"pred": "cat"});
        let v1_result2 = serde_json::json!({"pred": "dog"});
        let v2_result1 = serde_json::json!({"pred": "kitten"});
        let v2_result2 = serde_json::json!({"pred": "puppy"});

        PredictionService::store(&db, sample_id1, "v1.0", v1_result1.clone(), None)
            .expect("store v1 s1");
        PredictionService::store(&db, sample_id2, "v1.0", v1_result2.clone(), None)
            .expect("store v1 s2");
        PredictionService::store(&db, sample_id1, "v2.0", v2_result1.clone(), None)
            .expect("store v2 s1");
        PredictionService::store(&db, sample_id2, "v2.0", v2_result2.clone(), None)
            .expect("store v2 s2");

        let report =
            PredictionService::compare_models(&db, dataset_id, &["v1.0", "v2.0"]).expect("compare");

        assert_eq!(report.models, vec!["v1.0", "v2.0"]);
        assert_eq!(report.per_sample.len(), 2);

        let sc1 = report
            .per_sample
            .iter()
            .find(|sc| sc.sample_id == sample_id1)
            .expect("sample1");
        assert_eq!(sc1.predictions.get("v1.0"), Some(&v1_result1));
        assert_eq!(sc1.predictions.get("v2.0"), Some(&v2_result1));

        let sc2 = report
            .per_sample
            .iter()
            .find(|sc| sc.sample_id == sample_id2)
            .expect("sample2");
        assert_eq!(sc2.predictions.get("v1.0"), Some(&v1_result2));
        assert_eq!(sc2.predictions.get("v2.0"), Some(&v2_result2));
    }

    #[test]
    fn test_get_by_sample_empty() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, sample_id) = seed_sample(&db, "ds-empty-sample");

        let preds = PredictionService::get_by_sample(&db, sample_id).expect("get empty");
        assert!(preds.is_empty());
    }
}
