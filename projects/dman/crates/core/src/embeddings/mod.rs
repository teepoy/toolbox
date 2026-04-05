use crate::{Result, db::Database, types::Embedding};

pub struct EmbeddingService;

impl EmbeddingService {
    fn encode_vector(vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn decode_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn store(db: &Database, asset_id: i64, model_name: &str, vector: &[f32]) -> Result<i64> {
        let blob = Self::encode_vector(vector);
        db.conn.execute(
            "INSERT INTO embeddings (asset_id, model_name, vector) VALUES (?1, ?2, ?3)",
            rusqlite::params![asset_id, model_name, blob],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    pub fn get(db: &Database, asset_id: i64, model_name: &str) -> Result<Option<Embedding>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, asset_id, model_name, vector, metadata \
             FROM embeddings \
             WHERE asset_id = ?1 AND model_name = ?2 \
             LIMIT 1",
        )?;

        let mut rows = stmt.query(rusqlite::params![asset_id, model_name])?;

        if let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let aid: i64 = row.get(1)?;
            let model: String = row.get(2)?;
            let bytes: Vec<u8> = row.get(3)?;
            let metadata_str: Option<String> = row.get(4)?;

            let metadata = metadata_str
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?;

            Ok(Some(Embedding {
                id,
                asset_id: aid,
                model_name: model,
                vector: Self::decode_vector(&bytes),
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn list_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Embedding>> {
        let mut stmt = db.conn.prepare(
            "SELECT e.id, e.asset_id, e.model_name, e.vector, e.metadata \
             FROM embeddings e \
             JOIN assets a ON e.asset_id = a.id \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id], |row| {
            let id: i64 = row.get(0)?;
            let asset_id: i64 = row.get(1)?;
            let model_name: String = row.get(2)?;
            let bytes: Vec<u8> = row.get(3)?;
            let metadata_str: Option<String> = row.get(4)?;
            Ok((id, asset_id, model_name, bytes, metadata_str))
        })?;

        let mut embeddings = Vec::new();
        for row in rows {
            let (id, asset_id, model_name, bytes, metadata_str) = row?;
            let metadata = metadata_str
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(crate::DmanError::SerdeJson)?;
            embeddings.push(Embedding {
                id,
                asset_id,
                model_name,
                vector: Self::decode_vector(&bytes),
                metadata,
            });
        }
        Ok(embeddings)
    }

    pub fn delete_by_asset(db: &Database, asset_id: i64) -> Result<()> {
        db.conn.execute(
            "DELETE FROM embeddings WHERE asset_id = ?1",
            rusqlite::params![asset_id],
        )?;
        Ok(())
    }

    pub fn has_embeddings(db: &Database, dataset_id: i64) -> Result<bool> {
        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) \
             FROM embeddings e \
             JOIN assets a ON e.asset_id = a.id \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1",
            rusqlite::params![dataset_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    pub fn get_embedding_models(db: &Database, dataset_id: i64) -> Result<Vec<String>> {
        let mut stmt = db.conn.prepare(
            "SELECT DISTINCT e.model_name \
             FROM embeddings e \
             JOIN assets a ON e.asset_id = a.id \
             JOIN samples s ON a.sample_id = s.id \
             WHERE s.dataset_id = ?1 \
             ORDER BY e.model_name",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id], |row| row.get::<_, String>(0))?;

        let mut models = Vec::new();
        for row in rows {
            models.push(row?);
        }
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    fn seed_asset(db: &Database, dataset_name: &str) -> (i64, i64) {
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

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'image', ?2, ?3)",
                rusqlite::params![sample_id, "img001.jpg", "/tmp/img001.jpg"],
            )
            .expect("insert asset");
        let asset_id = db.conn.last_insert_rowid();

        (dataset_id, asset_id)
    }

    fn make_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| i as f32 * 0.01_f32).collect()
    }

    #[test]
    fn test_embeddings_store_and_retrieve() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, asset_id) = seed_asset(&db, "ds-store");

        let vec128 = make_vector(128);
        let row_id =
            EmbeddingService::store(&db, asset_id, "clip-vit-b32", &vec128).expect("store");
        assert!(row_id > 0);

        let emb = EmbeddingService::get(&db, asset_id, "clip-vit-b32")
            .expect("get")
            .expect("should exist");

        assert_eq!(emb.asset_id, asset_id);
        assert_eq!(emb.model_name, "clip-vit-b32");
        assert_eq!(emb.vector.len(), 128);

        for (a, b) in emb.vector.iter().zip(vec128.iter()) {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "vector mismatch at value {b}: got {a}"
            );
        }
    }

    #[test]
    fn test_embeddings_get_missing_returns_none() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, asset_id) = seed_asset(&db, "ds-miss");

        let result = EmbeddingService::get(&db, asset_id, "nonexistent-model").expect("get");
        assert!(result.is_none());
    }

    #[test]
    fn test_embeddings_list_by_dataset() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, asset_id) = seed_asset(&db, "ds-list");

        EmbeddingService::store(&db, asset_id, "model-a", &make_vector(64)).expect("store a");
        EmbeddingService::store(&db, asset_id, "model-b", &make_vector(64)).expect("store b");

        let list = EmbeddingService::list_by_dataset(&db, dataset_id).expect("list");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_embeddings_list_by_dataset_empty() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, _asset_id) = seed_asset(&db, "ds-empty");

        let list = EmbeddingService::list_by_dataset(&db, dataset_id).expect("list");
        assert!(list.is_empty());
    }

    #[test]
    fn test_embeddings_delete_by_asset() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, asset_id) = seed_asset(&db, "ds-delete");

        EmbeddingService::store(&db, asset_id, "clip", &make_vector(32)).expect("store");

        assert!(EmbeddingService::has_embeddings(&db, dataset_id).expect("has"));

        EmbeddingService::delete_by_asset(&db, asset_id).expect("delete");

        let result = EmbeddingService::get(&db, asset_id, "clip").expect("get after delete");
        assert!(result.is_none());
        assert!(!EmbeddingService::has_embeddings(&db, dataset_id).expect("has after delete"));
    }

    #[test]
    fn test_embeddings_has_embeddings() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, asset_id) = seed_asset(&db, "ds-has");

        assert!(!EmbeddingService::has_embeddings(&db, dataset_id).expect("before"));

        EmbeddingService::store(&db, asset_id, "resnet50", &make_vector(2048)).expect("store");

        assert!(EmbeddingService::has_embeddings(&db, dataset_id).expect("after"));
    }

    #[test]
    fn test_embeddings_get_embedding_models() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, asset_id) = seed_asset(&db, "ds-models");

        EmbeddingService::store(&db, asset_id, "alpha", &make_vector(8)).expect("alpha");
        EmbeddingService::store(&db, asset_id, "beta", &make_vector(8)).expect("beta");
        EmbeddingService::store(&db, asset_id, "alpha", &make_vector(8)).expect("alpha-dup");

        let models = EmbeddingService::get_embedding_models(&db, dataset_id).expect("models");

        assert_eq!(models, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_embeddings_byte_roundtrip() {
        let values = vec![0.0_f32, 1.0, -1.0, f32::MAX, f32::MIN_POSITIVE, f32::NAN];
        let encoded = EmbeddingService::encode_vector(&values);
        let decoded = EmbeddingService::decode_vector(&encoded);

        assert_eq!(decoded.len(), values.len());
        for (a, b) in decoded.iter().zip(values.iter()) {
            if b.is_nan() {
                assert!(a.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits(), "mismatch for value {b}");
            }
        }
    }
}
