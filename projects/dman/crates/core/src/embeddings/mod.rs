use crate::{db::Database, types::Embedding, Result};

/// Stateless service for storing and retrieving image embeddings.
///
/// All methods take a `&Database` reference; no instance state is held.
/// The embeddings table schema:
///   `embeddings(id INTEGER PK, image_id INTEGER FK, model_name TEXT, vector BLOB, metadata TEXT)`
pub struct EmbeddingService;

impl EmbeddingService {
    /// Encode a `&[f32]` slice into a `Vec<u8>` using little-endian byte order.
    fn encode_vector(vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// Decode a `Vec<u8>` (little-endian f32 bytes) back into a `Vec<f32>`.
    fn decode_vector(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Store an embedding vector for the given image and model.
    ///
    /// Returns the newly inserted row id.
    pub fn store(db: &Database, image_id: i64, model_name: &str, vector: &[f32]) -> Result<i64> {
        let blob = Self::encode_vector(vector);
        db.conn.execute(
            "INSERT INTO embeddings (image_id, model_name, vector) VALUES (?1, ?2, ?3)",
            rusqlite::params![image_id, model_name, blob],
        )?;
        Ok(db.conn.last_insert_rowid())
    }

    /// Retrieve the embedding for a specific image and model.
    ///
    /// Returns `None` if no matching row is found.
    pub fn get(db: &Database, image_id: i64, model_name: &str) -> Result<Option<Embedding>> {
        let mut stmt = db.conn.prepare(
            "SELECT id, image_id, model_name, vector, metadata \
             FROM embeddings \
             WHERE image_id = ?1 AND model_name = ?2 \
             LIMIT 1",
        )?;

        let mut rows = stmt.query(rusqlite::params![image_id, model_name])?;

        if let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let img_id: i64 = row.get(1)?;
            let model: String = row.get(2)?;
            let bytes: Vec<u8> = row.get(3)?;
            let metadata_str: Option<String> = row.get(4)?;

            let metadata = metadata_str
                .as_deref()
                .map(serde_json::from_str)
                .transpose()?;

            Ok(Some(Embedding {
                id,
                image_id: img_id,
                model_name: model,
                vector: Self::decode_vector(&bytes),
                metadata,
            }))
        } else {
            Ok(None)
        }
    }

    /// List all embeddings whose source image belongs to the given dataset.
    ///
    /// Uses a JOIN through the `images` table since the `embeddings` table
    /// does not store `dataset_id` directly.
    pub fn list_by_dataset(db: &Database, dataset_id: i64) -> Result<Vec<Embedding>> {
        let mut stmt = db.conn.prepare(
            "SELECT e.id, e.image_id, e.model_name, e.vector, e.metadata \
             FROM embeddings e \
             JOIN images i ON e.image_id = i.id \
             WHERE i.dataset_id = ?1",
        )?;

        let rows = stmt.query_map(rusqlite::params![dataset_id], |row| {
            let id: i64 = row.get(0)?;
            let image_id: i64 = row.get(1)?;
            let model_name: String = row.get(2)?;
            let bytes: Vec<u8> = row.get(3)?;
            let metadata_str: Option<String> = row.get(4)?;
            Ok((id, image_id, model_name, bytes, metadata_str))
        })?;

        let mut embeddings = Vec::new();
        for row in rows {
            let (id, image_id, model_name, bytes, metadata_str) = row?;
            let metadata = metadata_str
                .as_deref()
                .map(serde_json::from_str)
                .transpose()
                .map_err(crate::DmanError::SerdeJson)?;
            embeddings.push(Embedding {
                id,
                image_id,
                model_name,
                vector: Self::decode_vector(&bytes),
                metadata,
            });
        }
        Ok(embeddings)
    }

    /// Delete all embeddings for the given image.
    pub fn delete_by_image(db: &Database, image_id: i64) -> Result<()> {
        db.conn.execute(
            "DELETE FROM embeddings WHERE image_id = ?1",
            rusqlite::params![image_id],
        )?;
        Ok(())
    }

    /// Returns `true` if the dataset has at least one embedding stored.
    ///
    /// Joins through `images` since embeddings don't carry `dataset_id`.
    pub fn has_embeddings(db: &Database, dataset_id: i64) -> Result<bool> {
        let count: i64 = db.conn.query_row(
            "SELECT COUNT(*) \
             FROM embeddings e \
             JOIN images i ON e.image_id = i.id \
             WHERE i.dataset_id = ?1",
            rusqlite::params![dataset_id],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Return the distinct model names used for embeddings in a dataset.
    pub fn get_embedding_models(db: &Database, dataset_id: i64) -> Result<Vec<String>> {
        let mut stmt = db.conn.prepare(
            "SELECT DISTINCT e.model_name \
             FROM embeddings e \
             JOIN images i ON e.image_id = i.id \
             WHERE i.dataset_id = ?1 \
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

    fn make_vector(dim: usize) -> Vec<f32> {
        (0..dim).map(|i| i as f32 * 0.01_f32).collect()
    }

    #[test]
    fn test_embeddings_store_and_retrieve() {
        let db = Database::open_in_memory().expect("open db");
        let (_dataset_id, image_id) = seed_image(&db, "ds-store");

        let vec128 = make_vector(128);
        let row_id =
            EmbeddingService::store(&db, image_id, "clip-vit-b32", &vec128).expect("store");
        assert!(row_id > 0);

        let emb = EmbeddingService::get(&db, image_id, "clip-vit-b32")
            .expect("get")
            .expect("should exist");

        assert_eq!(emb.image_id, image_id);
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
        let (_dataset_id, image_id) = seed_image(&db, "ds-miss");

        let result = EmbeddingService::get(&db, image_id, "nonexistent-model").expect("get");
        assert!(result.is_none());
    }

    #[test]
    fn test_embeddings_list_by_dataset() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-list");

        EmbeddingService::store(&db, image_id, "model-a", &make_vector(64)).expect("store a");
        EmbeddingService::store(&db, image_id, "model-b", &make_vector(64)).expect("store b");

        let list = EmbeddingService::list_by_dataset(&db, dataset_id).expect("list");
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_embeddings_list_by_dataset_empty() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, _image_id) = seed_image(&db, "ds-empty");

        let list = EmbeddingService::list_by_dataset(&db, dataset_id).expect("list");
        assert!(list.is_empty());
    }

    #[test]
    fn test_embeddings_delete_by_image() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-delete");

        EmbeddingService::store(&db, image_id, "clip", &make_vector(32)).expect("store");

        assert!(EmbeddingService::has_embeddings(&db, dataset_id).expect("has"));

        EmbeddingService::delete_by_image(&db, image_id).expect("delete");

        let result = EmbeddingService::get(&db, image_id, "clip").expect("get after delete");
        assert!(result.is_none());
        assert!(!EmbeddingService::has_embeddings(&db, dataset_id).expect("has after delete"));
    }

    #[test]
    fn test_embeddings_has_embeddings() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-has");

        assert!(!EmbeddingService::has_embeddings(&db, dataset_id).expect("before"));

        EmbeddingService::store(&db, image_id, "resnet50", &make_vector(2048)).expect("store");

        assert!(EmbeddingService::has_embeddings(&db, dataset_id).expect("after"));
    }

    #[test]
    fn test_embeddings_get_embedding_models() {
        let db = Database::open_in_memory().expect("open db");
        let (dataset_id, image_id) = seed_image(&db, "ds-models");

        EmbeddingService::store(&db, image_id, "alpha", &make_vector(8)).expect("alpha");
        EmbeddingService::store(&db, image_id, "beta", &make_vector(8)).expect("beta");
        EmbeddingService::store(&db, image_id, "alpha", &make_vector(8)).expect("alpha-dup");

        let models = EmbeddingService::get_embedding_models(&db, dataset_id).expect("models");

        // DISTINCT + ORDER BY means we get ["alpha", "beta"]
        assert_eq!(models, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[test]
    fn test_embeddings_byte_roundtrip() {
        // Verify encode→decode is lossless for edge-case floats.
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
