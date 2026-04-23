use std::collections::HashMap;
use std::io::Write as IoWrite;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;

use axum::{
    Json,
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::{IntoResponse, Response},
};
use dman_core::{
    Annotation, Asset, Category,
    dataset::DatasetService,
    db::Database,
    embeddings::EmbeddingService,
    error::DmanError,
    formats::FormatRegistry,
    patches::PatchService,
    predictions::PredictionService,
    storage::StorageManager,
    types::{Embedding, Prediction, Sample, VirtualDataset, VirtualDatasetDef},
    virtual_dataset::VirtualDatasetService,
};
use rusqlite::params;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// ─── AppState ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct AppState {
    pub catalog_path: PathBuf,
}

// ─── Pagination ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct PaginationParams {
    #[serde(default = "default_page")]
    pub page: usize,
    #[serde(default = "default_per_page")]
    pub per_page: usize,
}

fn default_page() -> usize {
    1
}
fn default_per_page() -> usize {
    50
}

#[derive(Debug, Serialize)]
pub struct Pagination {
    pub page: usize,
    pub per_page: usize,
    pub total: usize,
}

fn paginate<T>(items: Vec<T>, page: usize, per_page: usize) -> (Vec<T>, Pagination) {
    let total = items.len();
    let page = page.max(1);
    let per_page = per_page.clamp(1, 500);
    let offset = (page - 1) * per_page;
    let data = items.into_iter().skip(offset).take(per_page).collect();
    let pagination = Pagination {
        page,
        per_page,
        total,
    };
    (data, pagination)
}

// ─── Response helpers ────────────────────────────────────────────────────────

fn api_ok<T: Serialize>(data: T) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({ "data": data })))
}

fn api_paged<T: Serialize>(data: T, pagination: Pagination) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(json!({ "data": data, "pagination": pagination })),
    )
}

fn api_error(code: &str, message: impl Into<String>) -> (StatusCode, Json<Value>) {
    let status = match code {
        "NOT_FOUND" => StatusCode::NOT_FOUND,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(json!({ "error": { "code": code, "message": message.into() } })),
    )
}

fn open_db(catalog_path: &StdPath) -> Result<Database, (StatusCode, Json<Value>)> {
    let db_path = catalog_path.join("catalog.db");
    Database::open(&db_path).map_err(|e| api_error("DB_ERROR", e.to_string()))
}

fn handle_dman_err(e: DmanError) -> (StatusCode, Json<Value>) {
    match &e {
        DmanError::DatasetNotFound(msg) => {
            api_error("NOT_FOUND", format!("Dataset '{}' not found", msg))
        }
        _ => api_error("INTERNAL_ERROR", e.to_string()),
    }
}

// ─── Dataset handlers ────────────────────────────────────────────────────────

/// GET /api/datasets
pub async fn list_datasets(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let datasets = match DatasetService::list(&db) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };
    let (page_data, pagination) = paginate(datasets, params.page, params.per_page);
    api_paged(page_data, pagination).into_response()
}

/// GET /api/datasets/{name}
pub async fn get_dataset(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    match DatasetService::get(&db, &name) {
        Ok(ds) => api_ok(ds).into_response(),
        Err(DmanError::DatasetNotFound(_)) => {
            api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response()
        }
        Err(e) => handle_dman_err(e).into_response(),
    }
}

/// GET /api/datasets/{name}/samples?category=X&page=1&per_page=50
pub async fn list_samples(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let category_filter = query.get("category").cloned();
    let page: usize = query.get("page").and_then(|v| v.parse().ok()).unwrap_or(1);
    let per_page: usize = query
        .get("per_page")
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);

    let samples = match fetch_assets_for_dataset(&db, ds.id, category_filter.as_deref()) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let (page_data, pagination) = paginate(samples, page, per_page);
    api_paged(page_data, pagination).into_response()
}

/// GET /api/datasets/{name}/assets/{id}
pub async fn get_asset(
    State(state): State<Arc<AppState>>,
    Path((name, asset_id)): Path<(String, i64)>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    // Verify dataset exists
    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    // Fetch asset, ensure it belongs to this dataset via sample
    let asset = match db.conn.query_row(
        "SELECT a.id, a.sample_id, a.asset_type, a.file_name, a.file_path, \
                a.width, a.height, a.hash, a.metadata \
         FROM assets a \
         JOIN samples s ON s.id = a.sample_id \
         WHERE a.id = ?1 AND s.dataset_id = ?2",
        params![asset_id, ds.id],
        row_to_asset,
    ) {
        Ok(a) => a,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return api_error(
                "NOT_FOUND",
                format!("Asset {} not found in dataset '{}'", asset_id, name),
            )
            .into_response();
        }
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    // Fetch annotations for this asset's sample
    let annotations = match fetch_annotations_for_sample(&db, asset.sample_id) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    api_ok(json!({ "asset": asset, "annotations": annotations })).into_response()
}

/// GET /api/datasets/{name}/categories
pub async fn list_categories(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let mut stmt = match db.conn.prepare(
        "SELECT id, dataset_id, name, supercategory FROM categories WHERE dataset_id = ?1 ORDER BY id",
    ) {
        Ok(s) => s,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    let categories: Vec<Category> = match stmt
        .query_map(params![ds.id], |row| {
            Ok(Category {
                id: row.get(0)?,
                dataset_id: row.get(1)?,
                name: row.get(2)?,
                supercategory: row.get(3)?,
            })
        })
        .and_then(|r| r.collect::<rusqlite::Result<Vec<_>>>())
    {
        Ok(v) => v,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    api_ok(categories).into_response()
}

/// GET /api/datasets/{name}/stats
pub async fn get_dataset_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    match DatasetService::inspect(&db, &name) {
        Ok(info) => {
            let stats = json!({
                "sample_count": info.sample_count,
                "asset_count": info.asset_count,
                "annotation_count": info.annotation_count,
                "category_count": info.category_count,
            });
            api_ok(stats).into_response()
        }
        Err(DmanError::DatasetNotFound(_)) => {
            api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response()
        }
        Err(e) => handle_dman_err(e).into_response(),
    }
}

// ─── Virtual Dataset handlers ─────────────────────────────────────────────────

/// GET /api/virtual-datasets
pub async fn list_virtual_datasets(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationParams>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let vds_list = match VirtualDatasetService::list(&db) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };
    let (page_data, pagination) = paginate(vds_list, params.page, params.per_page);
    api_paged(page_data, pagination).into_response()
}

/// GET /api/virtual-datasets/{name}
pub async fn get_virtual_dataset(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    match VirtualDatasetService::get(&db, &name) {
        Ok(vds) => api_ok(vds).into_response(),
        Err(DmanError::DatasetNotFound(_)) => {
            api_error("NOT_FOUND", format!("Virtual dataset '{}' not found", name)).into_response()
        }
        Err(e) => handle_dman_err(e).into_response(),
    }
}

// ─── DB query helpers ─────────────────────────────────────────────────────────

fn row_to_asset(row: &rusqlite::Row<'_>) -> rusqlite::Result<Asset> {
    use dman_core::types::AssetType;
    use std::path::PathBuf;
    let asset_type_str: String = row.get(2)?;
    let metadata_str: Option<String> = row.get(8)?;
    let metadata = metadata_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(Asset {
        id: row.get(0)?,
        sample_id: row.get(1)?,
        asset_type: asset_type_str
            .parse()
            .unwrap_or(AssetType::Other(asset_type_str.clone())),
        file_name: row.get(3)?,
        file_path: PathBuf::from(row.get::<_, String>(4)?),
        width: row.get(5)?,
        height: row.get(6)?,
        hash: row.get(7)?,
        metadata,
    })
}

fn row_to_annotation(row: &rusqlite::Row<'_>) -> rusqlite::Result<Annotation> {
    let bbox_str: Option<String> = row.get(4)?;
    let bbox = bbox_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let seg_str: Option<String> = row.get(5)?;
    let segmentation = seg_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    let kp_str: Option<String> = row.get(6)?;
    let keypoints = kp_str.as_deref().and_then(|s| serde_json::from_str(s).ok());
    let meta_str: Option<String> = row.get(7)?;
    let metadata = meta_str
        .as_deref()
        .and_then(|s| serde_json::from_str(s).ok());
    Ok(Annotation {
        id: row.get(0)?,
        sample_id: row.get(1)?,
        asset_id: row.get(2)?,
        category_id: row.get(3)?,
        bbox,
        segmentation,
        keypoints,
        metadata,
    })
}

fn fetch_assets_for_dataset(
    db: &Database,
    dataset_id: i64,
    category: Option<&str>,
) -> dman_core::Result<Vec<Asset>> {
    match category {
        None => {
            let mut stmt = db.conn.prepare(
                "SELECT a.id, a.sample_id, a.asset_type, a.file_name, a.file_path, \
                        a.width, a.height, a.hash, a.metadata \
                 FROM assets a \
                 JOIN samples s ON s.id = a.sample_id \
                 WHERE s.dataset_id = ?1 \
                 ORDER BY a.id",
            )?;
            let assets = stmt
                .query_map(params![dataset_id], row_to_asset)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(assets)
        }
        Some(cat) => {
            let mut stmt = db.conn.prepare(
                "SELECT DISTINCT a.id, a.sample_id, a.asset_type, a.file_name, a.file_path, \
                        a.width, a.height, a.hash, a.metadata \
                 FROM assets a \
                 JOIN samples s ON s.id = a.sample_id \
                 JOIN annotations ann ON ann.sample_id = s.id \
                 JOIN categories c ON c.id = ann.category_id \
                 WHERE s.dataset_id = ?1 AND c.name = ?2 \
                 ORDER BY a.id",
            )?;
            let assets = stmt
                .query_map(params![dataset_id, cat], row_to_asset)?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            Ok(assets)
        }
    }
}

fn fetch_annotations_for_sample(
    db: &Database,
    sample_id: i64,
) -> dman_core::Result<Vec<Annotation>> {
    let mut stmt = db.conn.prepare(
        "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
         FROM annotations WHERE sample_id = ?1 ORDER BY id",
    )?;
    let annotations = stmt
        .query_map(params![sample_id], row_to_annotation)?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(annotations)
}

// ─── Sample detail handler ────────────────────────────────────────────────────

/// GET /api/datasets/{name}/samples/{sample_id}
pub async fn get_sample(
    State(state): State<Arc<AppState>>,
    Path((name, sample_id)): Path<(String, i64)>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    // Fetch the sample and verify it belongs to this dataset.
    let sample = match db.conn.query_row(
        "SELECT id, dataset_id, name, metadata, created_at \
         FROM samples WHERE id = ?1 AND dataset_id = ?2",
        params![sample_id, ds.id],
        |row| {
            let meta_str: Option<String> = row.get(3)?;
            let metadata = meta_str.as_deref().and_then(|s| serde_json::from_str(s).ok());
            Ok(Sample {
                id: row.get(0)?,
                dataset_id: row.get(1)?,
                name: row.get(2)?,
                metadata,
                created_at: row.get(4)?,
            })
        },
    ) {
        Ok(s) => s,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return api_error(
                "NOT_FOUND",
                format!("Sample {} not found in dataset '{}'", sample_id, name),
            )
            .into_response();
        }
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    let assets = match DatasetService::get_assets(&db, sample_id) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let annotations = match fetch_annotations_for_sample(&db, sample_id) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    api_ok(json!({ "sample": sample, "assets": assets, "annotations": annotations }))
        .into_response()
}

// ─── Embeddings handlers ──────────────────────────────────────────────────────

/// GET /api/datasets/{name}/embeddings?model=<name>&page=1&per_page=50
pub async fn list_dataset_embeddings(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let model_filter = query.get("model").cloned();
    let page: usize = query.get("page").and_then(|v| v.parse().ok()).unwrap_or(1);
    let per_page: usize = query.get("per_page").and_then(|v| v.parse().ok()).unwrap_or(50);

    let embeddings = match EmbeddingService::list_by_dataset(&db, ds.id) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let filtered: Vec<Embedding> = match model_filter {
        Some(ref m) => embeddings.into_iter().filter(|e| &e.model_name == m).collect(),
        None => embeddings,
    };

    let (page_data, pagination) = paginate(filtered, page, per_page);
    api_paged(page_data, pagination).into_response()
}

/// GET /api/assets/{id}/embeddings?model=<name>
pub async fn list_asset_embeddings(
    State(state): State<Arc<AppState>>,
    Path(asset_id): Path<i64>,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    let model_filter = query.get("model").cloned();

    // Fetch all embeddings for this asset from DB.
    let mut stmt = match db.conn.prepare(
        "SELECT id, asset_id, model_name, vector, metadata \
         FROM embeddings WHERE asset_id = ?1 ORDER BY id",
    ) {
        Ok(s) => s,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    let rows: rusqlite::Result<Vec<Embedding>> = stmt
        .query_map(params![asset_id], |row| {
            let blob: Vec<u8> = row.get(3)?;
            let vector: Vec<f32> = blob
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let meta_str: Option<String> = row.get(4)?;
            let metadata = meta_str.as_deref().and_then(|s| serde_json::from_str(s).ok());
            Ok(Embedding {
                id: row.get(0)?,
                asset_id: row.get(1)?,
                model_name: row.get(2)?,
                vector,
                metadata,
            })
        })
        .and_then(|r| r.collect());

    let embeddings = match rows {
        Ok(v) => v,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    let filtered: Vec<Embedding> = match model_filter {
        Some(ref m) => embeddings.into_iter().filter(|e| &e.model_name == m).collect(),
        None => embeddings,
    };

    api_ok(filtered).into_response()
}

// ─── Predictions handlers ─────────────────────────────────────────────────────

/// GET /api/datasets/{name}/predictions?model_version=<v>&page=1&per_page=50
pub async fn list_dataset_predictions(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let model_version = query.get("model_version").cloned();
    let page: usize = query.get("page").and_then(|v| v.parse().ok()).unwrap_or(1);
    let per_page: usize = query.get("per_page").and_then(|v| v.parse().ok()).unwrap_or(50);

    let predictions: Vec<Prediction> = match model_version {
        Some(ref mv) => {
            match PredictionService::get_by_model(&db, ds.id, mv) {
                Ok(v) => v,
                Err(e) => return handle_dman_err(e).into_response(),
            }
        }
        None => {
            match PredictionService::get_by_dataset(&db, ds.id) {
                Ok(v) => v,
                Err(e) => return handle_dman_err(e).into_response(),
            }
        }
    };

    let (page_data, pagination) = paginate(predictions, page, per_page);
    api_paged(page_data, pagination).into_response()
}

// ─── Patches handlers ─────────────────────────────────────────────────────────

/// GET /api/datasets/{name}/patches?page=1&per_page=50
pub async fn list_dataset_patches(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<PaginationParams>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let ds = match DatasetService::get(&db, &name) {
        Ok(d) => d,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Dataset '{}' not found", name)).into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let patches = match PatchService::get_by_dataset(&db, ds.id) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let (page_data, pagination) = paginate(patches, params.page, params.per_page);
    api_paged(page_data, pagination).into_response()
}

/// GET /api/assets/{id}/patches
pub async fn list_asset_patches(
    State(state): State<Arc<AppState>>,
    Path(asset_id): Path<i64>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    match PatchService::get_by_asset(&db, asset_id) {
        Ok(patches) => api_ok(patches).into_response(),
        Err(e) => handle_dman_err(e).into_response(),
    }
}

/// GET /api/assets/{id}/annotations
pub async fn list_asset_annotations(
    State(state): State<Arc<AppState>>,
    Path(asset_id): Path<i64>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    // Resolve asset → sample_id to reuse existing helper.
    let sample_id: i64 = match db.conn.query_row(
        "SELECT sample_id FROM assets WHERE id = ?1",
        params![asset_id],
        |r| r.get(0),
    ) {
        Ok(id) => id,
        Err(rusqlite::Error::QueryReturnedNoRows) => {
            return api_error("NOT_FOUND", format!("Asset {} not found", asset_id)).into_response();
        }
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    // Return only annotations scoped to this specific asset.
    let mut stmt = match db.conn.prepare(
        "SELECT id, sample_id, asset_id, category_id, bbox, segmentation, keypoints, metadata \
         FROM annotations WHERE asset_id = ?1 ORDER BY id",
    ) {
        Ok(s) => s,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    let _ = sample_id; // kept for future sample-scoped filtering
    let annotations: Vec<Annotation> = match stmt
        .query_map(params![asset_id], row_to_annotation)
        .and_then(|r| r.collect::<rusqlite::Result<Vec<_>>>())
    {
        Ok(v) => v,
        Err(e) => return handle_dman_err(DmanError::Database(e)).into_response(),
    };

    api_ok(annotations).into_response()
}

// ─── Export handler ───────────────────────────────────────────────────────────

/// GET /api/datasets/{name}/export?format=yolo|coco|huggingface
pub async fn export_dataset(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(query): Query<HashMap<String, String>>,
) -> Response<Body> {
    let format_id = match query.get("format") {
        Some(f) => f.clone(),
        None => {
            let resp: Response<Body> = (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"code": "BAD_REQUEST", "message": "Missing required query parameter: format"}})),
            )
                .into_response();
            return resp;
        }
    };

    // Validate format is known before spinning up a thread.
    {
        let registry = FormatRegistry::default_registry();
        if registry.get_exporter(&format_id).is_none() {
            let msg = format!(
                "Unsupported export format '{}'. Built-in options: yolo, coco, huggingface",
                format_id
            );
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({"error": {"code": "BAD_REQUEST", "message": msg}})),
            )
                .into_response();
        }
    }

    let catalog_path = state.catalog_path.clone();
    let name_clone = name.clone();
    let fmt_clone = format_id.clone();

    let result = tokio::task::spawn_blocking(move || {
        run_export(&catalog_path, &name_clone, &fmt_clone)
    })
    .await;

    match result {
        Ok(Ok(zip_bytes)) => {
            let content_disposition =
                format!("attachment; filename=\"{}-{}.zip\"", name, format_id);
            let mut headers = HeaderMap::new();
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/zip"),
            );
            headers.insert(
                header::CONTENT_DISPOSITION,
                HeaderValue::from_str(&content_disposition)
                    .unwrap_or(HeaderValue::from_static("attachment")),
            );
            (StatusCode::OK, headers, zip_bytes).into_response()
        }
        Ok(Err(resp)) => resp,
        Err(join_err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "INTERNAL_ERROR", "message": join_err.to_string()}})),
        )
            .into_response(),
    }
}

fn run_export(
    catalog_path: &StdPath,
    dataset_name: &str,
    format_id: &str,
) -> Result<Vec<u8>, Response<Body>> {
    let db_path = catalog_path.join("catalog.db");
    let db = Database::open(&db_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "DB_ERROR", "message": e.to_string()}})),
        )
            .into_response()
    })?;

    let ds = DatasetService::get(&db, dataset_name).map_err(|e| match &e {
        DmanError::DatasetNotFound(_) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": {"code": "NOT_FOUND", "message": format!("Dataset '{}' not found", dataset_name)}})),
        )
            .into_response(),
        _ => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "INTERNAL_ERROR", "message": e.to_string()}})),
        )
            .into_response(),
    })?;

    let storage = StorageManager::new(catalog_path.join("data"));
    let registry = FormatRegistry::default_registry();
    let exporter = registry.get_exporter(format_id).ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": {"code": "BAD_REQUEST", "message": format!("No exporter found for format '{}'", format_id)}})),
        )
            .into_response()
    })?;

    let tmp = tempfile::tempdir().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "INTERNAL_ERROR", "message": e.to_string()}})),
        )
            .into_response()
    })?;
    let export_dir = tmp.path().join("export");
    std::fs::create_dir_all(&export_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "INTERNAL_ERROR", "message": e.to_string()}})),
        )
            .into_response()
    })?;

    exporter
        .export(&db, &storage, &ds, &export_dir)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(
                    json!({"error": {"code": "INTERNAL_ERROR", "message": format!("Export failed: {}", e)}}),
                ),
            )
                .into_response()
        })?;

    zip_directory(&export_dir).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"code": "INTERNAL_ERROR", "message": format!("Zip failed: {}", e)}})),
        )
            .into_response()
    })
}

fn zip_directory(dir: &StdPath) -> std::io::Result<Vec<u8>> {
    use std::io::Read;

    let buf = Vec::new();
    let cursor = std::io::Cursor::new(buf);
    let mut zip = zip::ZipWriter::new(cursor);
    let options: zip::write::FileOptions<'_, ()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    let walkdir = walkdir::WalkDir::new(dir).min_depth(1);
    for entry in walkdir {
        let entry = entry.map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::Other, e.to_string())
        })?;
        let path = entry.path();
        let relative = path.strip_prefix(dir).unwrap_or(path);
        let name = relative.to_string_lossy();

        if path.is_dir() {
            let dir_name = format!("{}/", name);
            zip.add_directory(dir_name, options.clone())?;
        } else {
            zip.start_file(name.as_ref(), options.clone())?;
            let mut f = std::fs::File::open(path)?;
            let mut contents = Vec::new();
            f.read_to_end(&mut contents)?;
            zip.write_all(&contents)?;
        }
    }
    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

// ─── Virtual dataset sample evaluation handlers ───────────────────────────────

/// GET /api/virtual-datasets/{name}/samples?page=1&per_page=50
pub async fn list_virtual_dataset_samples(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Query(params): Query<PaginationParams>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let vds = match VirtualDatasetService::get(&db, &name) {
        Ok(v) => v,
        Err(DmanError::DatasetNotFound(_)) => {
            return api_error("NOT_FOUND", format!("Virtual dataset '{}' not found", name))
                .into_response();
        }
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let samples = match VirtualDatasetService::evaluate(&db, &vds) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let (page_data, pagination) = paginate(samples, params.page, params.per_page);
    api_paged(page_data, pagination).into_response()
}

/// POST /api/virtual-datasets/evaluate
#[derive(Debug, Deserialize)]
pub struct EvaluateRequest {
    pub source_datasets: Vec<i64>,
    pub definition: VirtualDatasetDef,
    pub limit: Option<usize>,
}

pub async fn evaluate_virtual_dataset(
    State(state): State<Arc<AppState>>,
    Json(body): Json<EvaluateRequest>,
) -> impl IntoResponse {
    let db = match open_db(&state.catalog_path) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };

    let vds = VirtualDataset {
        id: 0,
        name: String::new(),
        source_datasets: body.source_datasets,
        definition: body.definition,
    };

    let samples = match VirtualDatasetService::evaluate(&db, &vds) {
        Ok(v) => v,
        Err(e) => return handle_dman_err(e).into_response(),
    };

    let result: Vec<Sample> = match body.limit {
        Some(n) => samples.into_iter().take(n).collect(),
        None => samples,
    };

    api_ok(result).into_response()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use axum::routing::{get, post};
    use dman_core::dataset::DatasetService;
    use dman_core::db::Database;
    use dman_core::types::DatasetFormat;
    use dman_core::types::VirtualDatasetDef;
    use dman_core::virtual_dataset::VirtualDatasetService;
    use rusqlite::params;
    use tempfile::tempdir;
    use tower::ServiceExt;

    use super::*;

    fn setup_catalog(tmp: &tempfile::TempDir) -> PathBuf {
        let catalog_path = tmp.path().to_path_buf();
        // Create catalog.db
        let db_path = catalog_path.join("catalog.db");
        Database::open(&db_path).expect("create catalog db");
        catalog_path
    }

    fn make_router(catalog_path: PathBuf) -> Router {
        let state = Arc::new(AppState { catalog_path });
        Router::new()
            .route("/api/datasets", get(list_datasets))
            .route("/api/datasets/{name}", get(get_dataset))
            .route("/api/datasets/{name}/samples", get(list_samples))
            .route("/api/datasets/{name}/samples/{sample_id}", get(get_sample))
            .route("/api/datasets/{name}/assets/{id}", get(get_asset))
            .route("/api/datasets/{name}/categories", get(list_categories))
            .route("/api/datasets/{name}/stats", get(get_dataset_stats))
            .route("/api/datasets/{name}/embeddings", get(list_dataset_embeddings))
            .route("/api/datasets/{name}/predictions", get(list_dataset_predictions))
            .route("/api/datasets/{name}/patches", get(list_dataset_patches))
            .route("/api/datasets/{name}/export", get(export_dataset))
            .route("/api/assets/{id}/embeddings", get(list_asset_embeddings))
            .route("/api/assets/{id}/patches", get(list_asset_patches))
            .route("/api/assets/{id}/annotations", get(list_asset_annotations))
            .route("/api/virtual-datasets", get(list_virtual_datasets))
            .route("/api/virtual-datasets/{name}", get(get_virtual_dataset))
            .route("/api/virtual-datasets/{name}/samples", get(list_virtual_dataset_samples))
            .route("/api/virtual-datasets/evaluate", post(evaluate_virtual_dataset))
            .with_state(state)
    }

    async fn body_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    // ─── Test 1: list_datasets empty ─────────────────────────────────────────
    #[tokio::test]
    async fn api_list_datasets_empty() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let app = make_router(catalog);

        let resp = app
            .oneshot(Request::get("/api/datasets").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"], serde_json::json!([]));
        assert_eq!(json["pagination"]["total"], 0);
        assert_eq!(json["pagination"]["page"], 1);
    }

    // ─── Test 2: list_datasets with data ─────────────────────────────────────
    #[tokio::test]
    async fn api_list_datasets_with_data() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("myds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        // Register a dataset
        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "my-dataset", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(Request::get("/api/datasets").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 1);
        assert_eq!(json["data"][0]["name"], "my-dataset");
    }

    // ─── Test 3: get_dataset 404 ──────────────────────────────────────────────
    #[tokio::test]
    async fn api_get_dataset_not_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let app = make_router(catalog);

        let resp = app
            .oneshot(
                Request::get("/api/datasets/nonexistent")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "NOT_FOUND");
    }

    // ─── Test 4: list_samples with category filter ────────────────────────────
    #[tokio::test]
    async fn api_list_samples_category_filter() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("filterds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "filter-ds", &ds_dir, DatasetFormat::coco()).unwrap();
        let dataset_id = ds.id;

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, 'sample-cat')",
                params![dataset_id],
            )
            .unwrap();
        let sample1_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'Image', 'cat.jpg', '/tmp/cat.jpg')",
                params![sample1_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, 'sample-dog')",
                params![dataset_id],
            )
            .unwrap();
        let sample2_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'Image', 'dog.jpg', '/tmp/dog.jpg')",
                params![sample2_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, 'sample-empty')",
                params![dataset_id],
            )
            .unwrap();
        let sample3_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'Image', 'empty.jpg', '/tmp/empty.jpg')",
                params![sample3_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'cat')",
                params![dataset_id],
            )
            .unwrap();
        let cat_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'dog')",
                params![dataset_id],
            )
            .unwrap();
        let dog_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, category_id) VALUES (?1, ?2)",
                params![sample1_id, cat_id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, category_id) VALUES (?1, ?2)",
                params![sample2_id, dog_id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/filter-ds/samples?category=cat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 1);
        assert_eq!(json["data"][0]["file_name"], "cat.jpg");
    }

    // ─── Test 5: list_samples no filter returns all ───────────────────────────
    #[tokio::test]
    async fn api_list_samples_no_filter() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("allimgds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "all-img-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let dataset_id = ds.id;

        for i in 0..3 {
            db.conn
                .execute(
                    "INSERT INTO samples (dataset_id, name) VALUES (?1, ?2)",
                    params![dataset_id, format!("sample-{}", i)],
                )
                .unwrap();
            let sample_id = db.conn.last_insert_rowid();
            db.conn
                .execute(
                    "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'Image', ?2, ?3)",
                    params![
                        sample_id,
                        format!("img{}.jpg", i),
                        format!("/tmp/img{}.jpg", i)
                    ],
                )
                .unwrap();
        }
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/all-img-ds/samples")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 3);
    }

    // ─── Test 6: list_samples for nonexistent dataset returns 404 ────────────
    #[tokio::test]
    async fn api_list_samples_dataset_not_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let app = make_router(catalog);

        let resp = app
            .oneshot(
                Request::get("/api/datasets/ghost/samples")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "NOT_FOUND");
    }

    // ─── Test 7: list_categories ──────────────────────────────────────────────
    #[tokio::test]
    async fn api_list_categories() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("catds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds = DatasetService::register(&db, "cat-ds", &ds_dir, DatasetFormat::coco()).unwrap();
        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name, supercategory) VALUES (?1, 'cat', 'animal')",
                params![ds.id],
            )
            .unwrap();
        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'dog')",
                params![ds.id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/cat-ds/categories")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        let names: Vec<&str> = data.iter().map(|v| v["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"cat"));
        assert!(names.contains(&"dog"));
    }

    // ─── Test 8: dataset stats ────────────────────────────────────────────────
    #[tokio::test]
    async fn api_dataset_stats() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("statsds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds = DatasetService::register(&db, "stats-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let dataset_id = ds.id;

        db.conn
            .execute(
                "INSERT INTO samples (dataset_id, name) VALUES (?1, 'sample-a')",
                params![dataset_id],
            )
            .unwrap();
        let sample_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) VALUES (?1, 'Image', 'a.jpg', '/tmp/a.jpg')",
                params![sample_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'cat')",
                params![dataset_id],
            )
            .unwrap();

        db.conn
            .execute(
                "INSERT INTO annotations (sample_id) VALUES (?1)",
                params![sample_id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/stats-ds/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"]["sample_count"], 1);
        assert_eq!(json["data"]["asset_count"], 1);
        assert_eq!(json["data"]["annotation_count"], 1);
        assert_eq!(json["data"]["category_count"], 1);
    }

    // ─── Test 9: list_virtual_datasets empty ─────────────────────────────────
    #[tokio::test]
    async fn api_list_virtual_datasets_empty() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let app = make_router(catalog);

        let resp = app
            .oneshot(
                Request::get("/api/virtual-datasets")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"], serde_json::json!([]));
        assert_eq!(json["pagination"]["total"], 0);
    }

    // ─── Test 10: get_virtual_dataset 404 ────────────────────────────────────
    #[tokio::test]
    async fn api_get_virtual_dataset_not_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let app = make_router(catalog);

        let resp = app
            .oneshot(
                Request::get("/api/virtual-datasets/ghost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "NOT_FOUND");
    }

    // ─── Test 11: get_virtual_dataset found ──────────────────────────────────
    #[tokio::test]
    async fn api_get_virtual_dataset_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        // Insert a real dataset first (needed for source_datasets)
        db.conn
            .execute(
                "INSERT INTO datasets (name, path, format) VALUES ('src-ds', '/tmp', 'Yolo')",
                [],
            )
            .unwrap();
        let ds_id = db.conn.last_insert_rowid();
        let def = VirtualDatasetDef::Sample { ratio: 0.5 };
        VirtualDatasetService::create(&db, "my-vds", vec![ds_id], &def).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/virtual-datasets/my-vds")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"]["name"], "my-vds");
    }

    // ─── Test 12: get_asset 404 ───────────────────────────────────────────────
    #[tokio::test]
    async fn api_get_asset_not_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("imgds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "img-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/img-ds/assets/9999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ─── Test 13: pagination ──────────────────────────────────────────────────
    #[tokio::test]
    async fn api_list_datasets_pagination() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("pgds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        for i in 0..5 {
            DatasetService::register(&db, &format!("ds-{}", i), &ds_dir, DatasetFormat::yolo())
                .unwrap();
        }
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets?page=2&per_page=2")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 5);
        assert_eq!(json["pagination"]["page"], 2);
        assert_eq!(json["pagination"]["per_page"], 2);
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
    }

    // ─── Test 14: get_sample found ────────────────────────────────────────────
    #[tokio::test]
    async fn api_get_sample_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("sampleds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds = DatasetService::register(&db, "sample-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s001", None).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get(format!("/api/datasets/sample-ds/samples/{}", sample_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"]["sample"]["name"], "s001");
    }

    // ─── Test 15: get_sample not found ────────────────────────────────────────
    #[tokio::test]
    async fn api_get_sample_not_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("sds2");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "sds2", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/sds2/samples/9999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    // ─── Test 16: list_dataset_embeddings empty ───────────────────────────────
    #[tokio::test]
    async fn api_list_embeddings_empty() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("embds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "emb-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/emb-ds/embeddings")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"], serde_json::json!([]));
        assert_eq!(json["pagination"]["total"], 0);
    }

    // ─── Test 17: list_dataset_predictions empty ──────────────────────────────
    #[tokio::test]
    async fn api_list_predictions_empty() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("predds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "pred-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/pred-ds/predictions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"], serde_json::json!([]));
    }

    // ─── Test 18: list_dataset_patches empty ──────────────────────────────────
    #[tokio::test]
    async fn api_list_patches_empty() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("patchds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "patch-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/patch-ds/patches")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"], serde_json::json!([]));
    }

    // ─── Test 21: get_asset found ─────────────────────────────────────────────
    #[tokio::test]
    async fn api_get_asset_found() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("assetds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds = DatasetService::register(&db, "asset-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id =
            DatasetService::add_sample(&db, ds.id, "sample-a", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get(format!("/api/datasets/asset-ds/assets/{}", asset_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"]["asset"]["id"], asset_id);
        assert_eq!(json["data"]["asset"]["file_name"], "img.jpg");
    }

    // ─── Test 22: list_dataset_embeddings with data ───────────────────────────
    #[tokio::test]
    async fn api_list_embeddings_with_data() {
        use dman_core::embeddings::EmbeddingService;

        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("embwithdata");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "emb-data-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        EmbeddingService::store(&db, asset_id, "clip-vit", &[0.1_f32, 0.2, 0.3]).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/emb-data-ds/embeddings")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 1);
        assert_eq!(json["data"][0]["model_name"], "clip-vit");
    }

    // ─── Test 23: list_asset_embeddings ───────────────────────────────────────
    #[tokio::test]
    async fn api_list_asset_embeddings() {
        use dman_core::embeddings::EmbeddingService;

        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("assetemb");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "asset-emb-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        EmbeddingService::store(&db, asset_id, "resnet", &[1.0_f32, 2.0]).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get(format!("/api/assets/{}/embeddings", asset_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["data"][0]["model_name"], "resnet");
    }

    // ─── Test 24: list_dataset_predictions with data ──────────────────────────
    #[tokio::test]
    async fn api_list_predictions_with_data() {
        use dman_core::predictions::PredictionService;

        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("predwithdata");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "pred-data-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        PredictionService::store(
            &db,
            sample_id,
            "v1.0",
            serde_json::json!({"label": "cat"}),
            Some(0.95),
        )
        .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/pred-data-ds/predictions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["data"][0]["model_version"], "v1.0");
    }

    // ─── Test 25: list_dataset_patches with data ──────────────────────────────
    #[tokio::test]
    async fn api_list_patches_with_data() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("patchwithdata");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "patch-data-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        // Insert a patch row directly (extract requires a real image file on disk)
        db.conn
            .execute(
                "INSERT INTO patches (asset_id, bbox, file_path) \
                 VALUES (?1, '{\"x\":0,\"y\":0,\"width\":10,\"height\":10}', '/tmp/patch.jpg')",
                params![asset_id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/patch-data-ds/patches")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
    }

    // ─── Test 26: list_asset_patches ──────────────────────────────────────────
    #[tokio::test]
    async fn api_list_asset_patches() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("assetpatches");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "asset-patch-ds", &ds_dir, DatasetFormat::yolo())
                .unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO patches (asset_id, bbox, file_path) \
                 VALUES (?1, '{\"x\":5,\"y\":5,\"width\":20,\"height\":20}', '/tmp/patch2.jpg')",
                params![asset_id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get(format!("/api/assets/{}/patches", asset_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
    }

    // ─── Test 27: list_asset_annotations ─────────────────────────────────────
    #[tokio::test]
    async fn api_list_asset_annotations() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("assetanns");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "asset-ann-ds", &ds_dir, DatasetFormat::coco()).unwrap();
        let sample_id = DatasetService::add_sample(&db, ds.id, "s1", None).unwrap();
        db.conn
            .execute(
                "INSERT INTO assets (sample_id, asset_type, file_name, file_path) \
                 VALUES (?1, 'Image', 'img.jpg', '/tmp/img.jpg')",
                params![sample_id],
            )
            .unwrap();
        let asset_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO categories (dataset_id, name) VALUES (?1, 'dog')",
                params![ds.id],
            )
            .unwrap();
        let cat_id = db.conn.last_insert_rowid();
        db.conn
            .execute(
                "INSERT INTO annotations (sample_id, asset_id, category_id) \
                 VALUES (?1, ?2, ?3)",
                params![sample_id, asset_id, cat_id],
            )
            .unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get(format!("/api/assets/{}/annotations", asset_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["data"].as_array().unwrap().len(), 1);
        assert_eq!(json["data"][0]["category_id"], cat_id);
    }

    // ─── Test 28: export_dataset — missing format param returns 400 ───────────
    #[tokio::test]
    async fn api_export_dataset_missing_format() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("exportds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "export-ds", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/export-ds/export")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "BAD_REQUEST");
    }

    // ─── Test 29: export_dataset — unsupported format returns 400 ────────────
    #[tokio::test]
    async fn api_export_dataset_unsupported_format() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("exportds2");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "export-ds2", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/export-ds2/export?format=notaformat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let json = body_json(resp).await;
        assert_eq!(json["error"]["code"], "BAD_REQUEST");
    }

    // ─── Test 30: export_dataset — valid format returns zip bytes ────────────
    #[tokio::test]
    async fn api_export_dataset_yolo_returns_zip() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("exportyolo");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        DatasetService::register(&db, "export-yolo", &ds_dir, DatasetFormat::yolo()).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/datasets/export-yolo/export?format=yolo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .unwrap(),
            "application/zip"
        );
        // Verify body starts with PK (ZIP magic bytes)
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert!(bytes.starts_with(b"PK"), "expected ZIP magic bytes");
    }

    // ─── Test 19: evaluate_virtual_dataset ad-hoc ─────────────────────────────
    #[tokio::test]
    async fn api_evaluate_ad_hoc_virtual_dataset() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("vdseval");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds = DatasetService::register(&db, "vds-eval", &ds_dir, DatasetFormat::yolo()).unwrap();
        for i in 0..4 {
            DatasetService::add_sample(&db, ds.id, &format!("s{}", i), None).unwrap();
        }
        let ds_id = ds.id;
        drop(db);

        let body = serde_json::json!({
            "source_datasets": [ds_id],
            "definition": { "Sample": { "ratio": 0.5 } }
        });

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::post("/api/virtual-datasets/evaluate")
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        // 4 samples × 0.5 ratio = 2
        assert_eq!(json["data"].as_array().unwrap().len(), 2);
    }

    // ─── Test 20: evaluate stored virtual dataset samples ────────────────────
    #[tokio::test]
    async fn api_evaluate_stored_vds_samples() {
        let tmp = tempdir().unwrap();
        let catalog = setup_catalog(&tmp);
        let ds_dir = tmp.path().join("storedevalvds");
        std::fs::create_dir_all(&ds_dir).unwrap();

        let db = Database::open(catalog.join("catalog.db")).unwrap();
        let ds =
            DatasetService::register(&db, "stored-vds-src", &ds_dir, DatasetFormat::yolo())
                .unwrap();
        for i in 0..6 {
            DatasetService::add_sample(&db, ds.id, &format!("s{}", i), None).unwrap();
        }
        let def = VirtualDatasetDef::Sample { ratio: 1.0 };
        VirtualDatasetService::create(&db, "stored-vds", vec![ds.id], &def).unwrap();
        drop(db);

        let app = make_router(catalog);
        let resp = app
            .oneshot(
                Request::get("/api/virtual-datasets/stored-vds/samples")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        let json = body_json(resp).await;
        assert_eq!(json["pagination"]["total"], 6);
    }
}
