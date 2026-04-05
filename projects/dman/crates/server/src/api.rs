use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};
use std::sync::Arc;

use axum::{
    Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use dman_core::{
    Annotation, Asset, Category, dataset::DatasetService, db::Database, error::DmanError,
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

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use axum::Router;
    use axum::body::{Body, to_bytes};
    use axum::http::{Request, StatusCode};
    use axum::routing::get;
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
            .route("/api/datasets/{name}/assets/{id}", get(get_asset))
            .route("/api/datasets/{name}/categories", get(list_categories))
            .route("/api/datasets/{name}/stats", get(get_dataset_stats))
            .route("/api/virtual-datasets", get(list_virtual_datasets))
            .route("/api/virtual-datasets/{name}", get(get_virtual_dataset))
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
}
