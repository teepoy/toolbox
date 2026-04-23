pub mod api;
pub mod label_studio;

use std::sync::Arc;

use axum::{
    Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, Response, StatusCode, Uri, header},
    response::IntoResponse,
    routing::{get, post},
};
use rust_embed::RustEmbed;
use serde_json::json;
use tower_http::cors::CorsLayer;

pub use api::AppState;

#[derive(RustEmbed)]
#[folder = "frontend/dist/"]
struct Frontend;

fn content_type_for_extension(ext: &str) -> &'static str {
    match ext.to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    }
}

async fn health_handler() -> impl IntoResponse {
    axum::Json(json!({"status": "ok"}))
}

async fn image_handler(
    State(state): State<Arc<AppState>>,
    Path((dataset_id, filename)): Path<(String, String)>,
    headers: HeaderMap,
) -> Response<Body> {
    let image_path = state
        .catalog_path
        .join("data")
        .join(&dataset_id)
        .join("images")
        .join(&filename);

    match tokio::fs::read(&image_path).await {
        Ok(bytes) => {
            let ext = std::path::Path::new(&filename)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let content_type = content_type_for_extension(ext);

            // Simple ETag derived from content length.
            // When the asset hash is stored in the DB it can be used instead.
            let etag_value = format!("\"{}\"", bytes.len());

            // Honour If-None-Match so downstream caches (browsers, dataloaders)
            // can avoid re-downloading unchanged files.
            if let Some(inm) = headers.get(header::IF_NONE_MATCH) {
                if inm.as_bytes() == etag_value.as_bytes() {
                    return Response::builder()
                        .status(StatusCode::NOT_MODIFIED)
                        .body(Body::empty())
                        .expect("304 response build");
                }
            }

            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, HeaderValue::from_static(content_type))
                .header(
                    header::ETAG,
                    HeaderValue::from_str(&etag_value)
                        .unwrap_or(HeaderValue::from_static("\"0\"")),
                )
                .header(
                    header::CACHE_CONTROL,
                    HeaderValue::from_static("public, max-age=86400"),
                )
                .body(Body::from(bytes))
                .unwrap_or_else(|_| {
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .expect("static response build")
                })
        }
        Err(_) => Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::empty())
            .expect("404 response build"),
    }
}

async fn spa_fallback(uri: Uri) -> Response<Body> {
    let path = uri.path().trim_start_matches('/');

    if let Some(content) = Frontend::get(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(content.data.into_owned()))
            .unwrap_or_else(|_| {
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Body::empty())
                    .expect("static response build")
            })
    } else {
        match Frontend::get("index.html") {
            Some(index) => Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
                .body(Body::from(index.data.into_owned()))
                .unwrap_or_else(|_| {
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .expect("static response build")
                }),
            None => Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
                .body(Body::from(
                    r#"<html><body><div id="root">dman</div></body></html>"#,
                ))
                .unwrap_or_else(|_| {
                    Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::empty())
                        .expect("static response build")
                }),
        }
    }
}

pub async fn create_router(state: AppState) -> Router {
    let state = Arc::new(state);
    Router::new()
        // ── System ─────────────────────────────────────────────────────────
        .route("/api/health", get(health_handler))
        // ── Datasets ───────────────────────────────────────────────────────
        .route("/api/datasets", get(api::list_datasets))
        .route("/api/datasets/{name}", get(api::get_dataset))
        .route("/api/datasets/{name}/stats", get(api::get_dataset_stats))
        // ── Samples ────────────────────────────────────────────────────────
        .route("/api/datasets/{name}/samples", get(api::list_samples))
        .route(
            "/api/datasets/{name}/samples/{sample_id}",
            get(api::get_sample),
        )
        // ── Categories ─────────────────────────────────────────────────────
        .route(
            "/api/datasets/{name}/categories",
            get(api::list_categories),
        )
        // ── Asset-level (legacy + annotations) ────────────────────────────
        .route("/api/datasets/{name}/assets/{id}", get(api::get_asset))
        .route(
            "/api/assets/{id}/annotations",
            get(api::list_asset_annotations),
        )
        // ── Embeddings ─────────────────────────────────────────────────────
        .route(
            "/api/datasets/{name}/embeddings",
            get(api::list_dataset_embeddings),
        )
        .route(
            "/api/assets/{id}/embeddings",
            get(api::list_asset_embeddings),
        )
        // ── Predictions ────────────────────────────────────────────────────
        .route(
            "/api/datasets/{name}/predictions",
            get(api::list_dataset_predictions),
        )
        // ── Patches ────────────────────────────────────────────────────────
        .route(
            "/api/datasets/{name}/patches",
            get(api::list_dataset_patches),
        )
        .route("/api/assets/{id}/patches", get(api::list_asset_patches))
        // ── Export ─────────────────────────────────────────────────────────
        .route("/api/datasets/{name}/export", get(api::export_dataset))
        // ── Virtual datasets ───────────────────────────────────────────────
        .route("/api/virtual-datasets", get(api::list_virtual_datasets))
        .route(
            "/api/virtual-datasets/{name}",
            get(api::get_virtual_dataset),
        )
        .route(
            "/api/virtual-datasets/{name}/samples",
            get(api::list_virtual_dataset_samples),
        )
        .route(
            "/api/virtual-datasets/evaluate",
            post(api::evaluate_virtual_dataset),
        )
        // ── Media ──────────────────────────────────────────────────────────
        .route("/images/{dataset_id}/{filename}", get(image_handler))
        .fallback(spa_fallback)
        .with_state(state)
        .layer(CorsLayer::permissive())
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::path::PathBuf;

    use axum::body::to_bytes;
    use axum::http::{Request, StatusCode};
    use tempfile::tempdir;
    use tower::ServiceExt;

    use super::*;

    async fn make_app(catalog_path: PathBuf) -> Router {
        create_router(AppState { catalog_path }).await
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let tmp = tempdir().unwrap();
        let app = make_app(tmp.path().to_path_buf()).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
    }

    #[tokio::test]
    async fn test_missing_image_404() {
        let tmp = tempdir().unwrap();
        let app = make_app(tmp.path().to_path_buf()).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/images/999/nonexistent.jpg")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_spa_fallback() {
        let tmp = tempdir().unwrap();
        let app = make_app(tmp.path().to_path_buf()).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/some/page")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);

        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(
            body_str.contains("dman"),
            "SPA fallback should contain 'dman'"
        );
    }

    #[tokio::test]
    async fn test_image_served_with_correct_content_type() {
        let tmp = tempdir().unwrap();
        let img_dir = tmp.path().join("data").join("1").join("images");
        std::fs::create_dir_all(&img_dir).unwrap();
        let mut f = std::fs::File::create(img_dir.join("test.png")).unwrap();
        f.write_all(b"\x89PNG\r\n\x1a\n").unwrap();

        let app = make_app(tmp.path().to_path_buf()).await;

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/images/1/test.png")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(ct, "image/png");
    }

    #[tokio::test]
    async fn test_cors_header_present() {
        let tmp = tempdir().unwrap();
        let app = make_app(tmp.path().to_path_buf()).await;

        let response = app
            .oneshot(
                Request::builder()
                    .method("OPTIONS")
                    .uri("/api/health")
                    .header("Origin", "http://localhost:3000")
                    .header("Access-Control-Request-Method", "GET")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let has_cors = response
            .headers()
            .contains_key("access-control-allow-origin");
        assert!(has_cors, "CORS headers should be present");
    }
}
