use std::net::SocketAddr;
use std::path::PathBuf;

use dman_server::{AppState, create_router};

fn parse_args() -> (u16, PathBuf) {
    let args: Vec<String> = std::env::args().collect();
    let mut port: u16 = 8080;
    let mut catalog: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => {
                i += 1;
                if let Some(p) = args.get(i) {
                    port = p.parse().unwrap_or(8080);
                }
            }
            "--catalog" => {
                i += 1;
                if let Some(c) = args.get(i) {
                    catalog = Some(PathBuf::from(c));
                }
            }
            _ => {}
        }
        i += 1;
    }

    let catalog_path = catalog.unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".dman")
    });

    (port, catalog_path)
}

#[tokio::main]
async fn main() {
    let (port, catalog_path) = parse_args();
    let state = AppState { catalog_path };
    let app = create_router(state).await;
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind address");
    axum::serve(listener, app)
        .await
        .expect("server error");
}
