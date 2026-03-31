use actix_web::{web, App, HttpServer, HttpResponse, Responder, get, post, put, delete, middleware};
use serde::{Deserialize, Serialize};
use rusqlite::{Connection, params};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::sync::Mutex;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Column {
    pub name: String,
    pub data_type: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Dataset {
    pub id: String,
    pub name: String,
    pub description: String,
    pub columns: Vec<Column>,
    pub row_count: usize,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateDatasetRequest {
    pub name: String,
    pub description: String,
    pub columns: Vec<Column>,
    pub data: Vec<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RowUpdate {
    pub row_index: usize,
    pub updates: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExtensionResult {
    pub success: bool,
    pub message: String,
    pub modified_rows: usize,
}

pub struct AppState {
    pub db: Mutex<Connection>,
    pub data_dir: PathBuf,
}

fn init_database(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute(
        "CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            columns TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )",
        [],
    )?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS dataset_rows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_id TEXT NOT NULL,
            row_index INTEGER NOT NULL,
            data TEXT NOT NULL,
            FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
            UNIQUE(dataset_id, row_index)
        )",
        [],
    )?;

    Ok(())
}

#[get("/api/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "version": "0.1.0"
    }))
}

#[get("/api/datasets")]
async fn list_datasets(data: web::Data<AppState>) -> impl Responder {
    let conn = data.db.lock().unwrap();
    let mut stmt = match conn.prepare(
        "SELECT d.id, d.name, d.description, d.columns, d.created_at, d.updated_at,
                (SELECT COUNT(*) FROM dataset_rows WHERE dataset_id = d.id) as row_count
         FROM datasets d ORDER BY d.updated_at DESC"
    ) {
        Ok(s) => s,
        Err(e) => return HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    };

    let datasets: Vec<Dataset> = stmt.query_map([], |row| {
        let columns_json: String = row.get(3)?;
        let columns: Vec<Column> = serde_json::from_str(&columns_json).unwrap_or_default();

        Ok(Dataset {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            columns,
            row_count: row.get(6)?,
            created_at: row.get::<_, String>(4)?.parse().unwrap_or_else(|_| Utc::now()),
            updated_at: row.get::<_, String>(5)?.parse().unwrap_or_else(|_| Utc::now()),
        })
    }).ok().map(|rows| rows.filter_map(|r| r.ok()).collect()).unwrap_or_default();

    HttpResponse::Ok().json(datasets)
}

#[post("/api/datasets")]
async fn create_dataset(
    data: web::Data<AppState>,
    body: web::Json<CreateDatasetRequest>
) -> impl Responder {
    let dataset_id = Uuid::new_v4().to_string();
    let now = Utc::now();
    let columns_json = serde_json::to_string(&body.columns).unwrap_or_default();

    let conn = data.db.lock().unwrap();

    if let Err(e) = conn.execute(
        "INSERT INTO datasets (id, name, description, columns, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![dataset_id, body.name, body.description, columns_json, now.to_rfc3339(), now.to_rfc3339()],
    ) {
        return HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}));
    }

    for (idx, row) in body.data.iter().enumerate() {
        let row_json = serde_json::to_string(row).unwrap_or_default();
        if let Err(e) = conn.execute(
            "INSERT OR REPLACE INTO dataset_rows (dataset_id, row_index, data) VALUES (?1, ?2, ?3)",
            params![dataset_id, idx, row_json],
        ) {
            return HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()}));
        }
    }

    let dataset = Dataset {
        id: dataset_id,
        name: body.name.clone(),
        description: body.description.clone(),
        columns: body.columns.clone(),
        row_count: body.data.len(),
        created_at: now,
        updated_at: now,
    };

    HttpResponse::Created().json(dataset)
}

#[get("/api/datasets/{id}")]
async fn get_dataset(
    data: web::Data<AppState>,
    path: web::Path<String>
) -> impl Responder {
    let dataset_id = path.into_inner();
    let conn = data.db.lock().unwrap();

    let mut stmt = match conn.prepare(
        "SELECT id, name, description, columns, created_at, updated_at FROM datasets WHERE id = ?1"
    ) {
        Ok(s) => s,
        Err(e) => return HttpResponse::InternalServerError().json(serde_json::json!({"error": e.to_string()})),
    };

    let dataset: Option<Dataset> = stmt.query_row(params![dataset_id], |row| {
        let columns_json: String = row.get(3)?;
        let columns: Vec<Column> = serde_json::from_str(&columns_json).unwrap_or_default();

        Ok(Dataset {
            id: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            columns,
            row_count: 0,
            created_at: row.get::<_, String>(4)?.parse().unwrap_or_else(|_| Utc::now()),
            updated_at: row.get::<_, String>(5)?.parse().unwrap_or_else(|_| Utc::now()),
        })
    }).ok();

    match dataset {
        Some(mut ds) => {
            let mut count_stmt = conn.prepare("SELECT COUNT(*) FROM dataset_rows WHERE dataset_id = ?1").unwrap();
            ds.row_count = count_stmt.query_row(params![ds.id], |r| r.get(0)).unwrap_or(0);

            let mut rows_stmt = conn.prepare("SELECT data FROM dataset_rows WHERE dataset_id = ?1 ORDER BY row_index").unwrap();
            let rows: Vec<HashMap<String, serde_json::Value>> = rows_stmt
                .query_map(params![ds.id], |row| {
                    let data: String = row.get(0)?;
                    Ok(serde_json::from_str(&data).unwrap_or_default())
                })
                .ok()
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                .unwrap_or_default();

            HttpResponse::Ok().json(serde_json::json!({
                "dataset": ds,
                "data": rows
            }))
        }
        None => HttpResponse::NotFound().json(serde_json::json!({"error": "Dataset not found"}))
    }
}

#[put("/api/datasets/{id}/rows")]
async fn update_rows(
    data: web::Data<AppState>,
    path: web::Path<String>,
    body: web::Json<Vec<RowUpdate>>
) -> impl Responder {
    let dataset_id = path.into_inner();
    let conn = data.db.lock().unwrap();
    let mut modified = 0;

    for update in body.iter() {
        let row_index = update.row_index;

        let existing: Option<String> = conn.query_row(
            "SELECT data FROM dataset_rows WHERE dataset_id = ?1 AND row_index = ?2",
            params![dataset_id, row_index],
            |row| row.get(0)
        ).ok();

        if let Some(existing_data) = existing {
            let mut row_data: HashMap<String, serde_json::Value> = serde_json::from_str(&existing_data).unwrap_or_default();

            for (key, value) in &update.updates {
                row_data.insert(key.clone(), value.clone());
            }

            let updated_json = serde_json::to_string(&row_data).unwrap_or_default();

            if conn.execute(
                "UPDATE dataset_rows SET data = ?1 WHERE dataset_id = ?2 AND row_index = ?3",
                params![updated_json, dataset_id, row_index]
            ).is_ok() {
                modified += 1;
            }
        }
    }

    let _ = conn.execute(
        "UPDATE datasets SET updated_at = ?1 WHERE id = ?2",
        params![Utc::now().to_rfc3339(), dataset_id]
    );

    HttpResponse::Ok().json(ExtensionResult {
        success: true,
        message: format!("Modified {} rows", modified),
        modified_rows: modified,
    })
}

#[post("/api/datasets/{id}/extensions/{ext_name}")]
async fn run_extension(
    data: web::Data<AppState>,
    path: web::Path<(String, String)>,
    body: web::Json<serde_json::Value>
) -> impl Responder {
    let (dataset_id, ext_name) = path.into_inner();

    let conn = data.db.lock().unwrap();

    let mut stmt = conn.prepare("SELECT data FROM dataset_rows WHERE dataset_id = ?1 ORDER BY row_index").unwrap();
    let rows: Vec<HashMap<String, serde_json::Value>> = stmt
        .query_map(params![dataset_id], |row| {
            let data: String = row.get(0)?;
            Ok(serde_json::from_str(&data).unwrap_or_default())
        })
        .ok()
        .map(|r| r.filter_map(|row| row.ok()).collect())
        .unwrap_or_default();

    let extension_path = data.data_dir.join("extensions").join(format!("{}.py", ext_name));

    if !extension_path.exists() {
        return HttpResponse::NotFound().json(ExtensionResult {
            success: false,
            message: format!("Extension '{}' not found", ext_name),
            modified_rows: 0,
        });
    }

    let modified_data = run_python_extension(&extension_path, rows, body.into_inner());

    let mut modified = 0;
    for (idx, row) in modified_data.iter().enumerate() {
        let row_json = serde_json::to_string(row).unwrap_or_default();
        if conn.execute(
            "INSERT OR REPLACE INTO dataset_rows (dataset_id, row_index, data) VALUES (?1, ?2, ?3)",
            params![dataset_id, idx, row_json]
        ).is_ok() {
            modified += 1;
        }
    }

    let _ = conn.execute(
        "UPDATE datasets SET updated_at = ?1 WHERE id = ?2",
        params![Utc::now().to_rfc3339(), dataset_id]
    );

    HttpResponse::Ok().json(ExtensionResult {
        success: true,
        message: format!("Extension '{}' executed successfully", ext_name),
        modified_rows: modified,
    })
}

fn run_python_extension(path: &PathBuf, data: Vec<HashMap<String, serde_json::Value>>, params: serde_json::Value) -> Vec<HashMap<String, serde_json::Value>> {
    let temp_file = std::env::temp_dir().join("dataset_input.json");
    let output_file = std::env::temp_dir().join("dataset_output.json");

    if let Ok(json_data) = serde_json::to_string(&data) {
        let _ = std::fs::write(&temp_file, json_data);
    }

    let mut cmd = std::process::Command::new("python3");
    cmd.arg("-c")
       .arg(format!(
           "import json; exec(open('{}').read())",
           path.display()
       ))
       .current_dir(path.parent().unwrap_or(&PathBuf::from(".")));

    let output = cmd.output();

    if let Ok(out) = output {
        if let Ok(result) = String::from_utf8(out.stdout) {
            if let Ok(processed) = serde_json::from_str(&result) {
                let _ = std::fs::remove_file(&temp_file);
                let _ = std::fs::remove_file(&output_file);
                return processed;
            }
        }
    }

    let _ = std::fs::remove_file(&temp_file);
    let _ = std::fs::remove_file(&output_file);
    data
}

#[delete("/api/datasets/{id}")]
async fn delete_dataset(
    data: web::Data<AppState>,
    path: web::Path<String>
) -> impl Responder {
    let dataset_id = path.into_inner();
    let conn = data.db.lock().unwrap();

    let _ = conn.execute("DELETE FROM dataset_rows WHERE dataset_id = ?1", params![dataset_id]);
    let _ = conn.execute("DELETE FROM datasets WHERE id = ?1", params![dataset_id]);

    HttpResponse::Ok().json(serde_json::json!({"success": true}))
}

#[get("/api/extensions")]
async fn list_extensions(data: web::Data<AppState>) -> impl Responder {
    let extensions_dir = data.data_dir.join("extensions");
    let mut extensions = Vec::new();

    if let Ok(entries) = std::fs::read_dir(extensions_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            if let Some(name) = entry.path().file_stem() {
                if entry.path().extension().map_or(false, |e| e == "py") {
                    extensions.push(name.to_string_lossy().to_string());
                }
            }
        }
    }

    HttpResponse::Ok().json(extensions)
}

#[get("/")]
async fn index() -> impl Responder {
    HttpResponse::Found()
        .insert_header(("Location", "/static/index.html"))
        .finish()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));

    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("dataset-tool");

    std::fs::create_dir_all(&data_dir).ok();
    std::fs::create_dir_all(data_dir.join("extensions")).ok();

    let ext_source = PathBuf::from("./extensions");
    if ext_source.exists() {
        let ext_dest = data_dir.join("extensions");
        if let Ok(entries) = std::fs::read_dir(&ext_source) {
            for entry in entries.filter_map(|e| e.ok()) {
                if entry.path().extension().map_or(false, |e| e == "py") {
                    let dest = ext_dest.join(entry.path().file_name().unwrap());
                    let _ = std::fs::copy(entry.path(), dest);
                }
            }
        }
    }

    let db_path = data_dir.join("data.db");
    let conn = Connection::open(&db_path).expect("Failed to open database");
    init_database(&conn).expect("Failed to initialize database");

    log::info!("Dataset Tool API starting on http://localhost:8080");
    log::info!("Data directory: {:?}", data_dir);

    let app_state = web::Data::new(AppState {
        db: Mutex::new(conn),
        data_dir: data_dir.clone(),
    });

    let frontend_path = data_dir.join("frontend");
    std::fs::create_dir_all(&frontend_path).ok();
    let _ = std::fs::copy("./frontend/index.html", frontend_path.join("index.html"));

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(health)
            .service(list_datasets)
            .service(create_dataset)
            .service(get_dataset)
            .service(update_rows)
            .service(delete_dataset)
            .service(run_extension)
            .service(list_extensions)
            .service(index)
            .service(actix_files::Files::new("/static", data_dir.join("frontend")))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
