use std::collections::HashMap;
use std::path::PathBuf;

use dman_core::db::Database;
use dman_core::error::{DmanError, Result};
use dman_core::types::{BBox, Dataset, DatasetFormat};
use rusqlite::params;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSProject {
    pub id: i64,
    pub title: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTask {
    pub id: i64,
    pub data: LSTaskData,
    #[serde(default)]
    pub annotations: Vec<LSAnnotation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSTaskData {
    pub image: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSAnnotation {
    #[serde(default)]
    pub result: Vec<LSResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LSResult {
    pub value: serde_json::Value,
    #[serde(rename = "type")]
    pub type_: String,
    #[serde(default)]
    pub from_name: String,
    #[serde(default)]
    pub to_name: String,
}

#[derive(Debug, Deserialize)]
struct LSProjectsResponse {
    results: Vec<LSProject>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAnnotation {
    pub label: String,
    pub bbox: BBox,
}

pub fn parse_ls_result(
    result: &LSResult,
    image_width: u32,
    image_height: u32,
) -> Option<ParsedAnnotation> {
    if result.type_ != "rectanglelabels" {
        return None;
    }

    let value = &result.value;
    let x_pct = value.get("x")?.as_f64()?;
    let y_pct = value.get("y")?.as_f64()?;
    let w_pct = value.get("width")?.as_f64()?;
    let h_pct = value.get("height")?.as_f64()?;

    let labels = value.get("rectanglelabels")?.as_array()?;
    let label = labels.first()?.as_str()?.to_string();

    let img_w = image_width as f64;
    let img_h = image_height as f64;

    Some(ParsedAnnotation {
        label,
        bbox: BBox {
            x: x_pct * img_w / 100.0,
            y: y_pct * img_h / 100.0,
            width: w_pct * img_w / 100.0,
            height: h_pct * img_h / 100.0,
        },
    })
}

pub fn build_export_task(image_url: &str) -> serde_json::Value {
    serde_json::json!({
        "data": {
            "image": image_url
        }
    })
}

pub fn build_image_url(dataset_id: i64, filename: &str, server_port: u16) -> String {
    format!(
        "http://localhost:{}/images/{}/{}",
        server_port, dataset_id, filename
    )
}

pub struct LabelStudioClient {
    base_url: String,
    api_key: String,
    client: reqwest::blocking::Client,
}

impl LabelStudioClient {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        let base_url = base_url.trim_end_matches('/').to_string();
        Self {
            base_url,
            api_key: api_key.to_string(),
            client: reqwest::blocking::Client::new(),
        }
    }

    fn auth_header(&self) -> String {
        format!("Token {}", self.api_key)
    }

    pub fn list_projects(&self) -> Result<Vec<LSProject>> {
        let url = format!("{}/api/projects/", self.base_url);
        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .map_err(|e| DmanError::StorageError(format!("Label Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(DmanError::StorageError(format!(
                "Label Studio API returned status {}",
                response.status()
            )));
        }

        let body: LSProjectsResponse = response.json().map_err(|e| {
            DmanError::StorageError(format!("Failed to parse projects response: {}", e))
        })?;

        Ok(body.results)
    }

    pub fn create_project(&self, title: &str, label_config: &str) -> Result<LSProject> {
        let url = format!("{}/api/projects/", self.base_url);
        let body = serde_json::json!({
            "title": title,
            "label_config": label_config,
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header())
            .json(&body)
            .send()
            .map_err(|e| DmanError::StorageError(format!("Label Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(DmanError::StorageError(format!(
                "Label Studio API returned status {}",
                response.status()
            )));
        }

        let project: LSProject = response.json().map_err(|e| {
            DmanError::StorageError(format!("Failed to parse project response: {}", e))
        })?;

        Ok(project)
    }

    fn fetch_tasks(&self, project_id: i64) -> Result<Vec<LSTask>> {
        let url = format!(
            "{}/api/projects/{}/tasks/?page_size=10000",
            self.base_url, project_id
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .map_err(|e| DmanError::StorageError(format!("Label Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(DmanError::StorageError(format!(
                "Label Studio API returned status {}",
                response.status()
            )));
        }

        let body: serde_json::Value = response.json().map_err(|e| {
            DmanError::StorageError(format!("Failed to parse tasks response: {}", e))
        })?;

        let tasks_value = if body.is_array() {
            body
        } else if let Some(arr) = body.get("tasks").or_else(|| body.get("results")) {
            arr.clone()
        } else {
            body
        };

        let tasks: Vec<LSTask> = serde_json::from_value(tasks_value)?;
        Ok(tasks)
    }

    pub fn import_project(
        &self,
        project_id: i64,
        db: &Database,
        dataset_name: &str,
    ) -> Result<Dataset> {
        let tasks = self.fetch_tasks(project_id)?;

        let data_dir = std::env::temp_dir().join(format!("dman-ls-{}", dataset_name));
        std::fs::create_dir_all(&data_dir)?;

        import_tasks_to_db(db, dataset_name, &data_dir, &tasks)
    }

    pub fn export_to_project(
        &self,
        db: &Database,
        dataset_name: &str,
        project_id: i64,
        server_port: u16,
    ) -> Result<()> {
        let tasks = build_export_tasks(db, dataset_name, server_port)?;

        let url = format!("{}/api/projects/{}/import", self.base_url, project_id);

        let response = self
            .client
            .post(&url)
            .header("Authorization", self.auth_header())
            .json(&tasks)
            .send()
            .map_err(|e| DmanError::StorageError(format!("Label Studio request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(DmanError::StorageError(format!(
                "Label Studio export failed with status {}",
                response.status()
            )));
        }

        Ok(())
    }
}

pub fn import_tasks_to_db(
    db: &Database,
    dataset_name: &str,
    data_dir: &std::path::Path,
    tasks: &[LSTask],
) -> Result<Dataset> {
    let exists: bool = db.conn.query_row(
        "SELECT COUNT(*) FROM datasets WHERE name = ?1",
        params![dataset_name],
        |row| row.get::<_, i64>(0),
    )? > 0;

    if exists {
        return Err(DmanError::DatasetAlreadyExists(dataset_name.to_string()));
    }

    let path_str = data_dir.to_string_lossy().to_string();
    let format_str = "LabelStudio";

    db.conn.execute(
        "INSERT INTO datasets (name, path, format) VALUES (?1, ?2, ?3)",
        params![dataset_name, path_str, format_str],
    )?;
    let dataset_id = db.conn.last_insert_rowid();

    let mut category_map: HashMap<String, i64> = HashMap::new();

    for task in tasks {
        let image_url = &task.data.image;
        let file_name = extract_filename_from_url(image_url);

        let default_width: u32 = 1920;
        let default_height: u32 = 1080;

        db.conn.execute(
            "INSERT INTO images (dataset_id, file_name, file_path, width, height) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![dataset_id, file_name, image_url, default_width, default_height],
        )?;
        let image_id = db.conn.last_insert_rowid();

        for annotation in &task.annotations {
            for result in &annotation.result {
                if let Some(parsed) = parse_ls_result(result, default_width, default_height) {
                    let category_id = match category_map.get(&parsed.label) {
                        Some(&id) => id,
                        None => {
                            db.conn.execute(
                                "INSERT INTO categories (dataset_id, name) VALUES (?1, ?2)",
                                params![dataset_id, parsed.label],
                            )?;
                            let id = db.conn.last_insert_rowid();
                            category_map.insert(parsed.label.clone(), id);
                            id
                        }
                    };

                    let bbox_json = serde_json::to_string(&parsed.bbox)?;

                    db.conn.execute(
                        "INSERT INTO annotations (image_id, category_id, bbox) VALUES (?1, ?2, ?3)",
                        params![image_id, category_id, bbox_json],
                    )?;
                }
            }
        }
    }

    let ds = db.conn.query_row(
        "SELECT id, name, path, format, schema_path, created_at, updated_at, metadata FROM datasets WHERE id = ?1",
        params![dataset_id],
        |row| {
            let id: i64 = row.get(0)?;
            let name: String = row.get(1)?;
            let path: String = row.get(2)?;
            let _format: String = row.get(3)?;
            let schema_path: Option<String> = row.get(4)?;
            let created_at: String = row.get(5)?;
            let updated_at: Option<String> = row.get(6)?;
            let metadata_str: Option<String> = row.get(7)?;
            let metadata = metadata_str
                .as_deref()
                .and_then(|s| serde_json::from_str(s).ok());

            Ok(Dataset {
                id,
                name,
                path: PathBuf::from(path),
                format: DatasetFormat::Custom("LabelStudio".to_string()),
                schema_path: schema_path.map(PathBuf::from),
                created_at,
                updated_at,
                metadata,
            })
        },
    )?;

    Ok(ds)
}

pub fn build_export_tasks(
    db: &Database,
    dataset_name: &str,
    server_port: u16,
) -> Result<Vec<serde_json::Value>> {
    let dataset_id: i64 = db
        .conn
        .query_row(
            "SELECT id FROM datasets WHERE name = ?1",
            params![dataset_name],
            |row| row.get(0),
        )
        .map_err(|e| match e {
            rusqlite::Error::QueryReturnedNoRows => {
                DmanError::DatasetNotFound(dataset_name.to_string())
            }
            other => DmanError::Database(other),
        })?;

    let mut stmt = db
        .conn
        .prepare("SELECT file_name FROM images WHERE dataset_id = ?1 ORDER BY id")?;

    let tasks: Vec<serde_json::Value> = stmt
        .query_map(params![dataset_id], |row| {
            let file_name: String = row.get(0)?;
            Ok(file_name)
        })?
        .filter_map(|r| r.ok())
        .map(|file_name| {
            let url = build_image_url(dataset_id, &file_name, server_port);
            build_export_task(&url)
        })
        .collect();

    Ok(tasks)
}

fn extract_filename_from_url(url: &str) -> String {
    url.rsplit('/')
        .next()
        .unwrap_or(url)
        .split('?')
        .next()
        .unwrap_or(url)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dman_core::db::Database;

    fn in_memory_db() -> Database {
        Database::open_in_memory().expect("in-memory DB")
    }

    #[test]
    fn test_import_task_conversion() {
        let task_json = r#"[
            {
                "id": 1,
                "data": {"image": "https://example.com/images/cat_001.jpg"},
                "annotations": [{
                    "result": [{
                        "value": {
                            "x": 10.0,
                            "y": 20.0,
                            "width": 30.0,
                            "height": 40.0,
                            "rotation": 0,
                            "rectanglelabels": ["cat"]
                        },
                        "type": "rectanglelabels",
                        "from_name": "label",
                        "to_name": "image"
                    }]
                }]
            },
            {
                "id": 2,
                "data": {"image": "https://example.com/images/dog_002.jpg"},
                "annotations": [{
                    "result": [
                        {
                            "value": {
                                "x": 5.0,
                                "y": 10.0,
                                "width": 50.0,
                                "height": 60.0,
                                "rotation": 0,
                                "rectanglelabels": ["dog"]
                            },
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image"
                        },
                        {
                            "value": {
                                "x": 60.0,
                                "y": 70.0,
                                "width": 20.0,
                                "height": 15.0,
                                "rotation": 0,
                                "rectanglelabels": ["cat"]
                            },
                            "type": "rectanglelabels",
                            "from_name": "label",
                            "to_name": "image"
                        }
                    ]
                }]
            }
        ]"#;

        let tasks: Vec<LSTask> = serde_json::from_str(task_json).expect("parse tasks");
        assert_eq!(tasks.len(), 2);

        let db = in_memory_db();
        let tmp = tempfile::tempdir().expect("tmp dir");

        let dataset =
            import_tasks_to_db(&db, "ls-import", tmp.path(), &tasks).expect("import tasks");

        assert_eq!(dataset.name, "ls-import");

        let image_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM images WHERE dataset_id = ?1",
                params![dataset.id],
                |row| row.get(0),
            )
            .expect("count images");
        assert_eq!(image_count, 2, "should have 2 images");

        let ann_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM annotations WHERE image_id IN (SELECT id FROM images WHERE dataset_id = ?1)",
                params![dataset.id],
                |row| row.get(0),
            )
            .expect("count annotations");
        assert_eq!(ann_count, 3, "should have 3 annotations");

        let cat_count: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM categories WHERE dataset_id = ?1",
                params![dataset.id],
                |row| row.get(0),
            )
            .expect("count categories");
        assert_eq!(cat_count, 2, "should have 2 categories: cat, dog");

        let bbox_str: String = db
            .conn
            .query_row(
                "SELECT bbox FROM annotations ORDER BY id LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("get first bbox");
        let bbox: BBox = serde_json::from_str(&bbox_str).expect("parse bbox");
        assert!(
            (bbox.x - 192.0).abs() < 0.01,
            "x should be 10% of 1920 = 192.0, got {}",
            bbox.x
        );
        assert!(
            (bbox.y - 216.0).abs() < 0.01,
            "y should be 20% of 1080 = 216.0, got {}",
            bbox.y
        );
        assert!(
            (bbox.width - 576.0).abs() < 0.01,
            "w should be 30% of 1920 = 576.0, got {}",
            bbox.width
        );
        assert!(
            (bbox.height - 432.0).abs() < 0.01,
            "h should be 40% of 1080 = 432.0, got {}",
            bbox.height
        );

        let file_name: String = db
            .conn
            .query_row(
                "SELECT file_name FROM images ORDER BY id LIMIT 1",
                [],
                |row| row.get(0),
            )
            .expect("get file_name");
        assert_eq!(file_name, "cat_001.jpg");
    }

    #[test]
    fn test_export_task_format() {
        let db = in_memory_db();
        let tmp = tempfile::tempdir().expect("tmp dir");
        let path_str = tmp.path().to_string_lossy().to_string();

        db.conn
            .execute(
                "INSERT INTO datasets (name, path, format) VALUES ('export-ds', ?1, 'Coco')",
                params![path_str],
            )
            .expect("insert dataset");
        let dataset_id = db.conn.last_insert_rowid();

        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'img_001.jpg', '/tmp/img_001.jpg')",
                params![dataset_id],
            )
            .expect("insert img1");
        db.conn
            .execute(
                "INSERT INTO images (dataset_id, file_name, file_path) VALUES (?1, 'img_002.png', '/tmp/img_002.png')",
                params![dataset_id],
            )
            .expect("insert img2");

        let tasks = build_export_tasks(&db, "export-ds", 8080).expect("build export tasks");

        assert_eq!(tasks.len(), 2, "should have 2 tasks");

        let task1 = &tasks[0];
        let img1_url = task1["data"]["image"].as_str().expect("image url");
        assert_eq!(
            img1_url,
            format!("http://localhost:8080/images/{}/img_001.jpg", dataset_id)
        );

        let task2 = &tasks[1];
        let img2_url = task2["data"]["image"].as_str().expect("image url");
        assert_eq!(
            img2_url,
            format!("http://localhost:8080/images/{}/img_002.png", dataset_id)
        );
    }

    #[test]
    fn test_api_error_handling() {
        let client = LabelStudioClient::new("http://127.0.0.1:1", "fake-key");

        let result = client.list_projects();
        assert!(
            result.is_err(),
            "should return error for unreachable server"
        );

        match result {
            Err(DmanError::StorageError(msg)) => {
                assert!(
                    msg.contains("Label Studio request failed"),
                    "error should mention request failure, got: {}",
                    msg
                );
            }
            other => panic!("expected StorageError, got: {:?}", other),
        }
    }

    #[test]
    fn test_parse_ls_result_rectanglelabels() {
        let result = LSResult {
            value: serde_json::json!({
                "x": 25.0,
                "y": 50.0,
                "width": 10.0,
                "height": 20.0,
                "rotation": 0,
                "rectanglelabels": ["person"]
            }),
            type_: "rectanglelabels".to_string(),
            from_name: "label".to_string(),
            to_name: "image".to_string(),
        };

        let parsed = parse_ls_result(&result, 800, 600).expect("should parse");
        assert_eq!(parsed.label, "person");
        assert!((parsed.bbox.x - 200.0).abs() < 0.01);
        assert!((parsed.bbox.y - 300.0).abs() < 0.01);
        assert!((parsed.bbox.width - 80.0).abs() < 0.01);
        assert!((parsed.bbox.height - 120.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_ls_result_skips_non_rectanglelabels() {
        let result = LSResult {
            value: serde_json::json!({"text": ["hello"]}),
            type_: "textarea".to_string(),
            from_name: "text".to_string(),
            to_name: "image".to_string(),
        };

        assert!(
            parse_ls_result(&result, 800, 600).is_none(),
            "non-rectanglelabels should return None"
        );
    }

    #[test]
    fn test_build_image_url() {
        let url = build_image_url(42, "photo.jpg", 9090);
        assert_eq!(url, "http://localhost:9090/images/42/photo.jpg");
    }

    #[test]
    fn test_extract_filename_from_url() {
        assert_eq!(
            extract_filename_from_url("https://example.com/data/img.jpg"),
            "img.jpg"
        );
        assert_eq!(
            extract_filename_from_url("https://example.com/data/img.jpg?token=abc"),
            "img.jpg"
        );
        assert_eq!(extract_filename_from_url("simple.png"), "simple.png");
    }

    #[test]
    fn test_export_dataset_not_found() {
        let db = in_memory_db();
        let result = build_export_tasks(&db, "nonexistent-ds", 8080);
        assert!(result.is_err());
        match result {
            Err(DmanError::DatasetNotFound(name)) => {
                assert_eq!(name, "nonexistent-ds");
            }
            other => panic!("expected DatasetNotFound, got: {:?}", other),
        }
    }

    #[test]
    fn test_import_duplicate_dataset() {
        let db = in_memory_db();
        let tmp = tempfile::tempdir().expect("tmp dir");
        let tasks: Vec<LSTask> = vec![];

        import_tasks_to_db(&db, "dup-ds", tmp.path(), &tasks).expect("first import");
        let result = import_tasks_to_db(&db, "dup-ds", tmp.path(), &tasks);
        assert!(
            matches!(result, Err(DmanError::DatasetAlreadyExists(_))),
            "second import should fail with DatasetAlreadyExists"
        );
    }

    #[test]
    fn test_ls_task_deserialization() {
        let json = r#"{
            "id": 123,
            "data": {"image": "http://example.com/img.jpg"},
            "annotations": [{
                "result": [{
                    "value": {
                        "x": 10,
                        "y": 20,
                        "width": 30,
                        "height": 40,
                        "rotation": 0,
                        "rectanglelabels": ["cat"]
                    },
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image"
                }]
            }]
        }"#;

        let task: LSTask = serde_json::from_str(json).expect("deserialize task");
        assert_eq!(task.id, 123);
        assert_eq!(task.data.image, "http://example.com/img.jpg");
        assert_eq!(task.annotations.len(), 1);
        assert_eq!(task.annotations[0].result.len(), 1);
        assert_eq!(task.annotations[0].result[0].type_, "rectanglelabels");
    }

    #[test]
    fn test_task_without_annotations() {
        let json = r#"{"id": 1, "data": {"image": "http://example.com/img.jpg"}}"#;
        let task: LSTask =
            serde_json::from_str(json).expect("deserialize task without annotations");
        assert_eq!(task.id, 1);
        assert!(task.annotations.is_empty());
    }
}
