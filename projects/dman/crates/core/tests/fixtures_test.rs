mod common;

#[test]
fn test_yolo_fixture_exists() {
    let tmp = common::create_temp_dir();
    assert!(tmp.path().exists(), "temp dir should exist");
    let data_yaml = common::fixture_path("yolo/data.yaml");
    assert!(data_yaml.exists(), "yolo/data.yaml should exist");
    let img = common::fixture_path("yolo/images/train/img001.jpg");
    assert!(img.exists(), "yolo image should exist");
    let label = common::fixture_path("yolo/labels/train/img001.txt");
    assert!(label.exists(), "yolo label should exist");
}

#[test]
fn test_coco_fixture_is_valid_json() {
    let ann = common::fixture_path("coco/annotations.json");
    assert!(ann.exists(), "coco/annotations.json should exist");
    let content = std::fs::read_to_string(&ann).expect("should read file");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("should be valid JSON");
    assert!(parsed["images"].is_array());
    assert!(parsed["annotations"].is_array());
    assert!(parsed["categories"].is_array());
}

#[test]
fn test_schema_fixture_is_valid_toml() {
    let schema = common::fixture_path("schema/basic.toml");
    assert!(schema.exists(), "schema/basic.toml should exist");
    let content = std::fs::read_to_string(&schema).expect("should read file");
    let parsed: toml::Value = toml::from_str(&content).expect("should be valid TOML");
    assert!(parsed.get("fields").is_some(), "schema should have fields");
}
