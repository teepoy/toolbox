use assert_cmd::Command;
use predicates::prelude::*;
use std::path::PathBuf;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

fn fixture_path(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../core/tests/fixtures")
        .join(relative)
}

#[test]
fn import_detects_and_loads_yolo_fixture() {
    let home = tempdir().expect("tempdir");
    let path = fixture_path("yolo");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "import",
            path.to_str().expect("utf8"),
            "--name",
            "yolo-import",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Imported"))
        .stdout(predicate::str::contains("format=yolo"));
}

#[test]
fn import_with_explicit_format_succeeds() {
    let home = tempdir().expect("tempdir");
    let path = fixture_path("coco");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "import",
            path.to_str().expect("utf8"),
            "--format",
            "coco",
            "--name",
            "coco-import",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("coco-import"));
}

#[test]
fn import_requires_detectable_format_or_explicit_format() {
    let home = tempdir().expect("tempdir");
    let path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "import",
            path.path().to_str().expect("utf8"),
            "--name",
            "my-import",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "no registered format provider detected",
        ));
}

#[test]
fn export_writes_yolo_layout() {
    let home = tempdir().expect("tempdir");
    let output_path = tempdir().expect("output tempdir");
    let input_path = fixture_path("yolo");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "import",
            input_path.to_str().expect("utf8"),
            "--name",
            "some-dataset",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "export",
            "some-dataset",
            output_path.path().to_str().expect("utf8"),
            "--format",
            "yolo",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Exported dataset"));

    assert!(output_path.path().join("data.yaml").exists());
    assert!(output_path.path().join("images").join("train").is_dir());
    assert!(output_path.path().join("labels").join("train").is_dir());
}

#[test]
fn export_nonexistent_dataset_fails() {
    let home = tempdir().expect("tempdir");
    let output_path = tempdir().expect("output tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "export",
            "named-dataset",
            output_path.path().to_str().expect("utf8"),
            "--format",
            "huggingface",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "failed to find dataset 'named-dataset'",
        ));
}
