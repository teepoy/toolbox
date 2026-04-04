use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

#[test]
fn import_stub_exits_zero_and_prints_stub_message() {
    let home = tempdir().expect("tempdir");
    let path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["import", path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn import_stub_with_format_flag_exits_zero() {
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
            "--format",
            "coco",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("import is not yet implemented"));
}

#[test]
fn import_stub_with_name_flag_exits_zero() {
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
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn export_stub_exits_zero_and_prints_stub_message() {
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
            "some-dataset",
            output_path.path().to_str().expect("utf8"),
            "--format",
            "yolo",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn export_stub_includes_dataset_name_in_message() {
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
            "hf",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("named-dataset"));
}
