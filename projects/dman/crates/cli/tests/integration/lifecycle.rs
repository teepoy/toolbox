use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

// ── init ──────────────────────────────────────────────────────────────────────

#[test]
fn init_creates_catalog_db() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
    assert!(
        home.path().join("catalog.db").exists(),
        "catalog.db should be created after init"
    );
}

#[test]
fn init_prints_initialized_message() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Catalog initialized at"));
}

#[test]
fn init_is_idempotent() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
}

// ── add ───────────────────────────────────────────────────────────────────────

#[test]
fn add_registers_dataset_successfully() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "my-dataset", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("my-dataset"))
        .stdout(predicate::str::contains("Registered dataset"));
}

#[test]
fn add_with_format_flag_succeeds() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args([
            "add",
            "yolo-ds",
            ds_path.path().to_str().expect("utf8"),
            "--format",
            "yolo",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("yolo-ds"));
}

#[test]
fn add_without_init_fails() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .args(["add", "my-dataset", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure();
}

#[test]
fn add_nonexistent_path_fails() {
    let home = tempdir().expect("tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "ghost-ds", "/nonexistent/path/that/does/not/exist"])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure();
}

// ── list ──────────────────────────────────────────────────────────────────────

#[test]
fn list_empty_catalog_shows_no_datasets_message() {
    let home = tempdir().expect("tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("No datasets registered"));
}

#[test]
fn list_shows_added_dataset() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "listed-ds", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("listed-ds"));
}

#[test]
fn list_json_format_is_valid_json() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "json-ds", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    let output = dman()
        .args(["list", "--format", "json"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let text = String::from_utf8(output).expect("utf8 stdout");
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("output must be valid JSON");
    assert!(parsed.is_array(), "JSON output should be an array");
}

#[test]
fn list_table_shows_count() {
    let home = tempdir().expect("tempdir");
    let ds1 = tempdir().expect("ds tempdir 1");
    let ds2 = tempdir().expect("ds tempdir 2");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "alpha", ds1.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "beta", ds2.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("alpha"))
        .stdout(predicate::str::contains("beta"))
        .stdout(predicate::str::contains("dataset(s) total"));
}

// ── inspect ───────────────────────────────────────────────────────────────────

#[test]
fn inspect_shows_dataset_details() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "inspect-me", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["inspect", "inspect-me"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("inspect-me"));
}

#[test]
fn inspect_nonexistent_dataset_fails() {
    let home = tempdir().expect("tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["inspect", "no-such-dataset"])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure();
}

// ── remove ────────────────────────────────────────────────────────────────────

#[test]
fn remove_with_yes_flag_removes_dataset() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "to-remove", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["remove", "to-remove", "--yes"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("Removed dataset"));

    // Dataset should no longer appear in list
    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("No datasets registered"));
}

#[test]
fn remove_nonexistent_dataset_fails() {
    let home = tempdir().expect("tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["remove", "ghost", "--yes"])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure();
}

#[test]
fn remove_abort_via_stdin_n_keeps_dataset() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["add", "keep-me", ds_path.path().to_str().expect("utf8")])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    dman()
        .args(["remove", "keep-me"])
        .env("DMAN_HOME", home.path())
        .write_stdin("n\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("Aborted"));

    // Dataset should still exist
    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("keep-me"));
}

// ── full lifecycle ─────────────────────────────────────────────────────────────

#[test]
fn full_lifecycle_init_add_inspect_remove() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    // 1. init
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 2. add
    dman()
        .args([
            "add",
            "lifecycle-ds",
            ds_path.path().to_str().expect("utf8"),
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 3. list confirms presence
    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("lifecycle-ds"));

    // 4. inspect shows details
    dman()
        .args(["inspect", "lifecycle-ds"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("lifecycle-ds"));

    // 5. remove
    dman()
        .args(["remove", "lifecycle-ds", "--yes"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 6. list confirms removal
    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("No datasets registered"));
}
