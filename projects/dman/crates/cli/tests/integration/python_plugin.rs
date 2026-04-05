#![cfg(feature = "python")]

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

const TEST_PLUGIN_SOURCE: &str = r#"
dman_plugin = {"name": "test-custom-format", "type": "format", "version": "1.0.0"}

class Sample:
    def __init__(self, name):
        self.name = name
        self.metadata = None
        self.assets = []
        self.annotations = []

def import_dataset(path):
    return [Sample("sample-0")]

def export_dataset(samples, path):
    pass
"#;

fn write_test_plugin(home: &std::path::Path) {
    let plugins_dir = home.join("plugins");
    std::fs::create_dir_all(&plugins_dir).expect("create plugins dir");
    std::fs::write(plugins_dir.join("test_format.py"), TEST_PLUGIN_SOURCE)
        .expect("write test plugin");
}

// ── import with python plugin ─────────────────────────────────────────────────

#[test]
fn python_plugin_import_registers_dataset_with_custom_format() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    // 1. init catalog
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 2. place plugin so the CLI can discover it
    write_test_plugin(home.path());

    // 3. import the dataset using the plugin's format ID
    dman()
        .args([
            "import",
            ds_path.path().to_str().expect("utf8"),
            "--format",
            "test-custom-format",
            "--name",
            "plugin-ds",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("plugin-ds"));

    // 4. list confirms the dataset appears in the catalog
    dman()
        .arg("list")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("plugin-ds"));

    // 5. inspect confirms the format ID is stored correctly
    dman()
        .args(["inspect", "plugin-ds"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("test-custom-format"));
}

// ── import with unregistered format fails ─────────────────────────────────────

#[test]
fn import_with_missing_format_fails_with_clear_error() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");

    // init without writing any plugin
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // attempt to import with an unregistered format ID
    dman()
        .args([
            "import",
            ds_path.path().to_str().expect("utf8"),
            "--format",
            "missing-format",
            "--name",
            "ghost-ds",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing-format"));
}

// ── export via python plugin ───────────────────────────────────────────────────

#[test]
fn python_plugin_export_succeeds_for_registered_dataset() {
    let home = tempdir().expect("tempdir");
    let ds_path = tempdir().expect("ds tempdir");
    let out_path = tempdir().expect("output tempdir");

    // 1. init catalog
    dman()
        .arg("init")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 2. place plugin
    write_test_plugin(home.path());

    // 3. import to create a dataset registered under the custom format
    dman()
        .args([
            "import",
            ds_path.path().to_str().expect("utf8"),
            "--format",
            "test-custom-format",
            "--name",
            "export-ds",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();

    // 4. export the dataset using the same custom format
    dman()
        .args([
            "export",
            "export-ds",
            out_path.path().to_str().expect("utf8"),
            "--format",
            "test-custom-format",
        ])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("export-ds"));
}
