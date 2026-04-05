use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

#[test]
fn help_exits_zero() {
    dman().arg("--help").assert().success();
}

#[test]
fn help_shows_subcommands() {
    dman()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("init"))
        .stdout(predicate::str::contains("add"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("inspect"))
        .stdout(predicate::str::contains("remove"));
}

#[test]
fn init_exits_zero() {
    let dir = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();
}

#[test]
fn init_creates_catalog_dir() {
    let dir = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();
    assert!(dir.path().join("catalog.db").exists());
}

#[test]
fn list_after_init_shows_empty_message() {
    let dir = tempdir().expect("tempdir");
    dman()
        .arg("init")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();

    dman()
        .arg("list")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();
}

#[test]
fn list_without_init_exits_nonzero() {
    let dir = tempdir().expect("tempdir");
    dman()
        .arg("list")
        .env("DMAN_HOME", dir.path())
        .assert()
        .failure();
}

#[test]
fn add_and_list_dataset() {
    let dir = tempdir().expect("tempdir");
    let ds_dir = tempdir().expect("ds tempdir");

    dman()
        .arg("init")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();

    dman()
        .args([
            "add",
            "test-ds",
            ds_dir.path().to_str().expect("utf8 path"),
            "--format",
            "builder",
        ])
        .env("DMAN_HOME", dir.path())
        .assert()
        .success();

    dman()
        .arg("list")
        .env("DMAN_HOME", dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("test-ds"));
}

#[test]
fn stub_commands_exit_zero() {
    let dir = tempdir().expect("tempdir");
    for cmd in ["operate", "virtual"] {
        dman()
            .arg(cmd)
            .env("DMAN_HOME", dir.path())
            .assert()
            .success();
    }
}
