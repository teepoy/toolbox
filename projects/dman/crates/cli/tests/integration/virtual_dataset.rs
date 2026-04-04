use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn dman() -> Command {
    Command::cargo_bin("dman-cli").expect("dman-cli binary must exist")
}

#[test]
fn virtual_stub_exits_zero() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("virtual")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
}

#[test]
fn virtual_stub_prints_not_implemented_to_stderr() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("virtual")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn virtual_stub_includes_command_name_in_message() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("virtual")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("virtual"));
}

#[test]
fn materialize_stub_exits_zero() {
    let home = tempdir().expect("tempdir");
    dman()
        .args(["materialize", "any-dataset"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
}

#[test]
fn materialize_stub_prints_not_implemented_to_stderr() {
    let home = tempdir().expect("tempdir");
    dman()
        .args(["materialize", "some-ds"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn materialize_stub_includes_dataset_name_in_message() {
    let home = tempdir().expect("tempdir");
    dman()
        .args(["materialize", "my-virtual-ds"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("my-virtual-ds"));
}

#[test]
fn operate_stub_exits_zero() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("operate")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
}

#[test]
fn operate_stub_prints_not_implemented_to_stderr() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("operate")
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}

#[test]
fn serve_stub_exits_zero() {
    let home = tempdir().expect("tempdir");
    dman()
        .arg("serve")
        .env("DMAN_HOME", home.path())
        .assert()
        .success();
}

#[test]
fn serve_stub_with_custom_port_exits_zero() {
    let home = tempdir().expect("tempdir");
    dman()
        .args(["serve", "--port", "9090"])
        .env("DMAN_HOME", home.path())
        .assert()
        .success()
        .stderr(predicate::str::contains("not yet implemented"));
}
