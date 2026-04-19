.PHONY: help build check test fmt fmt-check clippy verify \
	python-install python-build \
	frontend-install frontend-dev frontend-build frontend-lint \
	clean

DEFAULT_GOAL := help

CARGO := cargo
PYTHON := python3
PIP := $(PYTHON) -m pip
MATURIN := maturin
NPM := npm
FRONTEND_DIR := crates/server/frontend
PYTHON_MANIFEST := crates/python/Cargo.toml

help:
	@printf "Available targets:\n"
	@printf "  build            Build the Rust workspace\n"
	@printf "  check            Run cargo check on the workspace\n"
	@printf "  test             Run workspace tests\n"
	@printf "  fmt              Format Rust code\n"
	@printf "  fmt-check        Check Rust formatting\n"
	@printf "  clippy           Run clippy with warnings denied\n"
	@printf "  verify           Run formatting, lint, tests, and frontend build/lint\n"
	@printf "  python-install   Install the Python package in editable mode\n"
	@printf "  python-build     Build the Python package via maturin\n"
	@printf "  frontend-install Install frontend dependencies\n"
	@printf "  frontend-dev     Start the Vite dev server\n"
	@printf "  frontend-build   Build the frontend\n"
	@printf "  frontend-lint    Lint the frontend\n"
	@printf "  clean            Clean Rust and frontend build artifacts\n"

build:
	$(CARGO) build --workspace

check:
	$(CARGO) check --workspace

test:
	$(CARGO) test --workspace

fmt:
	$(CARGO) fmt --all

fmt-check:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --workspace --all-targets -- -D warnings

verify: fmt-check clippy test frontend-lint frontend-build

python-install:
	$(PIP) install -e .

python-build:
	$(MATURIN) build --manifest-path $(PYTHON_MANIFEST)

frontend-install:
	$(NPM) --prefix $(FRONTEND_DIR) install

frontend-dev:
	$(NPM) --prefix $(FRONTEND_DIR) run dev

frontend-build:
	$(NPM) --prefix $(FRONTEND_DIR) run build

frontend-lint:
	$(NPM) --prefix $(FRONTEND_DIR) run lint

clean:
	$(CARGO) clean
	$(NPM) --prefix $(FRONTEND_DIR) run --if-present clean
