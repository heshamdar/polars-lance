default:
    @just --list

setup:
    uv sync --group dev --no-install-project

develop: setup
    uv run maturin develop

# Lance guards parts of its 2.1 encoder with `debug_assert!`s that a nullable nested column can
# trip (lance-format/lance#8032, #8033). A debug build cannot write that data at all, so some
# tests pin storage version 2.0 or skip. This builds what a released wheel is, letting those
# tests cover the real behaviour.
develop-release: setup
    uv run maturin develop --release

test-rust:
    cargo test

# The conformance suite's own tests: its generator, its controls, and the mutation check that
# every deliberately-broken harness is caught by some case. Does not need the extension.
test-conformance-self:
    uv run --project polars-io-conformance pytest polars-io-conformance/tests

# Run the conformance corpus against this plugin. Needs a release build: a fixed-size-list
# column trips a debug assertion on the way in.
test-conformance: develop-release
    uv run pytest tests/test_conformance.py

# The capability matrix, with the reference formats beside this plugin so a failure attributes to
# one or the other.
capabilities: develop-release
    uv run python -m plioc.report CAPABILITIES.md tests.conformance_harness:LanceHarness

test: develop
    uv run pytest

test-no-docker: develop
    uv run pytest -m "not needs_docker"

test-release: develop-release
    uv run pytest -m "not needs_docker"

test-versions:
    UV_PYTHON=3.10 just test
    UV_PYTHON=3.11 just test
    UV_PYTHON=3.12 just test
    UV_PYTHON=3.13 just test
    UV_PYTHON=3.14 just test

lint-rust:
    cargo fmt --check
    cargo check
    cargo check --features pyo3

lint-pyth: setup
    uv run --no-project ruff check
    uv run --no-project ruff format --check
    uv run --no-project mypy

lint-conformance:
    uv run --project polars-io-conformance ruff check polars-io-conformance
    uv run --project polars-io-conformance ruff format --check polars-io-conformance
    uv run --project polars-io-conformance mypy --config-file polars-io-conformance/pyproject.toml

lint-workflows:
    actionlint

fmt-rust:
    cargo fmt

fmt-pyth:
    uv run --no-project ruff format

build-docs: develop
    rm -rf site
    uv run pdoc -o site -d numpy polars_lance
