# polars-lance

A Polars IO plugin for Lance datasets: `scan_lance` (lazy read), `write_lance` (eager
write), `sink_lance` (streaming write). A Rust `cdylib` built with maturin/PyO3 wraps the
`lance` crate; a thin Python package registers it with Polars and translates predicates.

## Commands

`just` is the entry point for everything; it drives `uv` and `maturin`.

| Command | What it does |
| --- | --- |
| `just develop` | `uv sync` + `maturin develop` (debug build) |
| `just develop-release` | release build — needed for some tests, see below |
| `just test` | debug build, full pytest suite (needs Docker) |
| `just test-no-docker` | skips the MinIO/Azurite integration tests |
| `just test-release` | release build, no-Docker suite |
| `just test-rust` | `cargo test` only |
| `just lint-rust` | `cargo fmt --check`, `cargo check` with and without `pyo3` |
| `just lint-pyth` | `ruff check`, `ruff format --check`, `mypy` |
| `just build-docs` | pdoc into `site/` |

Building compiles the whole `lance` tree; the first build is slow. `cargo check` needs
`--features pyo3` to cover `src/py.rs`, which is why `lint-rust` runs it twice.

## Layout

```
src/
  lib.rs      module wiring; re-exports the pure-Rust API
  py.rs       PyO3 layer (behind the `pyo3` feature): classes, pyfunctions, error mapping
  scan.rs     LanceReader / LanceScanner over lance::Dataset
  write.rs    WriteParams construction, storage-version resolution, batch production
  blob.rs     blob v2 extension type: mark on write, unwrap on read
  arrow.rs    arrow-rs <-> polars-arrow bridge over the Arrow C data interface
  err.rs      LanceScannerError / LanceWriterError (macro-generated, identical shapes)
  io.rs       StorageOptions = HashMap<String, String>
python/polars_lance/
  __init__.py       public API + the io_source closure handed to register_io_source
  _predicate.py     Polars predicate -> Lance SQL filter string
  _polars_lance.pyi type stubs for the extension module
tests/                pytest; `utils.py` holds the shared Arrow fixtures
examples/compare_with_pyarrow.py  benchmark vs pl.scan_pyarrow_dataset
```

## Invariants worth not breaking

**Predicate pushdown must be a superset, never a subset.** `_predicate.py` translates only
part of a predicate; `scan_lance`'s `io_source` re-applies the *full* Polars predicate to
every batch (Polars does not do this for IO plugins). So a translation that matches extra
rows is fine, one that misses rows silently loses data. Concretely:

- A conjunction may drop an untranslatable side; a disjunction may not.
- `Not` is never translated — negating a weakened translation inverts superset into subset.
- Anything that returns `None` just means "Polars will handle it". `None` is always safe.

**Predicates are read from `Expr.meta.serialize(format="json")`.** That IR is not a stable
Polars API. Breakage costs pushdown, not correctness — but it breaks silently, so the
equivalence suite is the safety net.

**Every new translated construct needs a case in `tests/test_predicate_equivalence.py`.**
That file runs each predicate through three engines (Polars, Lance SQL, Polars' own
`SQLContext`) over a table containing nulls, empty strings/lists, null structs, and regex
metacharacters. `test_translation_is_sound` is the correctness gate; `test_translation_is_exact`
catches needless weakening.

**Deliberately not translated**, because Polars and SQL disagree: `NaN` comparisons
(Polars says `NaN == NaN`), `IS NULL` on a struct or on a field read out of one
(lance#7908 — the filter and the scan can disagree per row), `list.get` (1- vs 0-based,
out-of-bounds raises vs null), tz-aware timestamp literals, `is_in` containing null.

**Limit pushdown is disabled when a predicate is pushed.** Rows are dropped after the scan,
so `n_rows` is honoured in Python instead. See the `n_rows=n_rows if predicate is None`
line in `scan_lance`.

**Aggregation pushdown is impossible, not merely absent.** `register_io_source` calls the
plugin with `(with_columns, predicate, n_rows, batch_size)` and nothing else, so an
aggregate is indistinguishable from an ordinary read. Measured against Polars 1.40:
`select(pl.len())` arrives as a single-column projection, `.count()` as a full scan
(`with_columns=None`), `select(col.sum())` as a plain column read. Do not go looking for a
hook; there isn't one. If Polars grows one, this is the note to revisit.

**The Arrow bridge in `arrow.rs` is load-bearing and subtle.** arrow-rs and polars-arrow are
distinct crates; conversion goes through the C data interface with `transmute` between the
two crates' FFI structs. Two shapes cannot go through FFI and are rebuilt by hand:

- Anything containing the **null type** — Polars exports a null array with one buffer,
  arrow-rs allows none (`contains_null_dtype` gates this).
- A **sliced struct with nulls** — Polars leaves validity at an offset while the children
  are already sliced, so arrow-rs applies the offset twice and reads past the end
  (`struct_offset_exceeds_children` / `compact_struct_validity`).

If you touch this file, `tests/test_nested.py` is what protects it.

**Storage version defaults to 2.2**, ahead of Lance's own 2.1 default
(`DEFAULT_DATA_STORAGE_VERSION`). 2.0 does not record struct validity; neither 2.0 nor 2.1
can store a blob column's nulls (lance#7955), so `blob_columns` is refused before 2.2.
Appending keeps the existing dataset's version.

**Blob columns are named, not typed.** A Polars schema carries no per-field metadata or
extension types, so `blob_columns=[...]` is applied to the Arrow schema on the way out
(`mark_blob_columns`), and the batch's binary column is wrapped in the `struct<data, uri>`
the extension type declares (`wrap_blob_v2_columns`). On read the extension name is stripped
(`unwrap_blob_v2_fields` / `unwrap_blob_v2_batch`) because Polars cannot build a series from
an extension type — a *filtered* scan restates the extension name on the batch, an
unfiltered one does not, hence both functions.

**Debug vs release builds behave differently.** Lance guards parts of its 2.1 encoder with
`debug_assert!`s that a nullable nested column trips (lance#8032, #8033). Tests read
`_polars_lance._debug_assertions` (exported from `py.rs`) and skip or pin version 2.0
accordingly. CI runs a dedicated release-build job so those paths are covered for real.

**`sink_lance` must not hold the GIL.** `write_lance_stream` calls `py.detach()` because
`PyDataFrameBatchReader::fill` re-acquires the GIL from a Tokio worker to pull the next
dataframe. Removing the detach deadlocks.

## Conventions

- Rust comments explain *why*, referencing upstream issue numbers where behaviour is a
  workaround. Keep that style; do not add narration of what the code does.
- Tests carry a docstring or comment stating the failure they prevent.
- Python is fully typed; `mypy` runs with `disallow_untyped_defs` over `python/` and `tests/`.
- Public Python docstrings are numpydoc (pdoc renders them with `-d numpy`).
- Errors: `LanceScannerError`/`LanceWriterError` wrap Lance/Arrow/Polars errors. Only the
  Polars variant maps to a Polars exception; everything else becomes `RuntimeError`.
- Cloud storage tests use testcontainers (MinIO, Azurite) and are marked `needs_docker`.
