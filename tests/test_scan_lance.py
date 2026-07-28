from datetime import date
from pathlib import Path

import lance
import polars as pl
import pyarrow as pa
import pytest
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer

from polars_lance import scan_lance
from tests.utils import (
    FILTER_ARROW_TABLE,
    SUPPORTED_DATA_TYPES_ARROW_TABLE,
    az_storage_options,
    s3_storage_options,
    to_polars_arrow,
)


def test_scan_lance_data_types(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    df = scan_lance(dataset_path).collect()
    assert df.to_arrow() == to_polars_arrow(SUPPORTED_DATA_TYPES_ARROW_TABLE)


def test_scan_lance_str_source(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    df = scan_lance(str(dataset_path)).collect()
    assert df.height == SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows


# This test fails if the `with_columns` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_project(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    with_columns = ["int32", "string"]

    df = scan_lance(dataset_path).select(with_columns).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        with_columns=with_columns,
    )


# This test fails if the `predicate` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_filter(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    predicate = pl.col("int32") > 0

    df = scan_lance(dataset_path).filter(predicate).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        predicate=predicate,
    )


# Predicates are translated into a Lance filter to reduce the rows that are read, and
# re-applied to every batch. Both paths must produce the same result as filtering in
# Polars, whether the predicate is translatable in full, in part, or not at all.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(pl.col("int32") > 2, id="gt"),
        pytest.param(pl.col("int32") != 2, id="neq"),
        pytest.param(pl.col("float64") <= 3.5, id="lt_eq_float"),
        pytest.param(pl.col("string") == "b", id="eq_string"),
        pytest.param(pl.col("date32") > date(2024, 1, 3), id="gt_date"),
        pytest.param((pl.col("int32") > 1) & (pl.col("float64") < 5.0), id="and"),
        pytest.param((pl.col("int32") == 1) | (pl.col("int32") == 5), id="or"),
        pytest.param(pl.col("int32").is_in([1, 3, 5]), id="is_in"),
        pytest.param(pl.col("int32").is_null(), id="is_null"),
        pytest.param(pl.col("int32").is_not_null(), id="is_not_null"),
        pytest.param(pl.col("int32").is_between(2, 4), id="is_between"),
        pytest.param(~(pl.col("int32") > 2), id="not"),
        # Only the `int32` comparison can be pushed down; the rest is applied by the
        # scanner, so a partially translated predicate must not drop or duplicate rows.
        pytest.param(
            (pl.col("int32") > 1) & pl.col("string").str.contains("x"),
            id="partially_translatable",
        ),
        # Nothing can be pushed down here.
        pytest.param(pl.col("string").str.starts_with("x"), id="untranslatable"),
    ],
)
def test_scan_lance_filter_predicates(tmp_path: Path, predicate: pl.Expr) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(FILTER_ARROW_TABLE, dataset_path)

    df = scan_lance(dataset_path).filter(predicate).collect()
    assert df.to_arrow() == to_polars_arrow(FILTER_ARROW_TABLE, predicate=predicate)


# A filter combined with a slice must not return fewer rows than requested, which would
# happen if the slice were pushed down to Lance and rows were filtered out afterwards.
def test_scan_lance_filter_and_slice(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(FILTER_ARROW_TABLE, dataset_path)

    predicate = pl.col("int32") > 1
    n_rows = 2

    df = scan_lance(dataset_path).filter(predicate).head(n_rows).collect()
    assert df.height == n_rows
    assert df.to_arrow() == to_polars_arrow(
        FILTER_ARROW_TABLE,
        predicate=predicate,
        n_rows=n_rows,
    )


# Filtering on a column that is not selected must still work, because Polars projects the
# columns the predicate needs.
def test_scan_lance_filter_on_unselected_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(FILTER_ARROW_TABLE, dataset_path)

    predicate = pl.col("int32") > 2
    with_columns = ["string"]

    df = scan_lance(dataset_path).filter(predicate).select(with_columns).collect()

    # The predicate is applied before the projection here, so the expected result cannot
    # be built with `to_polars_arrow`, which projects first.
    expected_df = pl.from_arrow(FILTER_ARROW_TABLE)
    assert isinstance(expected_df, pl.DataFrame)
    expected = expected_df.lazy().filter(predicate).select(with_columns).collect()
    assert df.to_arrow() == expected.to_arrow()


# This test fails if the `n_rows` arg in the `io_source` passed to
# `register_io_source` in `scan_lance` is not correctly applied in the Rust
# `LanceScanner` impl.
def test_scan_lance_slice(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(SUPPORTED_DATA_TYPES_ARROW_TABLE, dataset_path)

    n_rows = 1
    assert n_rows < SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows  # sanity check

    df = scan_lance(dataset_path).head(n_rows).collect()
    assert df.to_arrow() == to_polars_arrow(
        SUPPORTED_DATA_TYPES_ARROW_TABLE,
        n_rows=n_rows,
    )


@pytest.mark.needs_docker
def test_scan_lance_s3_storage_options(minio: tuple[MinioContainer, str]) -> None:
    minio_container, bucket_name = minio
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"s3://{bucket_name}/my_dataset.lance"
    storage_options = s3_storage_options(minio_container)
    lance.write_dataset(table, uri, storage_options=storage_options)

    df = scan_lance(uri, storage_options=storage_options).collect()

    assert df.to_arrow() == to_polars_arrow(table)


@pytest.mark.needs_docker
def test_scan_lance_az_storage_options(azurite: tuple[AzuriteContainer, str]) -> None:
    azurite_container, container_name = azurite
    table = pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"az://{container_name}/my_dataset.lance"
    storage_options = az_storage_options(azurite_container)
    lance.write_dataset(table, uri, storage_options=storage_options)

    df = scan_lance(uri, storage_options=storage_options).collect()

    assert df.to_arrow() == to_polars_arrow(table)
