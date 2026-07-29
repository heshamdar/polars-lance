from collections.abc import Callable
from pathlib import Path

import lance
import polars as pl
import pytest
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer

from polars_lance import scan_lance, sink_lance, write_lance
from tests.utils import (
    SUPPORTED_DATA_TYPES_ARROW_TABLE,
    az_storage_options,
    s3_storage_options,
)


def test_write_lance_data_types(tmp_path: Path) -> None:
    df = pl.DataFrame(SUPPORTED_DATA_TYPES_ARROW_TABLE)
    dataset_path = tmp_path / "test.lance"

    write_lance(df, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(df)


def test_write_lance_str_target(tmp_path: Path) -> None:
    df = pl.DataFrame(SUPPORTED_DATA_TYPES_ARROW_TABLE)
    dataset_path = tmp_path / "test.lance"

    write_lance(df, target=str(dataset_path))

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == SUPPORTED_DATA_TYPES_ARROW_TABLE.num_rows


# A null struct must stay null, rather than becoming a valid struct holding filler values.
# That needs data storage version 2.1 or later, where Lance's own default is 2.0.
@pytest.mark.parametrize("write", [write_lance, sink_lance], ids=["write", "sink"])
def test_null_structs_survive_a_round_trip(
    tmp_path: Path, write: Callable[..., None]
) -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "st": [{"x": 1, "name": "p"}, None, {"x": None, "name": None}],
        },
        schema={"id": pl.Int64, "st": pl.Struct({"x": pl.Int64, "name": pl.String})},
    )
    dataset_path = tmp_path / "structs.lance"

    write(df.lazy() if write is sink_lance else df, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert ds.data_storage_version == "2.2"
    # The null struct, and the valid struct whose fields are null, stay distinguishable.
    assert ds.to_table().column("st").null_count == 1
    assert pl.DataFrame(ds.to_table()).equals(df)
    assert scan_lance(dataset_path).collect().equals(df)


def test_write_lance_empty_dataframe(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"id": pl.Int64, "val": pl.String})
    assert df.is_empty()
    dataset_path = tmp_path / "empty.lance"

    write_lance(df, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == 0


# `sink_lance` streams the query's batches, so the rows never all sit in memory. These
# tests pin down that the streamed result matches collecting the query and writing that.
def test_sink_lance(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    dataset_path = tmp_path / "lazy.lance"

    sink_lance(lf, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(lf.collect())


def test_sink_lance_data_types(tmp_path: Path) -> None:
    lf = pl.DataFrame(SUPPORTED_DATA_TYPES_ARROW_TABLE).lazy()
    dataset_path = tmp_path / "lazy_types.lance"

    sink_lance(lf, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(lf.collect())


def test_sink_lance_applies_the_query(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"id": list(range(10)), "grp": [i % 3 for i in range(10)]})
    query = lf.filter(pl.col("grp") == 1).select(["id"])
    dataset_path = tmp_path / "query.lance"

    sink_lance(query, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(query.collect())


def test_sink_lance_empty(tmp_path: Path) -> None:
    lf = pl.LazyFrame(schema={"id": pl.Int64, "val": pl.String})
    dataset_path = tmp_path / "lazy_empty.lance"

    sink_lance(lf, target=dataset_path)

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == 0
    assert ds.schema.names == ["id", "val"]


# `chunk_size` makes the streaming engine produce several batches, all of which must be
# written rather than only the first.
def test_sink_lance_multiple_batches(tmp_path: Path) -> None:
    rows = 10_000
    lf = pl.LazyFrame({"id": list(range(rows))})
    dataset_path = tmp_path / "batches.lance"

    sink_lance(lf, target=dataset_path, chunk_size=1_000)

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == rows
    assert pl.DataFrame(ds.to_table())["id"].to_list() == list(range(rows))


def test_sink_lance_append_mode(tmp_path: Path) -> None:
    first = pl.LazyFrame({"id": [1], "val": ["a"]})
    second = pl.LazyFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "lazy_append.lance"
    sink_lance(first, target=dataset_path)

    sink_lance(second, target=dataset_path, mode="append")

    ds = lance.dataset(dataset_path)
    expected_df = pl.concat([first.collect(), second.collect()], how="vertical")
    assert pl.DataFrame(ds.to_table()).equals(expected_df)


def test_sink_lance_overwrite_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "lazy_overwrite.lance"
    sink_lance(pl.LazyFrame({"id": [1]}), target=dataset_path)
    overwriting = pl.LazyFrame({"id": [9]})

    sink_lance(overwriting, target=dataset_path, mode="overwrite")

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(overwriting.collect())


def test_sink_lance_error_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "lazy_error.lance"
    sink_lance(pl.LazyFrame({"id": [1]}), target=dataset_path)

    with pytest.raises(FileExistsError):
        sink_lance(pl.LazyFrame({"id": [2]}), target=dataset_path)


def test_sink_lance_max_rows_per_file(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"id": list(range(10))})
    dataset_path = tmp_path / "lazy_rows.lance"

    sink_lance(lf, target=dataset_path, max_rows_per_file=2, chunk_size=3)

    ds = lance.dataset(dataset_path)
    assert ds.count_rows() == 10
    assert len(ds.get_fragments()) == 5


# Streaming a scan straight into a sink means the streaming engine drives `scan_lance` from
# a worker thread, so the scanner has to be usable from a thread other than the one that
# created it.
def test_sink_lance_from_scan_lance(tmp_path: Path) -> None:
    source_path = tmp_path / "source.lance"
    frame = pl.DataFrame({"id": list(range(1000)), "grp": [i % 4 for i in range(1000)]})
    write_lance(frame, target=source_path)
    target_path = tmp_path / "target.lance"

    query = scan_lance(source_path).filter(pl.col("grp") == 2).select(["id"])
    sink_lance(query, target=target_path, chunk_size=64)

    ds = lance.dataset(target_path)
    assert pl.DataFrame(ds.to_table()).equals(query.collect())


# An error raised while producing batches must surface rather than be swallowed, which
# would leave a silently truncated dataset.
def test_sink_lance_propagates_query_errors(tmp_path: Path) -> None:
    lf = pl.LazyFrame({"id": [1, 2, 3]}).select(
        pl.col("id").map_elements(lambda value: 1 // 0, return_dtype=pl.Int64)
    )
    dataset_path = tmp_path / "lazy_error_propagation.lance"

    with pytest.raises(RuntimeError, match="ZeroDivisionError"):
        sink_lance(lf, target=dataset_path)

    assert not dataset_path.exists()


def test_write_lance_error_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    with pytest.raises(FileExistsError, match="already exists"):
        write_lance(second, target=dataset_path, mode="error")


def test_write_lance_append_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    write_lance(second, target=dataset_path, mode="append")

    ds = lance.dataset(dataset_path)
    expected_df = pl.concat([first, second], how="vertical")
    assert pl.DataFrame(ds.to_table()).equals(expected_df)


def test_write_lance_overwrite_mode(tmp_path: Path) -> None:
    first = pl.DataFrame({"id": [1], "val": ["a"]})
    second = pl.DataFrame({"id": [2], "val": ["b"]})
    dataset_path = tmp_path / "test.lance"
    write_lance(first, target=dataset_path)

    write_lance(second, target=dataset_path, mode="overwrite")

    ds = lance.dataset(dataset_path)
    assert pl.DataFrame(ds.to_table()).equals(second)


@pytest.mark.needs_docker
def test_write_lance_s3_storage_options(minio: tuple[MinioContainer, str]) -> None:
    minio_container, bucket_name = minio
    df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"s3://{bucket_name}/my_dataset.lance"
    storage_options = s3_storage_options(minio_container)

    write_lance(df, target=uri, storage_options=storage_options)

    ds = lance.dataset(uri, storage_options=storage_options)
    assert pl.DataFrame(ds.to_table()).equals(df)


@pytest.mark.needs_docker
def test_write_lance_az_storage_options(
    azurite: tuple[AzuriteContainer, str],
) -> None:
    azurite_container, container_name = azurite
    df = pl.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
    uri = f"az://{container_name}/my_dataset.lance"
    storage_options = az_storage_options(azurite_container)

    write_lance(df, target=uri, storage_options=storage_options)

    ds = lance.dataset(uri, storage_options=storage_options)
    assert pl.DataFrame(ds.to_table()).equals(df)


def test_write_lance_max_rows_per_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
    dataset_path = tmp_path / "test.lance"

    write_lance(df, target=dataset_path, max_rows_per_file=2)

    ds = lance.dataset(dataset_path)
    data_files = [
        data_file
        for fragment in ds.get_fragments()
        for data_file in fragment.data_files()
    ]
    assert len(data_files) == 3
