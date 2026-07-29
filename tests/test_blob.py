"""Blob columns.

Lance stores a large binary value out of line when its field is marked as a blob, so a scan
that does not select the column never reads the bytes. Polars carries no per-field metadata and
no extension types, so a column cannot say for itself that it is a blob; it is requested by name
through `blob_columns`.

Lance describes such a column with an extension type over `struct<data, uri>`, from data storage
version 2.2 on. An older layout using field metadata exists but cannot keep a null, so it is not
written at all: asking for both `blob_columns` and an earlier version is refused.
"""

from __future__ import annotations

from pathlib import Path

import lance
import polars as pl
import pyarrow as pa
import pytest

from polars_lance import scan_lance, sink_lance, write_lance

BLOB_V2_EXTENSION_NAME = "lance.blob.v2"


@pytest.fixture
def blob_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": [1, 2, 3], "blob": [b"a" * 5000, b"b" * 7000, b"c" * 100]},
        schema={"id": pl.Int64, "blob": pl.Binary},
    )


@pytest.fixture
def nullable_blob_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": [1, 2, 3], "blob": [b"a" * 500, None, b"cc"]},
        schema={"id": pl.Int64, "blob": pl.Binary},
    )


def test_write_uses_the_blob_extension_type(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "blob.lance"

    write_lance(blob_frame, target=dataset_path, blob_columns=["blob"])

    dataset = lance.dataset(dataset_path)
    assert dataset.data_storage_version == "2.2"
    assert dataset.schema.field("blob").type.extension_name == BLOB_V2_EXTENSION_NAME


# An older version cannot describe a blob column without risking its nulls, so asking for both
# is refused rather than quietly writing something lossy.
@pytest.mark.parametrize("version", ["2.0", "2.1", "stable"])
def test_blob_columns_before_2_2_are_refused(
    tmp_path: Path, blob_frame: pl.DataFrame, version: str
) -> None:
    with pytest.raises(ValueError, match="blob_columns needs data storage"):
        write_lance(
            blob_frame,
            target=tmp_path / "old.lance",
            blob_columns=["blob"],
            data_storage_version=version,
        )


def test_without_blob_columns_it_is_an_ordinary_column(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "plain.lance"

    write_lance(blob_frame, target=dataset_path)

    dataset = lance.dataset(dataset_path)
    assert dataset.schema.field("blob").metadata is None
    assert dataset.schema.field("blob").type == pa.large_binary()


# A scan asks Lance for the bytes, so the column arrives as the binary column the schema
# advertises rather than as a position and size.
@pytest.mark.parametrize("write", [write_lance, sink_lance], ids=["write", "sink"])
def test_blob_round_trip(
    tmp_path: Path, blob_frame: pl.DataFrame, write: object
) -> None:
    dataset_path = tmp_path / "round_trip.lance"
    frame = blob_frame.lazy() if write is sink_lance else blob_frame

    write(frame, target=dataset_path, blob_columns=["blob"])  # type: ignore[operator]

    scanned = scan_lance(dataset_path).collect()
    assert scanned.schema["blob"] == pl.Binary
    assert scanned.equals(blob_frame)


# The extension type records nullability per value, so it does not matter which batch a null
# lands in - one row per batch puts it in a batch of its own, which the older layout could not
# encode at all.
@pytest.mark.parametrize("chunk_size", [None, 1, 2], ids=["default", "one", "two"])
@pytest.mark.parametrize("write", [write_lance, sink_lance], ids=["write", "sink"])
def test_blob_nulls_survive(
    tmp_path: Path,
    nullable_blob_frame: pl.DataFrame,
    write: object,
    chunk_size: int | None,
) -> None:
    if write is write_lance and chunk_size is not None:
        pytest.skip("chunk_size only applies to a streamed write")
    dataset_path = tmp_path / "nulls.lance"
    frame = nullable_blob_frame.lazy() if write is sink_lance else nullable_blob_frame
    extra = {"chunk_size": chunk_size} if write is sink_lance else {}

    write(frame, target=dataset_path, blob_columns=["blob"], **extra)  # type: ignore[operator]

    assert scan_lance(dataset_path).collect().equals(nullable_blob_frame)


# The reason for preferring 2.2: compacting fragments that disagree about nullability rewrites a
# null as an empty value under the legacy layout (lance-format/lance#7955).
def test_blob_nulls_survive_compaction(
    tmp_path: Path, nullable_blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "compacted.lance"
    # The first fragment holds no nulls, the second does, which is the losing arrangement.
    write_lance(
        nullable_blob_frame.drop_nulls("blob"),
        target=dataset_path,
        blob_columns=["blob"],
    )
    sink_lance(
        nullable_blob_frame.lazy(),
        target=dataset_path,
        mode="append",
        blob_columns=["blob"],
    )
    before = scan_lance(dataset_path).collect()

    lance.dataset(dataset_path).optimize.compact_files()

    assert scan_lance(dataset_path).collect().equals(before)
    assert before["blob"].null_count() == 1


# A blob v2 dataset written by another tool has to be readable, which means presenting the
# extension type as the binary column a scan returns.
def test_reads_a_blob_v2_dataset_written_elsewhere(tmp_path: Path) -> None:
    dataset_path = tmp_path / "foreign.lance"
    schema = pa.schema(
        [pa.field("id", pa.int64()), lance.blob_field("blob", inline_size_threshold=1)]
    )
    lance.write_dataset(
        pa.table(
            [pa.array([0, 1], pa.int64()), lance.blob_array([b"a" * 400, None])],
            schema=schema,
        ),
        dataset_path,
        data_storage_version="2.2",
    )

    scanned = scan_lance(dataset_path).collect()

    assert scanned.schema["blob"] == pl.Binary
    assert scanned["blob"].to_list() == [b"a" * 400, None]


# Not selecting the column is the point of a blob: Lance does not read the bytes at all.
def test_projection_skips_the_blob(tmp_path: Path, blob_frame: pl.DataFrame) -> None:
    dataset_path = tmp_path / "projected.lance"
    write_lance(blob_frame, target=dataset_path, blob_columns=["blob"])

    scanned = scan_lance(dataset_path).select(["id"]).collect()

    assert scanned.equals(blob_frame.select(["id"]))


def test_filter_alongside_a_blob(tmp_path: Path, blob_frame: pl.DataFrame) -> None:
    dataset_path = tmp_path / "filtered.lance"
    write_lance(blob_frame, target=dataset_path, blob_columns=["blob"])

    scanned = scan_lance(dataset_path).filter(pl.col("id") > 1).collect()

    assert scanned.equals(blob_frame.filter(pl.col("id") > 1))


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        pytest.param(["id"], "has to be binary", id="not_binary"),
        pytest.param(["missing"], "not in the frame", id="unknown_column"),
    ],
)
def test_blob_columns_is_validated(
    tmp_path: Path, blob_frame: pl.DataFrame, columns: list[str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        write_lance(blob_frame, target=tmp_path / "bad.lance", blob_columns=columns)
