"""Blob columns.

Lance stores a large binary value out of line when its field is marked as a blob, so a scan
that does not select the column never reads the bytes. Polars carries no per-field metadata and
no extension types, so a column cannot say for itself that it is a blob; it is requested by name
through `blob_columns`.

Lance has two ways of describing such a column, and the data storage version picks between them:
an extension type over `struct<data, uri>` from version 2.2, and `lance-encoding:blob` field
metadata before it. Only the newer one keeps a null reliably, so `blob_columns` writes at 2.2 by
default. The tests below cover both, since a caller can still ask for an older version.
"""

from __future__ import annotations

from pathlib import Path

import lance
import polars as pl
import pytest

from polars_lance import scan_lance, sink_lance, write_lance

LEGACY_BLOB_METADATA = {b"lance-encoding:blob": b"true"}
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


def test_write_uses_blob_v2_by_default(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "blob.lance"

    write_lance(blob_frame, target=dataset_path, blob_columns=["blob"])

    dataset = lance.dataset(dataset_path)
    assert dataset.data_storage_version == "2.2"
    assert dataset.schema.field("blob").type.extension_name == BLOB_V2_EXTENSION_NAME


# A caller can still ask for an older version, which only understands the metadata marker.
def test_write_uses_the_legacy_marker_before_2_2(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "legacy.lance"

    write_lance(
        blob_frame,
        target=dataset_path,
        blob_columns=["blob"],
        data_storage_version="2.1",
    )

    dataset = lance.dataset(dataset_path)
    assert dataset.data_storage_version == "2.1"
    assert dataset.schema.field("blob").metadata == LEGACY_BLOB_METADATA


# Only a blob column moves to 2.2; everything else keeps the default that preserves struct
# validity.
def test_without_blob_columns_it_is_an_ordinary_column(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "plain.lance"

    write_lance(blob_frame, target=dataset_path)

    dataset = lance.dataset(dataset_path)
    assert dataset.data_storage_version == "2.1"
    assert dataset.schema.field("blob").metadata is None


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


# One row per batch puts the null in a batch of its own, which the legacy layout cannot encode.
# The extension type records nullability per value, so the batch boundaries stop mattering.
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


# Before 2.2, batches of one write that disagree about nullability cannot be encoded, so the
# write is refused rather than silently storing the null as an empty value.
def test_legacy_blob_nulls_across_disagreeing_batches_are_refused(
    tmp_path: Path, nullable_blob_frame: pl.DataFrame
) -> None:
    with pytest.raises(RuntimeError, match="has nulls in some batches"):
        sink_lance(
            nullable_blob_frame.lazy(),
            target=tmp_path / "legacy_nulls.lance",
            blob_columns=["blob"],
            data_storage_version="2.1",
            chunk_size=1,
        )


# Batches that agree are fine even under the legacy layout.
def test_legacy_blob_nulls_in_one_batch(
    tmp_path: Path, nullable_blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "legacy_one_batch.lance"

    write_lance(
        nullable_blob_frame,
        target=dataset_path,
        blob_columns=["blob"],
        data_storage_version="2.1",
    )

    assert scan_lance(dataset_path).collect().equals(nullable_blob_frame)


# A blob v2 dataset written by another tool has to be readable, which means presenting the
# extension type as the binary column a scan returns.
def test_reads_a_blob_v2_dataset_written_elsewhere(tmp_path: Path) -> None:
    import pyarrow as pa

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
    with pytest.raises(RuntimeError, match=message):
        write_lance(blob_frame, target=tmp_path / "bad.lance", blob_columns=columns)
