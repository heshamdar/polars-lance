"""Blob columns.

Lance stores a large binary value out of line when its field is marked as a blob, so a scan
that does not select the column never reads the bytes. Polars carries no per-field metadata,
so the marker cannot travel with the column and is instead requested by name through
`blob_columns`.
"""

from __future__ import annotations

from pathlib import Path

import lance
import polars as pl
import pytest

from polars_lance import scan_lance, sink_lance, write_lance

BLOB_METADATA = {b"lance-encoding:blob": b"true"}


@pytest.fixture
def blob_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"id": [1, 2, 3], "blob": [b"a" * 5000, b"b" * 7000, b"c" * 100]},
        schema={"id": pl.Int64, "blob": pl.Binary},
    )


def test_write_marks_the_column_as_a_blob(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "blob.lance"

    write_lance(blob_frame, target=dataset_path, blob_columns=["blob"])

    field = lance.dataset(dataset_path).schema.field("blob")
    assert field.metadata == BLOB_METADATA


def test_without_blob_columns_it_is_an_ordinary_column(
    tmp_path: Path, blob_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "plain.lance"

    write_lance(blob_frame, target=dataset_path)

    assert lance.dataset(dataset_path).schema.field("blob").metadata is None


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


def test_blob_nulls_survive(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "blob": [b"a" * 500, None]},
        schema={"id": pl.Int64, "blob": pl.Binary},
    )
    dataset_path = tmp_path / "nulls.lance"

    write_lance(frame, target=dataset_path, blob_columns=["blob"])

    assert scan_lance(dataset_path).collect().equals(frame)


# Streaming a blob column that holds a null trips an assertion inside Lance's encoder
# (`AllValidItem` against `NullableItem`). Writing the collected frame works.
@pytest.mark.xfail(
    reason="lance: streaming a nullable blob column fails an internal assertion"
)
def test_sink_blob_with_nulls(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        {"id": [1, 2], "blob": [b"a" * 500, None]},
        schema={"id": pl.Int64, "blob": pl.Binary},
    )
    dataset_path = tmp_path / "sink_nulls.lance"

    sink_lance(frame.lazy(), target=dataset_path, blob_columns=["blob"])

    assert scan_lance(dataset_path).collect().equals(frame)


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
