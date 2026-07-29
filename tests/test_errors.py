"""
Failures have to be distinguishable.

Every error used to arrive as `RuntimeError`, so a caller could not tell a missing dataset
from a write that refused to clobber one without matching on the message. These pin the
mapping down; `PolarsLanceError` still derives from `RuntimeError`, so the older, coarser
`except RuntimeError` keeps working.
"""

from pathlib import Path

import lance
import polars as pl
import pytest

from polars_lance import (
    CommitConflictError,
    PolarsLanceError,
    scan_lance,
    sink_lance,
    write_lance,
)


@pytest.fixture
def frame() -> pl.DataFrame:
    return pl.DataFrame({"id": [1, 2], "blob": [b"a", b"b"]})


def test_scanning_a_missing_dataset_is_a_file_not_found_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan_lance(tmp_path / "absent.lance")


def test_writing_over_an_existing_dataset_is_a_file_exists_error(
    tmp_path: Path, frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "test.lance"
    write_lance(frame, target=dataset_path)

    with pytest.raises(FileExistsError):
        write_lance(frame, target=dataset_path, mode="error")


def test_sinking_over_an_existing_dataset_is_a_file_exists_error(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "test.lance"
    sink_lance(pl.LazyFrame({"id": [1]}), target=dataset_path)

    with pytest.raises(FileExistsError):
        sink_lance(pl.LazyFrame({"id": [2]}), target=dataset_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"blob_columns": ["id"]}, "has to be binary", id="not_binary"),
        pytest.param(
            {"blob_columns": ["nope"]}, "not in the frame", id="unknown_column"
        ),
        pytest.param(
            {"blob_columns": ["blob"], "data_storage_version": "2.0"},
            "blob_columns needs data storage",
            id="blob_before_2_2",
        ),
        pytest.param(
            {"data_storage_version": "9.9"},
            "Unknown Lance storage version",
            id="unknown_version",
        ),
        pytest.param({"mode": "bogus"}, "`mode` must be one of", id="bad_mode"),
    ],
)
def test_bad_arguments_are_value_errors(
    tmp_path: Path, frame: pl.DataFrame, kwargs: dict[str, object], message: str
) -> None:
    """An argument the caller got wrong is a `ValueError`, not a `RuntimeError`."""
    with pytest.raises(ValueError, match=message):
        write_lance(frame, target=tmp_path / "bad.lance", **kwargs)  # type: ignore[arg-type]


def test_a_value_error_is_not_swallowed_by_the_runtime_error_base(
    tmp_path: Path, frame: pl.DataFrame
) -> None:
    """
    `ValueError` has to stay outside the `PolarsLanceError` tree.

    Rooting everything at `PolarsLanceError` would make `except ValueError` miss the argument
    errors, which is the case this mapping exists to serve.
    """
    with pytest.raises(ValueError) as caught:
        write_lance(frame, target=tmp_path / "bad.lance", blob_columns=["id"])

    assert not isinstance(caught.value, PolarsLanceError)


def test_the_base_error_is_still_a_runtime_error() -> None:
    """Code written against the old behaviour keeps catching what it used to."""
    assert issubclass(PolarsLanceError, RuntimeError)
    assert issubclass(CommitConflictError, PolarsLanceError)


def test_a_query_error_reaches_the_caller(tmp_path: Path) -> None:
    """
    An exception raised by the caller's own query must not be reshaped into an argument error.

    It travels out of a streaming write as one of Lance's internal Arrow failures, which is
    why those are left on the base class rather than mapped to `ValueError`.
    """
    lf = pl.LazyFrame({"id": [1, 2, 3]}).select(
        pl.col("id").map_elements(lambda value: 1 // 0, return_dtype=pl.Int64)
    )

    with pytest.raises(RuntimeError, match="ZeroDivisionError") as caught:
        sink_lance(lf, target=tmp_path / "boom.lance")

    assert not isinstance(caught.value, ValueError), (
        "a failure inside the caller's query was reported as a bad argument"
    )


def test_reading_a_dataset_whose_data_is_gone(
    tmp_path: Path, frame: pl.DataFrame
) -> None:
    """A dataset that opens but cannot be read fails at collect, not at scan."""
    dataset_path = tmp_path / "test.lance"
    write_lance(frame, target=dataset_path)
    for data_file in (dataset_path / "data").glob("*.lance"):
        data_file.unlink()

    lf = scan_lance(dataset_path)

    with pytest.raises(Exception):
        lf.collect()


def test_lance_written_elsewhere_still_scans(tmp_path: Path) -> None:
    """The mapping must not have broken the ordinary path."""
    dataset_path = tmp_path / "test.lance"
    lance.write_dataset(pl.DataFrame({"id": [1, 2, 3]}).to_arrow(), dataset_path)

    assert scan_lance(dataset_path).collect().height == 3
