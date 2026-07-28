"""Round trips and filters over nested data: lists of structs, structs of lists, and lists
of lists.

Nested types are where a null can sit at several levels at once — a null list, an empty list,
a null struct inside a list, a null field inside a struct — and where the Arrow bridge between
Polars and Lance is most likely to lose one. Each test writes both eagerly and by streaming,
because the two take different paths through the bridge.

Filters are compared against Polars applied to the data *as Lance stored it*, so that a
filtering mistake is not confused with a storage limitation. The round trip tests check
storage fidelity separately.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import lance
import polars as pl
import pytest

from polars_lance import scan_lance, sink_lance, write_lance
from tests.utils import DEBUG_ASSERTIONS, NESTED_ARROW_TABLE

WRITERS: list[Callable[..., None]] = [write_lance, sink_lance]

# Every version from 2.1 on trips a `debug_assert!` in Lance's repdef serializer when a list of
# structs holds a null struct alongside an empty or null list (lance-format/lance#8032; checked at
# 2.1, 2.2 and 2.3). It reproduces from hand-built arrow-rs arrays with no Polars involved, and
# being a debug assertion it does not fire in a release build, which is why the same data written
# through `pylance` succeeds. `just develop` builds debug, so most of these tests pin version 2.0,
# the one version that writes the data — at the cost of not recording a struct's own validity.
# What a release build does with this column is covered by
# `test_list_of_struct_keeps_a_null_struct_at_the_default_version`.
STORAGE_VERSION = "2.0"

# A version 2.0 dataset does not record a struct's own validity, so a null struct inside a list
# reads back as a struct of filler values.
FILLER_STRUCT = {"x": 0, "name": ""}


def write_frame(
    write: Callable[..., None], df: pl.DataFrame, target: Path, **kwargs: object
) -> None:
    """Write with `write_lance` or `sink_lance`, whichever was parametrized."""
    write(df.lazy() if write is sink_lance else df, target=target, **kwargs)


@pytest.fixture
def nested_frame() -> pl.DataFrame:
    """Nested columns that round trip exactly, at the default storage version.

    `list_of_struct` is excluded: the test build pins it to version 2.0 (see
    `STORAGE_VERSION`), which in turn cannot keep a struct's own validity, so the two cannot be
    exercised together.
    """
    frame = pl.from_arrow(NESTED_ARROW_TABLE)
    assert isinstance(frame, pl.DataFrame)
    return frame.drop("list_of_struct")


@pytest.fixture
def list_of_struct_frame() -> pl.DataFrame:
    frame = pl.from_arrow(NESTED_ARROW_TABLE)
    assert isinstance(frame, pl.DataFrame)
    return frame.select(["id", "list_of_struct"])


@pytest.fixture
def nested_dataset(tmp_path: Path, nested_frame: pl.DataFrame) -> Path:
    dataset_path = tmp_path / "nested.lance"
    write_lance(nested_frame, target=dataset_path)
    return dataset_path


@pytest.fixture
def list_of_struct_dataset(tmp_path: Path, list_of_struct_frame: pl.DataFrame) -> Path:
    dataset_path = tmp_path / "list_of_struct.lance"
    write_lance(
        list_of_struct_frame, target=dataset_path, data_storage_version=STORAGE_VERSION
    )
    return dataset_path


@pytest.mark.parametrize("write", WRITERS, ids=["write", "sink"])
def test_nested_round_trip(
    tmp_path: Path, nested_frame: pl.DataFrame, write: Callable[..., None]
) -> None:
    dataset_path = tmp_path / "round_trip.lance"

    write_frame(write, nested_frame, dataset_path)

    assert scan_lance(dataset_path).collect().equals(nested_frame)


# Written in several batches, so a nested value is split across batches rather than being
# converted in one piece.
def test_nested_round_trip_in_batches(
    tmp_path: Path, nested_frame: pl.DataFrame
) -> None:
    dataset_path = tmp_path / "batches.lance"

    sink_lance(nested_frame.lazy(), target=dataset_path, chunk_size=2)

    assert lance.dataset(dataset_path).count_rows() == nested_frame.height
    assert scan_lance(dataset_path).collect().equals(nested_frame)


# A slice leaves an offset on the underlying arrays, which the bridge has to account for.
@pytest.mark.parametrize("write", WRITERS, ids=["write", "sink"])
@pytest.mark.parametrize(("offset", "length"), [(1, 3), (2, 2), (3, 1), (0, 4)])
def test_sliced_nested_round_trip(
    tmp_path: Path,
    nested_frame: pl.DataFrame,
    write: Callable[..., None],
    offset: int,
    length: int,
) -> None:
    sliced = nested_frame.slice(offset, length)
    dataset_path = tmp_path / "sliced.lance"

    write_frame(write, sliced, dataset_path)

    assert scan_lance(dataset_path).collect().equals(sliced)


@pytest.mark.parametrize("write", WRITERS, ids=["write", "sink"])
def test_nested_nulls_stay_distinguishable(
    tmp_path: Path, nested_frame: pl.DataFrame, write: Callable[..., None]
) -> None:
    """An empty list, a null list, and a null field must not be conflated."""
    dataset_path = tmp_path / "nulls.lance"

    write_frame(write, nested_frame, dataset_path)
    scanned = scan_lance(dataset_path).collect()

    nested_lists = scanned["list_of_list"].to_list()
    assert nested_lists[1] == [[]]  # list holding an empty list
    assert nested_lists[2] is None  # null list
    assert nested_lists[3] == [None]  # list holding a null list
    assert nested_lists[4] == [[None]]  # list holding a list of one null

    structs = scanned["struct_of_list"].to_list()
    assert structs[1]["values"] == []  # empty list inside a struct
    assert structs[2] is None  # null struct
    assert structs[3]["values"] is None  # null list inside a struct
    assert structs[4]["values"] == [None, 5]  # null element inside a list


# `list.len` and `list.contains` are pushed into Lance; anything reaching inside a list of
# structs is applied by Polars. Either way the rows must match what Polars alone selects.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(
            pl.col("list_of_list").is_null(),
            id="null_list",
            # A filtered scan reports a null struct as a struct of nulls, while an
            # unfiltered scan reports it as null. Reproduced identically through
            # `pylance`, so it is Lance's behaviour rather than this bridge's.
            marks=pytest.mark.xfail(
                reason="lance#7908: filtered and unfiltered scans disagree on a null struct"
            ),
        ),
        pytest.param(pl.col("list_of_list").is_not_null(), id="not_null_list"),
        pytest.param(pl.col("struct_of_list").struct.field("label") == "a", id="field"),
        pytest.param(
            pl.col("struct_of_list").struct.field("values").list.len() > 1,
            id="field_list_len",
        ),
        pytest.param(pl.col("list_of_list").list.len() > 1, id="nested_list_len"),
        pytest.param(
            pl.col("struct_of_list").struct.field("label").is_null(), id="null_field"
        ),
    ],
)
def test_nested_filter_matches_polars(nested_dataset: Path, predicate: pl.Expr) -> None:
    stored = scan_lance(nested_dataset).collect()

    scanned = scan_lance(nested_dataset).filter(predicate).collect()

    assert scanned.equals(stored.filter(predicate))


# Appending nested data has to keep the existing rows alongside the new ones.
def test_nested_append(nested_dataset: Path, nested_frame: pl.DataFrame) -> None:
    stored = scan_lance(nested_dataset).collect()

    sink_lance(nested_frame.lazy(), target=nested_dataset, mode="append")

    scanned = scan_lance(nested_dataset).collect()
    assert scanned.height == 2 * stored.height
    assert scanned.equals(pl.concat([stored, stored], how="vertical"))


# Projecting one nested column must not disturb the others.
@pytest.mark.parametrize(
    "columns",
    [["id", "struct_of_list"], ["list_of_list"], ["struct_of_list", "list_of_list"]],
    ids=["with_id", "only_list_of_list", "two_nested"],
)
def test_nested_projection(nested_dataset: Path, columns: list[str]) -> None:
    stored = scan_lance(nested_dataset).collect()

    scanned = scan_lance(nested_dataset).select(columns).collect()

    assert scanned.equals(stored.select(columns))


# `list_of_struct` is pinned to version 2.0 here, so it gets its own tests. Filters are
# compared against the stored data, which is what a reader of the dataset sees.
@pytest.mark.parametrize("write", WRITERS, ids=["write", "sink"])
def test_list_of_struct_round_trip(
    tmp_path: Path, list_of_struct_frame: pl.DataFrame, write: Callable[..., None]
) -> None:
    dataset_path = tmp_path / "los.lance"

    write_frame(
        write,
        list_of_struct_frame,
        dataset_path,
        data_storage_version=STORAGE_VERSION,
    )

    scanned = scan_lance(dataset_path).collect()
    assert scanned.height == list_of_struct_frame.height
    assert scanned.schema == list_of_struct_frame.schema
    lists = scanned["list_of_struct"].to_list()
    assert lists[0] == [{"x": 1, "name": "a"}, {"x": 2, "name": "b"}]
    assert lists[1] == []  # empty list
    assert lists[2] is None  # null list
    assert lists[3][0] == FILLER_STRUCT  # the null struct that 2.0 cannot record
    assert lists[3][1] == {"x": 4, "name": None}  # null field inside a struct
    assert lists[4][0] == {"x": None, "name": "e"}


# What a released wheel does with this column, which the version 2.0 tests above cannot show:
# at the default version a null struct inside a list stays null. Skipped rather than xfailed
# because in a debug build there is nothing to check - the write cannot happen at all - and
# letting a Rust panic unwind through the extension on every run only adds noise.
@pytest.mark.skipif(
    DEBUG_ASSERTIONS,
    reason="lance#8032: writing this past 2.0 trips a debug assertion (see STORAGE_VERSION)",
)
@pytest.mark.parametrize("write", WRITERS, ids=["write", "sink"])
def test_list_of_struct_keeps_a_null_struct_at_the_default_version(
    tmp_path: Path, list_of_struct_frame: pl.DataFrame, write: Callable[..., None]
) -> None:
    dataset_path = tmp_path / "los_21.lance"

    write_frame(write, list_of_struct_frame, dataset_path)

    assert lance.dataset(dataset_path).data_storage_version == "2.2"
    lists = scan_lance(dataset_path).collect()["list_of_struct"].to_list()
    # `Series.equals` compares the values hidden underneath a null struct, which a round trip
    # does not preserve, so compare what the column reports instead.
    assert lists == list_of_struct_frame["list_of_struct"].to_list()
    assert lists[3][0] is None  # null struct, where 2.0 gives `FILLER_STRUCT`


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(pl.col("list_of_struct").list.len() > 1, id="list_len"),
        pytest.param(pl.col("list_of_struct").list.len() == 0, id="empty_list"),
        pytest.param(pl.col("list_of_struct").is_null(), id="null_list"),
        pytest.param(pl.col("list_of_struct").is_not_null(), id="not_null_list"),
        pytest.param(
            pl.col("list_of_struct")
            .list.eval(pl.element().struct.field("x") > 1)
            .list.any(),
            id="element_field",
        ),
        pytest.param(
            pl.col("list_of_struct")
            .list.eval(pl.element().struct.field("name") == "a")
            .list.any(),
            id="element_string_field",
        ),
    ],
)
def test_list_of_struct_filter_matches_polars(
    list_of_struct_dataset: Path, predicate: pl.Expr
) -> None:
    stored = scan_lance(list_of_struct_dataset).collect()

    scanned = scan_lance(list_of_struct_dataset).filter(predicate).collect()

    assert scanned.equals(stored.filter(predicate))


def test_list_of_struct_projection(list_of_struct_dataset: Path) -> None:
    stored = scan_lance(list_of_struct_dataset).collect()

    scanned = scan_lance(list_of_struct_dataset).select(["list_of_struct"]).collect()

    assert scanned.equals(stored.select(["list_of_struct"]))
