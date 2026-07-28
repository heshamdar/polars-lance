"""Differential tests: does the pushed Lance filter agree with the Polars predicate?

Reasoning about two dialects' semantics is not enough to trust the translation, so every
predicate here is checked against real data containing the cases the two engines could
disagree on: nulls, empty strings and lists, null structs and struct fields, regex
metacharacters, `LIKE` wildcards, case differences, and non-ASCII text.

Two properties are asserted for each predicate:

- **Soundness** (required): the rows Lance returns for the translated filter are a superset
  of the rows the predicate selects. Anything less is a wrong result, because Polars does
  not re-apply the predicate for IO plugins.
- **Exactness** (recorded): whether the filter selects *exactly* those rows. A filter that
  is merely sound still gives correct results, it just reads more than it has to.

A predicate that is not translated at all is reported, not failed: leaving it to Polars is
always correct.
"""

from __future__ import annotations

from datetime import date, datetime

import lance
import polars as pl
import pyarrow as pa
import pytest

from polars_lance._predicate import to_lance_sql
from tests.utils import delivered_predicate

STRUCT_TYPE = pa.struct([("x", pa.int32()), ("name", pa.string())])

# `rid` identifies rows so the two engines' results can be compared as sets.
EQUIVALENCE_TABLE = pa.table(
    {
        "rid": pa.array(range(12), pa.int32()),
        "i32": pa.array([1, 2, 3, -1, 0, None, 5, 5, 7, 8, 9, None], pa.int32()),
        "i64": pa.array([1, 2, 3, -1, 0, None, 5, 5, 7, 8, 9, None], pa.int64()),
        # `nan` and `-0.0` are included because Polars compares NaN as an ordinary value
        # that equals itself, while SQL does not.
        "f64": pa.array(
            [1.5, 2.5, -0.5, -0.0, float("nan"), None, 5.5, 6.5, 7.5, 8.5, 9.5, None],
            pa.float64(),
        ),
        "b": pa.array(
            [
                True,
                False,
                True,
                None,
                False,
                True,
                None,
                False,
                True,
                True,
                False,
                None,
            ],
            pa.bool_(),
        ),
        "s": pa.array(
            [
                "alpha",
                "beta",
                "gamma alpha",
                "",  # empty string
                "ALPHA",  # case difference
                None,  # null
                "a%b",  # LIKE wildcard
                "a_b",  # LIKE single-char wildcard
                "a.b",  # regex metacharacter
                "it's",  # quote to escape
                "héllo",  # non-ASCII
                "alpha beta",
            ],
            pa.string(),
        ),
        "d": pa.array(
            [date(2024, 1, 1 + i) for i in range(11)] + [None],
            pa.date32(),
        ),
        "t": pa.array(
            [datetime(2024, 1, 1, 0, 0, i) for i in range(11)] + [None],
            pa.timestamp("us"),
        ),
        "st": pa.array(
            [
                {"x": 1, "name": "p"},
                {"x": 2, "name": "q"},
                {"x": None, "name": "r"},  # null field inside a struct
                {"x": 4, "name": None},
                {"x": 5, "name": "p"},
                None,  # null struct
                {"x": 7, "name": "q"},
                {"x": 8, "name": "r"},
                {"x": 9, "name": "p"},
                {"x": 10, "name": "q"},
                {"x": 11, "name": "r"},
                None,
            ],
            STRUCT_TYPE,
        ),
        "lst": pa.array(
            [
                [1, 2],
                [3],
                [],  # empty list
                None,  # null list
                [4, 5, 6],
                [None],  # list containing a null
                [7],
                [8, 9],
                [10],
                [11, 12],
                [13],
                [],
            ],
            pa.list_(pa.int32()),
        ),
    }
)

SCHEMA = {
    name: dtype
    for name, dtype in zip(
        EQUIVALENCE_TABLE.column_names,
        pl.from_arrow(EQUIVALENCE_TABLE).schema.dtypes(),  # type: ignore[union-attr]
    )
}

PREDICATES = [
    # comparisons over nullable columns
    pl.col("i64") > 2,
    pl.col("i64") >= 2,
    pl.col("i64") < 2,
    pl.col("i64") <= 2,
    pl.col("i64") == 5,
    pl.col("i64") != 5,
    pl.col("i32") > 2,
    pl.col("f64") > 0.0,
    pl.col("f64") <= -0.5,
    pl.col("b") == True,  # noqa: E712
    pl.col("b") != True,  # noqa: E712
    # nulls
    pl.col("i64").is_null(),
    pl.col("i64").is_not_null(),
    pl.col("s").is_null(),
    pl.col("st").is_null(),
    pl.col("lst").is_null(),
    # boolean combinations, including ones mixing nulls on both sides
    (pl.col("i64") > 2) & (pl.col("f64") > 0.0),
    (pl.col("i64") > 2) | (pl.col("f64") > 0.0),
    (pl.col("i64") > 2) & pl.col("s").is_not_null(),
    (pl.col("i64").is_null()) | (pl.col("s").is_null()),
    (pl.col("i64") > 2) & (pl.col("s") == "alpha"),
    # negation
    ~(pl.col("i64") > 2),
    ~pl.col("i64").is_null(),
    # membership and ranges
    pl.col("i64").is_in([1, 5, 9]),
    pl.col("i64").is_between(2, 8),
    pl.col("i64").is_between(2, 8, closed="none"),
    pl.col("s").is_in(["alpha", "beta"]),
    # strings, including the awkward literals
    pl.col("s") == "alpha",
    pl.col("s") == "",
    pl.col("s") == "it's",
    pl.col("s") == "a%b",
    pl.col("s") == "héllo",
    pl.col("s") != "alpha",
    pl.col("s") > "b",
    # temporal
    pl.col("d") == date(2024, 1, 3),
    pl.col("d") > date(2024, 1, 5),
    pl.col("t") > datetime(2024, 1, 1, 0, 0, 5),
    # floats, where NaN could be compared as an ordinary value
    pl.col("f64") == float("nan"),
    pl.col("f64") != float("nan"),
    pl.col("f64") > float("nan"),
    pl.col("f64") == -0.0,
    pl.col("f64").is_null(),
    # string functions
    pl.col("s").str.contains("alpha"),
    pl.col("s").str.contains("a.b"),
    pl.col("s").str.contains("^al"),
    pl.col("s").str.contains("a.b", literal=True),
    pl.col("s").str.contains("a%b", literal=True),
    pl.col("s").str.starts_with("al"),
    pl.col("s").str.starts_with(""),
    pl.col("s").str.ends_with("ha"),
    pl.col("s").str.to_lowercase() == "alpha",
    pl.col("s").str.to_uppercase() == "ALPHA",
    pl.col("s").str.len_chars() > 4,
    pl.col("s").str.len_chars() == 0,
    (pl.col("s").str.contains("alpha")) & (pl.col("i64") > 2),
    (pl.col("s").str.starts_with("a")) | (pl.col("i64") > 8),
    # nested access
    pl.col("st").struct.field("x") > 4,
    pl.col("st").struct.field("x").is_null(),
    pl.col("st").struct.field("x").is_not_null(),
    pl.col("st").struct.field("name") == "p",
    pl.col("st").struct.field("name").is_null(),
    pl.col("st").struct.field("name").str.starts_with("p"),
    (pl.col("st").struct.field("x") > 4) & (pl.col("i64") > 2),
    # lists
    pl.col("lst").list.len() > 1,
    pl.col("lst").list.len() == 0,
    pl.col("lst").list.contains(3),
    pl.col("lst").list.contains(99),
]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> lance.LanceDataset:
    path = tmp_path_factory.mktemp("equivalence") / "test.lance"
    lance.write_dataset(EQUIVALENCE_TABLE, path)
    return lance.dataset(path)


@pytest.fixture(scope="module")
def stored_frame(dataset: lance.LanceDataset) -> pl.DataFrame:
    """
    The data as Lance stored it, which is what both engines evaluate the filter against.

    Reading it back rather than using the original table keeps this about the translation:
    Lance does not preserve a null *struct*, so comparing against the original table would
    fail for reasons that have nothing to do with the filter.
    """
    frame = pl.from_arrow(dataset.to_table())
    assert isinstance(frame, pl.DataFrame)
    return frame


def polars_matches(frame: pl.DataFrame, predicate: pl.Expr) -> set[int]:
    return set(frame.filter(predicate)["rid"].to_list())


def lance_matches(dataset: lance.LanceDataset, sql: str) -> set[int]:
    table = dataset.scanner(filter=sql, columns=["rid"]).to_table()
    return set(table.column("rid").to_pylist())


@pytest.mark.parametrize("predicate", PREDICATES, ids=lambda p: str(p)[:60])
def test_translation_is_sound(
    dataset: lance.LanceDataset, stored_frame: pl.DataFrame, predicate: pl.Expr
) -> None:
    sql = to_lance_sql(delivered_predicate(predicate, SCHEMA), SCHEMA)
    if sql is None:
        pytest.skip("not translated; Polars applies the predicate")

    expected = polars_matches(stored_frame, predicate)
    actual = lance_matches(dataset, sql)

    missing = expected - actual
    assert not missing, (
        f"filter {sql!r} misses rows the predicate selects: {sorted(missing)}. "
        "The pushed filter must match a superset of the predicate."
    )


@pytest.mark.parametrize("predicate", PREDICATES, ids=lambda p: str(p)[:60])
def test_translation_is_exact(
    dataset: lance.LanceDataset, stored_frame: pl.DataFrame, predicate: pl.Expr
) -> None:
    """
    The translated filter should select exactly the predicate's rows.

    Soundness is what correctness needs; this test pins down that the translations we do
    perform are not needlessly broad, so a regression to a weaker filter is visible.
    """
    sql = to_lance_sql(delivered_predicate(predicate, SCHEMA), SCHEMA)
    if sql is None:
        pytest.skip("not translated; Polars applies the predicate")

    expected = polars_matches(stored_frame, predicate)
    actual = lance_matches(dataset, sql)

    assert actual == expected, (
        f"filter {sql!r} selects {sorted(actual)}, predicate selects {sorted(expected)}"
    )


@pytest.mark.parametrize("predicate", PREDICATES, ids=lambda p: str(p)[:60])
def test_translation_agrees_with_polars_sql(
    stored_frame: pl.DataFrame, predicate: pl.Expr
) -> None:
    """
    Cross-check the filter with a second, independent SQL engine: Polars' own.

    Lance and Polars could in principle agree on a reading of the SQL that is not what the
    predicate means. Running the same string through `SQLContext` checks the translation
    without involving Lance at all. Functions Polars' SQL dialect lacks are skipped.
    """
    sql = to_lance_sql(delivered_predicate(predicate, SCHEMA), SCHEMA)
    if sql is None:
        pytest.skip("not translated; Polars applies the predicate")

    context = pl.SQLContext(t=stored_frame)
    try:
        selected = context.execute(f"SELECT rid FROM t WHERE {sql}", eager=True)
    except Exception as error:
        pytest.skip(f"Polars' SQL dialect cannot run this filter: {error}")

    assert set(selected["rid"].to_list()) == polars_matches(stored_frame, predicate), (
        f"Polars' SQL engine reads {sql!r} differently than the predicate"
    )
