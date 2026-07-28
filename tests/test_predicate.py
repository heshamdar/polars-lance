"""Tests for translating Polars predicates into Lance SQL filters.

The translation is only allowed to match a *superset* of the rows a predicate selects,
because `scan_lance` re-applies the predicate to every batch. These tests pin down both
what is translated and what is deliberately left untranslated.
"""

from datetime import date, datetime, time, timedelta, timezone

import polars as pl
import pytest
from polars.datatypes import DataTypeClass

from polars_lance._predicate import to_lance_sql
from tests.utils import delivered_predicate

# An expression Lance cannot evaluate, used to check that untranslatable parts are dropped
# from conjunctions and abort the translation everywhere else.
UNTRANSLATABLE = pl.col("float64").is_finite()

SCHEMA: dict[str, pl.DataType | DataTypeClass] = {
    "struct": pl.Struct({"x": pl.Int64, "name": pl.String}),
    "list": pl.List(pl.Int64),
    "int32": pl.Int32,
    "int64": pl.Int64,
    "uint32": pl.UInt32,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "bool": pl.Boolean,
    "string": pl.String,
    "date": pl.Date,
    "datetime": pl.Datetime("us"),
    "datetime_ns": pl.Datetime("ns"),
    "datetime_tz": pl.Datetime("us", time_zone="Europe/Amsterdam"),
    "time": pl.Time,
    "duration": pl.Duration("us"),
}


def assert_translates_to(predicate: pl.Expr, expected: str | None) -> None:
    assert to_lance_sql(delivered_predicate(predicate, SCHEMA), SCHEMA) == expected


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (pl.col("int64") == 1, "(`int64` = 1)"),
        (pl.col("int64") != 1, "(`int64` != 1)"),
        (pl.col("int64") < 1, "(`int64` < 1)"),
        (pl.col("int64") <= 1, "(`int64` <= 1)"),
        (pl.col("int64") > 1, "(`int64` > 1)"),
        (pl.col("int64") >= 1, "(`int64` >= 1)"),
    ],
)
def test_comparisons(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (pl.col("int32") > 1, "(`int32` > 1)"),
        (pl.col("int64") > -1, "(`int64` > -1)"),
        (pl.col("uint32") > 7, "(`uint32` > 7)"),
        (pl.col("float64") > 1.5, "(`float64` > 1.5)"),
        # Whole floats keep a decimal point, so they are not read as integers.
        (pl.col("float64") > 2.0, "(`float64` > 2.0)"),
        (pl.col("bool") == True, "(`bool` = true)"),  # noqa: E712
        (pl.col("string") == "x", "(`string` = 'x')"),
        # Single quotes are escaped by doubling them.
        (pl.col("string") == "it's", "(`string` = 'it''s')"),
    ],
)
def test_literals(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (pl.col("date") == date(2021, 1, 3), "(`date` = date '2021-01-03')"),
        (
            pl.col("datetime") > datetime(2021, 1, 1, 0, 0, 2),
            "(`datetime` > timestamp '2021-01-01 00:00:02.000000')",
        ),
        (
            pl.col("datetime_ns") > datetime(2021, 1, 1, 0, 0, 2),
            "(`datetime_ns` > timestamp '2021-01-01 00:00:02.000000')",
        ),
    ],
)
def test_temporal_literals(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


@pytest.mark.parametrize(
    "predicate",
    [
        # A time zone aware timestamp would need converting to the dataset's time zone,
        # which risks an off-by-offset filter. Comparing such a column against a naive
        # datetime is rejected by Polars itself, so the literal is aware here.
        pytest.param(
            pl.col("datetime_tz") > datetime(2021, 1, 1, 0, 0, 2, tzinfo=timezone.utc),
            id="datetime_tz",
        ),
        pytest.param(pl.col("time") > time(1, 2, 3), id="time"),
        pytest.param(pl.col("duration") > timedelta(seconds=5), id="duration"),
    ],
)
def test_untranslated_literal_types(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, None)


def test_null_checks() -> None:
    assert_translates_to(pl.col("int64").is_null(), "(`int64` IS NULL)")
    assert_translates_to(pl.col("int64").is_not_null(), "(`int64` IS NOT NULL)")


def test_is_in() -> None:
    assert_translates_to(pl.col("int64").is_in([1, 2, 3]), "(`int64` IN (1, 2, 3))")
    assert_translates_to(pl.col("string").is_in(["x", "y"]), "(`string` IN ('x', 'y'))")


# A null cannot be matched by SQL's `IN`, so such a filter would match a subset.
def test_is_in_with_null_is_not_translated() -> None:
    assert_translates_to(pl.col("int64").is_in([1, None]), None)


# Polars casts a narrower column to match the dtype of the values, and a cast operand is
# not translated: unwrapping it is only sound if the cast cannot lose information, which
# the serialized expression does not say. The predicate is still applied, just not pushed.
def test_is_in_on_narrower_column_is_not_translated() -> None:
    assert_translates_to(pl.col("int32").is_in([1, 2, 3]), None)


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("both", "(`int64` >= 2 AND `int64` <= 8)"),
        ("left", "(`int64` >= 2 AND `int64` < 8)"),
        ("right", "(`int64` > 2 AND `int64` <= 8)"),
        ("none", "(`int64` > 2 AND `int64` < 8)"),
    ],
)
def test_is_between(closed: str, expected: str) -> None:
    predicate = pl.col("int64").is_between(2, 8, closed=closed)  # type: ignore[arg-type]
    assert_translates_to(predicate, expected)


def test_conjunction_of_translatable_sides() -> None:
    predicate = (pl.col("int64") > 5) & (pl.col("int32") < 10)
    assert to_lance_sql(delivered_predicate(predicate, SCHEMA), SCHEMA) in {
        "((`int64` > 5) AND (`int32` < 10))",
        "((`int32` < 10) AND (`int64` > 5))",
    }


# A conjunction may be weakened to the translatable side: the pushed filter then matches a
# superset of the rows, and `scan_lance` re-applies the full predicate.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param((pl.col("int64") > 5) & UNTRANSLATABLE, id="right_untranslatable"),
        pytest.param(UNTRANSLATABLE & (pl.col("int64") > 5), id="left_untranslatable"),
    ],
)
def test_conjunction_drops_untranslatable_side(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, "(`int64` > 5)")


def test_disjunction_of_translatable_sides() -> None:
    predicate = (pl.col("int64") == 1) | (pl.col("int64") == 5)
    assert_translates_to(predicate, "((`int64` = 1) OR (`int64` = 5))")


# A disjunction must not be weakened: dropping a side would match a subset.
def test_disjunction_with_untranslatable_side_is_not_translated() -> None:
    assert_translates_to((pl.col("int64") > 5) | UNTRANSLATABLE, None)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        # A regex pattern is passed through, because both engines use the Rust regex crate.
        (pl.col("string").str.contains("a.b"), "regexp_like(`string`, 'a.b')"),
        # A literal pattern is a substring match instead.
        (
            pl.col("string").str.contains("a.b", literal=True),
            "contains(`string`, 'a.b')",
        ),
        (pl.col("string").str.starts_with("x"), "starts_with(`string`, 'x')"),
        (pl.col("string").str.ends_with("x"), "ends_with(`string`, 'x')"),
        (
            pl.col("string").str.to_lowercase() == "x",
            "(lower(`string`) = 'x')",
        ),
        (
            pl.col("string").str.len_chars() > 3,
            "(character_length(`string`) > 3)",
        ),
    ],
)
def test_string_functions(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        (pl.col("struct").struct.field("x") > 1, "(`struct`.`x` > 1)"),
        (pl.col("struct").struct.field("name") == "p", "(`struct`.`name` = 'p')"),
        (pl.col("list").list.len() > 1, "(array_length(`list`) > 1)"),
        (pl.col("list").list.contains(3), "array_has(`list`, 3)"),
    ],
)
def test_nested_access(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


# Lance records nulls per leaf field, so it reports neither a null struct nor a field read
# out of one, while Polars reports both. Translating these would drop rows.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(pl.col("struct").is_null(), id="struct_is_null"),
        pytest.param(pl.col("struct").is_not_null(), id="struct_is_not_null"),
        pytest.param(
            pl.col("struct").struct.field("x").is_null(), id="struct_field_is_null"
        ),
        pytest.param(
            pl.col("struct").struct.field("x").is_not_null(),
            id="struct_field_is_not_null",
        ),
    ],
)
def test_null_checks_on_structs_are_not_translated(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, None)


# Without the schema the dtype is unknown, so a struct column cannot be ruled out.
def test_null_check_needs_the_schema() -> None:
    predicate = delivered_predicate(pl.col("int64").is_null(), SCHEMA)
    assert to_lance_sql(predicate) is None
    assert to_lance_sql(predicate, SCHEMA) == "(`int64` IS NULL)"


@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(UNTRANSLATABLE, id="unsupported_function"),
        # Comparing against a computed operand is not translated.
        pytest.param(pl.col("int64") > pl.col("int32") + 1, id="computed_operand"),
        # Null-safe equality does not map onto SQL `=`.
        pytest.param(pl.col("int64").eq_missing(1), id="eq_missing"),
        # Polars compares NaN as a value that equals itself, SQL does not.
        pytest.param(pl.col("float64") == float("nan"), id="nan"),
        # Polars indexes lists from 0 and raises when out of bounds; SQL does neither.
        pytest.param(pl.col("list").list.get(0) > 1, id="list_get"),
    ],
)
def test_untranslatable_predicates(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, None)


# A value with no SQL literal form must abort the comparison rather than be approximated:
# rendering an infinity or a truncated nanosecond would change which rows match.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(pl.col("float64") > float("inf"), id="infinity"),
        pytest.param(pl.col("float64") < float("-inf"), id="negative_infinity"),
        pytest.param(
            # A nanosecond that is not a whole microsecond: a SQL timestamp literal cannot
            # express it, and truncating would move the comparison.
            pl.col("datetime_ns") > pl.lit(1_000_000_001, dtype=pl.Datetime("ns")),
            id="sub_microsecond",
        ),
    ],
)
def test_values_without_a_sql_literal_are_not_translated(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, None)


# A nanosecond literal landing exactly on a microsecond loses nothing, so it is translated.
def test_whole_microsecond_nanosecond_literal_is_translated() -> None:
    assert_translates_to(
        pl.col("datetime_ns") > pl.lit(1_000_000_000, dtype=pl.Datetime("ns")),
        "(`datetime_ns` > timestamp '1970-01-01 00:00:01.000000')",
    )


# `is_in` values arrive as an Arrow buffer rather than as literals, so each type is rendered by
# its own path and needs its own check.
@pytest.mark.parametrize(
    ("predicate", "expected"),
    [
        pytest.param(
            pl.col("bool").is_in([True, False]),
            "(`bool` IN (true, false))",
            id="bool",
        ),
        pytest.param(
            pl.col("float64").is_in([1.5, 2.5]),
            "(`float64` IN (1.5, 2.5))",
            id="float",
        ),
        pytest.param(
            pl.col("string").is_in(["a", "b'c"]),
            "(`string` IN ('a', 'b''c'))",
            id="string_with_a_quote",
        ),
        pytest.param(
            pl.col("date").is_in([date(2024, 1, 1), date(2024, 1, 2)]),
            "(`date` IN (date '2024-01-01', date '2024-01-02'))",
            id="date",
        ),
        pytest.param(
            pl.col("datetime").is_in([datetime(2024, 1, 1, 2, 3, 4)]),
            "(`datetime` IN (timestamp '2024-01-01 02:03:04.000000'))",
            id="datetime",
        ),
    ],
)
def test_is_in_value_types(predicate: pl.Expr, expected: str) -> None:
    assert_translates_to(predicate, expected)


# A value in the list that has no SQL literal form abandons the whole `IN`, because dropping
# one value would match a subset.
@pytest.mark.parametrize(
    "predicate",
    [
        pytest.param(pl.col("float64").is_in([1.5, float("nan")]), id="nan"),
        pytest.param(pl.col("float64").is_in([1.5, float("inf")]), id="infinity"),
        pytest.param(
            pl.col("datetime_tz").is_in([datetime(2024, 1, 1, tzinfo=timezone.utc)]),
            id="time_zone_aware",
        ),
    ],
)
def test_is_in_with_an_unrenderable_value_is_not_translated(predicate: pl.Expr) -> None:
    assert_translates_to(predicate, None)


# Lance does not support field names containing periods.
def test_column_name_with_period_is_not_translated() -> None:
    assert to_lance_sql(pl.col("a.b") > 1, {"a.b": pl.Int64}) is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("column name with space", "(`column name with space` > 1)"),
        # A backtick in a column name is escaped by doubling it.
        ("with`backtick", "(`with``backtick` > 1)"),
    ],
)
def test_column_names_are_quoted(name: str, expected: str) -> None:
    # These are translated without going through the optimizer, because the column does
    # not exist in the test schema.
    assert to_lance_sql(pl.col(name) > 1, {name: pl.Int64}) == expected
