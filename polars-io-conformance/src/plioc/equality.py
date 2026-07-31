"""The strictness ladder, and the normalisers a lossy path is compared through.

Two contracts are asserted, and they are different:

- **normalised round-trip** -- `scan(sink(df)) == normalize(df)`, which says the loss is the one
  the harness declared;
- **fixpoint** -- `rt(rt(df)) == rt(df)` and `normalize(normalize(df)) == normalize(df)`, which
  says the loss happens once and then stops. A timezone that shifts on every write is a bug that
  the first contract alone cannot see, because each individual round-trip looks like a
  legitimate normalisation.
"""

from __future__ import annotations

from enum import IntEnum

import polars as pl
from polars.testing import assert_frame_equal

from plioc.gen.core import IDX
from plioc.gen.layout import n_chunks


class Strictness(IntEnum):
    VALUES = 1
    DTYPES = 2
    COLUMN_ORDER = 3
    ROW_ORDER = 4
    PHYSICAL = 5
    METADATA = 6


def sort_key(df: pl.DataFrame) -> pl.DataFrame:
    """Put a frame in a deterministic order for comparison below `ROW_ORDER`.

    Uses the case's `__i` identity column. Without it there is no total order that is both
    deterministic and dtype-agnostic -- sorting by arbitrary columns puts nulls, NaNs and
    equal-comparing values in an implementation-defined order and would make the comparison
    itself flaky.
    """
    if IDX not in df.columns:
        raise ValueError(
            "cannot compare row-order-agnostically without the __i identity column; "
            "the case sets order_key=False, so it is only comparable at ROW_ORDER or above"
        )
    return df.sort(IDX)


def compare(
    observed: pl.DataFrame,
    expected: pl.DataFrame,
    strictness: Strictness,
    *,
    check_exact: bool = True,
) -> None:
    """Assert equality at `strictness`. Raises `AssertionError` with Polars' own diff."""
    left, right = observed, expected

    if strictness < Strictness.ROW_ORDER:
        left, right = sort_key(left), sort_key(right)
    if strictness < Strictness.COLUMN_ORDER:
        left = left.select(sorted(left.columns))
        right = right.select(sorted(right.columns))

    assert_frame_equal(
        left,
        right,
        check_dtypes=strictness >= Strictness.DTYPES,
        check_column_order=strictness >= Strictness.COLUMN_ORDER,
        check_row_order=True,  # already normalised above when it should not be checked
        check_exact=check_exact,
        # A Categorical compared as its physical code is only meaningful once both sides came
        # from the same string cache, which a round-trip through a file breaks by construction.
        categorical_as_str=strictness < Strictness.PHYSICAL,
    )

    if strictness >= Strictness.DTYPES:
        _compare_signed_zeros(left, right)

    if strictness >= Strictness.PHYSICAL:
        if n_chunks(observed) != n_chunks(expected):
            raise AssertionError(
                f"chunk layout differs: {n_chunks(observed)} != {n_chunks(expected)}"
            )
    if strictness >= Strictness.METADATA:
        _compare_flags(observed, expected)


def _negative_zero(name: str) -> pl.Expr:
    # `x == 0.0` is true for both zeros, so the sign has to be recovered from the reciprocal.
    return ((pl.col(name) == 0.0) & (pl.lit(1.0) / pl.col(name) < 0.0)).alias(name)


def _compare_signed_zeros(left: pl.DataFrame, right: pl.DataFrame) -> None:
    """`-0.0` and `0.0` are distinct floats that compare equal.

    `assert_frame_equal` cannot see the difference even with `check_exact`, so a path that
    normalises one to the other loses data silently and passes every value assertion. Top-level
    float columns only; a float nested inside a list or struct is not reachable this way.
    """
    floats = [
        name
        for name, dtype in right.schema.items()
        if dtype in (pl.Float32, pl.Float64) and name in left.columns
    ]
    if not floats:
        return
    for name in floats:
        want = right.select(_negative_zero(name)).to_series()
        got = left.select(_negative_zero(name)).to_series()
        if not got.equals(want):
            raise AssertionError(
                f"{name!r}: the sign of zero was not preserved "
                f"({int(got.sum() or 0)} negative zeros, expected {int(want.sum() or 0)})"
            )


def _compare_flags(observed: pl.DataFrame, expected: pl.DataFrame) -> None:
    for name in expected.columns:
        want = expected[name].flags
        got = observed[name].flags
        for flag in ("SORTED_ASC", "SORTED_DESC"):
            if want.get(flag) != got.get(flag):
                raise AssertionError(
                    f"{name!r}: {flag} is {got.get(flag)}, expected {want.get(flag)}"
                )


# ------------------------------------------------------------------------------- normalisers
#
# Composable descriptions of a loss. A harness names the ones that apply to it; the suite then
# knows the difference between "this path is lossy in the declared way" and "this path is wrong".


def compose(*fns: object) -> object:
    def run(df: pl.DataFrame) -> pl.DataFrame:
        for fn in fns:
            df = fn(df)  # type: ignore[operator]
        return df

    return run


def categorical_to_string(df: pl.DataFrame) -> pl.DataFrame:
    """For a format with no dictionary type of its own."""
    return df.with_columns(
        pl.col(name).cast(pl.String)
        for name, dtype in df.schema.items()
        if isinstance(dtype, (pl.Categorical, pl.Enum))
    )


def _retime(dtype: pl.DataType, unit: str) -> pl.DataType | None:
    if isinstance(dtype, pl.Datetime) and dtype.time_unit != unit:
        return pl.Datetime(unit, dtype.time_zone)  # type: ignore[arg-type]
    if isinstance(dtype, pl.Duration) and dtype.time_unit != unit:
        return pl.Duration(unit)  # type: ignore[arg-type]
    return None


def time_unit_to(unit: str) -> object:
    """For a format that stores timestamps at one resolution. Truncates, as the format does."""

    def run(df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.col(name).cast(target)
            for name, dtype in df.schema.items()
            if (target := _retime(dtype, unit)) is not None
        ]
        return df.with_columns(casts) if casts else df

    return run


def timezone_to_utc(df: pl.DataFrame) -> pl.DataFrame:
    """For a format that stores an instant but not the zone it was written in."""
    return df.with_columns(
        pl.col(name).dt.convert_time_zone("UTC")
        for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None
    )


def rechunked(df: pl.DataFrame) -> pl.DataFrame:
    """For a harness that claims `PHYSICAL` but reassembles its own batches. Chunk layout is a
    property of how the data was read, not of the data; a harness that reads it back in one
    piece is not wrong, it just cannot be compared chunk-for-chunk without this."""
    return df.rechunk()


#: Every integer dtype narrower than `Int64`, which is the set a format with a smaller vocabulary
#: than Polars can widen.
NARROW_INTS: tuple[type[pl.DataType], ...] = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
)


def widen_ints(df: pl.DataFrame) -> pl.DataFrame:
    """For a format with only 64-bit integers."""
    casts = [
        pl.col(name).cast(pl.Int64)
        for name, dtype in df.schema.items()
        if isinstance(dtype, NARROW_INTS)
    ]
    return df.with_columns(casts) if casts else df


def widen_ints_to(mapping: dict[pl.DataType, pl.DataType]) -> object:
    """For a format with a narrower integer vocabulary than Polars.

    Avro, for instance, has `int` and `long` and nothing else, so `Int8` comes back as `Int32` and
    `UInt32` as `Int64` -- while `Int32` and `Int64` are kept exactly. Only the dtypes the mapping
    names are touched: a default of "widen everything else to `Int64`" reads as a convenience and
    is a trap, because it silently describes a loss the format does not have and turns the
    round-trip assertion into a check that the *suite* is wrong in the same way.
    """

    def run(df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.col(name).cast(mapping[dtype])
            for name, dtype in df.schema.items()
            if dtype in mapping
        ]
        return df.with_columns(casts) if casts else df

    return run


def array_to_list(df: pl.DataFrame) -> pl.DataFrame:
    """For a format with variable-length arrays only, which is most of them: the fixed width is
    a Polars-side guarantee that nothing on disk records."""
    casts = [
        pl.col(name).cast(pl.List(dtype.inner))
        for name, dtype in df.schema.items()
        if isinstance(dtype, pl.Array)
    ]
    return df.with_columns(casts) if casts else df
