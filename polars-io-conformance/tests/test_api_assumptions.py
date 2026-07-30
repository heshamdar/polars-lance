"""The S0 findings, as assertions.

`docs/api-findings.md` records what Polars actually does. This file fails when any of it stops
being true, which is the point: the generator is built on these three APIs and the pushdown
suite is built on the fourth, and all four are undocumented behaviour rather than contract.
"""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
import pytest
from polars.io.plugins import register_io_source

from plioc.gen.core import GenContext, rnd, splitmix64


def test_lazy_row_index_needs_no_input_columns() -> None:
    """The generator's whole scale-free story rests on this working with no source frame."""
    out = pl.LazyFrame().select(pl.int_range(0, 5, dtype=pl.UInt64).alias("__i")).collect()
    assert out.to_series().to_list() == [0, 1, 2, 3, 4]


def test_splitmix64_in_expressions_matches_the_reference() -> None:
    """Determinism is load-bearing, so the mixer is arithmetic rather than `Expr.hash`.

    `Expr.hash` runs, but nothing documents its output as a fixed function of the input, and a
    corpus that changes when Polars changes its hasher is not a corpus.
    """
    ctx = GenContext(seed=0, key=12345)
    got = (
        pl.LazyFrame()
        .select(pl.int_range(0, 64, dtype=pl.UInt64).alias("__i"))
        .select(rnd(ctx).alias("v"))
        .collect()
        .to_series()
        .to_list()
    )
    key = splitmix64(12345)
    assert got == [splitmix64(i ^ key) for i in range(64)]


def test_unsigned_multiply_wraps_rather_than_raising() -> None:
    big = pl.select((pl.lit(2**63, pl.UInt64) * pl.lit(3, pl.UInt64)).alias("v")).item()
    assert big == (2**63 * 3) % 2**64


def test_expr_has_no_right_shift_operator() -> None:
    """Floor division stands in for it; if this ever starts working, simplify `_shr`."""
    with pytest.raises(TypeError):
        pl.col("a") >> 1  # type: ignore[operator]


@pytest.mark.parametrize(
    "dtype,values",
    [
        (pl.Enum(["a", "b"]), ["a", "b"]),
        (pl.Categorical(), ["a", "b"]),
        (pl.Decimal(38, 2), None),
        (pl.Datetime("us", "America/New_York"), None),
        (pl.Struct({"a": pl.Int64}), [{"a": 1}, None]),
        (pl.Array(pl.Float32, 2), [[1.0, 2.0], [3.0, 4.0]]),
        (pl.List(pl.Int64), [[1], []]),
        (pl.Null(), [None, None]),
        (pl.Int128(), [1, 2]),
    ],
)
def test_palette_gather_preserves_every_dtype(dtype: pl.DataType, values: list | None) -> None:
    """`pl.lit(Series).gather(...)` is the single palette path. It is only that because it
    works for the exotic dtypes too -- the alternatives (`replace_strict`, a palette join) do
    not, and would have forced a per-dtype branch through the whole generator."""
    if values is None:
        source = pl.Series("p", [1, 2], dtype=pl.Int64).cast(dtype, strict=False)
    else:
        source = pl.Series("p", values, dtype=dtype)
    out = (
        pl.LazyFrame()
        .select(pl.int_range(0, 6, dtype=pl.UInt64).alias("__i"))
        .select(pl.lit(source).gather(pl.col("__i") % 2).alias("v"))
        .collect()
    )
    assert out.schema["v"] == dtype
    assert out.height == 6


def test_list_head_accepts_an_expression() -> None:
    """This one expression replaces the plan's cross-join/group_by/rejoin pipeline for
    variable-length lists, and is the reason list generation stays prefix-stable."""
    out = (
        pl.LazyFrame()
        .select(pl.int_range(0, 5, dtype=pl.UInt64).alias("__i"))
        .select(
            pl.concat_list([pl.col("__i").cast(pl.Int64), pl.lit(9, pl.Int64)])
            .list.head((pl.col("__i") % 3).cast(pl.UInt32))
            .alias("v")
        )
        .collect()
    )
    assert out.to_series().to_list() == [[], [1], [2, 9], [], [4]]


def test_a_frame_with_no_columns_has_no_rows() -> None:
    """So the corpus's "zero columns" case is necessarily also a zero-row case."""
    assert pl.LazyFrame().select(pl.int_range(0, 5).alias("i")).select([]).collect().shape == (0, 0)


def test_empty_struct_is_not_constructible_from_pl_struct() -> None:
    with pytest.raises(Exception):
        pl.select(pl.struct([]).alias("v"))
    assert pl.Series("v", [{}], dtype=pl.Struct({})).dtype == pl.Struct({})


# -- what register_io_source actually delivers ------------------------------------------------


def _probing_plugin(df: pl.DataFrame, seen: list[dict]) -> pl.LazyFrame:
    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        seen.append({"cols": with_columns, "pred": predicate, "n": n_rows, "bs": batch_size})
        out = df
        if with_columns is not None:
            out = out.select(with_columns)
        if predicate is not None:
            out = out.filter(predicate)
        if n_rows is not None:
            out = out.head(n_rows)
        yield out

    return register_io_source(io_source=io_source, schema=df.schema, is_pure=True)


@pytest.fixture
def frame() -> pl.DataFrame:
    # Three columns, not two: with a projection covering everything Polars sends `None` rather
    # than the full list, and the projection assertions below would be vacuous.
    return pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": list("xyzwv"), "c": [1.0] * 5})


def test_a_count_arrives_as_a_one_column_projection(frame: pl.DataFrame) -> None:
    """So "empty projection, zero columns read" is not an assertion the suite can make."""
    seen: list[dict] = []
    assert _probing_plugin(frame, seen).select(pl.len()).collect().item() == 5
    assert seen[-1]["cols"] is not None
    assert len(seen[-1]["cols"]) == 1


def test_limit_is_withheld_when_a_predicate_is_pushed(frame: pl.DataFrame) -> None:
    """A plugin therefore cannot get "the limit applies after the filter" wrong through the
    plan; the mutant that does is only reachable by driving both explicitly."""
    seen: list[dict] = []
    _probing_plugin(frame, seen).filter(pl.col("a") > 2).head(1).collect()
    assert seen[-1]["pred"] is not None
    assert seen[-1]["n"] is None


def test_row_index_before_a_filter_suppresses_predicate_pushdown(frame: pl.DataFrame) -> None:
    seen: list[dict] = []
    _probing_plugin(frame, seen).with_row_index("ri").filter(pl.col("a") > 3).collect()
    assert seen[-1]["pred"] is None


def test_resolving_the_schema_does_not_call_the_plugin(frame: pl.DataFrame) -> None:
    seen: list[dict] = []
    _probing_plugin(frame, seen).collect_schema()
    assert seen == []


def test_a_predicate_only_column_is_added_to_the_projection(frame: pl.DataFrame) -> None:
    seen: list[dict] = []
    _probing_plugin(frame, seen).filter(pl.col("a") > 2).select("b").collect()
    assert set(seen[-1]["cols"]) == {"a", "b"}


def test_the_predicate_is_a_mandate(frame: pl.DataFrame) -> None:
    """The single fact the whole mandate suite rests on: Polars does not re-apply a predicate it
    pushed, so a plugin that ignores it returns wrong rows and nothing anywhere complains."""

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        yield frame if with_columns is None else frame.select(with_columns)

    lf = register_io_source(io_source=io_source, schema=frame.schema, is_pure=True)
    assert lf.filter(pl.col("a") > 3).collect().height == 5
