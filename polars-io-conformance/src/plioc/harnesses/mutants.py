"""Deliberately broken harnesses -- one defect each.

A conformance suite that is never mutation-tested is a suite that passes. Every mutant here must
be caught by at least one named case, and `tests/test_mutants.py` asserts exactly that and
prints which case does the catching. That is the suite's real coverage metric; line coverage of
the generator says nothing about whether the corpus finds bugs.

Each defect is one a real plugin has actually shipped.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import polars as pl

from plioc.harnesses.plugin import PluginHarness


class DropsPredicate(PluginHarness):
    """Receives the predicate and does not apply it.

    `register_io_source` hands the predicate over as an obligation: Polars does not re-apply it.
    Returning unfiltered rows is silent, unbounded data corruption in the user's result.
    """

    name = "mutant:drops-predicate"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        yield from super().produce(df, with_columns, None, n_rows, batch_size)


class DropsResidual(PluginHarness):
    """Translates one side of a conjunction, pushes it, and forgets the rest.

    The flagship defect. A plugin translating predicates into a backend's filter language hits an
    untranslatable node, keeps what it could translate, and never re-applies the remainder. Every
    row that satisfies the translated half and fails the other half is returned wrongly.
    """

    name = "mutant:drops-residual"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        yield from super().produce(df, with_columns, _first_conjunct(predicate), n_rows, batch_size)


def _first_conjunct(predicate: pl.Expr | None) -> pl.Expr | None:
    """Keep the left side of a top-level `&`, drop the right.

    Anything that is not a conjunction passes through untouched, so this mutant is wrong only on
    the shape it models -- which is what makes "some case catches it" a meaningful claim.
    """
    if predicate is None:
        return None
    try:
        node = json.loads(predicate.meta.serialize(format="json"))
        if node.get("BinaryExpr", {}).get("op") != "And":
            return predicate
        # `meta.pop()` yields the operands right-to-left.
        return predicate.meta.pop()[-1]
    except Exception:  # pragma: no cover - the Polars expression IR is not a stable API
        return predicate


class LimitBeforeFilter(PluginHarness):
    """Applies `n_rows` to the rows read rather than to the rows that survive the predicate.

    **Unreachable, and that is the finding.** Measured against Polars 1.43: `n_rows` is never
    delivered alongside a predicate, so a plugin cannot get their interaction wrong no matter how
    it is written, and no case catches this. It stays in the file as a live check on that fact --
    `tests/test_mutants.py` asserts it survives, so if Polars ever starts pushing both, the
    assertion flips and the pushdown suite needs a new case.
    """

    name = "mutant:limit-before-filter"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        out = df if n_rows is None else df.head(n_rows)
        yield from super().produce(out, with_columns, predicate, None, batch_size)


class IgnoresProjection(PluginHarness):
    """Reads and returns every column regardless of what was asked for.

    Correct output, wrong work: this is the defect that makes a "conformant" plugin unusably
    slow on a wide dataset, and only an engagement assertion catches it.
    """

    name = "mutant:ignores-projection"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        for batch in super().produce(df, None, predicate, n_rows, batch_size):
            yield batch if with_columns is None else batch.select(with_columns)


class ReordersRows(PluginHarness):
    """Returns rows in a different order than they were written."""

    name = "mutant:reorders-rows"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        yield from super().produce(df.reverse(), with_columns, predicate, n_rows, batch_size)


class TruncatesLastBatch(PluginHarness):
    """Loses the final row of every batch. Invisible at one batch and at round row counts, which
    is why the corpus carries row counts either side of every power-of-two boundary."""

    name = "mutant:truncates-last-batch"

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        for batch in super().produce(df, with_columns, predicate, n_rows, batch_size):
            yield batch.head(max(batch.height - 1, 0))


# -- write-side defects -------------------------------------------------------------------


class CoercesNanosecondsToMicroseconds(PluginHarness):
    """Stores every timestamp at microsecond resolution and does not say so."""

    name = "mutant:ns-to-us"

    def on_write(self, df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.col(name).cast(pl.Datetime("us", dtype.time_zone))
            for name, dtype in df.schema.items()
            if isinstance(dtype, pl.Datetime) and dtype.time_unit == "ns"
        ]
        return df.with_columns(casts) if casts else df


class EmptyListAsNull(PluginHarness):
    """Cannot tell an empty list from a null one. The single most common silent data-loss bug in
    columnar IO, and the reason the list generator makes the three shapes independent."""

    name = "mutant:empty-list-as-null"

    def on_write(self, df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.when(pl.col(name).list.len() == 0)
            .then(pl.lit(None, dtype=dtype))
            .otherwise(pl.col(name))
            .alias(name)
            for name, dtype in df.schema.items()
            if isinstance(dtype, pl.List)
        ]
        return df.with_columns(casts) if casts else df


class NullStructAsFilled(PluginHarness):
    """Has no per-struct validity, so a null struct is stored as a struct whose fields are all
    null. Exactly what a storage format without struct-level validity does."""

    name = "mutant:null-struct-as-filled"

    def on_write(self, df: pl.DataFrame) -> pl.DataFrame:
        casts = []
        for name, dtype in df.schema.items():
            if not isinstance(dtype, pl.Struct):
                continue
            empty = pl.struct([pl.lit(None, dtype=f.dtype).alias(f.name) for f in dtype.fields])
            casts.append(
                pl.when(pl.col(name).is_null()).then(empty).otherwise(pl.col(name)).alias(name)
            )
        return df.with_columns(casts) if casts else df


class NegativeZeroToZero(PluginHarness):
    """Normalises `-0.0` to `0.0`. They are distinct floats and `-0.0 == 0.0`, so no
    value-comparison based on `==` notices; only an exact bit comparison does."""

    name = "mutant:negative-zero"

    def on_write(self, df: pl.DataFrame) -> pl.DataFrame:
        casts = [
            pl.when(pl.col(name) == 0.0)
            .then(pl.lit(0.0, dtype=dtype))
            .otherwise(pl.col(name))
            .alias(name)
            for name, dtype in df.schema.items()
            if dtype in (pl.Float32, pl.Float64)
        ]
        return df.with_columns(casts) if casts else df


#: Mutants that model a defect Polars' plugin interface cannot expose. Asserted to survive, so
#: that a change in what Polars pushes shows up as a test failure rather than as a silent gap.
UNREACHABLE_MUTANTS: tuple[type[PluginHarness], ...] = (LimitBeforeFilter,)

ALL_MUTANTS: tuple[type[PluginHarness], ...] = (
    DropsPredicate,
    DropsResidual,
    IgnoresProjection,
    ReordersRows,
    TruncatesLastBatch,
    CoercesNanosecondsToMicroseconds,
    EmptyListAsNull,
    NullStructAsFilled,
    NegativeZeroToZero,
)
