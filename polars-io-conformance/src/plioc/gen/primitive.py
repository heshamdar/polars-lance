"""Numeric, boolean, string and binary generators.

Each mixes a palette draw with an ordinary bulk fill. `palette_bias` is the probability of the
palette branch: at 1.0 a case is nothing but edge values, which is unrepresentative of real data
and lets a bug that only shows up in bulk hide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import polars as pl

from plioc.gen import palettes
from plioc.gen.core import GenContext, as_column, idx, pick, uniform_int, unit

DEFAULT_BIAS = 0.6


def _blend(ctx: GenContext, bias: float, chosen: pl.Expr, fill: pl.Expr) -> pl.Expr:
    if bias >= 1.0:
        return chosen
    if bias <= 0.0:
        return fill
    return pl.when(unit(ctx, 1) < pl.lit(bias)).then(chosen).otherwise(fill)


@dataclass(frozen=True)
class IntGen:
    dtype: pl.DataType = field(default_factory=lambda: pl.Int64())
    palette_bias: float = DEFAULT_BIAS

    def expr(self, ctx: GenContext) -> pl.Expr:
        lo, hi = palettes.int_limits(self.dtype)
        pal = palettes.series(palettes.int_values(self.dtype), self.dtype)
        # The fill stays inside i64 so the draw is representable, then narrows: casts in Polars
        # are checked, so a wider fill would raise rather than wrap.
        fill_lo, fill_hi = max(lo, -(2**62)), min(hi, 2**62)
        fill = uniform_int(ctx, fill_lo, fill_hi, draw=2).cast(self.dtype)
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class FloatGen:
    dtype: pl.DataType = field(default_factory=lambda: pl.Float64())
    palette_bias: float = DEFAULT_BIAS

    def expr(self, ctx: GenContext) -> pl.Expr:
        groups = palettes.FLOAT32_GROUPS if self.dtype == pl.Float32 else palettes.FLOAT_GROUPS
        pal = palettes.series(palettes.grouped(groups), self.dtype)
        fill = ((unit(ctx, 2) - pl.lit(0.5)) * pl.lit(2e6)).cast(self.dtype)
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class BoolGen:
    palette_bias: float = 0.0

    @property
    def dtype(self) -> pl.DataType:
        return pl.Boolean()

    def expr(self, ctx: GenContext) -> pl.Expr:
        return unit(ctx, 2) < pl.lit(0.5)


@dataclass(frozen=True)
class StringGen:
    tags: frozenset[str] | None = None
    palette_bias: float = DEFAULT_BIAS
    huge: bool = False

    @property
    def dtype(self) -> pl.DataType:
        return pl.String()

    def expr(self, ctx: GenContext) -> pl.Expr:
        values = (
            list(palettes.HUGE_STRINGS)
            if self.huge
            else palettes.grouped(palettes.STRING_GROUPS, self.tags)
        )
        pal = palettes.series(values, pl.String())
        fill = pl.lit("row-") + idx().cast(pl.String)
        return _blend(ctx, 1.0 if self.huge else self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class BinaryGen:
    tags: frozenset[str] | None = None
    palette_bias: float = DEFAULT_BIAS

    @property
    def dtype(self) -> pl.DataType:
        return pl.Binary()

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = palettes.series(palettes.grouped(palettes.BINARY_GROUPS, self.tags), pl.Binary())
        fill = (pl.lit("b-") + idx().cast(pl.String)).cast(pl.Binary)
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class DecimalGen:
    precision: int = 38
    scale: int = 10

    @property
    def dtype(self) -> pl.DataType:
        return pl.Decimal(self.precision, self.scale)

    def expr(self, ctx: GenContext) -> pl.Expr:
        # Palette only: there is no arithmetic path from a u64 draw into an arbitrary
        # (precision, scale) that does not itself risk the rescale bug under test.
        pal = palettes.series(palettes.decimal_values(self.precision, self.scale), self.dtype)
        return pick(ctx, pal)


@dataclass(frozen=True)
class NullGen:
    """The `Null` dtype itself, which is not the same thing as an all-null column of some other
    dtype -- a writer that stores one as the other loses the schema."""

    @property
    def dtype(self) -> pl.DataType:
        return pl.Null()

    def expr(self, ctx: GenContext) -> pl.Expr:
        return pl.repeat(None, pl.len(), dtype=pl.Null())


@dataclass(frozen=True)
class ConstGen:
    """A fixed value in every row. Used by shape cases where the values are beside the point."""

    value: object = None
    dtype: pl.DataType = field(default_factory=lambda: pl.Int64())

    def expr(self, ctx: GenContext) -> pl.Expr:
        if isinstance(self.value, Decimal):
            return as_column(pl.lit(str(self.value)).cast(self.dtype))
        return as_column(pl.lit(self.value, dtype=self.dtype))
