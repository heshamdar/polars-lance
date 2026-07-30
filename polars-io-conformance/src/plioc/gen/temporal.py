"""Date, datetime, time and duration generators.

Time zones are handled by generating an instant and converting, never by attaching a zone to a
naive local time: `replace_time_zone` on an ambiguous or non-existent local time raises, and the
DST palette exists precisely to contain such local times.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timezone
from typing import Literal

import polars as pl

from plioc.gen import palettes
from plioc.gen.core import GenContext, pick, uniform_int
from plioc.gen.primitive import DEFAULT_BIAS, _blend

TimeUnit = Literal["ms", "us", "ns"]

#: Physical range of each time unit that stays inside i64 and inside what most formats accept.
_FILL_SPAN: dict[str, tuple[int, int]] = {
    "ms": (-62135596800_000, 253402300799_999),
    "us": (-62135596800_000_000, 253402300799_999_999),
    "ns": (-9223372036, 9223372036_000_000_000 // 1000),
}


@dataclass(frozen=True)
class DateGen:
    palette_bias: float = DEFAULT_BIAS

    @property
    def dtype(self) -> pl.DataType:
        return pl.Date()

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = palettes.series(palettes.DATE_VALUES, pl.Date())
        fill = uniform_int(ctx, -719162, 2932896, draw=2).cast(pl.Int32).cast(pl.Date)
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class DatetimeGen:
    time_unit: TimeUnit = "us"
    time_zone: str | None = None
    dst: bool = False
    palette_bias: float = DEFAULT_BIAS

    @property
    def dtype(self) -> pl.DataType:
        return pl.Datetime(self.time_unit, self.time_zone)

    def _palette(self) -> pl.Series:
        if self.dst:
            values = [v.replace(tzinfo=timezone.utc) for v in palettes.DST_INSTANTS_UTC]
        else:
            values = list(palettes.DATETIME_VALUES)
            if self.time_unit == "ns":
                # Years outside ~1677..2262 are not representable as i64 nanoseconds.
                values = [v for v in values if 1678 <= v.year <= 2261]
            if self.time_zone is not None:
                values = [v.replace(tzinfo=timezone.utc) for v in values]
        s = pl.Series("palette", values, dtype=pl.Datetime(self.time_unit, "UTC"))
        if self.time_zone is None:
            return s.dt.replace_time_zone(None)
        return s.dt.convert_time_zone(self.time_zone)

    def expr(self, ctx: GenContext) -> pl.Expr:
        lo, hi = _FILL_SPAN[self.time_unit]
        fill = uniform_int(ctx, lo, hi, draw=2).cast(pl.Datetime(self.time_unit))
        if self.time_zone is not None:
            fill = fill.dt.replace_time_zone("UTC").dt.convert_time_zone(self.time_zone)
        return _blend(ctx, self.palette_bias, pick(ctx, self._palette()), fill)


@dataclass(frozen=True)
class TimeGen:
    palette_bias: float = DEFAULT_BIAS

    @property
    def dtype(self) -> pl.DataType:
        return pl.Time()

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = palettes.series(palettes.TIME_VALUES, pl.Time())
        fill = uniform_int(ctx, 0, 86_399_999_999_999, draw=2).cast(pl.Time)
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)


@dataclass(frozen=True)
class DurationGen:
    time_unit: TimeUnit = "us"
    palette_bias: float = DEFAULT_BIAS

    @property
    def dtype(self) -> pl.DataType:
        return pl.Duration(self.time_unit)

    def expr(self, ctx: GenContext) -> pl.Expr:
        pal = palettes.series(palettes.duration_values(self.time_unit), pl.Int64()).cast(
            pl.Duration(self.time_unit)
        )
        fill = uniform_int(ctx, -(2**40), 2**40, draw=2).cast(pl.Duration(self.time_unit))
        return _blend(ctx, self.palette_bias, pick(ctx, pal), fill)
