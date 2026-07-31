"""`CaseSpec` -- the whole corpus, as data.

A case is a pure function of `(spec, seed)`. It is never a file. `build()` is referentially
transparent: call it twice and the two frames compare equal at the strictest level.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dc_fields
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any

import polars as pl

from plioc.gen.core import (
    IDX,
    IDX_DTYPE,
    GenContext,
    Generator,
    NullPattern,
    nullify,
    row_index,
)
from plioc.gen.layout import Layout


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    gen: Generator
    nulls: NullPattern = NullPattern.SPARSE
    tags: frozenset[str] = frozenset()

    @property
    def dtype(self) -> pl.DataType:
        return self.gen.dtype

    def expr(self, seed: int, n_rows: int) -> pl.Expr:
        ctx = GenContext.for_column(self.name, seed, n_rows)
        value = self.gen.expr(ctx)
        return nullify(value, self.nulls, ctx, self.gen.dtype).alias(self.name)


@dataclass(frozen=True)
class CaseSpec:
    id: str
    columns: tuple[ColumnSpec, ...]
    n_rows: int = 1000
    seed: int = 0
    layout: Layout = field(default_factory=Layout)
    #: Emit `__i` as a leading `Int64` column, so a harness that does not preserve row order can
    #: still be compared after a deterministic sort. Cases testing schemas without an integer
    #: column turn it off and are only comparable below `Strictness.ROW_ORDER`.
    order_key: bool = True
    tags: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.n_rows < 0:
            raise ValueError("n_rows cannot be negative")
        names = [c.name for c in self.columns]
        if self.order_key and IDX in names:
            raise ValueError(f"{IDX!r} is reserved when order_key is set")
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate column names in {self.id}: {sorted(names)}")

    def build(self) -> pl.LazyFrame:
        exprs = [c.expr(self.seed, self.n_rows) for c in self.columns]
        if self.order_key:
            exprs.insert(0, pl.col(IDX))
        if not exprs:
            # A frame with no columns has no rows: Polars cannot represent (n, 0).
            return pl.LazyFrame()
        if self.n_rows == 0:
            # Built at one row and truncated, rather than at zero. Polars' broadcasting rules
            # differ at zero rows -- a length-1 literal expands to one row instead of none, and
            # `list.head` with a row-shaped length either raises or erases the element dtype --
            # so a zero-row frame built directly can end up with a schema no other row count
            # produces. There are no values to lose in the truncation.
            return row_index(1).select(exprs).head(0)
        return row_index(self.n_rows).select(exprs)

    def schema(self) -> pl.Schema:
        fields: list[tuple[str, pl.DataType]] = []
        if self.order_key:
            fields.append((IDX, IDX_DTYPE()))
        fields.extend((c.name, c.dtype) for c in self.columns)
        return pl.Schema(fields)

    @property
    def all_tags(self) -> frozenset[str]:
        return self.tags | frozenset().union(*(c.tags for c in self.columns), frozenset())

    def with_rows(self, n_rows: int) -> CaseSpec:
        return replace(self, n_rows=n_rows)

    def with_layout(self, layout: Layout) -> CaseSpec:
        return replace(self, layout=layout)

    @property
    def depends_on_n(self) -> bool:
        """True when the case's values are not a prefix-stable function of `__i` alone.

        Walks the whole spec, not just the top-level columns: a `NullPattern.LAST` on a struct
        field or a list's elements makes the case just as `n`-dependent, and checking only the
        column would have let those through the prefix-stability test as false passes.
        """
        return any(_uses_row_count(c) for c in self.columns)

    def digest(self, n_rows: int = 64) -> str:
        """A stable fingerprint of what this case generates.

        Guards against a Polars upgrade quietly changing the corpus. Deliberately not
        `hash_rows()` and not IPC bytes -- neither is a stable value across versions -- and
        deliberately not `repr`, which for a nested value is itself a Polars artefact.
        """
        frame = self.with_rows(min(n_rows, self.n_rows) if self.n_rows else 0).build().collect()
        h = hashlib.sha256()
        h.update(b"plioc-digest-v1\n")
        for name, dtype in frame.schema.items():
            h.update(f"{name}\x1f{dtype}\x1e".encode())
        # Sub-microsecond timestamps do not survive the trip through a Python `datetime`, so
        # they are digested as their physical integer instead. Nested ones are not reachable
        # this way and digest at microsecond resolution; this is a drift alarm, not a checksum.
        frame = frame.with_columns(
            pl.col(name).to_physical()
            for name, dtype in frame.schema.items()
            if isinstance(dtype, (pl.Datetime, pl.Duration, pl.Time))
        )
        for row in frame.iter_rows():
            for value in row:
                h.update(canonical(value).encode())
                h.update(b"\x1f")
            h.update(b"\x1e")
        return h.hexdigest()


def _uses_row_count(node: Any) -> bool:
    if isinstance(node, NullPattern):
        return node.depends_on_n
    if is_dataclass(node) and not isinstance(node, type):
        return any(_uses_row_count(getattr(node, f.name)) for f in dc_fields(node))
    if isinstance(node, (tuple, list)):
        return any(_uses_row_count(v) for v in node)
    return False


def canonical(value: Any) -> str:
    """Encode a Python value pulled out of a frame, without going through `str` of anything
    whose formatting a library owns.

    Floats use `float.hex`, which is exact and has no formatting choices in it -- so `-0.0` and
    `0.0` differ here, as they must.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"f:{value.hex()}"
    if isinstance(value, int):
        return f"i:{value}"
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        return f"d:{sign}:{''.join(map(str, digits))}:{exponent}"
    if isinstance(value, str):
        return "s:" + value.encode("utf-8", "surrogatepass").hex()
    if isinstance(value, (bytes, bytearray)):
        return "b:" + bytes(value).hex()
    if isinstance(value, datetime):
        offset = value.utcoffset()
        secs = "naive" if offset is None else str(int(offset.total_seconds()))
        return f"dt:{value.year:04d}{value.month:02d}{value.day:02d}T{value.hour:02d}{value.minute:02d}{value.second:02d}.{value.microsecond:06d}:{secs}"
    if isinstance(value, date):
        return f"D:{value.year:04d}{value.month:02d}{value.day:02d}"
    if isinstance(value, time):
        return f"T:{value.hour:02d}{value.minute:02d}{value.second:02d}.{value.microsecond:06d}"
    if isinstance(value, timedelta):
        return f"td:{value.days}:{value.seconds}:{value.microseconds}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}={canonical(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"no canonical encoding for {type(value).__name__}")
