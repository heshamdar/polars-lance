"""The six primitives every generator is built from.

Everything here is an expression over the row index `__i`. Nothing reads the clock, the
environment, or `random`, and nothing depends on the row count -- see `NullPattern.LAST` for
the single documented exception.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, runtime_checkable

import polars as pl

#: Name of the row-index column every generated frame is built from.
IDX = "__i"

#: Dtype of the row index, and therefore of the identity column a case emits. `Int64`, not the
#: obvious `UInt64`: several real formats -- Avro and Delta Lake among them -- have no unsigned
#: 64-bit type at all, and an identity column they cannot store makes every case in the corpus
#: fail for a reason that has nothing to do with what the case tests. Parquet and IPC carry
#: `UInt64` happily, which is exactly why this went unnoticed until a third-party plugin ran.
IDX_DTYPE = pl.Int64

#: The mixer's working type. Unrelated to the index dtype -- the index is cast into it.
U64 = pl.UInt64

# splitmix64. Chosen over `Expr.hash` because its output is a documented, fixed function of the
# input rather than whatever Polars' hasher happens to be this release; see docs/api-findings.md.
_GOLDEN = 0x9E3779B97F4A7C15
_MIX_1 = 0xBF58476D1CE4E5B9
_MIX_2 = 0x94D049BB133111EB
_U64_MASK = (1 << 64) - 1


def row_index(n: int) -> pl.LazyFrame:
    """A lazy frame of `n` rows carrying only the row index."""
    return pl.LazyFrame().select(pl.int_range(0, n, dtype=IDX_DTYPE).alias(IDX))


def idx() -> pl.Expr:
    return pl.col(IDX)


def stream(name: str, seed: int) -> int:
    """Map a column name to its own 64-bit stream key.

    Keyed by name rather than by position so that adding a column to a case leaves every other
    column's values untouched. `hash()` cannot be used: CPython randomises `str` hashing per
    process, which would make `build()` non-deterministic across runs.
    """
    h = hashlib.blake2b(f"{seed}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


def _shr(e: pl.Expr, k: int) -> pl.Expr:
    # `Expr` has no `__rshift__`, and `bitwise_xor`/`bitwise_or` are reductions. Floor division
    # by a power of two is an exact logical shift for unsigned integers.
    return e // pl.lit(1 << k, U64)


def _splitmix64(x: pl.Expr) -> pl.Expr:
    x = x * pl.lit(_GOLDEN, U64)
    z = (x ^ _shr(x, 30)) * pl.lit(_MIX_1, U64)
    z = (z ^ _shr(z, 27)) * pl.lit(_MIX_2, U64)
    return z ^ _shr(z, 31)


def splitmix64(x: int) -> int:
    """The reference implementation the expression form is asserted equal to."""
    x = (x * _GOLDEN) & _U64_MASK
    z = ((x ^ (x >> 30)) * _MIX_1) & _U64_MASK
    z = ((z ^ (z >> 27)) * _MIX_2) & _U64_MASK
    return (z ^ (z >> 31)) & _U64_MASK


@dataclass(frozen=True)
class GenContext:
    """What a generator needs to know: which stream to draw from, and how deep it is."""

    seed: int
    key: int
    depth: int = 0
    n_rows: int = 0

    @classmethod
    def for_column(cls, name: str, seed: int, n_rows: int) -> GenContext:
        return cls(seed=seed, key=stream(name, seed), depth=0, n_rows=n_rows)

    def at(self, position: int) -> GenContext:
        """A child context for element `position` of a container.

        Without this every element of a fixed-size list is built from the same expression and
        therefore holds the same value, which hides any bug in element ordering.
        """
        return replace(self, key=splitmix64(self.key ^ (position + 1)), depth=self.depth + 1)

    def named(self, name: str) -> GenContext:
        """A child context for a named field, keyed by the field name for the same reason
        column streams are: adding a struct field must not disturb its siblings."""
        return replace(self, key=self.key ^ stream(name, self.seed), depth=self.depth + 1)


@runtime_checkable
class Generator(Protocol):
    """A composable value, not a class with behaviour.

    `expr` returns a never-null expression; nulls are a separate axis applied on top of it by
    `nullify`. Composition is the whole nesting story -- `ListGen(child=StructGen(...))` recurses
    by threading a derived context, so `List<Struct<List<Int64>>>` costs nothing extra.
    """

    @property
    def dtype(self) -> pl.DataType: ...

    def expr(self, ctx: GenContext) -> pl.Expr: ...


def rnd(ctx: GenContext, draw: int = 0) -> pl.Expr:
    """A `UInt64` pseudo-random draw, one independent value per `(ctx, draw)`."""
    return _splitmix64(
        idx().cast(U64) ^ pl.lit(splitmix64(ctx.key ^ (draw * _GOLDEN & _U64_MASK)), U64)
    )


def unit(ctx: GenContext, draw: int = 0) -> pl.Expr:
    """A `Float64` draw in `[0, 1)`."""
    return _shr(rnd(ctx, draw), 11).cast(pl.Float64) / pl.lit(float(1 << 53))


def uniform_int(ctx: GenContext, lo: int, hi: int, draw: int = 0) -> pl.Expr:
    """A draw in `[lo, hi]`, inclusive, as `Int64`.

    Modulo bias is irrelevant here -- the point is coverage, not statistical quality -- and the
    modulo keeps the value inside the range so the later cast to a narrow width cannot raise.
    """
    span = hi - lo + 1
    return (rnd(ctx, draw) % pl.lit(span, U64)).cast(pl.Int64) + pl.lit(lo, pl.Int64)


def pick(ctx: GenContext, palette: pl.Series, draw: int = 0) -> pl.Expr:
    """Draw from a palette, preserving its dtype exactly."""
    if palette.len() == 0:
        raise ValueError("cannot pick from an empty palette")
    return pl.lit(palette).gather(rnd(ctx, draw) % pl.lit(palette.len(), U64))


class NullPattern(Enum):
    """Where the nulls go. Orthogonal to dtype, and its own axis in the corpus."""

    NONE = "none"
    ALL = "all"
    SPARSE = "sparse"
    DENSE = "dense"
    FIRST = "first"
    LAST = "last"
    ALTERNATING = "alternating"
    BOUNDARY = "boundary"

    @property
    def depends_on_n(self) -> bool:
        """`LAST` is the one pattern that cannot be a function of `__i` alone.

        Prefix stability (`build(n=10) == build(n=100).head(10)`) does not hold for a column
        using it. `tests/test_determinism.py` asserts this is the only such pattern.
        """
        return self is NullPattern.LAST


#: Chunk size the `BOUNDARY` pattern aims at. Writers commonly batch at 1024 or a multiple of
#: it, so a null landing on 0 and 1023 of every block lands on their batch edges.
BOUNDARY_BLOCK = 1024


def null_mask(pattern: NullPattern, ctx: GenContext, draw: int = 900) -> pl.Expr | None:
    """The boolean mask selecting the rows that are null, or `None` for never."""
    i = idx()
    if pattern is NullPattern.NONE:
        return None
    if pattern is NullPattern.ALL:
        # Row-shaped rather than `pl.lit(True)`: a scalar mask makes the whole `when` scalar, and
        # a scalar branch is broadcast to one row instead of to none in a zero-row frame.
        return i >= pl.lit(0, IDX_DTYPE)
    if pattern is NullPattern.SPARSE:
        return rnd(ctx, draw) % pl.lit(100, U64) < pl.lit(5, U64)
    if pattern is NullPattern.DENSE:
        return rnd(ctx, draw) % pl.lit(100, U64) < pl.lit(95, U64)
    if pattern is NullPattern.FIRST:
        return i == pl.lit(0, IDX_DTYPE)
    if pattern is NullPattern.LAST:
        return i == pl.lit(max(ctx.n_rows - 1, 0), IDX_DTYPE)
    if pattern is NullPattern.ALTERNATING:
        return i % pl.lit(2, IDX_DTYPE) == pl.lit(0, IDX_DTYPE)
    if pattern is NullPattern.BOUNDARY:
        pos = i % pl.lit(BOUNDARY_BLOCK, IDX_DTYPE)
        return (pos == pl.lit(0, IDX_DTYPE)) | (pos == pl.lit(BOUNDARY_BLOCK - 1, IDX_DTYPE))
    raise AssertionError(f"unhandled null pattern: {pattern}")


def nullify(value: pl.Expr, pattern: NullPattern, ctx: GenContext, dtype: pl.DataType) -> pl.Expr:
    """Punch nulls into a value expression.

    The dtype on the null literal is not optional: without it the two branches of the `when`
    have different types and the result is either an error or a silently widened column.
    """
    mask = null_mask(pattern, ctx)
    if mask is None:
        return value
    # `ALL` goes through the same branch as every other pattern. Short-circuiting it to a bare
    # `pl.lit(None, dtype)` looks like an obvious saving and is not one: a literal is a length-1
    # expression, so inside a `concat_list` in a zero-row frame it both mismatches the other
    # operands and loses the element dtype.
    return pl.when(mask).then(pl.lit(None, dtype=dtype)).otherwise(value)


def as_column(scalar: pl.Expr) -> pl.Expr:
    """Turn a length-1 expression into one row per row of the frame.

    Needed by the generators whose value is a constant. A literal alone is broadcast to a single
    row, which silently turns a zero-row case into a one-row case.
    """
    return scalar.gather(pl.zeros(pl.len(), dtype=pl.UInt32, eager=False))
