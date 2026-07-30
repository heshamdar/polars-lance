"""List, Array and Struct -- the recursive core.

Variable-length lists are built as `concat_list` of `max_len` children truncated by a per-row
length expression. The obvious alternative (cross-join to a flat child frame, `group_by().agg()`,
join back) works but costs both determinism properties this suite is built on: `group_by` order
is an optimiser decision, and the cross-join makes every row's value depend on the row count, so
`build(n=10)` stops being a prefix of `build(n=100)`. See PLAN-REVIEW.md C2.

Truncation evaluates `max_len` children per row regardless of the length drawn. `max_len` is
bounded by the spec and small, so this is cheaper than the machinery it replaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from plioc.gen.core import GenContext, Generator, NullPattern, as_column, nullify, pick

#: Lengths worth drawing: the two degenerate ones, the small ones where an off-by-one shows,
#: and one long enough to cross a writer's inline/out-of-line threshold.
DEFAULT_LENGTHS: tuple[int, ...] = (0, 1, 2, 3, 17)


def _child_expr(child: Generator, nulls: NullPattern, ctx: GenContext) -> pl.Expr:
    return nullify(child.expr(ctx), nulls, ctx, child.dtype)


def _pack(elements: list[pl.Expr], child_dtype: pl.DataType) -> pl.Expr:
    """Gather element expressions into one list per row, without flattening them.

    `pl.concat_list` is the obvious call and is wrong here: given list-typed operands it
    *concatenates* them, so `List<List<T>>` silently comes out as `List<T>` -- the declared dtype
    and the built dtype disagree and every nested-list case tests the wrong shape. `concat_arr`
    nests, and it also keeps the element dtype in a zero-row frame, which `concat_list` does not.
    """
    packed_dtype = child_dtype
    if isinstance(child_dtype, pl.Array):
        # `concat_arr` nests a `List` child but flattens an `Array` one -- `Array<T, 2>` twice over
        # comes out as `Array<T, 4>`. So an array child is packed through its list equivalent and
        # cast back at the end, which nests and keeps the values in place.
        packed_dtype = pl.List(child_dtype.inner)
        elements = [e.cast(packed_dtype) for e in elements]

    if len(elements) == 1:
        # `concat_arr` with one operand returns the operand unwrapped, so there is nothing to
        # cast. Pack two and drop the second: a one-element container is not worth a second path.
        packed = pl.concat_arr([elements[0], elements[0]]).cast(pl.List(packed_dtype)).list.head(1)
    else:
        packed = pl.concat_arr(elements).cast(pl.List(packed_dtype))

    return packed if packed_dtype is child_dtype else packed.cast(pl.List(child_dtype))


@dataclass(frozen=True)
class ListGen:
    """Variable-length list.

    The three knobs are deliberately independent, because the three shapes they produce --
    a null list, an empty list, and a list whose elements are null -- are the distinction most
    often lost in columnar IO, and a case can only prove a harness keeps them apart if it can
    ask for each one separately.

    - a **null list** comes from the column's own `NullPattern`, applied outside this generator
    - an **empty list** comes from `0` being in `lengths`
    - a **list of nulls** comes from `child_nulls`
    """

    child: Generator = field(default=None)  # type: ignore[assignment]
    lengths: tuple[int, ...] = DEFAULT_LENGTHS
    child_nulls: NullPattern = NullPattern.SPARSE

    def __post_init__(self) -> None:
        if self.child is None:
            raise ValueError("ListGen needs a child generator")
        if not self.lengths:
            raise ValueError("ListGen needs at least one length")
        if min(self.lengths) < 0:
            raise ValueError("list lengths cannot be negative")

    @property
    def dtype(self) -> pl.DataType:
        return pl.List(self.child.dtype)

    def expr(self, ctx: GenContext) -> pl.Expr:
        width = max(max(self.lengths), 1)
        elements = [_child_expr(self.child, self.child_nulls, ctx.at(k)) for k in range(width)]
        lengths = pl.Series("lengths", list(self.lengths), dtype=pl.Int64)
        return _pack(elements, self.child.dtype).list.head(
            pick(ctx, lengths, draw=3).cast(pl.UInt32)
        )


@dataclass(frozen=True)
class ArrayGen:
    """Fixed-size list. `Array(Float32, 128)` is the vector-column path for a vector store and
    is worth a case of its own rather than being folded into the list cases."""

    child: Generator = field(default=None)  # type: ignore[assignment]
    size: int = 3
    child_nulls: NullPattern = NullPattern.SPARSE

    def __post_init__(self) -> None:
        if self.child is None:
            raise ValueError("ArrayGen needs a child generator")
        if self.size < 1:
            raise ValueError("array size must be at least 1")

    @property
    def dtype(self) -> pl.DataType:
        return pl.Array(self.child.dtype, self.size)

    def expr(self, ctx: GenContext) -> pl.Expr:
        elements = [_child_expr(self.child, self.child_nulls, ctx.at(k)) for k in range(self.size)]
        return _pack(elements, self.child.dtype).cast(self.dtype)


@dataclass(frozen=True)
class StructField:
    name: str
    gen: Generator
    nulls: NullPattern = NullPattern.SPARSE


@dataclass(frozen=True)
class StructGen:
    """A struct, and the case that distinguishes a null struct from a struct whose fields are
    all null. Those two are different values; a writer without per-struct validity stores both
    as the second, and the corpus is what notices."""

    fields: tuple[StructField, ...] = ()

    @property
    def dtype(self) -> pl.DataType:
        return pl.Struct({f.name: f.gen.dtype for f in self.fields})

    def expr(self, ctx: GenContext) -> pl.Expr:
        if not self.fields:
            # `pl.struct([])` raises; a zero-field struct is only constructible as a literal,
            # and a literal has to be spread over the frame's rows by hand.
            return as_column(pl.lit(pl.Series("s", [{}], dtype=pl.Struct({}))))
        return pl.struct(
            [_child_expr(f.gen, f.nulls, ctx.named(f.name)).alias(f.name) for f in self.fields]
        )
