"""Queries as data.

A predicate is an AST, not a `pl.Expr`, for three reasons: it survives shrinking to something a
human can read, it serialises into a regression file, and the same value can be rendered both as
an expression and as a name for a test id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl

CmpOp = Literal["eq", "ne", "lt", "le", "gt", "ge"]


@runtime_checkable
class Pred(Protocol):
    def expr(self) -> pl.Expr: ...
    def label(self) -> str: ...
    def columns(self) -> frozenset[str]: ...


@dataclass(frozen=True)
class Cmp:
    column: str
    op: CmpOp = "gt"
    value: Any = 0

    def expr(self) -> pl.Expr:
        col = pl.col(self.column)
        return {
            "eq": col.eq,
            "ne": col.ne,
            "lt": col.lt,
            "le": col.le,
            "gt": col.gt,
            "ge": col.ge,
        }[self.op](pl.lit(self.value))

    def label(self) -> str:
        return f"{self.column}.{self.op}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class IsNull:
    column: str
    negate: bool = False

    def expr(self) -> pl.Expr:
        col = pl.col(self.column)
        return col.is_not_null() if self.negate else col.is_null()

    def label(self) -> str:
        return f"{self.column}.{'is_not_null' if self.negate else 'is_null'}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class IsIn:
    column: str
    values: tuple[Any, ...] = ()

    def expr(self) -> pl.Expr:
        # An empty `is_in` is a real query, and it is where a backend that renders the list into
        # `IN ()` produces a syntax error rather than an empty result. `implode` rather than a
        # bare Series literal: Polars deprecated the ambiguous same-dtype form.
        return pl.col(self.column).is_in(pl.lit(pl.Series(list(self.values))).implode())

    def label(self) -> str:
        return f"{self.column}.is_in[{len(self.values)}]"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class Between:
    column: str
    low: Any = 0
    high: Any = 1
    closed: Literal["both", "left", "right", "none"] = "both"

    def expr(self) -> pl.Expr:
        return pl.col(self.column).is_between(
            pl.lit(self.low), pl.lit(self.high), closed=self.closed
        )

    def label(self) -> str:
        return f"{self.column}.between"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class StrMatch:
    column: str
    kind: Literal["contains", "starts_with", "ends_with"] = "contains"
    value: str = ""
    literal: bool = True

    def expr(self) -> pl.Expr:
        s = pl.col(self.column).str
        if self.kind == "contains":
            return s.contains(self.value, literal=self.literal)
        if self.kind == "starts_with":
            return s.starts_with(self.value)
        return s.ends_with(self.value)

    def label(self) -> str:
        return f"{self.column}.str.{self.kind}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class ListContains:
    column: str
    value: Any = 0

    def expr(self) -> pl.Expr:
        return pl.col(self.column).list.contains(pl.lit(self.value))

    def label(self) -> str:
        return f"{self.column}.list.contains"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class Field:
    """A predicate on a struct field, which several backends address with a dotted path -- and
    which collides with a column whose name genuinely contains a dot."""

    column: str
    path: tuple[str, ...] = ()
    inner: Pred | None = None

    def expr(self) -> pl.Expr:
        col = pl.col(self.column)
        for part in self.path:
            col = col.struct.field(part)
        if self.inner is None:
            return col.is_not_null()
        return _rebind(self.inner, col)

    def label(self) -> str:
        return f"{self.column}.{'.'.join(self.path)}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


def _rebind(pred: Pred, target: pl.Expr) -> pl.Expr:
    """Apply a single-column predicate to an arbitrary sub-expression."""
    if isinstance(pred, Cmp):
        return {
            "eq": target.eq,
            "ne": target.ne,
            "lt": target.lt,
            "le": target.le,
            "gt": target.gt,
            "ge": target.ge,
        }[pred.op](pl.lit(pred.value))
    if isinstance(pred, IsNull):
        return target.is_not_null() if pred.negate else target.is_null()
    raise TypeError(f"cannot rebind {type(pred).__name__} onto a sub-expression")


@dataclass(frozen=True)
class And:
    left: Pred
    right: Pred

    def expr(self) -> pl.Expr:
        return self.left.expr() & self.right.expr()

    def label(self) -> str:
        return f"({self.left.label()}&{self.right.label()})"

    def columns(self) -> frozenset[str]:
        return self.left.columns() | self.right.columns()


@dataclass(frozen=True)
class Or:
    left: Pred
    right: Pred

    def expr(self) -> pl.Expr:
        return self.left.expr() | self.right.expr()

    def label(self) -> str:
        return f"({self.left.label()}|{self.right.label()})"

    def columns(self) -> frozenset[str]:
        return self.left.columns() | self.right.columns()


@dataclass(frozen=True)
class Not:
    inner: Pred

    def expr(self) -> pl.Expr:
        return ~self.inner.expr()

    def label(self) -> str:
        return f"~{self.inner.label()}"

    def columns(self) -> frozenset[str]:
        return self.inner.columns()


@dataclass(frozen=True)
class Opaque:
    """A predicate no translator can see into.

    Its job is to be *untranslatable*: a plugin must recognise that and fall back, and the result
    must still be right. Paired with a translatable side under `And`, it is the residual that
    `DropsResidual` throws away.

    Built from `Expr.hash`, not from `map_elements`. A UDF is the more obvious opaque node, but
    Polars serialises one with `cloudpickle` to send it through the plugin boundary, so a
    UDF-based predicate turns a missing optional dependency into a suite-wide failure. `hash` is
    equally untranslatable, needs nothing, and is deterministic within a process -- which is all
    the differential comparison requires, since both sides run in the same one.
    """

    column: str
    modulus: int = 2

    def expr(self) -> pl.Expr:
        return (pl.col(self.column).hash(seed=17) % pl.lit(self.modulus, pl.UInt64)) == 0

    def label(self) -> str:
        return f"{self.column}.opaque%{self.modulus}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class Udf:
    """The other kind of opaque node: a genuine Python callable.

    Kept separate from `Opaque` because Polars needs `cloudpickle` to hand one to an IO plugin,
    so the queries using it are skipped rather than failed when it is absent.
    """

    column: str
    modulus: int = 2

    def expr(self) -> pl.Expr:
        return (
            pl.col(self.column)
            .map_elements(
                lambda v: v is not None and int(v) % self.modulus == 0,
                return_dtype=pl.Boolean,
            )
            .fill_null(False)
        )

    def label(self) -> str:
        return f"{self.column}.udf%{self.modulus}"

    def columns(self) -> frozenset[str]:
        return frozenset({self.column})


@dataclass(frozen=True)
class Always:
    value: bool = True

    def expr(self) -> pl.Expr:
        return pl.lit(self.value)

    def label(self) -> str:
        return "all" if self.value else "none"

    def columns(self) -> frozenset[str]:
        return frozenset()


@dataclass(frozen=True)
class QuerySpec:
    """One query shape, applied identically to the harness frame and to the oracle frame."""

    id: str
    projection: tuple[str, ...] | None = None
    predicate: Pred | None = None
    limit: int | None = None
    offset: int | None = None
    tail: int | None = None
    #: `with_row_index` before the filter rather than after. Measured: this suppresses predicate
    #: pushdown entirely, so engagement must not be asserted for it.
    row_index: Literal["none", "before", "after"] = "none"
    count_only: bool = False
    tags: frozenset[str] = frozenset()

    def apply(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        out = lf
        if self.row_index == "before":
            out = out.with_row_index("__ri")
        if self.predicate is not None:
            out = out.filter(self.predicate.expr())
        if self.row_index == "after":
            out = out.with_row_index("__ri")
        if self.offset is not None:
            out = out.slice(self.offset, self.limit)
        elif self.limit is not None:
            out = out.head(self.limit)
        if self.tail is not None:
            out = out.tail(self.tail)
        if self.count_only:
            return out.select(pl.len())
        if self.projection is not None:
            out = out.select(list(self.projection))
        return out

    @property
    def pushdown_observable(self) -> bool:
        """False for shapes Polars does not push through the plugin interface at all."""
        return self.row_index != "before"

    def with_predicate(self, predicate: Pred | None) -> QuerySpec:
        return QuerySpec(**{**self.__dict__, "predicate": predicate})


def projection_of(schema: pl.Schema, *names: str) -> tuple[str, ...]:
    missing = [n for n in names if n not in schema]
    if missing:
        raise KeyError(f"not in schema: {missing}")
    return tuple(names)
