"""Hypothesis strategies.

Both layers generate **specs**, never data. That is what makes shrinking useful: a counterexample
shrinks to a frozen dataclass roughly ten lines long, which `plioc.regressions.record` writes to
a file and the corpus picks up forever after. A strategy over frames would shrink to a frame,
which is a data file by another name.

`polars.testing.parametric` is a reasonable source of dtype strategies and a poor source of
values -- it generates plausible data, and this suite needs adversarial data.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from hypothesis import strategies as st

from plioc.corpus import AWKWARD_NAMES, FLOAT_DTYPES, INT_DTYPES
from plioc.gen import categorical as C
from plioc.gen import nested as N
from plioc.gen import primitive as P
from plioc.gen import temporal as T
from plioc.gen.core import Generator, NullPattern
from plioc.gen.layout import Layout
from plioc.query import (
    And,
    Between,
    Cmp,
    IsIn,
    IsNull,
    Not,
    Opaque,
    Or,
    Pred,
    QuerySpec,
    StrMatch,
)
from plioc.spec import CaseSpec, ColumnSpec

#: Small and drawn from a fixed set, so shrinking lands on a row count that is quick to run and
#: still crosses a batch boundary.
ROW_COUNTS = (0, 1, 2, 63, 64, 65, 1023, 1024, 1025)

null_patterns = st.sampled_from(list(NullPattern))


def leaf_generators() -> st.SearchStrategy[Generator]:
    return st.one_of(
        st.builds(P.IntGen, dtype=st.sampled_from(INT_DTYPES)),
        st.builds(P.FloatGen, dtype=st.sampled_from(FLOAT_DTYPES)),
        st.builds(P.BoolGen),
        st.builds(
            P.StringGen,
            tags=st.sampled_from([None, frozenset({"escaping"}), frozenset({"unicode"})]),
        ),
        st.builds(P.BinaryGen),
        st.builds(P.NullGen),
        st.builds(T.DateGen),
        st.builds(T.TimeGen),
        st.builds(
            T.DatetimeGen,
            time_unit=st.sampled_from(["ms", "us", "ns"]),
            time_zone=st.sampled_from([None, "UTC", "America/New_York"]),
        ),
        st.builds(T.DurationGen, time_unit=st.sampled_from(["ms", "us", "ns"])),
        st.builds(C.EnumGen),
        st.builds(C.CategoricalGen),
        # Scale cannot exceed precision, so it is drawn relative to it rather than independently.
        st.integers(1, 38).flatmap(
            lambda p: st.builds(
                P.DecimalGen, precision=st.just(p), scale=st.integers(0, min(p, 10))
            )
        ),
    )


def generators(max_depth: int = 2) -> st.SearchStrategy[Generator]:
    """Recursive composition.

    Depth and width are both capped, and the widths here are smaller than the curated corpus's on
    purpose: element counts multiply through nesting, so a 17-element list inside an array inside
    a list is a thousand child expressions per row and turns one example into minutes. Long lists
    belong in the curated cases, where the nesting around them is shallow and known.
    """

    def extend(children: st.SearchStrategy[Generator]) -> st.SearchStrategy[Generator]:
        return st.one_of(
            st.builds(
                N.ListGen,
                child=children,
                lengths=st.sampled_from([(0, 1, 2), (0,), (1,), (0, 3)]),
                child_nulls=null_patterns,
            ),
            st.builds(
                N.ArrayGen, child=children, size=st.integers(1, 3), child_nulls=null_patterns
            ),
            st.builds(
                N.StructGen,
                fields=st.lists(
                    st.builds(
                        N.StructField,
                        name=st.sampled_from(["a", "b", "with.dot", ""]),
                        gen=children,
                        nulls=null_patterns,
                    ),
                    min_size=0,
                    max_size=3,
                    unique_by=lambda f: f.name,
                ).map(tuple),
            ),
        )

    return st.recursive(leaf_generators(), extend, max_leaves=max_depth + 1)


@st.composite
def column_specs(draw: Any, name: str) -> ColumnSpec:
    return ColumnSpec(name=name, gen=draw(generators()), nulls=draw(null_patterns))


@st.composite
def case_specs(draw: Any, max_columns: int = 4) -> CaseSpec:
    n = draw(st.integers(1, max_columns))
    names = draw(
        st.lists(
            st.one_of(st.sampled_from(AWKWARD_NAMES), st.text(min_size=1, max_size=8)),
            min_size=n,
            max_size=n,
            unique=True,
        )
    )
    columns = tuple(draw(column_specs(name)) for name in names)
    return CaseSpec(
        id="hypothesis/" + draw(st.uuids()).hex[:12],
        columns=columns,
        n_rows=draw(st.sampled_from(ROW_COUNTS)),
        seed=draw(st.integers(0, 2**32 - 1)),
        layout=draw(
            st.sampled_from(
                [
                    Layout(),
                    Layout(chunks=(7,), rechunk=False),
                    Layout(chunks=(0, 33), rechunk=False),
                ]
            )
        ),
    )


_NUMERIC = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.Int128,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)


def _leaf_predicates(schema: pl.Schema) -> list[st.SearchStrategy[Pred]]:
    out: list[st.SearchStrategy[Pred]] = []
    for name, dtype in schema.items():
        out.append(st.builds(IsNull, column=st.just(name), negate=st.booleans()))
        if isinstance(dtype, _NUMERIC):
            values = st.one_of(st.integers(-(2**20), 2**20), st.floats(allow_nan=True, width=32))
            out.append(
                st.builds(
                    Cmp,
                    column=st.just(name),
                    op=st.sampled_from(["eq", "ne", "lt", "le", "gt", "ge"]),
                    value=values,
                )
            )
            out.append(
                st.builds(
                    IsIn, column=st.just(name), values=st.lists(values, max_size=8).map(tuple)
                )
            )
            out.append(
                st.builds(
                    Between,
                    column=st.just(name),
                    low=values,
                    high=values,
                    closed=st.sampled_from(["both", "left", "right", "none"]),
                )
            )
            if isinstance(dtype, (pl.Int32, pl.Int64)):
                out.append(st.builds(Opaque, column=st.just(name), modulus=st.integers(2, 5)))
        if dtype == pl.String:
            literals = st.sampled_from(["", "a", "'", "%", "é", "\\"])
            out.append(
                st.builds(
                    Cmp,
                    column=st.just(name),
                    op=st.sampled_from(["eq", "ne", "lt", "gt"]),
                    value=literals,
                )
            )
            out.append(
                st.builds(
                    StrMatch,
                    column=st.just(name),
                    kind=st.sampled_from(["contains", "starts_with", "ends_with"]),
                    value=literals,
                    literal=st.just(True),
                )
            )
    return out


def predicates(schema: pl.Schema, max_depth: int = 3) -> st.SearchStrategy[Pred]:
    leaves = _leaf_predicates(schema)
    if not leaves:
        return st.builds(IsNull, column=st.just(next(iter(schema))))
    base = st.one_of(*leaves)
    return st.recursive(
        base,
        lambda children: st.one_of(
            st.builds(And, left=children, right=children),
            st.builds(Or, left=children, right=children),
            st.builds(Not, inner=children),
        ),
        max_leaves=max_depth,
    )


@st.composite
def queries(draw: Any, schema: pl.Schema) -> QuerySpec:
    names = list(schema)
    projection = draw(
        st.one_of(
            st.none(), st.lists(st.sampled_from(names), min_size=1, max_size=len(names)).map(tuple)
        )
    )
    return QuerySpec(
        id="hypothesis/" + draw(st.uuids()).hex[:12],
        projection=projection,
        predicate=draw(st.one_of(st.none(), predicates(schema))),
        limit=draw(st.one_of(st.none(), st.sampled_from([0, 1, 7, 64, 10_000]))),
        offset=draw(st.one_of(st.none(), st.sampled_from([0, 1, 63, 10_000]))),
        row_index=draw(st.sampled_from(["none", "before", "after"])),
    )
