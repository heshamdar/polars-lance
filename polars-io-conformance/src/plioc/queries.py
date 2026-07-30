"""The curated query matrix.

Generated predicates cover breadth; these cover the places where a backend's semantics and
Polars' semantics are known to part company, and no generator would think to look. They are
written against `corpus.query_fixture()`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from plioc.gen import palettes
from plioc.query import (
    Always,
    And,
    Between,
    Cmp,
    Field,
    IsIn,
    IsNull,
    ListContains,
    Not,
    Opaque,
    Or,
    QuerySpec,
    StrMatch,
    Udf,
)


def kleene_queries() -> Iterator[QuerySpec]:
    """Three-valued logic. A backend that folds unknown to false disagrees with Polars on every
    row where the predicate is null, and disagrees *silently*."""
    yield QuerySpec(
        "kleene/ne_with_nulls", predicate=Cmp("i64", "ne", 0), tags=frozenset({"kleene"})
    )
    yield QuerySpec("kleene/not_ne", predicate=Not(Cmp("i64", "ne", 0)), tags=frozenset({"kleene"}))
    yield QuerySpec(
        "kleene/is_in_with_null",
        predicate=IsIn("i64", (0, 1, None)),
        tags=frozenset({"kleene"}),
    )
    yield QuerySpec(
        "kleene/or_with_is_null",
        predicate=Or(Cmp("i64", "gt", 1), IsNull("s")),
        tags=frozenset({"kleene"}),
    )
    yield QuerySpec(
        "kleene/and_with_is_not_null",
        predicate=And(IsNull("s", negate=True), Cmp("i64", "lt", 0)),
        tags=frozenset({"kleene"}),
    )
    yield QuerySpec("kleene/not_is_null", predicate=Not(IsNull("s")), tags=frozenset({"kleene"}))


def nan_queries() -> Iterator[QuerySpec]:
    """`NaN` is not null, is not equal to itself in IEEE, and *is* equal to itself in Polars.
    Any backend that agrees with only one of those is wrong for the other."""
    yield QuerySpec("nan/gt_zero", predicate=Cmp("f64", "gt", 0.0), tags=frozenset({"nan"}))
    yield QuerySpec("nan/eq_nan", predicate=Cmp("f64", "eq", float("nan")), tags=frozenset({"nan"}))
    yield QuerySpec("nan/is_null", predicate=IsNull("f64"), tags=frozenset({"nan"}))
    yield QuerySpec(
        "nan/is_in_nan",
        predicate=IsIn("f64", (float("nan"), 0.0, -0.0)),
        tags=frozenset({"nan"}),
    )
    yield QuerySpec(
        "nan/negative_zero",
        predicate=Cmp("f64", "eq", -0.0),
        tags=frozenset({"nan"}),
    )
    yield QuerySpec(
        "nan/infinity",
        predicate=Cmp("f64", "ge", float("inf")),
        tags=frozenset({"nan"}),
    )


def collation_queries() -> Iterator[QuerySpec]:
    """Polars orders strings by bytes. A SQL backend may order them by a collation, and the two
    disagree the moment a non-ASCII value appears."""
    for literal in ("z", "é", "Z", ""):
        yield QuerySpec(
            f"collation/lt/{literal or 'empty'}",
            predicate=Cmp("s", "lt", literal),
            tags=frozenset({"collation", "unicode"}),
        )


def escaping_queries() -> Iterator[QuerySpec]:
    """Literals that have to survive being spliced into a filter string.

    For a plugin that renders predicates as SQL this is the single highest-value group here: a
    quoting bug is either a crash or, worse, a filter that silently means something else.
    """
    for i, literal in enumerate(palettes.ESCAPING_STRINGS):
        yield QuerySpec(
            f"escaping/eq/{i}",
            predicate=Cmp("esc", "eq", literal),
            tags=frozenset({"escaping"}),
        )
    for i, literal in enumerate(("%", "_", "'", "\\", "a b")):
        yield QuerySpec(
            f"escaping/contains/{i}",
            predicate=StrMatch("esc", "contains", literal, literal=True),
            tags=frozenset({"escaping"}),
        )
    yield QuerySpec(
        "escaping/starts_with_quote",
        predicate=StrMatch("esc", "starts_with", "'"),
        tags=frozenset({"escaping"}),
    )
    yield QuerySpec(
        "escaping/is_in_quotes",
        predicate=IsIn("esc", ("'", '"', "\\", "';DROP TABLE t;--")),
        tags=frozenset({"escaping"}),
    )


def coercion_queries() -> Iterator[QuerySpec]:
    """Literal types that do not match the column's. Each is legal in Polars and each is a place
    a translator can widen, narrow, or reinterpret."""
    yield QuerySpec(
        "coercion/i32_vs_float", predicate=Cmp("i32", "gt", 0.5), tags=frozenset({"coercion"})
    )
    yield QuerySpec(
        "coercion/i64_vs_big", predicate=Cmp("i64", "lt", 2**70), tags=frozenset({"coercion"})
    )
    yield QuerySpec(
        "coercion/naive_literal_vs_naive_col",
        predicate=Cmp("dt", "gt", datetime(2000, 1, 1)),
        tags=frozenset({"coercion", "temporal"}),
    )
    yield QuerySpec(
        "coercion/bool_eq",
        predicate=Cmp("bool", "eq", True),
        tags=frozenset({"coercion"}),
    )


def scale_queries() -> Iterator[QuerySpec]:
    """An `is_in` list long enough to hit a backend's statement-length limit, and one short
    enough to hit its empty-list syntax."""
    yield QuerySpec("scale/is_in_empty", predicate=IsIn("i64", ()), tags=frozenset({"scale"}))
    yield QuerySpec(
        "scale/is_in_large",
        predicate=IsIn("i64", tuple(range(100_000))),
        tags=frozenset({"scale", "slow"}),
    )


def selectivity_queries() -> Iterator[QuerySpec]:
    yield QuerySpec("selectivity/none", predicate=Always(False), tags=frozenset({"selectivity"}))
    yield QuerySpec("selectivity/all", predicate=Always(True), tags=frozenset({"selectivity"}))
    yield QuerySpec(
        "selectivity/impossible",
        predicate=And(Cmp("i64", "gt", 0), Cmp("i64", "lt", 0)),
        tags=frozenset({"selectivity"}),
    )


def nested_predicate_queries() -> Iterator[QuerySpec]:
    yield QuerySpec(
        "nested/list_contains",
        predicate=ListContains("lst", 0),
        tags=frozenset({"nested"}),
    )
    yield QuerySpec(
        "nested/struct_field_cmp",
        predicate=Field("st", ("a",), Cmp("st", "gt", 0)),
        tags=frozenset({"nested"}),
    )
    yield QuerySpec(
        "nested/struct_field_is_null",
        predicate=Field("st", ("a",), IsNull("st")),
        tags=frozenset({"nested"}),
    )
    yield QuerySpec(
        "nested/list_is_null",
        predicate=IsNull("lst"),
        tags=frozenset({"nested"}),
    )


def mandate_queries() -> Iterator[QuerySpec]:
    """The flagship. Predicates spanning fully-translatable to entirely-opaque.

    `register_io_source` delivers the predicate as an obligation. A plugin that translates the
    part it understands, pushes that, and drops the residual returns wrong rows with no error
    anywhere in the stack -- no exception, no warning, just fewer or more rows than the user
    asked for. Every entry here pairs something a translator will recognise with something it
    cannot, in both orders and under both connectives.
    """
    translatable = Cmp("i64", "gt", 0)
    opaque = Opaque("i32", 3)
    yield QuerySpec("mandate/translatable", predicate=translatable, tags=frozenset({"mandate"}))
    yield QuerySpec("mandate/opaque", predicate=opaque, tags=frozenset({"mandate"}))
    yield QuerySpec(
        "mandate/and_residual",
        predicate=And(translatable, opaque),
        tags=frozenset({"mandate"}),
    )
    yield QuerySpec(
        "mandate/and_residual_reversed",
        predicate=And(opaque, translatable),
        tags=frozenset({"mandate"}),
    )
    yield QuerySpec(
        "mandate/or_residual",
        predicate=Or(translatable, opaque),
        tags=frozenset({"mandate"}),
    )
    yield QuerySpec(
        "mandate/nested_residual",
        predicate=And(And(translatable, Cmp("i32", "lt", 1000)), Or(opaque, IsNull("s"))),
        tags=frozenset({"mandate"}),
    )
    yield QuerySpec(
        "mandate/negated_conjunction",
        predicate=Not(And(translatable, Cmp("s", "eq", "alpha"))),
        tags=frozenset({"mandate"}),
    )
    yield QuerySpec(
        "mandate/between_and_opaque",
        predicate=And(Between("i64", -1000, 1000), opaque),
        tags=frozenset({"mandate"}),
    )
    # A real Python callable, the other kind of node no translator can see into. Tagged so it can
    # be skipped: Polars needs `cloudpickle` to hand a UDF to an IO plugin, and a missing optional
    # dependency should not read as a conformance failure.
    yield QuerySpec("mandate/udf", predicate=Udf("i32", 3), tags=frozenset({"mandate", "udf"}))
    yield QuerySpec(
        "mandate/and_udf_residual",
        predicate=And(translatable, Udf("i32", 3)),
        tags=frozenset({"mandate", "udf"}),
    )


def projection_queries() -> Iterator[QuerySpec]:
    yield QuerySpec("projection/single", projection=("i64",), tags=frozenset({"projection"}))
    yield QuerySpec("projection/reordered", projection=("s", "i64"), tags=frozenset({"projection"}))
    yield QuerySpec(
        "projection/duplicated", projection=("i64", "i64"), tags=frozenset({"projection"})
    )
    yield QuerySpec(
        "projection/predicate_on_unprojected",
        projection=("s",),
        predicate=Cmp("i64", "gt", 0),
        tags=frozenset({"projection"}),
    )
    yield QuerySpec("projection/count_only", count_only=True, tags=frozenset({"projection"}))
    yield QuerySpec(
        "projection/count_with_filter",
        count_only=True,
        predicate=Cmp("i64", "gt", 0),
        tags=frozenset({"projection"}),
    )
    yield QuerySpec(
        "projection/struct_only", projection=("st",), tags=frozenset({"projection", "nested"})
    )


def limit_queries() -> Iterator[QuerySpec]:
    for n in (0, 1, 7, 1024, 10_000):
        yield QuerySpec(f"limit/head/{n}", limit=n, tags=frozenset({"limit"}))
    yield QuerySpec("limit/tail", tail=13, tags=frozenset({"limit"}))
    yield QuerySpec("limit/slice_past_end", offset=10_000, limit=5, tags=frozenset({"limit"}))
    yield QuerySpec("limit/slice_mid", offset=1000, limit=50, tags=frozenset({"limit"}))
    # Polars withholds `n_rows` whenever it pushes a predicate, so this shape exists to prove the
    # limit is applied to the rows that survive the filter and never to the rows read.
    yield QuerySpec(
        "limit/with_filter", limit=10, predicate=Cmp("i64", "gt", 0), tags=frozenset({"limit"})
    )
    yield QuerySpec(
        "limit/with_opaque_filter",
        limit=10,
        predicate=Opaque("i32", 3),
        tags=frozenset({"limit", "mandate"}),
    )


def row_index_queries() -> Iterator[QuerySpec]:
    yield QuerySpec(
        "row_index/before_filter",
        row_index="before",
        predicate=Cmp("i64", "gt", 0),
        tags=frozenset({"row_index"}),
    )
    yield QuerySpec(
        "row_index/after_filter",
        row_index="after",
        predicate=Cmp("i64", "gt", 0),
        tags=frozenset({"row_index"}),
    )


GROUPS = (
    kleene_queries,
    nan_queries,
    collation_queries,
    escaping_queries,
    coercion_queries,
    scale_queries,
    selectivity_queries,
    nested_predicate_queries,
    mandate_queries,
    projection_queries,
    limit_queries,
    row_index_queries,
)


def all_queries(include_slow: bool = True) -> list[QuerySpec]:
    out: list[QuerySpec] = []
    seen: set[str] = set()
    for group in GROUPS:
        for q in group():
            if q.id in seen:
                raise ValueError(f"duplicate query id: {q.id}")
            if not include_slow and "slow" in q.tags:
                continue
            seen.add(q.id)
            out.append(q)
    return out
