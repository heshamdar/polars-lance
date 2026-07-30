"""The curated corpus: one axis per case.

Deliberately not one wide kitchen-sink frame. A failure has to localise to a dtype, a null
pattern, or a shape, and several axes are mutually exclusive anyway -- a case cannot be both
zero rows and a specific chunk layout.

Cases are generated over parameter grids rather than written out, so adding a dtype adds a row
to every axis it belongs on.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import product
from typing import Literal

import polars as pl

from plioc.gen import categorical as C
from plioc.gen import nested as N
from plioc.gen import primitive as P
from plioc.gen import temporal as T
from plioc.gen.core import NullPattern
from plioc.gen.layout import Layout
from plioc.spec import CaseSpec, ColumnSpec

#: Small enough that the whole corpus runs in seconds, large enough to cross a 1024 boundary.
DEFAULT_ROWS = 1100

INT_DTYPES: tuple[pl.DataType, ...] = (
    pl.Int8(),
    pl.Int16(),
    pl.Int32(),
    pl.Int64(),
    pl.Int128(),
    pl.UInt8(),
    pl.UInt16(),
    pl.UInt32(),
    pl.UInt64(),
)
FLOAT_DTYPES: tuple[pl.DataType, ...] = (pl.Float32(), pl.Float64())
TIME_UNITS: tuple[Literal["ms", "us", "ns"], ...] = ("ms", "us", "ns")


def numeric_cases() -> Iterator[CaseSpec]:
    for dtype in INT_DTYPES:
        for nulls in NullPattern:
            yield CaseSpec(
                id=f"numeric/int/{dtype}/{nulls.name.lower()}",
                columns=(ColumnSpec("v", P.IntGen(dtype), nulls),),
                n_rows=DEFAULT_ROWS,
                tags=frozenset({"numeric", "boundary"}),
            )
    for dtype in FLOAT_DTYPES:
        for nulls in (NullPattern.NONE, NullPattern.SPARSE, NullPattern.ALTERNATING):
            yield CaseSpec(
                id=f"numeric/float/{dtype}/{nulls.name.lower()}",
                columns=(ColumnSpec("v", P.FloatGen(dtype), nulls),),
                n_rows=DEFAULT_ROWS,
                tags=frozenset({"numeric", "float"}),
            )
    yield CaseSpec(
        id="numeric/bool/alternating",
        columns=(ColumnSpec("v", P.BoolGen(), NullPattern.ALTERNATING),),
        n_rows=DEFAULT_ROWS,
        tags=frozenset({"numeric"}),
    )
    for precision, scale in ((38, 0), (38, 38), (38, 10), (10, 2), (1, 0)):
        yield CaseSpec(
            id=f"numeric/decimal/{precision}_{scale}",
            columns=(ColumnSpec("v", P.DecimalGen(precision, scale), NullPattern.SPARSE),),
            n_rows=256,
            tags=frozenset({"numeric", "decimal"}),
        )


def text_cases() -> Iterator[CaseSpec]:
    for tag in ("coercion", "escaping", "unicode", "collision"):
        yield CaseSpec(
            id=f"string/{tag}",
            columns=(
                ColumnSpec(
                    "v", P.StringGen(frozenset({tag}), palette_bias=1.0), NullPattern.SPARSE
                ),
            ),
            n_rows=512,
            tags=frozenset({"string", tag}),
        )
    yield CaseSpec(
        id="string/empty_vs_null",
        columns=(
            # An empty string and a null are different values, and a format that stores strings
            # as "length then bytes" with no validity conflates them.
            ColumnSpec(
                "v", P.StringGen(frozenset({"coercion"}), palette_bias=1.0), NullPattern.ALTERNATING
            ),
        ),
        n_rows=512,
        tags=frozenset({"string"}),
    )
    yield CaseSpec(
        id="string/huge",
        columns=(ColumnSpec("v", P.StringGen(huge=True), NullPattern.SPARSE),),
        n_rows=8,
        tags=frozenset({"string", "slow"}),
    )
    for tag in ("empty", "invalid_utf8"):
        yield CaseSpec(
            id=f"binary/{tag}",
            columns=(
                ColumnSpec(
                    "v", P.BinaryGen(frozenset({tag}), palette_bias=1.0), NullPattern.SPARSE
                ),
            ),
            n_rows=512,
            tags=frozenset({"binary", tag}),
        )


def temporal_cases() -> Iterator[CaseSpec]:
    yield CaseSpec(
        id="temporal/date",
        columns=(ColumnSpec("v", T.DateGen(), NullPattern.SPARSE),),
        n_rows=512,
        tags=frozenset({"temporal"}),
    )
    for unit in TIME_UNITS:
        yield CaseSpec(
            id=f"temporal/datetime/{unit}/naive",
            columns=(ColumnSpec("v", T.DatetimeGen(unit), NullPattern.SPARSE),),
            n_rows=512,
            tags=frozenset({"temporal"}),
        )
    # `Asia/Kolkata` rather than a literal `+05:30`: Polars validates a time zone against the
    # zone database and rejects a bare offset, and the half-hour offset is the point.
    for unit, zone in product(TIME_UNITS[1:], ("UTC", "America/New_York", "Asia/Kolkata")):
        yield CaseSpec(
            id=f"temporal/datetime/{unit}/{zone}",
            columns=(ColumnSpec("v", T.DatetimeGen(unit, zone), NullPattern.SPARSE),),
            n_rows=512,
            tags=frozenset({"temporal", "timezone"}),
        )
    for zone in ("America/New_York", "Europe/London"):
        yield CaseSpec(
            id=f"temporal/dst/{zone}",
            columns=(ColumnSpec("v", T.DatetimeGen("us", zone, dst=True), NullPattern.NONE),),
            n_rows=64,
            tags=frozenset({"temporal", "timezone", "dst"}),
        )
    yield CaseSpec(
        id="temporal/time",
        columns=(ColumnSpec("v", T.TimeGen(), NullPattern.SPARSE),),
        n_rows=512,
        tags=frozenset({"temporal"}),
    )
    for unit in TIME_UNITS:
        yield CaseSpec(
            id=f"temporal/duration/{unit}",
            columns=(ColumnSpec("v", T.DurationGen(unit), NullPattern.SPARSE),),
            n_rows=512,
            tags=frozenset({"temporal"}),
        )


def categorical_cases() -> Iterator[CaseSpec]:
    yield CaseSpec(
        id="categorical/enum",
        columns=(ColumnSpec("v", C.EnumGen(), NullPattern.SPARSE),),
        n_rows=512,
        tags=frozenset({"categorical"}),
    )
    yield CaseSpec(
        id="categorical/enum_unused_categories",
        columns=(ColumnSpec("v", C.EnumGen(unused=("a", "b", "c")), NullPattern.NONE),),
        n_rows=512,
        tags=frozenset({"categorical"}),
    )
    yield CaseSpec(
        id="categorical/enum_empty",
        columns=(ColumnSpec("v", C.EmptyEnumGen(), NullPattern.NONE),),
        n_rows=64,
        tags=frozenset({"categorical"}),
    )
    yield CaseSpec(
        id="categorical/categorical",
        columns=(ColumnSpec("v", C.CategoricalGen(), NullPattern.SPARSE),),
        n_rows=512,
        tags=frozenset({"categorical"}),
    )


def nested_cases() -> Iterator[CaseSpec]:
    yield CaseSpec(
        id="nested/list_int",
        columns=(ColumnSpec("v", N.ListGen(child=P.IntGen()), NullPattern.SPARSE),),
        n_rows=512,
        tags=frozenset({"nested"}),
    )
    # The flagship nested case: a null list, an empty list and a list of nulls are three
    # different values, and each of the three knobs produces one of them.
    yield CaseSpec(
        id="nested/list_three_way",
        columns=(
            ColumnSpec(
                "v",
                N.ListGen(child=P.IntGen(), lengths=(0, 1, 3), child_nulls=NullPattern.ALL),
                NullPattern.ALTERNATING,
            ),
            ColumnSpec(
                "w",
                N.ListGen(child=P.IntGen(), lengths=(0, 2), child_nulls=NullPattern.SPARSE),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=512,
        tags=frozenset({"nested", "null-vs-empty"}),
    )
    yield CaseSpec(
        id="nested/list_empty_only",
        columns=(
            ColumnSpec("v", N.ListGen(child=P.StringGen(), lengths=(0,)), NullPattern.SPARSE),
        ),
        n_rows=256,
        tags=frozenset({"nested", "null-vs-empty"}),
    )
    yield CaseSpec(
        id="nested/struct",
        columns=(
            ColumnSpec(
                "v",
                N.StructGen(
                    (
                        N.StructField("a", P.IntGen(), NullPattern.SPARSE),
                        N.StructField("b", P.StringGen(), NullPattern.SPARSE),
                    )
                ),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=512,
        tags=frozenset({"nested"}),
    )
    # A null struct is not a struct whose fields are all null. A format without struct-level
    # validity stores both as the second; this case is the only thing that notices.
    yield CaseSpec(
        id="nested/struct_null_vs_all_null_fields",
        columns=(
            ColumnSpec(
                "outer_null",
                N.StructGen((N.StructField("a", P.IntGen(), NullPattern.NONE),)),
                NullPattern.ALTERNATING,
            ),
            ColumnSpec(
                "fields_null",
                N.StructGen((N.StructField("a", P.IntGen(), NullPattern.ALL),)),
                NullPattern.NONE,
            ),
        ),
        n_rows=512,
        tags=frozenset({"nested", "null-vs-empty"}),
    )
    yield CaseSpec(
        id="nested/struct_empty",
        columns=(ColumnSpec("v", N.StructGen(()), NullPattern.NONE),),
        n_rows=64,
        tags=frozenset({"nested"}),
    )
    yield CaseSpec(
        id="nested/array_f32_128",
        columns=(
            # The vector-column shape. First-class for a vector store, and the one fixed-size
            # list width a plugin is most likely to have special-cased.
            ColumnSpec(
                "v",
                N.ArrayGen(child=P.FloatGen(pl.Float32()), size=128, child_nulls=NullPattern.NONE),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=256,
        tags=frozenset({"nested", "vector"}),
    )
    yield CaseSpec(
        id="nested/array_nullable_elements",
        columns=(
            ColumnSpec(
                "v",
                N.ArrayGen(child=P.IntGen(), size=3, child_nulls=NullPattern.ALTERNATING),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=256,
        tags=frozenset({"nested"}),
    )
    yield CaseSpec(
        id="nested/deep",
        columns=(
            ColumnSpec(
                "v",
                N.ListGen(
                    child=N.StructGen(
                        (
                            N.StructField("x", N.ListGen(child=P.IntGen()), NullPattern.SPARSE),
                            N.StructField("y", P.StringGen(), NullPattern.SPARSE),
                        )
                    ),
                    lengths=(0, 1, 2),
                ),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=256,
        tags=frozenset({"nested", "deep"}),
    )
    yield CaseSpec(
        id="nested/null_dtype",
        columns=(ColumnSpec("v", P.NullGen(), NullPattern.NONE),),
        n_rows=256,
        tags=frozenset({"nested"}),
    )
    yield CaseSpec(
        id="nested/list_of_null_dtype",
        columns=(ColumnSpec("v", N.ListGen(child=P.NullGen()), NullPattern.SPARSE),),
        n_rows=128,
        tags=frozenset({"nested"}),
    )


#: Names that break path escaping, SQL identifier quoting, or case-insensitive lookup.
AWKWARD_NAMES: tuple[str, ...] = (
    "",
    " ",
    " leading",
    "trailing ",
    "with.dot",
    "with[bracket]",
    'with"quote',
    "with'apostrophe",
    "with`backtick",
    "with space",
    "with/slash",
    "with\\backslash",
    "select",  # reserved SQL word
    "SELECT",
    "group by",
    "café",
    "日本語",
    "x" * 200,
    "Mixed",
    "mixed",  # differs from the previous only by case
)


def name_cases() -> Iterator[CaseSpec]:
    for i, name in enumerate(AWKWARD_NAMES):
        yield CaseSpec(
            id=f"names/single/{i}",
            columns=(ColumnSpec(name, P.IntGen(), NullPattern.SPARSE),),
            n_rows=64,
            tags=frozenset({"names"}),
        )
    yield CaseSpec(
        id="names/all",
        columns=tuple(ColumnSpec(name, P.IntGen(), NullPattern.SPARSE) for name in AWKWARD_NAMES),
        n_rows=64,
        tags=frozenset({"names"}),
    )


#: Either side of every power of two a writer is likely to batch at, plus the powers themselves.
BOUNDARY_ROWS: tuple[int, ...] = (
    0,
    1,
    2,
    63,
    64,
    65,
    1023,
    1024,
    1025,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
)


def shape_cases() -> Iterator[CaseSpec]:
    for n in BOUNDARY_ROWS:
        yield CaseSpec(
            id=f"shape/rows/{n}",
            columns=(
                ColumnSpec("i", P.IntGen(), NullPattern.SPARSE),
                ColumnSpec("s", P.StringGen(), NullPattern.BOUNDARY),
            ),
            n_rows=n,
            tags=frozenset({"shape", "boundary"}),
        )
    yield CaseSpec(
        id="shape/empty_with_schema",
        columns=(
            ColumnSpec("i", P.IntGen(), NullPattern.NONE),
            ColumnSpec("l", N.ListGen(child=P.IntGen()), NullPattern.NONE),
        ),
        n_rows=0,
        tags=frozenset({"shape"}),
    )
    # A frame with no columns has no rows -- Polars cannot represent (n, 0) -- so "zero columns"
    # is necessarily also "zero rows".
    yield CaseSpec(
        id="shape/no_columns",
        columns=(),
        n_rows=0,
        order_key=False,
        tags=frozenset({"shape"}),
    )
    yield CaseSpec(
        id="shape/single_row",
        columns=(ColumnSpec("i", P.IntGen(), NullPattern.NONE),),
        n_rows=1,
        tags=frozenset({"shape"}),
    )
    yield CaseSpec(
        id="shape/wide",
        columns=tuple(ColumnSpec(f"c{i:04d}", P.IntGen(), NullPattern.SPARSE) for i in range(1000)),
        n_rows=16,
        tags=frozenset({"shape", "wide", "slow"}),
    )
    yield CaseSpec(
        id="shape/no_order_key",
        columns=(ColumnSpec("s", P.StringGen(), NullPattern.SPARSE),),
        n_rows=64,
        order_key=False,
        tags=frozenset({"shape"}),
    )


def layout_cases() -> Iterator[CaseSpec]:
    base = (
        ColumnSpec("i", P.IntGen(), NullPattern.BOUNDARY),
        ColumnSpec("s", P.StringGen(), NullPattern.SPARSE),
    )
    layouts = {
        "single_chunk": Layout(rechunk=True),
        "many_small": Layout(chunks=(7,), rechunk=False),
        # 1024 is what writers batch at, so chunks of 1000 put every chunk edge inside a batch.
        "misaligned": Layout(chunks=(1000,), rechunk=False),
        "zero_length_chunk": Layout(chunks=(0, 500), rechunk=False),
        "uneven": Layout(chunks=(1, 2, 3, 1024), rechunk=False),
    }
    for name, layout in layouts.items():
        yield CaseSpec(
            id=f"layout/{name}",
            columns=base,
            n_rows=2100,
            layout=layout,
            tags=frozenset({"layout"}),
        )
    yield CaseSpec(
        id="layout/sorted_flag",
        columns=base,
        n_rows=1100,
        layout=Layout(sorted_by="__i"),
        tags=frozenset({"layout", "metadata"}),
    )


def query_fixture() -> CaseSpec:
    """The case the pushdown suite runs against.

    One frame with a column of every kind a predicate can address, so the whole query matrix
    shares a single write and failures still attribute to a column.
    """
    return CaseSpec(
        id="query/fixture",
        columns=(
            ColumnSpec("i64", P.IntGen(), NullPattern.SPARSE),
            ColumnSpec("i32", P.IntGen(pl.Int32()), NullPattern.NONE),
            ColumnSpec("f64", P.FloatGen(), NullPattern.SPARSE),
            ColumnSpec("bool", P.BoolGen(), NullPattern.SPARSE),
            ColumnSpec("s", P.StringGen(), NullPattern.SPARSE),
            ColumnSpec(
                "esc", P.StringGen(frozenset({"escaping"}), palette_bias=1.0), NullPattern.SPARSE
            ),
            ColumnSpec("dt", T.DatetimeGen("us"), NullPattern.SPARSE),
            ColumnSpec("cat", C.EnumGen(), NullPattern.SPARSE),
            ColumnSpec("lst", N.ListGen(child=P.IntGen(), lengths=(0, 1, 3)), NullPattern.SPARSE),
            ColumnSpec(
                "st",
                N.StructGen(
                    (
                        N.StructField("a", P.IntGen(), NullPattern.SPARSE),
                        N.StructField("b", P.StringGen(), NullPattern.SPARSE),
                    )
                ),
                NullPattern.SPARSE,
            ),
        ),
        n_rows=1100,
        tags=frozenset({"query"}),
    )


GENERATORS = (
    numeric_cases,
    text_cases,
    temporal_cases,
    categorical_cases,
    nested_cases,
    name_cases,
    shape_cases,
    layout_cases,
)


def all_cases(include_slow: bool = True) -> dict[str, CaseSpec]:
    """Every curated case plus every recorded regression, keyed by id."""
    from plioc.regressions import load_regressions

    cases: dict[str, CaseSpec] = {}
    for generator in GENERATORS:
        for case in generator():
            if case.id in cases:
                raise ValueError(f"duplicate case id: {case.id}")
            if not include_slow and "slow" in case.all_tags:
                continue
            cases[case.id] = case
    for case in load_regressions():
        cases[case.id] = case
    fixture = query_fixture()
    cases[fixture.id] = fixture
    return cases


def select(tags: frozenset[str] | None = None, include_slow: bool = True) -> list[CaseSpec]:
    cases = all_cases(include_slow=include_slow).values()
    if tags is None:
        return list(cases)
    return [c for c in cases if c.all_tags & tags]
