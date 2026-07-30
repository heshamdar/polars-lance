"""Value tables weighted toward pathology.

A uniform draw over a dtype's range almost never produces the values that break IO: it does not
produce `-0.0`, it does not produce `""`, and it does not produce a string that a SQL filter has
to escape. Generators draw from these tables most of the time and fill the rest with ordinary
bulk data, so a case exercises both.

Entries are grouped by tag. A case can ask for one group (`tags={"escaping"}`) so that a failure
attributes to a value class rather than to "strings".
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal, localcontext
from typing import Any

import polars as pl

# --------------------------------------------------------------------------------------- ints

_INT_LIMITS: dict[str, tuple[int, int]] = {
    "Int8": (-(2**7), 2**7 - 1),
    "Int16": (-(2**15), 2**15 - 1),
    "Int32": (-(2**31), 2**31 - 1),
    "Int64": (-(2**63), 2**63 - 1),
    "Int128": (-(2**127), 2**127 - 1),
    "UInt8": (0, 2**8 - 1),
    "UInt16": (0, 2**16 - 1),
    "UInt32": (0, 2**32 - 1),
    "UInt64": (0, 2**64 - 1),
}


def int_limits(dtype: pl.DataType) -> tuple[int, int]:
    return _INT_LIMITS[_dtype_key(dtype)]


def _dtype_key(dtype: pl.DataType) -> str:
    return type(dtype).__name__ if isinstance(dtype, pl.DataType) else dtype.__name__  # type: ignore[unreachable]


def int_values(dtype: pl.DataType) -> list[int]:
    """Zero, the ends, one step in from the ends, and every power-of-two boundary in range.

    The power-of-two boundaries are where a writer that narrows a column (i64 -> i32 because
    "the values fit") gets it wrong by exactly one.
    """
    lo, hi = int_limits(dtype)
    out = {0, lo, hi, lo + 1, hi - 1}
    if lo < 0:
        out |= {-1, 1}
    else:
        out |= {1}
    bit = 8
    while bit < 128:
        for base in (1 << bit, -(1 << bit)):
            for delta in (-1, 0, 1):
                v = base + delta
                if lo <= v <= hi:
                    out.add(v)
        bit *= 2
    return sorted(out)


# ------------------------------------------------------------------------------------- floats

_F64_SUBNORMAL = 5e-324
_F32_SUBNORMAL = float.fromhex("0x1p-149")

FLOAT_GROUPS: Mapping[str, Sequence[float]] = {
    "zero": [0.0, -0.0],
    "special": [math.inf, -math.inf, math.nan],
    "extreme": [1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308],
    "subnormal": [_F64_SUBNORMAL, -_F64_SUBNORMAL],
    # 0.1 is not representable in f32; 0.5 and 3.0 are. A path that round-trips through f32
    # keeps the second pair bit-identical and perturbs the first, which is the tell.
    "precision": [0.1, 0.2, 1e308, 1e-308, 0.5, 3.0, 16777217.0],
    "ordinary": [1.0, -1.0, 42.0, -273.15, 3.141592653589793],
}

FLOAT32_GROUPS: Mapping[str, Sequence[float]] = {
    "zero": [0.0, -0.0],
    "special": [math.inf, -math.inf, math.nan],
    "extreme": [3.4028234663852886e38, -3.4028234663852886e38, 1.1754943508222875e-38],
    "subnormal": [_F32_SUBNORMAL, -_F32_SUBNORMAL],
    "precision": [0.5, 3.0, 16777216.0, 16777217.0],
    "ordinary": [1.0, -1.0, 42.0, 3.1415927410125732],
}

# ------------------------------------------------------------------------------------ strings

#: Values a naive writer or a naive reader mistakes for something other than a string.
COERCION_STRINGS: Sequence[str] = [
    "",
    " ",
    "  leading and trailing  ",
    "NULL",
    "null",
    "None",
    "nil",
    "NaN",
    "inf",
    "-inf",
    "true",
    "False",
    "0",
    "1",
    "-1",
    "1e10",
    "0x0",
    "1/2",
    "2020-01-01",
    "2020-01-01T00:00:00Z",
    "00:00:00",
    "1,000",
    "$1.00",
    "0.1",
    "007",
]

#: Anything that has to survive being spliced into a SQL filter string, a path, or a JSON blob.
#: For a plugin that translates predicates to SQL this is the highest-value group in the file.
ESCAPING_STRINGS: Sequence[str] = [
    "'",
    "''",
    '"',
    '""',
    "\\",
    "\\\\",
    "\\'",
    "`",
    "%",
    "_",
    "%_%",
    "';DROP TABLE t;--",
    "' OR '1'='1",
    "--",
    "/*",
    "*/",
    "{}",
    "[]",
    "()",
    "\x00",
    "a\x00b",
    "\n",
    "\r\n",
    "\t",
    "\n\r\t",
    "a b",
    "a b",
    "\x1b[31m",
    "${HOME}",
    "$(whoami)",
    "../../etc/passwd",
    "﻿bom",
]

#: NFC and NFD forms of the same text are distinct strings. A round-trip that normalises them
#: has silently changed the user's data, and byte-equality is the only way to notice.
UNICODE_STRINGS: Sequence[str] = [
    "é",  # e-acute, NFC
    "é",  # e-acute, NFD -- must stay distinct from the entry above
    "ẛ̣",  # NFC/NFD/NFKC all differ
    "\U0001f469‍\U0001f4bb",  # ZWJ sequence: one grapheme, several code points
    "\U0001f1e6\U0001f1f6",  # regional indicators
    "‮abc",  # RTL override
    "​",  # zero-width space
    "\ufffd",  # the replacement character itself, so a lossy decode is not silent
    "a\u0301",  # combining acute after a plain a: NFC would fold it to the entry above
    "日本語",
    "עברית",
    "क्ष",
    "ß",
    "SS",
    "ﬀ",  # NFKC-folds to "ff"
    "İ",  # uppercases/lowercases asymmetrically
    "",
    "🯰",
]

#: Values that only differ by case or by whitespace, so a path that folds either collides them.
COLLISION_STRINGS: Sequence[str] = ["a", "A", "a ", " a", "a\n", "ss", "ß"]

STRING_GROUPS: Mapping[str, Sequence[str]] = {
    "coercion": COERCION_STRINGS,
    "escaping": ESCAPING_STRINGS,
    "unicode": UNICODE_STRINGS,
    "collision": COLLISION_STRINGS,
    "ordinary": ["alpha", "beta", "gamma", "delta", "epsilon"],
}

#: Not in `STRING_GROUPS` on purpose. At the default 1000 rows a 1 MiB entry drawn 60% of the
#: time materialises hundreds of megabytes per case. Reachable only from a `slow`-tagged case
#: with a tiny row count.
HUGE_STRINGS: Sequence[str] = ["x" * (1 << 20), "é" * (1 << 18), ""]

# ------------------------------------------------------------------------------------- binary

BINARY_GROUPS: Mapping[str, Sequence[bytes]] = {
    "empty": [b"", b"\x00", b"\x00\x00"],
    # Sequences no UTF-8 decoder accepts. A path that round-trips binary through a string type
    # either raises here or replaces bytes with U+FFFD; both are data loss and both are caught.
    "invalid_utf8": [
        b"\xff",
        b"\xfe\xff",
        b"\xc0\x80",  # overlong NUL
        b"\xed\xa0\x80",  # lone surrogate, WTF-8
        b"\xf4\x90\x80\x80",  # beyond U+10FFFF
        b"\x80",  # bare continuation
    ],
    "ordinary": [b"\x01\x02\x03", bytes(range(256)), b"payload"],
}

# ----------------------------------------------------------------------------------- temporal

DATE_VALUES: Sequence[date] = [
    date(1970, 1, 1),
    date(1969, 12, 31),
    date(1, 1, 1),
    date(9999, 12, 31),
    date(2000, 2, 29),
    date(2020, 2, 29),
    date(1900, 3, 1),  # 1900 is not a leap year, unlike 2000
    date(2262, 4, 11),  # last date representable as ns since epoch
    date(1677, 9, 22),
    date(2024, 12, 31),
]

DATETIME_VALUES: Sequence[datetime] = [
    datetime(1970, 1, 1, 0, 0, 0),
    datetime(1969, 12, 31, 23, 59, 59, 999999),
    datetime(1, 1, 1, 0, 0, 0),
    datetime(9999, 12, 31, 23, 59, 59, 999999),
    datetime(1970, 1, 1, 0, 0, 0, 1),
    datetime(2262, 4, 11, 23, 47, 16),  # i64 nanoseconds overflows just after this
    datetime(1677, 9, 21, 0, 12, 44),
    datetime(2024, 2, 29, 12, 0, 0),
    datetime(2000, 1, 1, 0, 0, 0),
]

#: Local times that are ambiguous or do not exist. `America/New_York` and `Europe/London` both
#: observe DST; a fixed-offset zone such as `Asia/Seoul` does not and proves nothing here.
DST_ZONES: Sequence[str] = ["America/New_York", "Europe/London"]

#: UTC instants chosen so that, in `America/New_York`, the first pair is the same local time
#: twice over (01:30 on the fall-back day) and the third is the instant a wall clock skips.
DST_INSTANTS_UTC: Sequence[datetime] = [
    datetime(2020, 11, 1, 5, 30),  # 01:30 EDT -- first occurrence
    datetime(2020, 11, 1, 6, 30),  # 01:30 EST -- second occurrence, same local time
    datetime(2020, 3, 8, 7, 0),  # 03:00 EDT; 02:30 local never happens
    datetime(2020, 3, 8, 6, 59, 59, 999999),
]

TIME_VALUES: Sequence[time] = [
    time(0, 0, 0),
    time(23, 59, 59, 999999),
    time(12, 0, 0),
    time(0, 0, 0, 1),
    time(1, 2, 3, 4),
]


#: Durations are given as their physical integer, not as `timedelta`. A `timedelta` carries no
#: time unit, so the same table would overflow at `ns` and be silently rounded at `ms`; and the
#: largest `timedelta` is not representable at either.
def duration_values(time_unit: str) -> list[int]:
    per_second = {"ms": 10**3, "us": 10**6, "ns": 10**9}[time_unit]
    extreme = (2**63 - 1) // per_second * per_second
    return [
        0,
        1,
        -1,
        per_second,
        -per_second,
        86_400 * per_second,
        -86_400 * per_second,
        extreme,
        -extreme,
        60 * per_second - 1,
    ]


# ------------------------------------------------------------------------------------ decimal


def decimal_values(precision: int, scale: int) -> list[Decimal]:
    """Zero, the extremes at this precision, and values that need a rescale to store.

    Evaluated in a local context wide enough for the requested precision: the default decimal
    context is 28 digits and silently refuses to `quantize` past it.
    """
    if not 1 <= precision <= 38 or not 0 <= scale <= precision:
        raise ValueError(f"unsupported decimal({precision}, {scale})")
    with localcontext() as ctx:
        ctx.prec = precision + scale + 8
        unit = Decimal(1).scaleb(-scale)
        biggest = Decimal((10**precision) - 1).scaleb(-scale)
        out = [
            Decimal(0).quantize(unit),
            unit,
            -unit,
            biggest,
            -biggest,
            biggest - unit,
        ]
        if precision > scale:
            out += [
                Decimal(1).quantize(unit),
                Decimal(-1).quantize(unit),
                Decimal(10 ** (precision - scale) - 1).quantize(unit),
            ]
        return out


# --------------------------------------------------------------------------------------- api


def series(values: Iterable[Any], dtype: pl.DataType, name: str = "palette") -> pl.Series:
    return pl.Series(name, list(values), dtype=dtype)


def grouped(groups: Mapping[str, Sequence[Any]], tags: frozenset[str] | None = None) -> list[Any]:
    """Flatten selected tag groups, preserving declaration order for stable digests."""
    if tags is None:
        return [v for vs in groups.values() for v in vs]
    unknown = tags - set(groups)
    if unknown:
        raise ValueError(f"unknown palette tags: {sorted(unknown)}")
    return [v for tag, vs in groups.items() if tag in tags for v in vs]
