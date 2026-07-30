"""Specs to text and back.

A Hypothesis counterexample is only worth anything if it survives the process that found it.
This encodes a `CaseSpec` or a `QuerySpec` into a nested mapping of plain scalars, which is
what makes a shrunk failure a permanent ten-line regression file rather than a data file.

JSON rather than the TOML the plan asked for: a predicate is a recursive tree with nulls in it,
which TOML expresses badly and has no writer for in the standard library. The property that
mattered -- a committed, diffable, hand-editable spec instead of committed bytes -- is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import polars as pl

from plioc import query as Q
from plioc.gen import categorical as C
from plioc.gen import nested as N
from plioc.gen import primitive as P
from plioc.gen import temporal as T
from plioc.gen.core import NullPattern
from plioc.gen.layout import Layout
from plioc.spec import CaseSpec, ColumnSpec

_CLASSES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        P.IntGen,
        P.FloatGen,
        P.BoolGen,
        P.StringGen,
        P.BinaryGen,
        P.DecimalGen,
        P.NullGen,
        P.ConstGen,
        T.DateGen,
        T.DatetimeGen,
        T.TimeGen,
        T.DurationGen,
        C.EnumGen,
        C.CategoricalGen,
        C.EmptyEnumGen,
        N.ListGen,
        N.ArrayGen,
        N.StructGen,
        N.StructField,
        Q.Cmp,
        Q.IsNull,
        Q.IsIn,
        Q.Between,
        Q.StrMatch,
        Q.ListContains,
        Q.Field,
        Q.And,
        Q.Or,
        Q.Not,
        Q.Opaque,
        Q.Udf,
        Q.Always,
        Q.QuerySpec,
        ColumnSpec,
        CaseSpec,
        Layout,
    )
}

#: Only dtypes a generator can actually carry in a field. Encoding these by name keeps the
#: format readable and keeps a Polars dtype repr change from silently breaking old regressions.
_DTYPES: dict[str, pl.DataType] = {
    str(dt): dt
    for dt in (
        pl.Int8(),
        pl.Int16(),
        pl.Int32(),
        pl.Int64(),
        pl.Int128(),
        pl.UInt8(),
        pl.UInt16(),
        pl.UInt32(),
        pl.UInt64(),
        pl.Float32(),
        pl.Float64(),
        pl.Boolean(),
        pl.String(),
        pl.Binary(),
        pl.Null(),
        pl.Date(),
        pl.Time(),
    )
}


def encode(value: Any) -> Any:
    if isinstance(value, NullPattern):
        return {"@nulls": value.value}
    if isinstance(value, pl.DataType):
        name = str(value)
        if name not in _DTYPES:
            raise TypeError(f"dtype {name} is not encodable in a spec field")
        return {"@dtype": name}
    if is_dataclass(value) and not isinstance(value, type):
        out: dict[str, Any] = {"@": type(value).__name__}
        if out["@"] not in _CLASSES:
            raise TypeError(f"{out['@']} is not in the codec registry")
        for f in fields(value):
            out[f.name] = encode(getattr(value, f.name))
        return out
    if isinstance(value, frozenset):
        return {"@set": sorted(encode(v) for v in value)}
    if isinstance(value, tuple):
        return {"@tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, bytes):
        return {"@bytes": value.hex()}
    # Temporal literals appear in predicates. ISO 8601 keeps the file readable, and `fromisoformat`
    # is exact for what `isoformat` writes -- no format string to get subtly wrong.
    if isinstance(value, datetime):
        return {"@datetime": value.isoformat()}
    if isinstance(value, date):
        return {"@date": value.isoformat()}
    if isinstance(value, time):
        return {"@time": value.isoformat()}
    if isinstance(value, timedelta):
        return {"@timedelta": [value.days, value.seconds, value.microseconds]}
    if isinstance(value, Decimal):
        return {"@decimal": str(value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__}")


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "@nulls" in value:
        return NullPattern(value["@nulls"])
    if "@dtype" in value:
        return _DTYPES[value["@dtype"]]
    if "@set" in value:
        return frozenset(decode(v) for v in value["@set"])
    if "@tuple" in value:
        return tuple(decode(v) for v in value["@tuple"])
    if "@bytes" in value:
        return bytes.fromhex(value["@bytes"])
    if "@datetime" in value:
        return datetime.fromisoformat(value["@datetime"])
    if "@date" in value:
        return date.fromisoformat(value["@date"])
    if "@time" in value:
        return time.fromisoformat(value["@time"])
    if "@timedelta" in value:
        days, seconds, microseconds = value["@timedelta"]
        return timedelta(days=days, seconds=seconds, microseconds=microseconds)
    if "@decimal" in value:
        return Decimal(value["@decimal"])
    if "@" in value:
        cls = _CLASSES[value["@"]]
        kwargs = {k: decode(v) for k, v in value.items() if k != "@"}
        return cls(**kwargs)
    return {k: decode(v) for k, v in value.items()}


def dumps(obj: Any) -> str:
    return json.dumps(encode(obj), indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def loads(text: str) -> Any:
    return decode(json.loads(text))


def save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj), encoding="utf-8")


def load(path: Path) -> Any:
    return loads(path.read_text(encoding="utf-8"))
