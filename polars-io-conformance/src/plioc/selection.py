"""Which cases and queries apply to a given harness.

Shared by the pytest surface (`suite.py`) and the reporting surface (`run.py`) on purpose: the
two must agree about what counts as a skip, or the HTML report and the test run tell different
stories about the same plugin.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import Any

import polars as pl

from plioc.harness import Capabilities
from plioc.query import QuerySpec
from plioc.spec import CaseSpec


def flatten_dtype(dtype: Any) -> list[pl.DataType]:
    """A dtype and everything nested inside it.

    Typed loosely because `Schema.items()` is annotated as yielding `DataType | DataTypeClass`,
    and a nested `inner`/`field.dtype` is the same union again.
    """
    out: list[pl.DataType] = [dtype]
    if isinstance(dtype, (pl.List, pl.Array)):
        out += flatten_dtype(dtype.inner)
    elif isinstance(dtype, pl.Struct):
        for f in dtype.fields:
            out += flatten_dtype(f.dtype)
    return out


def unsupported(caps: Capabilities, case: CaseSpec) -> list[str]:
    """Dtypes in this case the harness does not claim, as strings, for a skip message."""
    missing = []
    for name, dtype in case.schema().items():
        for part in flatten_dtype(dtype):
            if not caps.supports(part):
                missing.append(f"{part} ({name!r})")
    return missing


def narrow(caps: Capabilities, case: CaseSpec) -> CaseSpec:
    """Drop the columns whose dtype the harness does not claim.

    Used for the query fixture only. Narrowed rather than skipped because one unsupported dtype in
    a ten-column fixture would otherwise cost the entire query matrix, which is the part of the
    suite a plugin most needs to see.
    """
    keep = tuple(c for c in case.columns if all(caps.supports(p) for p in flatten_dtype(c.dtype)))
    return case if len(keep) == len(case.columns) else replace(case, columns=keep)


def missing_columns(query: QuerySpec, case: CaseSpec) -> list[str]:
    """Columns the query needs that the (possibly narrowed) case does not have."""
    referenced = set(query.projection or ()) | set(
        query.predicate.columns() if query.predicate is not None else ()
    )
    return sorted(referenced - set(case.schema()))


def query_skip_reason(query: QuerySpec, case: CaseSpec, *, include_slow: bool) -> str | None:
    """Why this query cannot be a verdict about the harness, or `None` if it can.

    Consulted by both surfaces so that the HTML report and the pytest run skip the same things.
    The last check is the expensive one and is deliberately last: it runs the query against the
    in-memory oracle, and a query Polars itself rejects has no right answer for a harness to
    produce -- counting it either way would be a claim about Polars, not about the plugin.
    """
    if not include_slow and "slow" in query.tags:
        return "slow; not selected"
    if "udf" in query.tags and importlib.util.find_spec("cloudpickle") is None:
        # Polars serialises a UDF with cloudpickle to hand it to an IO plugin. A missing optional
        # dependency is not a conformance result in either direction.
        return "cloudpickle is not installed, so a UDF cannot reach an IO plugin"
    missing = missing_columns(query, case)
    if missing:
        return f"the fixture has no {missing}: dtype not claimed"
    from plioc import oracle

    if not oracle.is_runnable(query, case):
        return "Polars itself rejects this query, so there is no right answer to check"
    return None
