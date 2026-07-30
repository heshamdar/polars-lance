"""What a plugin author implements, and what the suite is allowed to assume.

Three required methods. Everything else the suite needs it derives.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import polars as pl

from plioc.equality import Strictness

#: Whatever the harness needs to name a written dataset: a path, a URI, a key in a dict. The
#: suite only ever passes back what `IOHarness.target` handed it.
Target = Any

Pushdown = Literal["projection", "predicate", "limit"]


@dataclass(frozen=True)
class ScanCall:
    """One invocation of the plugin's scan callback, as Polars made it."""

    columns: tuple[str, ...] | None
    predicate: str | None
    n_rows: int | None
    batch_size: int | None
    rows_produced: int
    #: Rows read from storage, when the harness can tell them apart from the rows it returned.
    rows_read: int | None = None


@dataclass
class Probe:
    """What the harness saw during one query.

    Scoped, not cumulative: `IOHarness.probing()` clears the record and yields it, so an
    assertion about "this query" means something. A single set of counters shared across queries
    silently stops discriminating after the first one.

    `probe()` returning `None` is allowed and is not a failure -- but it degrades the pushdown
    suite to correctness only, and correctness is exactly what a plugin that ignores every
    predicate and filters in memory already has. That configuration reports a loud skip.
    """

    calls: list[ScanCall] = field(default_factory=list)

    @property
    def was_called(self) -> bool:
        return bool(self.calls)

    @property
    def rows_scanned(self) -> int:
        """Rows the harness pulled out of storage.

        Equal to the rows it yielded only for a harness that applies the predicate at the source.
        One that reads everything and filters in Python is correct but not engaged, and should
        report the larger number here -- otherwise the engagement assertions measure nothing.
        """
        return sum(c.rows_produced if c.rows_read is None else c.rows_read for c in self.calls)

    @property
    def columns_read(self) -> frozenset[str] | None:
        """Columns the harness was asked for, or `None` if any call asked for all of them."""
        if any(c.columns is None for c in self.calls):
            return None
        return frozenset(col for c in self.calls if c.columns for col in c.columns)

    @property
    def predicates_received(self) -> list[str]:
        return [c.predicate for c in self.calls if c.predicate is not None]

    @property
    def limit_received(self) -> int | None:
        limits = [c.n_rows for c in self.calls if c.n_rows is not None]
        return min(limits) if limits else None


def identity(df: pl.DataFrame) -> pl.DataFrame:
    return df


#: Exceptions that mean "stop", not "this harness failed". Everything else counts as a failure,
#: including `BaseException` subclasses -- a Rust panic surfaced through PyO3 is a
#: `pyo3_runtime.PanicException`, which derives from `BaseException` specifically so that
#: `except Exception` does not swallow it. A suite that only catches `Exception` therefore reports
#: a panicking harness as an infrastructure error rather than as the declared failure it is.
CONTROL_FLOW: tuple[type[BaseException], ...] = (KeyboardInterrupt, SystemExit, GeneratorExit)


def is_failure(exc: BaseException) -> bool:
    """Whether an exception should be read as "the harness got this wrong"."""
    if isinstance(exc, CONTROL_FLOW):
        return False
    # pytest's own outcomes (Skipped, Failed, Exit) are control flow too, and they are
    # `BaseException` subclasses carrying this attribute.
    return not getattr(exc, "__module__", "").startswith("_pytest")


@dataclass(frozen=True)
class Capabilities:
    """The harness's declaration of what it does and does not preserve.

    Kept honest by `known_failures` becoming a *strict* xfail: a loss that gets fixed upstream
    breaks CI until the declaration is updated. Without that, a capability matrix decays into
    pessimism within a release or two and stops being worth reading.
    """

    strictness: Strictness = Strictness.DTYPES
    #: Dtypes the harness claims to support. Cases using anything outside this set are skipped
    #: rather than failed -- an unsupported type is a documented limit, not a defect.
    dtypes: frozenset[type[pl.DataType]] | None = None
    pushdown: frozenset[Pushdown] = frozenset()
    preserves_row_order: bool = True
    #: Applied to the *expected* frame before comparing, to describe a loss rather than chase
    #: identity. Must be idempotent; the suite asserts that.
    normalize: Callable[[pl.DataFrame], pl.DataFrame] = identity
    #: case id -> reason. Becomes `xfail(strict=True)`.
    known_failures: Mapping[str, str] = field(default_factory=dict)
    #: case id -> the strictness at which it is *known to stop* matching. Opt-in, per case: a
    #: blanket "the level above the declared one must fail" rule would force a harness with one
    #: lossy dtype to be wrong about every case it actually gets right.
    exact_at: Mapping[str, Strictness] = field(default_factory=dict)

    def supports(self, dtype: pl.DataType) -> bool:
        if self.dtypes is None:
            return True
        return type(dtype) in self.dtypes


@runtime_checkable
class IOHarness(Protocol):
    name: str

    def target(self, name: str) -> Target: ...
    def sink(self, lf: pl.LazyFrame, target: Target) -> None: ...
    def scan(self, target: Target) -> pl.LazyFrame: ...
    def capabilities(self) -> Capabilities: ...

    def probe(self) -> Probe | None: ...
    def probing(self) -> Any: ...


class BaseHarness:
    """Optional convenience base. Gives a working (empty) probe implementation so a harness that
    cannot observe its own scans still satisfies the protocol."""

    name = "unnamed"

    def __init__(self) -> None:
        self._probe: Probe | None = None

    def probe(self) -> Probe | None:
        return self._probe

    @contextmanager
    def probing(self) -> Iterator[Probe | None]:
        """Scope a probe to one query. Yields the record, already cleared."""
        if self._probe is None:
            yield None
            return
        self._probe.calls.clear()
        yield self._probe

    def record(self, call: ScanCall) -> None:
        if self._probe is not None:
            self._probe.calls.append(call)

    def capabilities(self) -> Capabilities:  # pragma: no cover - overridden
        raise NotImplementedError
