"""Running the suite outside pytest, and recording what happened.

The pytest surface reports pass/fail per test; this one produces a value -- a `SuiteRun` -- that
can be rendered, diffed between commits, or checked into a build log. `html.py` renders it.

The semantics are deliberately identical to `suite.py`'s: same skips, same strict-xfail
treatment, same engagement degradation. `selection.py` holds what both consult so they cannot
drift apart.
"""

from __future__ import annotations

import platform
import sys
import time
from functools import partial
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

import polars as pl

from plioc import codec, oracle, selection
from plioc.corpus import all_cases, query_fixture
from plioc.harness import Capabilities, IOHarness, is_failure
from plioc.queries import all_queries
from plioc.query import Cmp, QuerySpec
from plioc.spec import CaseSpec


class Status(Enum):
    PASS = "pass"
    XFAIL = "xfail"
    XPASS = "xpass"
    SKIP = "skip"
    FAIL = "fail"

    @property
    def is_bad(self) -> bool:
        """`XPASS` counts as bad: a declared failure that now passes means the declaration is
        stale, and letting it slide is how a capability matrix rots into pessimism."""
        return self in (Status.FAIL, Status.XPASS)


#: The contracts, in report order, each with the question it answers.
CONTRACTS: dict[str, str] = {
    "roundtrip": "Does everything written come back, at the declared strictness?",
    "fixpoint": "Does a declared loss happen once, or on every write?",
    "schema": "Does the schema read back match the data, and does planning read none of it?",
    "query": "Does every query match in-memory Polars? (this is where the mandate is enforced)",
    "engagement": "Did the harness use the projection, predicate and limit it was pushed?",
    "law": "Do the metamorphic laws hold?",
}


@dataclass(frozen=True)
class Check:
    contract: str
    subject: str
    status: Status
    detail: str = ""
    tags: frozenset[str] = frozenset()
    duration_ms: float = 0.0

    @property
    def group(self) -> str:
        """The leading path segment of the subject, which is how the corpus is organised."""
        return self.subject.split("/", 1)[0] if "/" in self.subject else self.subject


@dataclass
class HarnessRun:
    name: str
    capabilities: Capabilities
    checks: list[Check] = field(default_factory=list)
    #: Populated for failing cases: the spec that reproduces it, as JSON.
    repros: dict[str, str] = field(default_factory=dict)

    def by_status(self, status: Status) -> list[Check]:
        return [c for c in self.checks if c.status is status]

    def count(self, status: Status) -> int:
        return sum(1 for c in self.checks if c.status is status)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status.is_bad]

    @property
    def verdict(self) -> str:
        return "fail" if self.failures else "pass"


@dataclass
class SuiteRun:
    runs: list[HarnessRun] = field(default_factory=list)
    started: str = ""
    duration_s: float = 0.0
    polars_version: str = ""
    python_version: str = ""
    platform: str = ""
    case_count: int = 0
    query_count: int = 0
    row_count: int = 0
    include_slow: bool = False

    @property
    def failed(self) -> bool:
        return any(r.failures for r in self.runs)


# -- the engine -------------------------------------------------------------------------------


def _timed(fn: Callable[[], None]) -> tuple[Status, str, float]:
    """Run one assertion and classify the outcome.

    `BaseException`, not `Exception`: a Rust panic surfaced through PyO3 is a
    `pyo3_runtime.PanicException` and derives from `BaseException` specifically so that
    `except Exception` does not swallow it.
    """
    start = time.perf_counter()
    try:
        fn()
    except BaseException as exc:
        if not is_failure(exc):
            raise
        return Status.FAIL, _one_line(exc), (time.perf_counter() - start) * 1000
    return Status.PASS, "", (time.perf_counter() - start) * 1000


def _one_line(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ")
    return text[:400]


def _classify(run: HarnessRun, subject: str, status: Status, detail: str) -> tuple[Status, str]:
    """Fold a raw outcome against the harness's declaration.

    A declared known failure that fails becomes `XFAIL`; one that passes becomes `XPASS`, which
    counts as a failure. That asymmetry is the whole mechanism keeping a declaration honest.
    """
    reason = run.capabilities.known_failures.get(subject)
    if reason is None:
        return status, detail
    if status is Status.FAIL:
        return Status.XFAIL, f"declared: {reason} ({detail})"
    return (
        Status.XPASS,
        f"declared a known failure ({reason}) but passes -- update the declaration",
    )


def _record(
    run: HarnessRun,
    contract: str,
    subject: str,
    tags: frozenset[str],
    fn: Callable[[], None],
) -> Check:
    status, detail, ms = _timed(fn)
    status, detail = _classify(run, subject, status, detail)
    check = Check(contract, subject, status, detail, tags, ms)
    run.checks.append(check)
    return check


def _skip(run: HarnessRun, contract: str, subject: str, tags: frozenset[str], detail: str) -> None:
    run.checks.append(Check(contract, subject, Status.SKIP, detail, tags))


def run_harness(
    harness: IOHarness,
    *,
    cases: Sequence[CaseSpec] | None = None,
    queries: Sequence[QuerySpec] | None = None,
    include_slow: bool = False,
) -> HarnessRun:
    """Drive one harness through every contract and record the result of each."""
    caps = harness.capabilities()
    run = HarnessRun(name=harness.name, capabilities=caps)

    corpus = list(cases if cases is not None else all_cases(include_slow=include_slow).values())
    matrix = list(queries if queries is not None else all_queries(include_slow=include_slow))
    fixture = selection.narrow(caps, query_fixture())

    for case in corpus:
        tags = case.all_tags
        if not include_slow and "slow" in tags:
            _skip(run, "roundtrip", case.id, tags, "slow; not selected")
            _skip(run, "fixpoint", case.id, tags, "slow; not selected")
            continue
        missing = selection.unsupported(caps, case)
        if missing:
            detail = f"dtype not claimed: {', '.join(sorted(set(missing)))}"
            _skip(run, "roundtrip", case.id, tags, detail)
            _skip(run, "fixpoint", case.id, tags, detail)
            continue
        check = _record(
            run, "roundtrip", case.id, tags, partial(oracle.check_roundtrip, harness, case)
        )
        if check.status.is_bad:
            run.repros.setdefault(case.id, codec.dumps(case))
        _record(run, "fixpoint", case.id, tags, partial(oracle.check_fixpoint, harness, case))

    if fixture.columns:
        _run_schema_checks(harness, run, fixture)
        _run_queries(harness, run, fixture, matrix, include_slow=include_slow)
        _run_laws(harness, run, fixture)
    else:
        _skip(run, "schema", "query/fixture", frozenset(), "no claimed dtype in the fixture")

    return run


def _run_schema_checks(harness: IOHarness, run: HarnessRun, fixture: CaseSpec) -> None:
    _record(
        run,
        "schema",
        "planning/reads-no-data",
        frozenset({"planning"}),
        lambda: oracle.check_planning_reads_nothing(harness, fixture),
    )

    def schema_matches() -> None:
        target = oracle.materialise(harness, fixture)
        observed = dict(harness.scan(target).collect_schema())
        expected = dict(oracle.oracle_frame(harness, fixture).schema)
        if observed != expected:
            differing = {
                k for k in set(observed) | set(expected) if observed.get(k) != expected.get(k)
            }
            raise AssertionError(
                "schema differs on "
                + ", ".join(
                    f"{k}: {observed.get(k)} != {expected.get(k)}" for k in sorted(differing)
                )
            )

    _record(run, "schema", "schema/matches-data", frozenset({"schema"}), schema_matches)


def _run_queries(
    harness: IOHarness,
    run: HarnessRun,
    fixture: CaseSpec,
    matrix: Sequence[QuerySpec],
    *,
    include_slow: bool,
) -> None:
    caps = harness.capabilities()
    target = oracle.materialise(harness, fixture, "/queries")
    engagement_possible = harness.probe() is not None and bool(caps.pushdown)

    for query in matrix:
        tags = query.tags
        reason = selection.query_skip_reason(query, fixture, include_slow=include_slow)
        if reason is not None:
            _skip(run, "query", query.id, tags, reason)
            _skip(run, "engagement", query.id, tags, reason)
            continue

        holder: dict[str, oracle.Run] = {}

        def correctness(q: QuerySpec = query) -> None:
            holder["run"] = oracle.check_query_correctness(harness, fixture, q, target)

        check = _record(run, "query", query.id, tags, correctness)
        if check.status.is_bad:
            run.repros.setdefault(query.id, codec.dumps(query))

        if not engagement_possible:
            _skip(
                run,
                "engagement",
                query.id,
                tags,
                "harness reports no probe"
                if harness.probe() is None
                else "harness claims no pushdown",
            )
            continue
        if "run" not in holder:
            _skip(run, "engagement", query.id, tags, "the query did not complete")
            continue

        # Not routed through `_record`: `EngagementSkipped` is a skip, and `_record` would
        # classify any exception as a failure.
        start = time.perf_counter()
        try:
            oracle.check_engagement(harness, fixture, query, holder["run"])
        except oracle.EngagementSkipped as exc:
            _skip(run, "engagement", query.id, tags, str(exc))
            continue
        except BaseException as exc:
            if not is_failure(exc):
                raise
            status, detail = Status.FAIL, _one_line(exc)
        else:
            status, detail = Status.PASS, ""
        status, detail = _classify(run, query.id, status, detail)
        run.checks.append(
            Check(
                "engagement",
                query.id,
                status,
                detail,
                tags,
                (time.perf_counter() - start) * 1000,
            )
        )


def _run_laws(harness: IOHarness, run: HarnessRun, fixture: CaseSpec) -> None:
    schema = list(fixture.schema())
    numeric = next(
        (
            name
            for name, dtype in fixture.schema().items()
            if dtype in (pl.Int64, pl.Int32, pl.Float64) and name != "__i"
        ),
        None,
    )
    if numeric is None:
        _skip(run, "law", "laws", frozenset({"law"}), "no numeric column in the fixture")
        return
    target = oracle.materialise(harness, fixture, "/laws")
    predicate = Cmp(numeric, "gt", 0)
    other = Cmp(numeric, "lt", 1000)
    tags = frozenset({"law"})

    _record(
        run,
        "law",
        "law/partition",
        tags,
        lambda: oracle.law_partition(harness, fixture, predicate, target),
    )
    _record(
        run,
        "law",
        "law/filter-conjunction",
        tags,
        lambda: oracle.law_filter_conjunction(harness, fixture, predicate, other, target),
    )
    for a, b in ((0, 5), (5, 0), (3, 9), (10_000, 7)):
        _record(
            run,
            "law",
            f"law/head-composition/{a}-{b}",
            tags,
            partial(oracle.law_head_composition, harness, fixture, a, b, target),
        )
    subset = tuple(n for n in schema[:2])
    _record(
        run,
        "law",
        "law/projection-composition",
        tags,
        lambda: oracle.law_projection_composition(harness, fixture, subset, target),
    )
    _record(
        run,
        "law",
        "law/filter-projection-commute",
        tags,
        lambda: oracle.law_filter_projection_commute(harness, fixture, predicate, numeric, target),
    )


def run_suite(
    harnesses: Sequence[IOHarness],
    *,
    include_slow: bool = False,
    cases: Sequence[CaseSpec] | None = None,
    queries: Sequence[QuerySpec] | None = None,
) -> SuiteRun:
    corpus = list(cases if cases is not None else all_cases(include_slow=include_slow).values())
    matrix = list(queries if queries is not None else all_queries(include_slow=include_slow))
    started = time.perf_counter()
    out = SuiteRun(
        started=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        polars_version=pl.__version__,
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()} (CPython {sys.version_info.major}.{sys.version_info.minor})",
        case_count=len(corpus),
        query_count=len(matrix),
        row_count=sum(c.n_rows for c in corpus),
        include_slow=include_slow,
    )
    for harness in harnesses:
        out.runs.append(
            run_harness(harness, cases=corpus, queries=matrix, include_slow=include_slow)
        )
    out.duration_s = time.perf_counter() - started
    return out
