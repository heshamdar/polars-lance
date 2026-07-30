"""The runners: round-trip, fixpoint, differential query, metamorphic law.

Correctness and engagement are separate assertions and neither ever stands in for the other. A
plugin that ignores every predicate and filters the rows in Python is perfectly correct and
delivers none of the value; a plugin that pushes a predicate it translated wrongly is engaged
and returns the wrong rows. Only asking both questions distinguishes the four outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from plioc.equality import Strictness, compare
from plioc.harness import IOHarness, Probe
from plioc.query import Always, And, Cmp, Pred, QuerySpec
from plioc.spec import CaseSpec


class EngagementSkipped(Exception):
    """Raised when a harness cannot be asked what it received.

    Not an error and not a pass: the pushdown suite has been degraded to correctness only, and
    the caller is expected to turn this into a visible skip rather than swallow it.
    """


@dataclass(frozen=True)
class Run:
    observed: pl.DataFrame
    expected: pl.DataFrame
    probe: Probe | None


#: Building a case is pure, so the same spec never needs building twice in one session. Keyed by
#: the spec itself, which is why every part of a spec is a frozen dataclass.
_BUILT: dict[CaseSpec, pl.DataFrame] = {}


def built(case: CaseSpec) -> pl.DataFrame:
    """The case as Polars holds it, laid out as the spec asks."""
    if case not in _BUILT:
        _BUILT[case] = case.layout.apply(case.build().collect())
    return _BUILT[case]


def materialise(harness: IOHarness, case: CaseSpec, suffix: str = "") -> object:
    """Write the case through the harness and return the target."""
    target = harness.target(case.id + suffix)
    harness.sink(built(case).lazy(), target)
    return target


def oracle_frame(harness: IOHarness, case: CaseSpec) -> pl.DataFrame:
    """Ground truth: the case as Polars itself holds it, put through the harness's declared
    normalisation so that a declared loss is described rather than chased."""
    return harness.capabilities().normalize(built(case))


def run_query(harness: IOHarness, case: CaseSpec, query: QuerySpec, target: object) -> Run:
    expected = query.apply(oracle_frame(harness, case).lazy()).collect()
    with harness.probing() as probe:
        observed = query.apply(harness.scan(target)).collect()
    return Run(observed=observed, expected=expected, probe=probe)


# -- contracts ------------------------------------------------------------------------------


def check_roundtrip(harness: IOHarness, case: CaseSpec) -> None:
    """`scan(sink(df)) == normalize(df)` at the harness's declared strictness."""
    target = materialise(harness, case)
    observed = harness.scan(target).collect()
    strictness = _effective_strictness(harness, case)
    compare(observed, oracle_frame(harness, case), strictness)


def check_fixpoint(harness: IOHarness, case: CaseSpec) -> None:
    """`rt(rt(df)) == rt(df)`, and `normalize(normalize(df)) == normalize(df)`.

    Idempotence is a real contract even where identity is impossible, and it catches the drift
    that normalisation hides: a timezone that shifts by an hour on every write looks like a
    legitimate normalisation each time round and is only visible by going round twice.
    """
    caps = harness.capabilities()
    once = caps.normalize(built(case))
    twice = caps.normalize(once)
    compare(twice, once, Strictness.METADATA)

    first_target = materialise(harness, case)
    first = harness.scan(first_target).collect()

    second_target = harness.target(case.id + "/rt2")
    harness.sink(first.lazy(), second_target)
    second = harness.scan(second_target).collect()

    compare(second, first, _effective_strictness(harness, case))


def check_query_correctness(
    harness: IOHarness, case: CaseSpec, query: QuerySpec, target: object | None = None
) -> Run:
    """The differential assertion. In-memory Polars is the oracle."""
    if target is None:
        target = materialise(harness, case)
    run = run_query(harness, case, query, target)
    strictness = _effective_strictness(harness, case)
    if query.count_only:
        strictness = max(strictness, Strictness.ROW_ORDER)
    elif strictness < Strictness.ROW_ORDER and "__i" not in run.expected.columns:
        # Nothing to sort by once the identity column has been projected away, so the comparison
        # has to accept the harness's order. Only sound because the query is applied identically
        # to both sides and the oracle side is in Polars' own order.
        strictness = max(strictness, Strictness.ROW_ORDER)
    compare(run.observed, run.expected, strictness)
    return run


def check_engagement(harness: IOHarness, case: CaseSpec, query: QuerySpec, run: Run) -> None:
    """Did the harness *use* what it was given, or merely survive it?"""
    caps = harness.capabilities()
    if run.probe is None:
        raise EngagementSkipped(f"{harness.name} does not implement probe()")
    if not query.pushdown_observable:
        return
    if not run.probe.was_called:
        if run.expected.height:
            raise AssertionError("the harness was never asked to scan")
        # Polars short-circuits a query it can prove empty -- `head(0)`, a slice past the end --
        # without touching the plugin. Nothing was pushed, so there is nothing to have engaged
        # with, and demanding a scan here would assert a fact about Polars rather than the plugin.
        return

    if "projection" in caps.pushdown and query.projection is not None:
        needed = set(query.projection)
        if query.predicate is not None:
            needed |= set(query.predicate.columns())
        if needed < set(case.schema()):
            read = run.probe.columns_read
            if read is None:
                raise AssertionError("projection ignored: the harness read every column")
            if not read <= needed:
                raise AssertionError(f"read {sorted(read - needed)} outside the projection")

    if "predicate" in caps.pushdown and query.predicate is not None:
        if not run.probe.predicates_received:
            raise AssertionError("Polars pushed a predicate and the harness recorded none")
        if run.expected.height < case.n_rows and run.probe.rows_scanned >= case.n_rows:
            raise AssertionError(
                f"predicate not applied at the source: scanned {run.probe.rows_scanned} "
                f"of {case.n_rows} rows for a filter matching {run.expected.height}"
            )

    if (
        "limit" in caps.pushdown
        and query.limit is not None
        and query.predicate is None
        and query.offset is None
    ):
        # Only meaningful without a predicate -- Polars withholds `n_rows` once it pushes one --
        # and only without an offset, where the limit Polars pushes is `offset + limit` rather
        # than the limit the query asked for.
        pushed = run.probe.limit_received
        if pushed is None:
            raise AssertionError("Polars pushed a limit and the harness recorded none")
        if run.probe.rows_scanned > max(pushed * 2, pushed + 1):
            raise AssertionError(
                f"limit not honoured early: scanned {run.probe.rows_scanned} "
                f"for a pushed limit of {pushed}"
            )


def check_planning_reads_nothing(harness: IOHarness, case: CaseSpec) -> None:
    """`collect_schema()` must not touch data. Measured: Polars does not invoke the plugin at
    all for it, so the assertion is that no scan happened, not that it read zero rows."""
    target = materialise(harness, case)
    with harness.probing() as probe:
        schema = harness.scan(target).collect_schema()
    if probe is not None and probe.was_called:
        raise AssertionError("resolving the schema scanned data")
    if len(schema) != len(case.schema()):
        raise AssertionError(f"schema width {len(schema)} != {len(case.schema())}")


# -- metamorphic laws -----------------------------------------------------------------------
#
# Cheap, need no oracle frame, and generalise over any generated predicate.


def law_partition(harness: IOHarness, case: CaseSpec, predicate: Pred, target: object) -> None:
    """`count(p) + count(~p) + count(p is null) == n`.

    Kleene logic is where a SQL backend and Polars most often disagree: a row where the
    predicate is null belongs to neither `p` nor `~p`, and a backend that folds null to false
    puts it in `~p` and breaks the sum.
    """
    lf = harness.scan(target)
    n = lf.select(pl.len()).collect().item()
    expr = predicate.expr()
    matched = lf.filter(expr).select(pl.len()).collect().item()
    rejected = lf.filter(~expr).select(pl.len()).collect().item()
    unknown = lf.filter(expr.is_null()).select(pl.len()).collect().item()
    if matched + rejected + unknown != n:
        raise AssertionError(
            f"partition law: {matched} + {rejected} + {unknown} != {n} for {predicate.label()}"
        )


def law_filter_conjunction(
    harness: IOHarness, case: CaseSpec, left: Pred, right: Pred, target: object
) -> None:
    """`filter(p).filter(q) == filter(p & q)`."""
    lf = harness.scan(target)
    chained = lf.filter(left.expr()).filter(right.expr()).collect()
    combined = lf.filter(And(left, right).expr()).collect()
    compare(chained, combined, Strictness.ROW_ORDER)


def law_head_composition(
    harness: IOHarness, case: CaseSpec, a: int, b: int, target: object
) -> None:
    """`head(a).head(b) == head(min(a, b))`."""
    lf = harness.scan(target)
    compare(lf.head(a).head(b).collect(), lf.head(min(a, b)).collect(), Strictness.ROW_ORDER)


def law_projection_composition(
    harness: IOHarness, case: CaseSpec, outer: tuple[str, ...], target: object
) -> None:
    """`select(A).select(B) == select(B)` for `B` a subset of `A`."""
    lf = harness.scan(target)
    everything = tuple(case.schema())
    compare(
        lf.select(list(everything)).select(list(outer)).collect(),
        lf.select(list(outer)).collect(),
        Strictness.ROW_ORDER,
    )


def law_filter_projection_commute(
    harness: IOHarness, case: CaseSpec, predicate: Pred, column: str, target: object
) -> None:
    """`filter(p).select(c) == select(c + cols(p)).filter(p).select(c)`."""
    lf = harness.scan(target)
    needed = sorted({column} | set(predicate.columns()))
    direct = lf.filter(predicate.expr()).select(column).collect()
    routed = lf.select(needed).filter(predicate.expr()).select(column).collect()
    compare(direct, routed, Strictness.ROW_ORDER)


METAMORPHIC_LAWS = (
    law_partition,
    law_filter_conjunction,
    law_head_composition,
    law_projection_composition,
    law_filter_projection_commute,
)


# -- helpers --------------------------------------------------------------------------------


def _effective_strictness(harness: IOHarness, case: CaseSpec) -> Strictness:
    caps = harness.capabilities()
    level = caps.strictness
    if not case.order_key:
        # Nothing to sort by, so row order has to be trusted rather than normalised away.
        level = max(level, Strictness.ROW_ORDER)
    return Strictness(min(level, caps.exact_at.get(case.id, Strictness.METADATA)))


def is_runnable(query: QuerySpec, case: CaseSpec) -> bool:
    """Whether Polars will run this query at all against this case.

    Some curated queries are only legal for some schemas -- a duplicated projection, a literal
    Polars refuses to coerce. A query Polars rejects has no right answer for a harness to
    produce, so it is neither a pass nor a failure and must not be counted as either. Without
    this check a mutant can appear "caught" by an error that has nothing to do with its defect.
    """
    try:
        query.apply(built(case).lazy()).collect()
    except Exception:  # noqa: BLE001 - any error means the query is not well-formed here
        return False
    return True


def selective_predicate(case: CaseSpec) -> Pred:
    """A predicate over the case's identity column that keeps roughly half the rows.

    Used by the laws, which need *a* predicate rather than a particular one, and by engagement
    assertions that need one whose selectivity is known without consulting the data.
    """
    if not case.order_key:
        return Always(True)
    return Cmp("__i", "lt", max(case.n_rows // 2, 1))
