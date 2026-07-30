"""What a plugin author subclasses.

    from plioc import ConformanceSuite

    class TestLance(ConformanceSuite):
        @pytest.fixture
        def harness(self, tmp_path):
            return LanceHarness(tmp_path)

That is the whole integration. The capability declaration lives on the harness, so the same
object can be used outside pytest -- by the report generator, for instance.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from typing import Any

import polars as pl
import pytest

from plioc import oracle
from plioc.corpus import all_cases, query_fixture
from plioc.equality import Strictness
from plioc.harness import Capabilities, IOHarness, is_failure
from plioc.queries import all_queries
from plioc.query import Cmp, QuerySpec
from plioc.spec import CaseSpec

_CASES = all_cases()
_CASE_IDS = sorted(_CASES)
_QUERIES = all_queries()
_QUERY_IDS = [q.id for q in _QUERIES]
_FIXTURE = query_fixture()


def _dtypes_of(dtype: Any) -> list[pl.DataType]:
    """Flatten a dtype into itself and everything nested inside it.

    Typed loosely because `Schema.items()` is annotated as yielding `DataType | DataTypeClass`,
    and a nested `inner`/`field.dtype` is the same union again.
    """
    out: list[pl.DataType] = [dtype]
    if isinstance(dtype, (pl.List, pl.Array)):
        out += _dtypes_of(dtype.inner)
    elif isinstance(dtype, pl.Struct):
        for f in dtype.fields:
            out += _dtypes_of(f.dtype)
    return out


def expect(caps: Capabilities, case_id: str, run: Any) -> None:
    """Run an assertion under the harness's declaration.

    A declared known failure is a *strict* xfail: if it starts passing, this fails loudly and
    forces the declaration to be updated. Without that, a capability matrix rots into pessimism
    and stops telling anyone anything.
    """
    reason = caps.known_failures.get(case_id)
    if reason is None:
        run()
        return
    try:
        run()
    except BaseException as exc:  # noqa: BLE001 - see `is_failure`: a Rust panic is not an Exception
        if not is_failure(exc):
            raise
        pytest.xfail(f"declared: {reason}")
    pytest.fail(
        f"{case_id} is declared a known failure ({reason}) but now passes -- "
        "update the capability declaration"
    )


class ConformanceSuite:
    """The suite. Subclass it, provide a `harness` fixture, and run pytest."""

    #: Set to False to skip the cases that materialise a lot of data.
    include_slow: bool = False

    # -- fixtures --------------------------------------------------------------------------

    @pytest.fixture
    def harness(self) -> IOHarness:  # pragma: no cover - overridden
        raise NotImplementedError("provide a `harness` fixture")

    @pytest.fixture
    def caps(self, harness: IOHarness) -> Capabilities:
        return harness.capabilities()

    @pytest.fixture
    def fixture_case(self, caps: Capabilities) -> CaseSpec:
        """The query fixture, narrowed to the dtypes the harness claims.

        Narrowed rather than skipped: one unsupported dtype in a ten-column fixture would
        otherwise cost the entire query matrix, which is the part of the suite a plugin most needs.
        Queries that reference a dropped column skip individually.
        """
        return _supported_columns(caps, _FIXTURE)

    @pytest.fixture
    def written_fixture(self, harness: IOHarness, fixture_case: CaseSpec) -> Any:
        return oracle.materialise(harness, fixture_case)

    # -- helpers ---------------------------------------------------------------------------

    def _skip_slow(self, case: CaseSpec) -> None:
        if not self.include_slow and "slow" in case.all_tags:
            pytest.skip("slow case; set include_slow = True to run it")

    def _require_dtypes(self, caps: Capabilities, case: CaseSpec) -> None:
        for name, dtype in case.schema().items():
            for part in _dtypes_of(dtype):
                if not caps.supports(part):
                    pytest.skip(f"the harness does not claim {part} (column {name!r})")

    def _require_columns(self, query: QuerySpec, case: CaseSpec) -> None:
        referenced = set(query.projection or ()) | set(
            query.predicate.columns() if query.predicate is not None else ()
        )
        missing = referenced - set(case.schema())
        if missing:
            pytest.skip(f"the fixture has no {sorted(missing)}: dtype not claimed")

    # -- round-trip ------------------------------------------------------------------------

    @pytest.mark.parametrize("case_id", _CASE_IDS)
    def test_roundtrip(self, harness: IOHarness, caps: Capabilities, case_id: str) -> None:
        """Everything written comes back, at the strictness the harness claims."""
        case = _CASES[case_id]
        self._skip_slow(case)
        self._require_dtypes(caps, case)
        expect(caps, case_id, lambda: oracle.check_roundtrip(harness, case))

    @pytest.mark.parametrize("case_id", _CASE_IDS)
    def test_fixpoint(self, harness: IOHarness, caps: Capabilities, case_id: str) -> None:
        """A declared loss happens once. A loss that happens on every write is a bug that
        normalisation hides, because each round-trip in isolation looks legitimate."""
        case = _CASES[case_id]
        self._skip_slow(case)
        self._require_dtypes(caps, case)
        expect(caps, case_id, lambda: oracle.check_fixpoint(harness, case))

    @pytest.mark.parametrize("case_id", _CASE_IDS)
    def test_exactness_is_declared(
        self, harness: IOHarness, caps: Capabilities, case_id: str
    ) -> None:
        """Where a case is declared lossy at a level, it must still *be* lossy there.

        Opt-in and per case, deliberately: a blanket "the level above the declared one must
        fail" rule would force a harness with one lossy dtype to be wrong about every case it
        gets right.
        """
        if case_id not in caps.exact_at:
            pytest.skip("no exactness declared for this case")
        case = _CASES[case_id]
        self._skip_slow(case)
        above = Strictness(min(caps.exact_at[case_id] + 1, Strictness.METADATA))
        if above == caps.exact_at[case_id]:
            pytest.skip("declared exact at the top of the ladder")
        target = oracle.materialise(harness, case)
        observed = harness.scan(target).collect()
        with pytest.raises(AssertionError):
            from plioc.equality import compare

            compare(observed, oracle.oracle_frame(harness, case), above)

    # -- planning --------------------------------------------------------------------------

    def test_planning_reads_no_data(self, harness: IOHarness, fixture_case: CaseSpec) -> None:
        """Resolving the schema must not scan. Measured: Polars does not even call the plugin."""
        oracle.check_planning_reads_nothing(harness, fixture_case)

    def test_schema_matches_declaration(self, harness: IOHarness, fixture_case: CaseSpec) -> None:
        target = oracle.materialise(harness, fixture_case)
        observed = harness.scan(target).collect_schema()
        expected = oracle.oracle_frame(harness, fixture_case).schema
        assert dict(observed) == dict(expected)

    # -- queries ---------------------------------------------------------------------------

    @pytest.mark.parametrize("query_id", _QUERY_IDS)
    def test_query_correctness(
        self,
        harness: IOHarness,
        caps: Capabilities,
        fixture_case: CaseSpec,
        written_fixture: Any,
        query_id: str,
    ) -> None:
        """The differential assertion: in-memory Polars is the oracle.

        This is where the mandate is enforced. A predicate handed to an IO plugin is an
        obligation, not a hint -- Polars will not re-apply it -- so a plugin that pushes half of
        a conjunction and forgets the residual fails here and nowhere else.
        """
        query = _query(query_id)
        self._require_columns(query, fixture_case)
        _skip_unrunnable(query, fixture_case, self.include_slow)
        expect(
            caps,
            query.id,
            lambda: oracle.check_query_correctness(harness, fixture_case, query, written_fixture),
        )

    @pytest.mark.parametrize("query_id", _QUERY_IDS)
    def test_query_engagement(
        self,
        harness: IOHarness,
        caps: Capabilities,
        fixture_case: CaseSpec,
        written_fixture: Any,
        query_id: str,
    ) -> None:
        """Did the harness *use* what it was pushed?

        Separate from correctness on purpose. A plugin that ignores every predicate and filters
        in Python passes the correctness suite completely while delivering none of its value.
        """
        query = _query(query_id)
        if not caps.pushdown:
            pytest.skip("harness claims no pushdown")
        self._require_columns(query, fixture_case)
        _skip_unrunnable(query, fixture_case, self.include_slow)
        run = oracle.run_query(harness, fixture_case, query, written_fixture)
        try:
            oracle.check_engagement(harness, fixture_case, query, run)
        except oracle.EngagementSkipped as exc:
            pytest.skip(
                f"{exc}: the pushdown suite is degraded to correctness only, which a plugin "
                "that ignores every predicate also passes"
            )

    # -- metamorphic laws ------------------------------------------------------------------

    def test_law_partition(
        self, harness: IOHarness, fixture_case: CaseSpec, written_fixture: Any
    ) -> None:
        oracle.law_partition(harness, fixture_case, Cmp("i64", "gt", 0), written_fixture)

    def test_law_filter_conjunction(
        self, harness: IOHarness, fixture_case: CaseSpec, written_fixture: Any
    ) -> None:
        oracle.law_filter_conjunction(
            harness, fixture_case, Cmp("i64", "gt", 0), Cmp("s", "ne", "alpha"), written_fixture
        )

    @pytest.mark.parametrize("a,b", [(0, 5), (5, 0), (3, 9), (9, 3), (10_000, 7)])
    def test_law_head_composition(
        self, harness: IOHarness, fixture_case: CaseSpec, written_fixture: Any, a: int, b: int
    ) -> None:
        oracle.law_head_composition(harness, fixture_case, a, b, written_fixture)

    def test_law_projection_composition(
        self, harness: IOHarness, fixture_case: CaseSpec, written_fixture: Any
    ) -> None:
        oracle.law_projection_composition(harness, fixture_case, ("s", "i64"), written_fixture)

    def test_law_filter_projection_commute(
        self, harness: IOHarness, fixture_case: CaseSpec, written_fixture: Any
    ) -> None:
        oracle.law_filter_projection_commute(
            harness, fixture_case, Cmp("i64", "gt", 0), "s", written_fixture
        )


def _query(query_id: str) -> QuerySpec:
    for q in _QUERIES:
        if q.id == query_id:
            return q
    raise KeyError(query_id)


def _skip_unrunnable(query: QuerySpec, case: CaseSpec, include_slow: bool) -> None:
    if "slow" in query.tags and not include_slow:
        pytest.skip("slow query; set include_slow = True to run it")
    if "udf" in query.tags and importlib.util.find_spec("cloudpickle") is None:
        # Polars serialises a UDF with cloudpickle to hand it to an IO plugin. Without it the
        # query fails inside Polars, which is a missing optional dependency and not a
        # conformance result either way.
        pytest.skip("cloudpickle is not installed, so a UDF cannot reach an IO plugin")
    if not oracle.is_runnable(query, case):
        pytest.skip("Polars itself rejects this query, so there is no right answer to check")


def _supported_columns(caps: Capabilities, case: CaseSpec) -> CaseSpec:
    keep = tuple(
        c for c in case.columns if all(caps.supports(part) for part in _dtypes_of(c.dtype))
    )
    return case if len(keep) == len(case.columns) else replace(case, columns=keep)
