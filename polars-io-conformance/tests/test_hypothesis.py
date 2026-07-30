"""The generative layers, and the loop that turns a discovery into a permanent regression.

Two properties: a generated spec round-trips, and a generated query agrees with the in-memory
oracle. Both run against the reference harnesses, where any failure is the suite's own bug.

The last test is the one that matters most: it injects a known defect, lets Hypothesis find it,
and records the shrunk counterexample as a regression file. If that loop does not work, the
generative layer produces failures nobody can reproduce next week.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from plioc import codec, oracle
from plioc.corpus import query_fixture
from plioc.harnesses.files import IpcHarness
from plioc.harnesses.memory import MemoryHarness
from plioc.harnesses.mutants import EmptyListAsNull
from plioc.harnesses.plugin import PluginHarness
from plioc.regressions import record
from plioc.spec import CaseSpec
from plioc.strategies import case_specs, generators, queries

FIXTURE = query_fixture()

# Deadlines are meaningless here: an example's cost is dominated by materialising a frame, which
# varies by orders of magnitude across specs.
PROFILE = settings(
    deadline=None,
    max_examples=15,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


@PROFILE
@given(case=case_specs())
def test_generated_specs_round_trip_through_memory(case: CaseSpec) -> None:
    """The positive control, generatively. `MemoryHarness` is the identity, so a failure here is
    a spec the generator should not have produced."""
    oracle.check_roundtrip(MemoryHarness(), case)


@PROFILE
@given(case=case_specs())
def test_generated_specs_round_trip_through_the_plugin(case: CaseSpec) -> None:
    oracle.check_roundtrip(PluginHarness(), case)


@PROFILE
@given(case=case_specs())
def test_generated_specs_are_prefix_stable(case: CaseSpec) -> None:
    if case.n_rows < 4 or case.depends_on_n:
        return
    from polars.testing import assert_frame_equal

    assert_frame_equal(case.with_rows(4).build().collect(), case.build().collect().head(4))


@PROFILE
@given(case=case_specs())
def test_generated_specs_survive_the_codec(case: CaseSpec) -> None:
    assert codec.dumps(codec.loads(codec.dumps(case))) == codec.dumps(case)


@PROFILE
@given(data=st.data())
def test_generated_queries_agree_with_the_oracle(data: st.DataObject) -> None:
    harness = PluginHarness()
    target = oracle.materialise(harness, FIXTURE)
    query = data.draw(queries(FIXTURE.schema()))
    if not oracle.is_runnable(query, FIXTURE):
        return
    oracle.check_query_correctness(harness, FIXTURE, query, target)


@PROFILE
@given(gen=generators())
def test_every_generated_generator_produces_its_declared_dtype(gen: object) -> None:
    from plioc.gen.core import NullPattern
    from plioc.spec import ColumnSpec

    case = CaseSpec(
        id="hypothesis/dtype",
        columns=(ColumnSpec("v", gen, NullPattern.SPARSE),),  # type: ignore[arg-type]
        n_rows=8,
    )
    assert case.build().collect_schema()["v"] == gen.dtype  # type: ignore[attr-defined]


def test_an_injected_bug_is_found_and_recorded(tmp_path: object) -> None:
    """The whole point of the generative layer, end to end.

    A harness with a real defect is handed to Hypothesis, which finds a spec that exposes it; the
    shrunk spec is written to `regressions/` and loaded back. That closes the loop: every future
    discovery costs a ten-line file rather than a data file, and runs deterministically on every
    commit thereafter.
    """
    found: list[CaseSpec] = []

    @settings(deadline=None, max_examples=120, suppress_health_check=list(HealthCheck))
    @given(case=case_specs(max_columns=2))
    def hunt(case: CaseSpec) -> None:
        try:
            oracle.check_roundtrip(EmptyListAsNull(), case)
        except AssertionError:
            found.append(case)
            raise

    try:
        hunt()
    except AssertionError:
        pass

    assert found, "Hypothesis did not reach the injected empty-list-as-null defect"
    shrunk = found[-1]
    path = record(shrunk, None, "injected empty-list-as-null defect, recorded by the test suite")
    try:
        revived = codec.loads(codec.dumps(shrunk))
        try:
            oracle.check_roundtrip(EmptyListAsNull(), revived)
        except AssertionError:
            pass
        else:
            raise AssertionError("the recorded spec no longer reproduces the failure")
    finally:
        path.unlink()


@settings(deadline=None, max_examples=8, suppress_health_check=list(HealthCheck))
@given(case=case_specs(max_columns=2))
def test_generated_specs_round_trip_through_ipc(case: CaseSpec) -> None:
    """A real format, generatively, on a short leash.

    Kept small on purpose: writing a file per example costs orders of magnitude more than an
    in-memory round-trip, and the curated corpus already covers IPC thoroughly. The generative
    layer's job here is reaching shapes nobody thought to name, not re-testing the format.
    Parquet is deliberately absent -- its own limits (no zero-field struct) turn generated shapes
    into noise that has to be pattern-matched out of the failure, which is worse than not running.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        oracle.check_roundtrip(IpcHarness(Path(tmp)), case)
