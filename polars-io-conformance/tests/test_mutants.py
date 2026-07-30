"""The suite's own coverage metric.

Line coverage of the generator says nothing about whether the corpus finds bugs. This does: every
deliberately-broken harness must be caught by at least one named case or query, and the pairing is
printed so it can be read.

A mutant that nothing catches is a defect the suite would ship past, which makes this file the
first place to look when adding a case.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from plioc import oracle
from plioc.corpus import all_cases
from plioc.harnesses.mutants import ALL_MUTANTS, UNREACHABLE_MUTANTS
from plioc.harnesses.plugin import PluginHarness
from plioc.queries import all_queries
from plioc.spec import CaseSpec

CASES = [c for c in all_cases(include_slow=False).values() if c.n_rows]
FIXTURE = all_cases()["query/fixture"]
# Only queries Polars will actually run: an error it raises itself is not evidence about a
# mutant's defect, and counting one as a catch would make this whole file lie.
QUERIES = [
    q
    for q in all_queries(include_slow=False)
    if "udf" not in q.tags and oracle.is_runnable(q, FIXTURE)
]


@dataclass(frozen=True)
class Catch:
    kind: str
    id: str
    detail: str


def _catch(mutant: PluginHarness) -> Catch | None:
    for case in CASES:
        try:
            oracle.check_roundtrip(mutant, case)
        except BaseException as exc:  # noqa: BLE001 - any failure is a catch, panics included
            return Catch("roundtrip", case.id, _first_line(exc))
    for case in CASES:
        try:
            oracle.check_fixpoint(mutant, case)
        except BaseException as exc:  # noqa: BLE001
            return Catch("fixpoint", case.id, _first_line(exc))

    target = oracle.materialise(mutant, FIXTURE, "/queries")
    for query in QUERIES:
        try:
            run = oracle.check_query_correctness(mutant, FIXTURE, query, target)
        except BaseException as exc:  # noqa: BLE001
            return Catch("query", query.id, _first_line(exc))
        try:
            oracle.check_engagement(mutant, FIXTURE, query, run)
        except oracle.EngagementSkipped:
            continue
        except BaseException as exc:  # noqa: BLE001
            return Catch("engagement", query.id, _first_line(exc))
    return None


def _first_line(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".splitlines()[0][:100]


@pytest.fixture(scope="module")
def catches() -> dict[str, Catch | None]:
    return {cls.name: _catch(cls()) for cls in ALL_MUTANTS}


@pytest.mark.parametrize("mutant", ALL_MUTANTS, ids=lambda c: c.name)
def test_mutant_is_caught(catches: dict[str, Catch | None], mutant: type[PluginHarness]) -> None:
    catch = catches[mutant.name]
    assert catch is not None, (
        f"{mutant.name} survives the entire corpus. Either it models a defect no case exercises "
        "-- add one -- or it is not actually broken."
    )
    print(f"{mutant.name}: caught by {catch.kind} {catch.id} ({catch.detail})")


@pytest.mark.parametrize("mutant", UNREACHABLE_MUTANTS, ids=lambda c: c.name)
def test_unreachable_mutant_survives(mutant: type[PluginHarness]) -> None:
    """A defect Polars' plugin interface cannot expose.

    Asserting the mutant survives is not a concession -- it pins down a measured fact about what
    Polars pushes. If this starts failing, Polars began delivering a limit together with a
    predicate, the mutant became a real defect, and the pushdown suite needs a case for it.
    """
    catch = _catch(mutant())
    assert catch is None, (
        f"{mutant.name} is now caught by {catch.kind} {catch.id} -- Polars' pushdown behaviour "
        "changed, so move it into ALL_MUTANTS and revisit the limit assertions"
    )


def test_the_reference_plugin_is_not_caught() -> None:
    """The control on the control. If the correct harness also "fails", the catches above prove
    nothing about the mutants."""
    assert _catch(PluginHarness()) is None


def test_every_case_builds_and_matches_its_declared_schema() -> None:
    for case in all_cases().values():
        assert case.build().collect_schema() == case.schema(), case.id


def test_case_ids_are_unique_and_filesystem_safe() -> None:
    """Ids become file names for the file-backed harnesses, so a collision after sanitising is a
    silent overwrite of one case by another."""
    ids = [c.id for c in all_cases().values()]
    assert len(ids) == len(set(ids))
    sanitised = [i.replace("/", "_") for i in ids]
    assert len(sanitised) == len(set(sanitised))


def test_query_ids_are_unique() -> None:
    ids = [q.id for q in all_queries()]
    assert len(ids) == len(set(ids))


def test_cases_without_an_order_key_are_declared() -> None:
    """A case with no `__i` cannot be compared row-order-agnostically, so it must be the case's
    own decision rather than an accident of how its columns were written."""
    for case in all_cases().values():
        if not case.order_key:
            assert case.id in {"shape/no_columns", "shape/no_order_key"}, case.id


def _ids(cases: list[CaseSpec]) -> list[str]:
    return [c.id for c in cases]
