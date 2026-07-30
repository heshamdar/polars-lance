"""Specs must survive the trip to text and back.

If they do not, a Hypothesis counterexample cannot become a committed regression, and the only
remaining way to keep one is to commit the data -- which is the thing this suite exists to avoid.
"""

from __future__ import annotations

import pytest

from plioc import codec
from plioc.corpus import all_cases
from plioc.gen.core import NullPattern
from plioc.gen.nested import ListGen, StructField, StructGen
from plioc.gen.primitive import IntGen, StringGen
from plioc.queries import all_queries
from plioc.regressions import load_all, record
from plioc.spec import CaseSpec, ColumnSpec

CASES = all_cases()


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_every_case_round_trips_through_the_codec(case_id: str) -> None:
    case = CASES[case_id]
    revived = codec.loads(codec.dumps(case))
    assert revived == case
    # Equality of the spec is necessary but not sufficient: what matters is that the revived spec
    # generates the same data, which is what a regression file is for.
    assert revived.digest() == case.digest()


@pytest.mark.parametrize("query", all_queries(), ids=lambda q: q.id)
def test_every_query_round_trips_through_the_codec(query: object) -> None:
    # Compared as text, not with `==`: several predicates carry `NaN` literals, and `NaN != NaN`
    # makes dataclass equality report a difference between a value and itself. Stable
    # serialisation is the property a regression file actually needs.
    assert codec.dumps(codec.loads(codec.dumps(query))) == codec.dumps(query)


def test_a_nested_spec_survives() -> None:
    case = CaseSpec(
        id="codec/nested",
        columns=(
            ColumnSpec(
                "v",
                ListGen(
                    child=StructGen(
                        (
                            StructField("a", IntGen(), NullPattern.ALTERNATING),
                            StructField("with.dot", StringGen(frozenset({"escaping"}))),
                        )
                    ),
                    lengths=(0, 2),
                    child_nulls=NullPattern.BOUNDARY,
                ),
                NullPattern.LAST,
            ),
        ),
        n_rows=32,
    )
    assert codec.loads(codec.dumps(case)) == case


def test_an_unregistered_class_is_refused_rather_than_silently_dropped() -> None:
    class Sneaky:
        pass

    with pytest.raises(TypeError):
        codec.encode(Sneaky())


def test_recorded_regressions_load(tmp_path: object) -> None:
    """The regression directory is loaded on import of the corpus, so a malformed file breaks
    every run rather than one test."""
    for regression in load_all():
        assert regression.case.build().collect_schema() == regression.case.schema()
        assert regression.note, f"{regression.path} has no note explaining why it exists"


def test_record_writes_a_loadable_file() -> None:
    case = CaseSpec(
        id="codec/recorded", columns=(ColumnSpec("v", IntGen(), NullPattern.FIRST),), n_rows=4
    )
    path = record(case, None, "written by test_codec")
    try:
        loaded = [r for r in load_all() if r.path == path]
        assert len(loaded) == 1
        assert loaded[0].case == case
    finally:
        path.unlink()
