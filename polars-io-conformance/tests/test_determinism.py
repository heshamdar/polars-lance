"""The five properties from the thesis, asserted.

Each of these is a property a fixture-file corpus cannot have, so each of them is a property
worth failing CI over. They are also the ones most easily broken by an innocent-looking change
to a generator.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from polars.testing import assert_frame_equal

from plioc.corpus import all_cases
from plioc.gen.core import NullPattern, stream
from plioc.gen.primitive import IntGen, StringGen
from plioc.spec import CaseSpec, ColumnSpec

CASES = all_cases(include_slow=False)
IDS = sorted(CASES)


def test_stream_keys_are_stable_across_processes() -> None:
    """`hash(name)` would be a silent determinism bug: CPython randomises `str` hashing per
    process, so a corpus keyed on it differs between runs and every digest is meaningless."""
    assert stream("v", 0) == 0x62AFC9955F5D82CB
    assert stream("other", 7) == 0x420EF15AA69E927F

    script = "from plioc.gen.core import stream; print(stream('v', 0), stream('other', 7))"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        # A different hash seed is exactly the condition under which `hash()` would diverge.
        env={"PYTHONHASHSEED": "1", "PYTHONPATH": ":".join(sys.path)},
    ).stdout.split()
    assert [int(x) for x in out] == [stream("v", 0), stream("other", 7)]


@pytest.mark.parametrize("case_id", IDS)
def test_build_is_referentially_transparent(case_id: str) -> None:
    case = CASES[case_id]
    assert_frame_equal(case.build().collect(), case.build().collect())


@pytest.mark.parametrize("case_id", IDS)
def test_prefix_stability(case_id: str) -> None:
    """`build(n=k)` is literally `build(n=N).head(k)`, so a fast CI run is a prefix of a nightly
    one rather than a different corpus."""
    case = CASES[case_id]
    if case.n_rows < 8:
        pytest.skip("too few rows for a meaningful prefix")
    if case.depends_on_n:
        pytest.skip("uses NullPattern.LAST, the one pattern that is a function of n")
    short = case.with_rows(8).build().collect()
    long = case.build().collect().head(8)
    assert_frame_equal(short, long)


def test_last_is_the_only_pattern_that_depends_on_n() -> None:
    """If another one grows an `n` dependency, prefix stability quietly stops holding for the
    cases using it and the skip above starts hiding it."""
    depends = {p for p in NullPattern if p.depends_on_n}
    assert depends == {NullPattern.LAST}
    for pattern in NullPattern:
        if pattern is NullPattern.LAST:
            continue
        case = CaseSpec(id="probe", columns=(ColumnSpec("v", IntGen(), pattern),), n_rows=64)
        assert_frame_equal(case.with_rows(8).build().collect(), case.build().collect().head(8))


def test_columns_draw_from_independent_streams() -> None:
    """Adding a column to a case must not perturb the others -- otherwise every digest and every
    recorded regression is invalidated by an unrelated edit."""
    a = ColumnSpec("a", IntGen(), NullPattern.SPARSE)
    b = ColumnSpec("b", StringGen(), NullPattern.DENSE)
    c = ColumnSpec("c", IntGen(), NullPattern.ALTERNATING)
    before = CaseSpec(id="x", columns=(a, b), n_rows=128).build().collect()
    after = CaseSpec(id="x", columns=(a, c, b), n_rows=128).build().collect()
    assert_frame_equal(before, after.select(before.columns))


def test_column_values_depend_on_the_name_not_the_position() -> None:
    a = ColumnSpec("a", IntGen(), NullPattern.NONE)
    renamed = ColumnSpec("z", IntGen(), NullPattern.NONE)
    one = CaseSpec(id="x", columns=(a,), n_rows=64).build().collect()["a"]
    two = CaseSpec(id="x", columns=(renamed,), n_rows=64).build().collect()["z"]
    assert one.to_list() != two.to_list()


def test_the_seed_reaches_the_stream_key() -> None:
    """The seed is mixed into the stream key, not into the values, so one seed change moves
    every column at once. A generator that ignored it would still look random."""
    columns = (
        ColumnSpec("a", IntGen(), NullPattern.SPARSE),
        ColumnSpec("b", StringGen(), NullPattern.SPARSE),
    )
    one = CaseSpec(id="x", columns=columns, n_rows=128, seed=0)
    two = CaseSpec(id="x", columns=columns, n_rows=128, seed=1)
    assert one.digest() != two.digest()
    assert stream("a", 0) != stream("a", 1)


@pytest.mark.parametrize("case_id", IDS)
def test_digest_is_reproducible(case_id: str) -> None:
    """The drift alarm has to be stable within a run before it can be stable across releases."""
    case = CASES[case_id]
    assert case.digest() == case.digest()
