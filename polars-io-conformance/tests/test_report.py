"""Tests of the report itself.

A report is a deliverable, so it gets the same treatment as the suite: the runner's verdicts must
match what pytest would say, the mutants must show up as failures, and the HTML must be
self-contained -- a report that silently needs a CDN is not readable from a CI artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from plioc import html, run
from plioc.corpus import all_cases
from plioc.harness import Capabilities
from plioc.harnesses.files import IpcHarness
from plioc.harnesses.memory import MemoryHarness
from plioc.harnesses.mutants import DropsResidual, EmptyListAsNull
from plioc.harnesses.plugin import PluginHarness
from plioc.queries import all_queries
from plioc.run import Status

# A slice of the corpus: these tests are about the reporting layer, and the corpus itself is
# covered by `test_reference_harnesses.py`.
CASES = [
    all_cases()[i]
    for i in (
        "numeric/int/Int64/sparse",
        "nested/list_three_way",
        "nested/struct_null_vs_all_null_fields",
        "temporal/datetime/ns/naive",
        "shape/rows/0",
        "categorical/enum",
    )
]
QUERIES = [q for q in all_queries(include_slow=False) if q.id.startswith(("mandate/", "limit/"))]


def _run(harness: object) -> run.HarnessRun:
    return run.run_harness(harness, cases=CASES, queries=QUERIES)  # type: ignore[arg-type]


def test_the_identity_harness_has_no_failures() -> None:
    """The positive control, through the reporting path rather than through pytest. If the two
    surfaces disagree about `MemoryHarness`, one of them is wrong."""
    result = _run(MemoryHarness())
    assert result.failures == []
    assert result.count(Status.PASS) > 0
    assert result.verdict == "pass"


def test_every_contract_is_exercised() -> None:
    result = _run(PluginHarness())
    contracts = {c.contract for c in result.checks}
    assert contracts == set(run.CONTRACTS)


def test_a_mutant_shows_up_as_a_failure() -> None:
    result = _run(EmptyListAsNull())
    assert result.failures, "the reporting path did not notice the injected defect"
    assert any(c.contract == "roundtrip" for c in result.failures)


def test_a_failure_carries_a_spec_that_reproduces_it() -> None:
    """The reproduction spec is the report's most useful column: a conformance failure nobody can
    re-run gets triaged into a backlog and forgotten."""
    from plioc import codec, oracle

    result = _run(EmptyListAsNull())
    failing = result.failures[0]
    assert failing.subject in result.repros
    revived = codec.loads(result.repros[failing.subject])
    with pytest.raises(AssertionError):
        oracle.check_roundtrip(EmptyListAsNull(), revived)


def test_a_declared_failure_is_reported_as_declared_not_as_a_failure() -> None:
    harness = EmptyListAsNull()
    reason = "the injected defect, declared"
    original = harness.capabilities

    def declared() -> Capabilities:
        caps = original()
        return Capabilities(
            strictness=caps.strictness,
            pushdown=caps.pushdown,
            known_failures={
                **{c.id: reason for c in CASES},
                **{q.id: reason for q in QUERIES},
            },
        )

    harness.capabilities = declared  # type: ignore[method-assign]
    result = _run(harness)
    assert result.count(Status.XFAIL) > 0
    assert [c for c in result.checks if c.status is Status.FAIL] == []


def test_a_declaration_that_no_longer_fails_is_a_failure() -> None:
    """The ratchet. A stale declaration is what turns a capability matrix into fiction, so it is
    reported as a failure rather than as a pass."""
    harness = MemoryHarness()
    original = harness.capabilities

    def declared() -> Capabilities:
        caps = original()
        return Capabilities(
            strictness=caps.strictness,
            known_failures={CASES[0].id: "not actually broken"},
        )

    harness.capabilities = declared  # type: ignore[method-assign]
    result = _run(harness)
    stale = [c for c in result.checks if c.status is Status.XPASS]
    assert stale, "a declaration that passes was not reported"
    assert stale[0] in result.failures


def test_engagement_degrades_to_a_skip_without_a_probe() -> None:
    """A harness that cannot be asked what it received must not silently pass the engagement
    checks -- passing them would mean nothing."""
    result = _run(PluginHarness(probing=False))
    engagement = [c for c in result.checks if c.contract == "engagement"]
    assert engagement
    assert all(c.status is Status.SKIP for c in engagement)
    assert any("probe" in c.detail for c in engagement)


def test_unclaimed_dtypes_skip_rather_than_fail(tmp_path: Path) -> None:
    import polars as pl

    harness = IpcHarness(tmp_path)
    original = harness.capabilities

    def declared() -> Capabilities:
        caps = original()
        return Capabilities(strictness=caps.strictness, dtypes=frozenset({pl.Int64}))

    harness.capabilities = declared  # type: ignore[method-assign]
    result = _run(harness)
    assert result.failures == []
    assert result.count(Status.SKIP) > 0


def test_the_suite_run_records_its_environment() -> None:
    result = run.run_suite([MemoryHarness()], cases=CASES, queries=QUERIES)
    assert result.polars_version and result.python_version
    assert result.case_count == len(CASES)
    assert result.query_count == len(QUERIES)
    assert result.duration_s >= 0


# -- rendering --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rendered() -> str:
    result = run.run_suite(
        [MemoryHarness(), PluginHarness(), DropsResidual()], cases=CASES, queries=QUERIES
    )
    return html.document(result, title="test report")


def test_the_page_is_self_contained(rendered: str) -> None:
    """A report that needs a CDN is unreadable from a CI artifact or an air-gapped machine."""
    for pattern in (r"src\s*=\s*[\"']https?:", r"href\s*=\s*[\"']https?:", r"@import"):
        assert not re.search(pattern, rendered), pattern


def test_the_page_is_well_formed(rendered: str) -> None:
    from html.parser import HTMLParser

    class Check(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.stack: list[str] = []
            self.void = {"meta", "br", "hr", "input", "img", "link"}

        def handle_starttag(self, tag: str, attrs: object) -> None:
            if tag not in self.void:
                self.stack.append(tag)

        def handle_endtag(self, tag: str) -> None:
            if tag in self.void:
                return
            assert self.stack and self.stack[-1] == tag, f"</{tag}> closes {self.stack[-1:]}"
            self.stack.pop()

    parser = Check()
    parser.feed(rendered)
    assert parser.stack == []


def test_the_page_reports_the_failures_and_the_declarations(rendered: str) -> None:
    assert "mutant:drops-residual" in rendered
    assert "Failures" in rendered
    # The mandate queries are what catch a dropped residual, so at least one has to be named.
    assert "mandate/" in rendered


def test_dangerous_content_is_escaped() -> None:
    """Case ids and failure messages contain quotes, angle brackets and control characters --
    they come from the escaping palette on purpose."""
    result = run.run_suite([MemoryHarness()], cases=CASES, queries=QUERIES)
    result.runs[0].checks.append(
        run.Check("roundtrip", "<script>alert(1)</script>", Status.FAIL, 'x" onload="y')
    )
    page = html.document(result)
    assert "<script>alert(1)</script>" not in page.replace(html._JS, "")
    assert "&lt;script&gt;" in page


def test_json_output_round_trips() -> None:
    import json

    result = run.run_suite([MemoryHarness()], cases=CASES, queries=QUERIES)
    payload = json.loads(html.to_json(result))
    assert payload["harnesses"][0]["name"] == "memory"
    assert sum(payload["harnesses"][0]["totals"].values()) == len(result.runs[0].checks)


def test_the_cli_writes_a_report_and_signals_the_verdict(tmp_path: Path) -> None:
    from plioc.__main__ import main

    out = tmp_path / "report.html"
    code = main(["--no-reference", "--html", str(out), "plioc.harnesses.memory:MemoryHarness"])
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")

    broken = tmp_path / "broken.html"
    code = main(
        ["--no-reference", "--html", str(broken), "plioc.harnesses.mutants:EmptyListAsNull"]
    )
    assert code == 1, "the CLI must fail the build on an undeclared failure"
