"""Run the generative IO-plugin conformance suite against this plugin.

The suite lives in `polars-io-conformance/` and knows nothing about Lance: it generates its corpus
from specs, writes it through `sink_lance`, reads it back through `scan_lance`, and compares
against in-memory Polars. Its value here is the part hand-written tests do not reach -- the
null-versus-empty distinctions, the row counts either side of every batch boundary, the escaping
palette against the predicate translator, and the predicate mandate.

What this plugin does and does not preserve is declared in `tests/conformance_harness.py`. Every
entry there is a strict xfail, so an upstream fix breaks this file until the declaration is updated.
"""

from pathlib import Path

import pytest

from plioc import ConformanceSuite
from polars_lance import _polars_lance
from tests.conformance_harness import LanceHarness

# The corpus writes nullable nested columns as a matter of course, which is exactly the data Lance
# guards with `debug_assert!`s in its 2.1 encoder (lance-format/lance#8032, #8033). A debug build
# cannot write it at all, so rather than pin a storage version for the whole corpus and test
# something other than the default, the suite runs against a release build -- `just
# test-conformance`, and the release-build CI job.
pytestmark = pytest.mark.skipif(
    _polars_lance._debug_assertions,
    reason="needs a release build: Lance's 2.1 encoder debug-asserts on nullable nested columns",
)


class TestLance(ConformanceSuite):
    @pytest.fixture
    def harness(self, tmp_path: Path) -> LanceHarness:
        return LanceHarness(tmp_path)
