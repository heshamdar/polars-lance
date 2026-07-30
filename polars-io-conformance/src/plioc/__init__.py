"""`plioc` -- a generative conformance suite for Polars IO plugins.

The corpus is a pure function of `(spec, seed)`, materialised on demand as a `LazyFrame` built
entirely from expressions over a row index. Nothing is generated in Python loops and nothing is
committed as bytes, which is what buys determinism, prefix stability, stream independence and
scale-freedom all at once.

Start at `ConformanceSuite`.
"""

from plioc.equality import Strictness
from plioc.gen.core import Generator, NullPattern
from plioc.gen.layout import Layout
from plioc.harness import BaseHarness, Capabilities, IOHarness, Probe, ScanCall
from plioc.query import QuerySpec
from plioc.spec import CaseSpec, ColumnSpec
from plioc.suite import ConformanceSuite

__all__ = [
    "BaseHarness",
    "Capabilities",
    "CaseSpec",
    "ColumnSpec",
    "ConformanceSuite",
    "Generator",
    "IOHarness",
    "Layout",
    "NullPattern",
    "Probe",
    "QuerySpec",
    "ScanCall",
    "Strictness",
]
