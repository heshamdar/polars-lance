"""The identity harness.

Positive control: it must pass every case at `Strictness.METADATA`. When it does not, the suite
is wrong -- a case that `MemoryHarness` fails is asserting something that is not true of Polars
itself, let alone of a format.
"""

from __future__ import annotations

import polars as pl

from plioc.equality import Strictness
from plioc.harness import BaseHarness, Capabilities, Target


class MemoryHarness(BaseHarness):
    name = "memory"

    def __init__(self) -> None:
        super().__init__()
        self._store: dict[str, pl.DataFrame] = {}

    def target(self, name: str) -> Target:
        return name

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        self._store[target] = lf.collect()

    def scan(self, target: Target) -> pl.LazyFrame:
        return self._store[target].lazy()

    def capabilities(self) -> Capabilities:
        return Capabilities(
            strictness=Strictness.METADATA,
            # No pushdown at all: `LazyFrame.lazy()` is an ordinary in-memory scan, so Polars
            # does the projecting and filtering itself. Correct, and unobservable.
            pushdown=frozenset(),
            preserves_row_order=True,
        )
