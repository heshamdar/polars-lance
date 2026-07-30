"""Parquet and IPC reference harnesses.

Their value to a plugin author is the diff: a case that fails for Parquet as well is a
format-inherent loss, not their bug, and the report prints the two columns side by side so that
is visible at a glance rather than after an afternoon.

The declarations below are measured against the pinned Polars, not assumed --
`tests/test_reference_harnesses.py` fails if a loss appears or disappears.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


from plioc.equality import Strictness
from plioc.harness import BaseHarness, Capabilities, Target


class _FileHarness(BaseHarness):
    suffix = ""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def target(self, name: str) -> Target:
        return self.root / f"{name.replace('/', '_')}{self.suffix}"


class ParquetHarness(_FileHarness):
    name = "parquet"
    suffix = ".parquet"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        lf.sink_parquet(target)

    def scan(self, target: Target) -> pl.LazyFrame:
        return pl.scan_parquet(target)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Chunking becomes row groups on the way in and batches on the way out; neither is
            # the frame's original chunk layout, so PHYSICAL is not claimable.
            strictness=Strictness.ROW_ORDER,
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
            # No normaliser. `Enum` and `Categorical` were expected to come back as strings --
            # Parquet has no dictionary type that carries a category list, let alone an unused
            # category -- and measurably they do not: Polars round-trips both. Declaring the loss
            # anyway is exactly the pessimism `exact_at` and strict xfail exist to prevent, and
            # the suite caught it.
            known_failures={
                # Format-inherent, and the useful kind of entry: a plugin author seeing their own
                # harness fail this case can see Parquet fails it too and stop looking for a bug.
                "nested/struct_empty": (
                    "Parquet cannot represent a struct with no child field: "
                    "'Unable to write struct type with no child field to Parquet'"
                ),
            },
        )


class IpcHarness(_FileHarness):
    name = "ipc"
    suffix = ".arrow"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        lf.sink_ipc(target)

    def scan(self, target: Target) -> pl.LazyFrame:
        return pl.scan_ipc(target)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            strictness=Strictness.ROW_ORDER,
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
        )
