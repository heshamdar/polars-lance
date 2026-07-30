"""A correct, probeable IO plugin over an in-memory store.

This is the positive control for the pushdown suite. Parquet and IPC are real formats but their
scans are Polars' own, so nothing can observe what was pushed into them; only a harness built on
`register_io_source` can be asked "what did you actually receive". Every mutant in
`harnesses/mutants.py` is this class with exactly one thing broken.
"""

from __future__ import annotations

from collections.abc import Iterator

import polars as pl
from polars.io.plugins import register_io_source

from plioc.equality import Strictness
from plioc.harness import BaseHarness, Capabilities, Probe, ScanCall, Target


class PluginHarness(BaseHarness):
    name = "plugin"

    def __init__(self, probing: bool = True) -> None:
        super().__init__()
        self._store: dict[str, pl.DataFrame] = {}
        self._probe = Probe() if probing else None
        self._read_columns: tuple[str, ...] | None = None

    # -- write side ------------------------------------------------------------------------

    def target(self, name: str) -> Target:
        return name

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        self._store[target] = self.on_write(lf.collect())

    def on_write(self, df: pl.DataFrame) -> pl.DataFrame:
        """Hook: what the format does to the data on the way in. Correct here, broken in the
        write-side mutants."""
        return df

    # -- read side -------------------------------------------------------------------------

    def scan(self, target: Target) -> pl.LazyFrame:
        df = self._store[target]

        def io_source(
            with_columns: list[str] | None,
            predicate: pl.Expr | None,
            n_rows: int | None,
            batch_size: int | None,
        ) -> Iterator[pl.DataFrame]:
            produced = 0
            self._read_columns = None if with_columns is None else tuple(with_columns)
            for batch in self.produce(df, with_columns, predicate, n_rows, batch_size):
                produced += batch.height
                yield batch
            self.record(
                ScanCall(
                    # What the harness *read*, not what it was asked for. Recording the request
                    # would make every projection assertion vacuous: a plugin that reads all the
                    # columns and then subsets them in memory would report the narrow projection
                    # it was handed and look perfectly engaged.
                    columns=self._read_columns,
                    predicate=None if predicate is None else str(predicate),
                    n_rows=n_rows,
                    batch_size=batch_size,
                    rows_produced=produced,
                )
            )

        return register_io_source(io_source=io_source, schema=df.schema, is_pure=True)

    def produce(
        self,
        df: pl.DataFrame,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """Hook: the scan itself.

        The order here is the contract. The predicate is a *mandate* -- Polars will not re-apply
        it -- so it must be honoured in full, and the limit must be applied to what survives it,
        never to what went in.
        """
        self._read_columns = None if with_columns is None else tuple(with_columns)
        remaining = n_rows
        for batch in self.batches(df, batch_size):
            out = batch
            if with_columns is not None:
                out = out.select(with_columns)
            if predicate is not None:
                out = out.filter(predicate)
            if remaining is not None:
                out = out.head(remaining)
                remaining -= out.height
            yield out
            if remaining == 0:
                return

    @staticmethod
    def batches(df: pl.DataFrame, batch_size: int | None) -> Iterator[pl.DataFrame]:
        if batch_size is None or batch_size >= df.height:
            yield df
            return
        for offset in range(0, df.height, batch_size):
            yield df.slice(offset, batch_size)

    # -- declaration -----------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Not METADATA: a projection or a filter builds a new frame, so chunk layout and
            # sortedness flags are properties of this scan rather than of the stored data.
            strictness=Strictness.ROW_ORDER,
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
        )
