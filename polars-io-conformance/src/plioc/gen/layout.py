"""Physical properties of a materialised frame.

Chunking is not expressible lazily -- it is a property of memory, not of a plan -- so `Layout` is
a transform applied to the collected oracle frame immediately before it is handed to a sink. It
is invisible to `CaseSpec.build()`, and two cases differing only in `Layout` generate identical
values.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Layout:
    #: Explicit chunk sizes. Sizes that do not sum to the row count are cycled; a `0` produces a
    #: zero-length chunk, which several writers mishandle.
    chunks: tuple[int, ...] | None = None
    rechunk: bool = True
    #: Column to mark as sorted, to see whether the flag survives a round-trip. Setting it is a
    #: promise, not a sort: only pass a column the frame is genuinely ordered by.
    sorted_by: str | None = None

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        out = df
        if self.chunks is not None:
            out = _chunked(out, self.chunks)
        elif self.rechunk:
            out = out.rechunk()
        if self.sorted_by is not None:
            out = out.with_columns(pl.col(self.sorted_by).set_sorted())
        return out

    @property
    def is_default(self) -> bool:
        return self.chunks is None and self.rechunk and self.sorted_by is None


DEFAULT = Layout()


def _chunked(df: pl.DataFrame, sizes: tuple[int, ...]) -> pl.DataFrame:
    if not sizes:
        return df
    pieces: list[pl.DataFrame] = []
    offset = 0
    k = 0
    while offset < df.height:
        size = sizes[k % len(sizes)]
        k += 1
        if size == 0:
            pieces.append(df.head(0))
            # A run of zero-sized chunks would never terminate; only emit one per cycle.
            if all(s == 0 for s in sizes):
                break
            continue
        pieces.append(df.slice(offset, size))
        offset += size
    if not pieces:
        return df.head(0)
    return pl.concat(pieces, rechunk=False)


def n_chunks(df: pl.DataFrame) -> int:
    """Chunk count of the frame, taken from its widest column.

    Columns of a frame need not agree on chunking; the maximum is the one that matters for a
    writer, since it is what forces a copy.
    """
    return max((s.n_chunks() for s in df.get_columns()), default=0)
