from __future__ import annotations

from collections.abc import Iterator
from typing import Literal

import polars as pl

class LanceReader:
    def __init__(
        self,
        uri: str,
        storage_options: dict[str, str] | None = None,
    ) -> None: ...
    def schema(self) -> dict[str, pl.DataType]: ...
    def scanner(
        self,
        with_columns: list[str] | None = None,
        filter: str | None = None,
        n_rows: int | None = None,
        batch_size: int | None = None,
    ) -> LanceScanner: ...

class LanceScanner:
    def next(self) -> pl.DataFrame | None: ...

def write_lance_stream(
    dataframes: Iterator[pl.DataFrame],
    schema: pl.DataFrame,
    target: str,
    *,
    mode: str = "error",
    storage_options: dict[str, str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    data_storage_version: str | None = None,
    blob_columns: list[str] | None = None,
) -> None: ...
def write_lance(
    df: pl.DataFrame,
    target: str,
    *,
    mode: Literal["error", "append", "overwrite"] = "error",
    storage_options: dict[str, str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    data_storage_version: str | None = None,
    blob_columns: list[str] | None = None,
) -> None: ...
