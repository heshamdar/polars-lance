from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import polars as pl
from polars.io.plugins import register_io_source

from polars_lance import _polars_lance

__all__ = ["scan_lance", "write_lance"]


def scan_lance(
    source: str | Path,
    *,
    storage_options: dict[str, str] | None = None,
) -> pl.LazyFrame:
    """
    Lazily read from a Lance dataset.

    Parameters
    ----------
    source
        Path or URI to a Lance dataset.
    storage_options
        Cloud storage configuration to read remote datasets on AWS S3,
        Azure Blob Storage, or Google Cloud Storage. Supported keys:
        - [aws](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html)
        - [azure](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html)
        - [gcp](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html)

    Returns
    -------
    LazyFrame

    Examples
    --------
    Scan a local Lance dataset.

    >>> scan_lance("example.lance")

    Scan a remote Lance dataset on AWS S3.

    >>> source = "s3://bucket/example.lance"
    >>> storage_options = {
    ...     "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    ...     "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ...     "aws_region": "us-east-1",
    ... }
    >>> scan_lance(source, storage_options=storage_options)
    """
    source_str = str(source)

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        lance_scanner = _polars_lance.LanceScanner(
            uri=source_str,
            with_columns=with_columns,
            predicate=predicate,
            n_rows=n_rows,
            batch_size=batch_size,
            storage_options=storage_options,
        )

        while (df := lance_scanner.next()) is not None:
            yield df

    return register_io_source(
        io_source=io_source,
        schema=_polars_lance.LanceScanner.schema_for_uri(
            uri=source_str,
            storage_options=storage_options,
        ),
    )


def write_lance(
    df: pl.DataFrame,
    target: str | Path,
    *,
    mode: Literal["error", "append", "overwrite"] = "error",
    storage_options: dict[str, str] | None = None,
) -> None:
    """
    Write dataframe to a Lance dataset.

    Parameters
    ----------
    df
        Dataframe to write.
    target
        Path or URI to the Lance dataset.
    mode : {'error', 'append', 'overwrite'}
        How to behave if the target dataset already exists.
        - `error`: raise an error
        - `append`: append to the existing dataset
        - `overwrite`: replace the existing dataset
    storage_options
        Cloud storage configuration to write remote datasets on AWS S3,
        Azure Blob Storage, or Google Cloud Storage. Supported keys:
        - [aws](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html)
        - [azure](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html)
        - [gcp](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html)
    max_rows_per_file
        Maximum number of rows to write before starting a new data file.
    max_bytes_per_file
        Maximum number of bytes to write before starting a new data file. This is a soft
        limit that is checked after a group is written, meaning that the actual file
        size may exceed this limit.

    Examples
    --------
    Write a local Lance dataset.

    >>> df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    >>> write_lance(df, "example.lance")

    Write a remote Lance dataset on AWS S3.

    >>> df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    >>> target = "s3://bucket/example.lance"
    >>> storage_options = {
    ...     "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    ...     "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    ...     "aws_region": "us-east-1",
    ... }
    >>> write_lance(df, target, storage_options=storage_options)
    """
    _polars_lance.write_lance(
        df,
        target=str(target),
        mode=mode,
        storage_options=storage_options,
    )
