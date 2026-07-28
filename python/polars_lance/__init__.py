from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import polars as pl
from polars.io.plugins import register_io_source

from polars_lance import _polars_lance
from polars_lance._predicate import to_lance_sql

__all__ = ["scan_lance", "sink_lance", "write_lance"]


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
    reader = _polars_lance.LanceReader(
        uri=str(source),
        storage_options=storage_options,
    )
    schema = reader.schema()

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        if n_rows == 0:
            return

        # The translated filter only reduces the rows Lance reads; it may match a superset
        # of the predicate, so the predicate is applied to every batch below. Polars does
        # not re-apply it for IO plugins.
        lance_scanner = reader.scanner(
            with_columns=with_columns,
            filter=to_lance_sql(predicate, schema) if predicate is not None else None,
            # A limit can only be pushed down if no rows are dropped afterwards. With a
            # predicate, rows are filtered below, so `n_rows` is honored there instead.
            n_rows=n_rows if predicate is None else None,
            batch_size=batch_size,
        )

        remaining_rows = n_rows

        while (df := lance_scanner.next()) is not None:
            if predicate is not None:
                df = df.filter(predicate)

            if remaining_rows is not None:
                df = df.head(remaining_rows)
                remaining_rows -= df.height

            yield df

            if remaining_rows == 0:
                return

    return register_io_source(io_source=io_source, schema=schema)


def write_lance(
    df: pl.DataFrame,
    target: str | Path,
    *,
    mode: Literal["error", "append", "overwrite"] = "error",
    storage_options: dict[str, str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    data_storage_version: str | None = None,
    blob_columns: list[str] | None = None,
) -> None:
    """
    Write dataframe to a Lance dataset.

    To write the result of a query without collecting it into memory first, use
    [`sink_lance`][polars_lance.sink_lance].

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
    data_storage_version : {'2.2', '2.1', '2.0'}
        Lance file format version to write a new dataset with. Defaults to `2.2`, the newest
        version Lance calls stable, which is ahead of its own default of `2.1`. Earlier
        versions lose information: `2.0` does not record the validity of a struct, so a null
        struct reads back as a struct of filler values, and neither `2.0` nor `2.1` can store
        a blob column's nulls, so `blob_columns` is refused with them. Ignored when
        appending, which keeps the existing dataset's version.
    blob_columns
        Names of binary columns to store out of line, so that a scan not selecting them
        never reads the bytes. Each has to be a binary column of the frame. Needs
        `data_storage_version` `2.2` or later, which is the default.

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
        max_rows_per_file=max_rows_per_file,
        max_bytes_per_file=max_bytes_per_file,
        data_storage_version=data_storage_version,
        blob_columns=blob_columns,
    )


def sink_lance(
    lf: pl.LazyFrame,
    target: str | Path,
    *,
    mode: Literal["error", "append", "overwrite"] = "error",
    storage_options: dict[str, str] | None = None,
    max_rows_per_file: int | None = None,
    max_bytes_per_file: int | None = None,
    chunk_size: int | None = None,
    data_storage_version: str | None = None,
    blob_columns: list[str] | None = None,
) -> None:
    """
    Stream a lazy query into a Lance dataset.

    The query runs on Polars' streaming engine and each batch is written as it is produced,
    so peak memory does not grow with the number of rows and a result larger than memory
    can be written. To write a dataframe that is already in memory, use
    [`write_lance`][polars_lance.write_lance].

    Parameters
    ----------
    lf
        Lazy frame to execute and write.
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
        Maximum number of rows to write before starting a new data file. Also bounds how
        many rows the writer buffers.
    max_bytes_per_file
        Maximum number of bytes to write before starting a new data file. This is a soft
        limit that is checked after a group is written, meaning that the actual file
        size may exceed this limit.
    chunk_size
        Number of rows per batch to request from the streaming engine.
    data_storage_version : {'2.2', '2.1', '2.0'}
        Lance file format version to write a new dataset with. Defaults to `2.2`, the newest
        version Lance calls stable, which is ahead of its own default of `2.1`. Earlier
        versions lose information: `2.0` does not record the validity of a struct, so a null
        struct reads back as a struct of filler values, and neither `2.0` nor `2.1` can store
        a blob column's nulls, so `blob_columns` is refused with them. Ignored when
        appending, which keeps the existing dataset's version.
    blob_columns
        Names of binary columns to store out of line, so that a scan not selecting them
        never reads the bytes. Each has to be a binary column of the frame. Needs
        `data_storage_version` `2.2` or later, which is the default.

    Examples
    --------
    Stream a query into a Lance dataset without collecting it first.

    >>> lf = pl.scan_parquet("large.parquet").filter(pl.col("id") > 1000)
    >>> sink_lance(lf, "example.lance")
    """
    # The batches are pulled by the writer, so only one is held at a time. An empty frame
    # carries the schema, which Lance needs before the first batch arrives.
    _polars_lance.write_lance_stream(
        lf.collect_batches(chunk_size=chunk_size, engine="streaming"),
        pl.DataFrame(schema=lf.collect_schema()),
        target=str(target),
        mode=mode,
        storage_options=storage_options,
        max_rows_per_file=max_rows_per_file,
        max_bytes_per_file=max_bytes_per_file,
        data_storage_version=data_storage_version,
        blob_columns=blob_columns,
    )
