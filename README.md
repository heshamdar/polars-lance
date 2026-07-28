# polars-lance

Polars plugin for reading Lance datasets into Polars dataframes and writing Polars dataframes to Lance datasets.

## Installation

```bash
pip install polars-lance
```

## Read

```python
from polars_lance import scan_lance

lf = scan_lance("example.lance")
df = lf.collect()
```

### Pushdown

`scan_lance` returns a lazy frame, and Polars pushes work from the query into the scan:

- **Projection**: only the columns the query needs are read.
- **Slice**: `head`/`limit` becomes a Lance limit, so the scan stops early.
- **Predicate**: `filter` is translated into a Lance filter, so Lance evaluates it while
  reading. This matters most for selective filters on wide tables, where Lance only reads
  the other columns for matching rows, and for columns with a
  [scalar index](https://lance.org/guide/indexing/), where it can skip data entirely.

```python
import polars as pl

from polars_lance import scan_lance

# The comparison is evaluated by Lance; only matching rows reach Polars.
df = scan_lance("example.lance").filter(pl.col("id") > 1000).collect()
```

These are translated into a Lance filter:

- comparisons, `AND`/`OR`, `IS NULL`, `is_in`, `is_between`
- string functions: `str.contains` (as a regex, or a substring when `literal=True`),
  `str.starts_with`, `str.ends_with`, `str.to_lowercase`, `str.to_uppercase`,
  `str.len_chars`
- struct fields, e.g. `pl.col("meta").struct.field("id") > 5`
- `list.len` and `list.contains`

Anything else — casts, arithmetic, `list.get` — is applied by Polars instead. A predicate
that mixes both pushes down the part it can, so
`filter((pl.col("id") > 1000) & pl.col("name").str.contains("x"))` still narrows the scan
by `id`. Results are identical either way; pushdown only changes how much data is read.

Polars and SQL disagree in a few places, and those cases are deliberately left to Polars so
that results stay correct:

- Comparing against `NaN`, which Polars treats as equal to itself and SQL does not.
- `is_null` on a struct, or on a field read out of one. Datasets written elsewhere may use
  version 2.0, which does not record a struct's own validity, and the filter and the scan can
  [disagree](https://github.com/lance-format/lance/issues/7908) about the same row.
  Comparisons on a struct field are pushed down; only the null checks are not.
- A time zone aware timestamp literal, and `is_in` on a column narrower than its values.

The translation is checked against both engines on data containing nulls, empty strings and
lists, regex metacharacters, and non-ASCII text — see `tests/test_predicate_equivalence.py`.

## Write

```python
import polars as pl

from polars_lance import write_lance

df = pl.DataFrame({"id": [1, 2], "val": ["a", "b"]})
write_lance(df, "example.lance")
```

Pass `mode` to control behavior in case the target dataset already exists. Valid values are `error` (default), `append`, and `overwrite`.

```python
write_lance(df, "example.lance", mode="append")
```

### Streaming writes

`sink_lance` runs a query with Polars' streaming engine and writes each batch as it is
produced, so a result larger than memory can be written without collecting it. This mirrors
Polars' own naming: `write_*` takes a `DataFrame`, `sink_*` takes a `LazyFrame`.

```python
from polars_lance import scan_lance, sink_lance

lf = pl.scan_parquet("large.parquet").filter(pl.col("id") > 1000)
sink_lance(lf, "example.lance")

# Lance to Lance, without collecting in between.
sink_lance(scan_lance("example.lance").filter(pl.col("id") > 2000), "filtered.lance")
```

Peak memory stays flat as the data grows rather than holding the whole result: writing 8M
and then 16M rows peaked at 383 MB and 417 MB, against 798 MB and 1354 MB when collecting
first. Use `chunk_size` to ask for smaller batches, and `max_rows_per_file` to bound how
much the writer buffers before rolling over to a new file.

### Blob columns

Lance can store a large binary value out of line, so a scan that does not select the column
never reads the bytes. Name the columns to store that way:

```python
write_lance(df, "example.lance", blob_columns=["image"])
```

The marker Lance uses lives in Arrow field metadata, which a Polars schema does not carry, so
it has to be named rather than travelling with the column. Reading needs nothing extra: a scan
asks Lance for the bytes, so the column arrives as the `Binary` column the schema advertises.

Streaming a blob column that contains nulls is not supported — Lance's encoder fails an
internal assertion — so write such a frame with `write_lance` rather than `sink_lance`.

### Nulls in structs

New datasets are written with data storage version 2.1, so a null struct is still null when
read back. Version 2.0, which Lance still writes by default, does not record the validity of
a struct itself and returns a valid struct holding filler values instead.

## Cloud storage

Pass `storage_options` to work with Lance datasets stored in AWS S3, Azure Blob Storage, or Google Cloud Storage.

```python
from polars_lance import scan_lance, write_lance

uri = "s3://my-bucket/example.lance"
storage_options = {
    "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
    "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "aws_region": "us-east-1",
}

write_lance(df, uri, storage_options=storage_options)

lf = scan_lance(uri, storage_options=storage_options)
df = lf.collect()
```

Cloud storage IO is powered by the `object_store` Rust crate. These are the supported schemes and options:
- `s3://`: [AWS options](https://docs.rs/object_store/latest/object_store/aws/enum.AmazonS3ConfigKey.html)
- `az://`: [Azure options](https://docs.rs/object_store/latest/object_store/azure/enum.AzureConfigKey.html)
- `gs://`: [GCP options](https://docs.rs/object_store/latest/object_store/gcp/enum.GoogleConfigKey.html)

## Example

`examples/compare_with_pyarrow.py` runs the same queries through `scan_lance` and through
`pl.scan_pyarrow_dataset`, printing what gets pushed into Lance, how much each reads, and how
the two compare on memory and null handling.

```bash
uv run python examples/compare_with_pyarrow.py
```

## API reference

See the [API reference](https://jorritsandbrink.github.io/polars-lance/polars_lance.html) for full function signatures and parameter details.
