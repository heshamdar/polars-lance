from collections.abc import Iterator, Mapping
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

import polars as pl
import pyarrow as pa
from polars.datatypes import DataTypeClass
from polars.io.plugins import register_io_source
from testcontainers.azurite import AzuriteContainer
from testcontainers.minio import MinioContainer

from polars_lance import _polars_lance

# Lance guards parts of its 2.1 encoder with `debug_assert!`s that a nullable nested column can
# trip (lance-format/lance#8032, #8033). A release build skips them and writes the same data
# correctly, so what these tests can cover depends on how the extension was compiled: `just
# develop` builds debug, released wheels are release builds.
DEBUG_ASSERTIONS: bool = _polars_lance._debug_assertions

# we exclude some types because they are currently not supported by Lance and/or Polars
# all data types: https://arrow.apache.org/docs/python/api/datatypes.html
# excluded types: month_day_nano_interval, binary_view, string_view, decimal256,
#   list_view, large_list_view, map_, run_end_encoded, fixed_shape_tensor, union,
#   dense_union, sparse_union, opaque, bool8, uuid, json_
SUPPORTED_DATA_TYPES_COLUMNS = [
    ("null", pa.null(), [None, None]),
    ("bool", pa.bool_(), [True, False]),
    ("int8", pa.int8(), [1, -2]),
    ("int16", pa.int16(), [1, -2]),
    ("int32", pa.int32(), [1, -2]),
    ("int64", pa.int64(), [1, -2]),
    ("uint8", pa.uint8(), [1, 2]),
    ("uint16", pa.uint16(), [1, 2]),
    ("uint32", pa.uint32(), [1, 2]),
    ("uint64", pa.uint64(), [1, 2]),
    ("float16", pa.float16(), [1.5, -2.5]),
    ("float32", pa.float32(), [1.5, -2.5]),
    ("float64", pa.float64(), [1.5, -2.5]),
    ("date32", pa.date32(), [date(2024, 1, 1), date(2024, 1, 2)]),
    ("date64", pa.date64(), [date(2024, 1, 1), date(2024, 1, 2)]),
    ("time32_s", pa.time32("s"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time32_ms", pa.time32("ms"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time64_us", pa.time64("us"), [time(1, 2, 3), time(4, 5, 6)]),
    ("time64_ns", pa.time64("ns"), [time(1, 2, 3), time(4, 5, 6)]),
    (
        "timestamp_s",
        pa.timestamp("s"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_ms",
        pa.timestamp("ms"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_us",
        pa.timestamp("us"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_ns",
        pa.timestamp("ns"),
        [datetime(2024, 1, 1, 1, 2, 3), datetime(2024, 1, 2, 4, 5, 6)],
    ),
    (
        "timestamp_tz",
        pa.timestamp("us", tz="UTC"),
        [
            datetime(2024, 1, 1, 1, 2, 3, tzinfo=timezone.utc),
            datetime(2024, 1, 2, 4, 5, 6, tzinfo=timezone.utc),
        ],
    ),
    ("duration_s", pa.duration("s"), [timedelta(seconds=1), timedelta(seconds=2)]),
    (
        "duration_ms",
        pa.duration("ms"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    (
        "duration_us",
        pa.duration("us"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    (
        "duration_ns",
        pa.duration("ns"),
        [timedelta(seconds=1), timedelta(seconds=2)],
    ),
    ("binary_variable_size", pa.binary(), [b"a", b"bb"]),
    ("binary_fixed_size", pa.binary(3), [b"aaa", b"bbb"]),
    ("string", pa.string(), ["a", "bb"]),
    ("utf8", pa.utf8(), ["a", "bb"]),
    ("large_binary", pa.large_binary(), [b"a", b"bb"]),
    ("large_string", pa.large_string(), ["a", "bb"]),
    ("large_utf8", pa.large_utf8(), ["a", "bb"]),
    ("decimal128", pa.decimal128(10, 2), [Decimal("1.23"), Decimal("4.56")]),
    ("list_variable_size_int32", pa.list_(pa.int32()), [[1, 2], [3]]),
    ("list_fixed_size_int32", pa.list_(pa.int32(), 2), [[1, 2], [3, 4]]),
    ("large_list_int32", pa.large_list(pa.int32()), [[1, 2], [3]]),
    (
        "struct",
        pa.struct(
            [
                ("x", pa.int32()),
                (
                    "nested",
                    pa.struct(
                        [
                            ("flag", pa.bool_()),
                            ("values", pa.list_(pa.int32())),
                        ]
                    ),
                ),
            ]
        ),
        [
            {"x": 1, "nested": {"flag": True, "values": [1, 2]}},
            {"x": 2, "nested": {"flag": False, "values": [3]}},
        ],
    ),
    ("dictionary", pa.dictionary(pa.int8(), pa.string()), ["a", "b"]),
]
SUPPORTED_DATA_TYPES_ARROW_TABLE = pa.table(
    {
        name: pa.array(values, type=dtype)
        for name, dtype, values in SUPPORTED_DATA_TYPES_COLUMNS
    }
)


# A small table for exercising predicates. It includes nulls, and values that are only
# matched by part of a predicate, so that filtering mistakes change the result.
FILTER_ARROW_TABLE = pa.table(
    {
        "int32": pa.array([1, 2, 3, 4, 5, None], pa.int32()),
        "float64": pa.array([1.5, 2.5, 3.5, 4.5, 5.5, None], pa.float64()),
        "string": pa.array(["a", "b", "xc", "d", None, "xe"], pa.string()),
        "date32": pa.array(
            [date(2024, 1, day) for day in range(1, 7)],
            pa.date32(),
        ),
    }
)


# Nested data with a null at every level that can hold one: a null list, an empty list, a
# null struct inside a list, a null field inside a struct, and a null list element.
NESTED_ARROW_TABLE = pa.table(
    {
        "id": pa.array([1, 2, 3, 4, 5], pa.int64()),
        "list_of_struct": pa.array(
            [
                [{"x": 1, "name": "a"}, {"x": 2, "name": "b"}],
                [],
                None,
                [None, {"x": 4, "name": None}],
                [{"x": None, "name": "e"}],
            ],
            pa.list_(pa.struct([("x", pa.int64()), ("name", pa.string())])),
        ),
        "struct_of_list": pa.array(
            [
                {"values": [1, 2], "label": "a"},
                {"values": [], "label": "b"},
                None,
                {"values": None, "label": None},
                {"values": [None, 5], "label": "e"},
            ],
            pa.struct([("values", pa.list_(pa.int64())), ("label", pa.string())]),
        ),
        "list_of_list": pa.array(
            [[[1, 2], [3]], [[]], None, [None], [[None]]],
            pa.list_(pa.list_(pa.int64())),
        ),
    }
)


def delivered_predicate(
    predicate: pl.Expr,
    schema: Mapping[str, pl.DataType | DataTypeClass],
) -> pl.Expr:
    """
    Return the predicate as Polars delivers it to an IO plugin.

    Building an expression with the Python API is not enough: the optimizer resolves
    literal dtypes and rewrites some expressions, and the translation runs on the result.
    """
    delivered: list[pl.Expr | None] = []

    def io_source(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        delivered.append(predicate)
        columns = with_columns if with_columns is not None else list(schema)
        yield pl.DataFrame(
            {name: pl.Series([], dtype=schema[name]) for name in columns}
        )

    register_io_source(io_source=io_source, schema=schema).filter(predicate).collect()

    assert delivered, "predicate was not pushed into the scan"
    assert delivered[-1] is not None, "predicate was not pushed into the scan"
    return delivered[-1]


def to_polars_arrow(
    table: pa.Table,
    *,
    with_columns: list[str] | None = None,
    predicate: pl.Expr | None = None,
    n_rows: int | None = None,
) -> pa.Table:
    expected_df = pl.from_arrow(table)
    assert isinstance(expected_df, pl.DataFrame)

    lf = expected_df.lazy()

    if with_columns is not None:
        lf = lf.select(with_columns)

    if predicate is not None:
        lf = lf.filter(predicate)

    if n_rows is not None:
        lf = lf.head(n_rows)

    return lf.collect().to_arrow()


def s3_storage_options(minio_container: MinioContainer) -> dict[str, str]:
    config = minio_container.get_config()
    return {
        "aws_endpoint": f"http://{config['endpoint']}",
        "aws_access_key_id": config["access_key"],
        "aws_secret_access_key": config["secret_key"],
        # Including the region is not strictly necessary, but it makes requests
        # much faster.
        "aws_region": "us-east-1",
        "aws_allow_http": "true",
    }


def az_storage_options(azurite_container: AzuriteContainer) -> dict[str, str]:
    conn_str = azurite_container.get_connection_string()
    endpoint = next(
        v
        for p in conn_str.split(";")
        if "=" in p
        for k, v in [p.split("=", 1)]
        if k == "BlobEndpoint"
    )
    return {
        "azure_endpoint": endpoint,
        "azure_storage_account_name": azurite_container.account_name,
        "azure_storage_account_key": azurite_container.account_key,
        "azure_allow_http": "true",
    }
