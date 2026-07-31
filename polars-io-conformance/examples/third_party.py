"""Harnesses for IO plugins that are not this project.

The suite claims to work against *any* Polars IO plugin. These are the evidence: three
third-party plugins, none of which know this suite exists, each wired up in about fifteen lines.

    pip install polars-avro polars-fastavro deltalake
    python -m plioc --html report.html \\
        examples.third_party:AvroHarness \\
        examples.third_party:FastAvroHarness \\
        examples.third_party:DeltaHarness

Every declaration below was written from a run, not from a specification. The first run of each
harness declared nothing, failed a few hundred checks, and each limit was added only once a
one-column probe had shown what the format actually does with that dtype. That is the workflow
the suite is for, and it is why these are worth reading before writing your own.

One declaration that is deliberately absent: an over-declaration copied from one
Avro plugin to the other was reported as a *stale declaration* and removed, which is the whole
point of that mechanism.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl

from plioc import equality
from plioc.equality import Strictness
from plioc.harness import BaseHarness, Capabilities, Target

#: Avro has `int` (32-bit) and `long` (64-bit) and no unsigned types at all, so Polars' narrower
#: integers come back widened. Measured, not assumed -- both Avro plugins agree on this mapping.
_AVRO_INT_WIDENING = {
    pl.Int8(): pl.Int32(),
    pl.Int16(): pl.Int32(),
    pl.UInt8(): pl.Int32(),
    pl.UInt16(): pl.Int32(),
    pl.UInt32(): pl.Int64(),
}

#: Every one of these formats stores a variable-length array; the fixed width is a Polars-side
#: guarantee that nothing on disk records.
_AVRO_NORMALIZE = equality.compose(
    equality.widen_ints_to(_AVRO_INT_WIDENING),
    equality.array_to_list,
    equality.timezone_to_utc,
)


#: `polars-avro` mangles a column name to fit Avro's `[A-Za-z_][A-Za-z0-9_]*` identifier rule --
#: `""` becomes `"_"`, `"with.dot"` becomes `"with_dot"`, `"café"` becomes `"caf_"` -- and the
#: write reports success. The scan then looks for the original name and fails. Renaming to fit the
#: format is defensible; doing it silently and then being unable to read the result back is the
#: finding, and it is the same defect for every name in the list.
_AVRO_NAME_FAILURES = {
    f"names/single/{i}": "the writer silently mangles the column name and the scan cannot find it"
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16)
}
_AVRO_NAME_FAILURES["names/all"] = "contains the names above"

#: Parquet cannot represent a zero-field struct, and both Avro and Delta inherit that here -- the
#: reference `parquet` column in the report shows the same cell, which is the point of having it.
_EMPTY_STRUCT = {"nested/struct_empty": "the format cannot represent a struct with no fields"}


class _DirectoryHarness(BaseHarness):
    """Common plumbing: one file or table per case, under a root the caller supplies."""

    suffix = ""

    def __init__(self, root: Path) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def target(self, name: str) -> Target:
        return self.root / f"{name.replace('/', '_')}{self.suffix}"


class AvroHarness(_DirectoryHarness):
    """`polars-avro` -- a Rust IO plugin over the `apache-avro` crate.

    https://github.com/hafaio/polars-avro
    """

    name = "polars-avro"
    suffix = ".avro"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        from polars_avro import write_avro

        write_avro(lf.collect(), target)

    def scan(self, target: Target) -> pl.LazyFrame:
        from polars_avro import scan_avro

        return scan_avro(target)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # Avro carries no chunk layout and no Polars sortedness flag, so `PHYSICAL` and
            # `METADATA` are not claimable by construction.
            strictness=Strictness.ROW_ORDER,
            # Absent, all measured with a one-column round-trip: `UInt64` (no unsigned `long`, and
            # the writer refuses values past `i64::MAX`), `Int128`, `Categorical`/`Enum` (the
            # writer rejects an Arrow dictionary outright), `Duration` (comes back as a bare
            # `Int64`), and `Time` (Avro's time-micros truncates Polars' nanoseconds).
            dtypes=frozenset(
                {
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.Float32,
                    pl.Float64,
                    pl.Boolean,
                    pl.String,
                    pl.Binary,
                    pl.Null,
                    pl.Decimal,
                    pl.Date,
                    pl.Datetime,
                    pl.List,
                    pl.Array,
                    pl.Struct,
                }
            ),
            normalize=_AVRO_NORMALIZE,  # type: ignore[arg-type]
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
            known_failures={**_AVRO_NAME_FAILURES, **_EMPTY_STRUCT},
        )


class FastAvroHarness(_DirectoryHarness):
    """`polars-fastavro` -- the same format through a pure-Python `fastavro` bridge.

    Interesting precisely because it targets the same format by a different route: where the two
    disagree, the difference is the implementation rather than Avro.

    https://pypi.org/project/polars-fastavro/
    """

    name = "polars-fastavro"
    suffix = ".avro"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        from polars_fastavro import write_avro

        write_avro(lf.collect(), target)

    def scan(self, target: Target) -> pl.LazyFrame:
        from polars_fastavro import scan_avro

        return scan_avro(target)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            strictness=Strictness.ROW_ORDER,
            # A strictly smaller set than `polars-avro`'s, for the same format: this bridge also
            # refuses `Decimal`, `Time`, a tz-aware `Datetime`, and `Duration`. Where two plugins
            # over one format disagree, the difference is the implementation, not Avro.
            dtypes=frozenset(
                {
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.UInt8,
                    pl.UInt16,
                    pl.UInt32,
                    pl.Float32,
                    pl.Float64,
                    pl.Boolean,
                    pl.String,
                    pl.Binary,
                    pl.Null,
                    pl.Date,
                    pl.Datetime,
                    pl.List,
                    pl.Array,
                    pl.Struct,
                }
            ),
            normalize=_AVRO_NORMALIZE,  # type: ignore[arg-type]
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
            known_failures={
                # No name failures here, and that is the finding: the pure-Python bridge writes
                # every awkward column name in the corpus and reads it back, where the Rust plugin
                # over the same format mangles them. Declaring them by copy-paste is exactly what
                # the stale-declaration ratchet caught.
                #
                # `Capabilities.dtypes` is keyed by dtype *class*, so "supports `Datetime(us)`
                # but not `Datetime(ns)`, and no time zone at all" is not expressible there and
                # has to be declared case by case.
                **{
                    f"temporal/datetime/ns/{z}": "the fastavro bridge rejects nanosecond timestamps"
                    for z in ("naive", "UTC", "America/New_York", "Asia/Kolkata")
                },
                **{
                    f"temporal/datetime/us/{z}": "the fastavro bridge rejects a tz-aware timestamp"
                    for z in ("America/New_York", "Asia/Kolkata")
                },
                **{
                    f"temporal/dst/{z}": "the fastavro bridge rejects a tz-aware timestamp"
                    for z in ("America/New_York", "Europe/London")
                },
            },
        )


class DeltaHarness(_DirectoryHarness):
    """Delta Lake through `deltalake` and Polars' own `scan_delta`.

    A table format rather than a file format -- the write goes through Arrow and a transaction
    log, so its losses are a different shape from the Avro ones.

    https://delta-io.github.io/delta-rs/
    """

    name = "delta"
    suffix = ".delta"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        # The directory is removed first rather than overwritten. A Delta overwrite replaces the
        # data and keeps the table's *schema*, so writing a frame whose dtypes differ from an
        # earlier write to the same target -- which is exactly what a fixpoint check does, since
        # the format normalised them on the way in -- is rejected against a schema the format
        # itself produced. `sink` means "this target now holds this frame", which for a table
        # format is a new table.
        if target.exists():
            shutil.rmtree(target)
        lf.collect().write_delta(str(target))

    def scan(self, target: Target) -> pl.LazyFrame:
        return pl.scan_delta(str(target))

    def capabilities(self) -> Capabilities:
        return Capabilities(
            strictness=Strictness.ROW_ORDER,
            # Delta's type system is Spark's: signed integers only, no `Null`, no `Time`, no
            # `Duration`, no fixed-size list, no dictionary, no 128-bit integer. Unlike Avro it
            # does keep the narrow signed widths.
            dtypes=frozenset(
                {
                    pl.Int8,
                    pl.Int16,
                    pl.Int32,
                    pl.Int64,
                    pl.Float32,
                    pl.Float64,
                    pl.Boolean,
                    pl.String,
                    pl.Binary,
                    pl.Decimal,
                    pl.Date,
                    pl.Datetime,
                    pl.List,
                    pl.Struct,
                }
            ),
            # Timestamps are normalised to microseconds in UTC on the way in, which is Delta's
            # `timestamp` type doing exactly what it says.
            normalize=equality.compose(  # type: ignore[arg-type]
                equality.timezone_to_utc,
                equality.time_unit_to("us"),
            ),
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
            known_failures={
                **_EMPTY_STRUCT,
                "names/all": (
                    "Delta compares field names case-insensitively, so a schema holding both "
                    "'select' and 'SELECT' is rejected as a duplicate"
                ),
                "shape/no_columns": "a Delta table must define at least one column",
            },
        )
