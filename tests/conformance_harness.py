"""`plioc` harness for this plugin.

Everything the conformance suite needs to drive `sink_lance`/`scan_lance`, plus this plugin's
declaration of what it does and does not preserve. Kept out of the test module so that
`plioc.report` can import it without pytest.

Every `known_failures` entry is a strict xfail: when one of these is fixed upstream,
`tests/test_conformance.py` fails until the entry is removed. That is the point -- it is what
stops this list from quietly outliving the bugs it describes.
"""

from pathlib import Path

import polars as pl
from plioc.equality import Strictness
from plioc.harness import BaseHarness, Capabilities, Target

from polars_lance import scan_lance, sink_lance

#: Dtypes this plugin claims. Cases using anything else skip rather than fail.
#:
#: `Int128` is absent: Polars exports it over the Arrow C data interface as the extension type
#: `_pli128`, which arrow-rs rejects outright ("The datatype "_pli128" is still not supported in
#: Rust implementation"), so neither `arrow.rs` nor Lance ever sees an integer.
#:
#: `Enum` is absent for a different reason -- the write succeeds and the *read* fails. See the
#: note under `known_failures`. `Categorical` round-trips and is claimed.
SUPPORTED_DTYPES: frozenset[type[pl.DataType]] = frozenset(
    {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
        pl.Boolean,
        pl.String,
        pl.Binary,
        pl.Null,
        pl.Date,
        pl.Datetime,
        pl.Time,
        pl.Duration,
        pl.Decimal,
        pl.Categorical,
        pl.List,
        pl.Array,
        pl.Struct,
    }
)

# A fixed-size-list column panics in the Arrow bridge ("the offset of the new Buffer cannot exceed
# the existing length") against Polars 1.40 through 1.42, and round-trips against 1.43. The
# declaration is gated on the measured boundary rather than asserted for every version, because a
# strict xfail that is wrong in either direction is worse than none.
_ARRAY_ROUNDTRIPS = tuple(int(p) for p in pl.__version__.split(".")[:2]) >= (1, 43)

_ARRAY_PANIC = (
    "the Arrow bridge panics on a fixed-size-list column ('the offset of the new Buffer "
    f"cannot exceed the existing length') with Polars {pl.__version__}; fixed in 1.43"
)

KNOWN_FAILURES: dict[str, str] = {
    # Column names Lance cannot address. The write is accepted for the first and third, and the
    # scan then cannot resolve the field, so the failure surfaces as a read error rather than as a
    # rejected write.
    "names/single/0": "Lance cannot project a field whose name is the empty string",
    "names/single/4": "Lance rejects `.` in a top-level field name (it means struct nesting)",
    "names/single/8": "Lance cannot project a field whose name contains a backtick",
    "names/all": "contains the names above",
    # The one that matters. `col == 0.0` is true for both `0.0` and `-0.0` in Polars, and the
    # translated filter `(v = -0.0)` distinguishes them -- so the pushed filter matches a *subset*
    # of the predicate and rows Polars would have kept never reach it. `_predicate.py` already
    # refuses to translate NaN comparisons for the same reason; a zero literal needs the same
    # treatment.
    "nan/negative_zero": "signed-zero comparison translates to a subset, not a superset",
}

if not _ARRAY_ROUNDTRIPS:
    KNOWN_FAILURES["nested/array_f32_128"] = _ARRAY_PANIC
    KNOWN_FAILURES["nested/array_nullable_elements"] = _ARRAY_PANIC


class LanceHarness(BaseHarness):
    name = "lance"

    def __init__(self, root: Path, data_storage_version: str | None = None) -> None:
        super().__init__()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_storage_version = data_storage_version

    def target(self, name: str) -> Target:
        return self.root / f"{name.replace('/', '_')}.lance"

    def sink(self, lf: pl.LazyFrame, target: Target) -> None:
        sink_lance(
            lf,
            target,
            mode="overwrite",
            data_storage_version=self.data_storage_version,
        )

    def scan(self, target: Target) -> pl.LazyFrame:
        return scan_lance(target)

    def capabilities(self) -> Capabilities:
        return Capabilities(
            # A Lance scan reassembles its own batches, so the frame's chunk layout is not
            # recoverable and `PHYSICAL` is not claimable. Row order is preserved.
            strictness=Strictness.ROW_ORDER,
            dtypes=SUPPORTED_DTYPES,
            pushdown=frozenset({"projection", "predicate", "limit"}),
            preserves_row_order=True,
            known_failures=KNOWN_FAILURES,
        )

    # `probe()` stays `None`, so the suite degrades its pushdown tests to correctness only and
    # says so out loud. Nothing in `scan_lance` reports what it was handed; threading a
    # rows-read counter out of `LanceScanner` is what would let the engagement assertions run.
