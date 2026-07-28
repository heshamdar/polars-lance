"""polars-lance vs. reading Lance through PyArrow.

A Lance dataset can already be read into Polars without this package, by handing
`LanceDataset.scanner(...)` to `pl.scan_pyarrow_dataset` or `pl.from_arrow`. This script
runs the same queries both ways and prints what differs: which work at all, what gets
pushed into Lance, how much is read, and whether nulls survive a round trip.

Run it with:

    uv run python examples/compare_with_pyarrow.py

It writes a few hundred MB to a temporary directory and deletes it on the way out.
"""

from __future__ import annotations

import re
import resource
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import lance
import polars as pl

import polars_lance
from polars_lance import scan_lance, sink_lance, write_lance

ROWS = 2_000_000
PAYLOAD_COLUMNS = 6


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "─" * len(title))


@contextmanager
def measured(label: str) -> Iterator[None]:
    start = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    yield
    elapsed = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = after / (1024 * 1024)  # bytes on macOS
    print(
        f"    {label:<34} {elapsed:6.2f}s   peak RSS {peak_mb:6.0f} MB"
        + ("  (new high)" if after > before else "")
    )


def build_dataset(path: Path) -> None:
    """Write a wide dataset by streaming, so it is never held in memory."""
    payload = {
        f"pay{index}": pl.format(
            "value-{}-{}", pl.lit(index), pl.int_range(ROWS) % 1000
        )
        for index in range(PAYLOAD_COLUMNS)
    }
    query = pl.LazyFrame({"id": pl.int_range(ROWS, eager=True)}).with_columns(
        grp=pl.col("id") % 10,
        **payload,
    )

    with measured("sink_lance (streaming write)"):
        sink_lance(query, path, mode="overwrite", max_rows_per_file=200_000)


def lance_rows_scanned(path: Path, sql: str | None, columns: list[str]) -> str:
    """Ask Lance how much it actually read for a filter."""
    plan = lance.dataset(path).scanner(filter=sql, columns=columns).analyze_plan()
    line = next(line for line in plan.splitlines() if "LanceRead" in line)
    metrics = dict(
        pair.split("=", 1)
        for pair in (re.search(r"metrics=\[(.*?)\]", line) or re.match("", ""))
        .group(1)
        .split(", ")
        if "=" in pair
    )
    return f"rows_scanned={metrics.get('rows_scanned', '?'):>9} bytes_read={metrics.get('bytes_read', '?'):>10}"


def pyarrow_scan(path: Path) -> pl.LazyFrame:
    """The route available without this package."""
    return pl.scan_pyarrow_dataset(lance.dataset(path))


def main() -> None:
    directory = Path(tempfile.mkdtemp(prefix="polars-lance-example-"))
    path = directory / "example.lance"

    try:
        rule(f"1. Write {ROWS:,} rows x {PAYLOAD_COLUMNS + 2} columns")
        build_dataset(path)
        dataset = lance.dataset(path)
        print(
            f"    rows={dataset.count_rows():,}  fragments={len(dataset.get_fragments())}"
        )
        print(f"    storage version={dataset.data_storage_version}")

        rule("2. Same query, both routes")
        predicate = (pl.col("grp") == 3) & (pl.col("id") > ROWS - 1000)
        # Include payload columns: Lance then only reads them for matching rows.
        selected = ["id", "grp", "pay0", "pay1", "pay2"]

        with measured("scan_lance"):
            ours = scan_lance(path).filter(predicate).select(selected).collect()
        with measured("scan_pyarrow_dataset"):
            theirs = pyarrow_scan(path).filter(predicate).select(selected).collect()

        print(f"    same result: {ours.equals(theirs)}  ({ours.height} rows)")

        rule("3. What reaches Lance")
        # Capture the SQL scan_lance pushes down.
        pushed: list[str | None] = []
        translate = polars_lance.to_lance_sql
        polars_lance.to_lance_sql = lambda expr, schema=None: (
            lambda sql: (pushed.append(sql), sql)[1]
        )(translate(expr, schema))
        try:
            scan_lance(path).filter(predicate).select(selected).collect()
        finally:
            polars_lance.to_lance_sql = translate

        print(f"    scan_lance pushes : {pushed[-1]}")
        print(
            f"      with filter    -> {lance_rows_scanned(path, pushed[-1], selected)}"
        )
        print(f"      no filter      -> {lance_rows_scanned(path, None, selected)}")
        print(
            "    Lance still scans the filter column, but only reads the payload columns for\n"
            "    matching rows, so bytes_read drops sharply. A scalar index would cut\n"
            "    rows_scanned too."
        )
        print(
            "\n    scan_pyarrow_dataset receives the predicate as a PyArrow expression and\n"
            "    filters in PyArrow, so Lance itself sees no filter."
        )

        rule("4. Predicates the SQL translation covers")
        for label, expr in [
            ("comparison", pl.col("id") > ROWS - 10),
            ("AND", (pl.col("grp") == 3) & (pl.col("id") > 100)),
            ("is_in", pl.col("grp").is_in([1, 2])),
            ("str.contains", pl.col("pay0").str.contains("value-0-99")),
            ("str.starts_with", pl.col("pay0").str.starts_with("value-0-1")),
            ("is_null", pl.col("grp").is_null()),
        ]:
            sql = translate(
                _delivered(path, expr), pl.Schema(scan_lance(path).collect_schema())
            )
            print(f"    {label:<18} {sql if sql else '(applied by Polars)'}")

        rule("5. Null structs survive a round trip")
        struct_path = directory / "structs.lance"
        frame = pl.DataFrame(
            {"id": [1, 2, 3], "st": [{"x": 1}, None, {"x": None}]},
            schema={"id": pl.Int64, "st": pl.Struct({"x": pl.Int64})},
        )
        write_lance(frame, struct_path)
        print(f"    written        : {frame['st'].to_list()}")
        print(
            f"    scan_lance     : {scan_lance(struct_path).collect()['st'].to_list()}"
        )
        print(
            f"    via pyarrow    : {pyarrow_scan(struct_path).collect()['st'].to_list()}"
        )
        print(
            "    A null struct stays null because the dataset is written with data storage\n"
            f"    version {lance.dataset(struct_path).data_storage_version}; version 2.0"
            " turns it into a struct of filler values."
        )

        rule("6. Memory: streaming vs collecting")
        print("    Peak RSS of a fresh process writing the same 2M rows each way:")
        for label, call in [
            ("sink_lance (streams)", "sink_lance(query, target)"),
            ("write_lance (collects)", "write_lance(query.collect(), target)"),
        ]:
            peak = _peak_rss_of(call, path, directory / f"{label.split()[0]}.lance")
            print(f"    {label:<34} peak RSS {peak:6.0f} MB")
        print(
            "    Streaming holds one batch at a time, so its peak does not grow with the\n"
            "    number of rows; collecting has to hold the whole result."
        )

    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _peak_rss_of(call: str, source: Path, target: Path) -> float:
    """Run one write in a fresh process and report its peak RSS in MB."""
    script = textwrap.dedent(f"""
        import resource
        from polars_lance import scan_lance, sink_lance, write_lance

        query = scan_lance({str(source)!r}).select(["id", "grp", "pay0"])
        target = {str(target)!r}
        {call}
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024))
    """)
    output = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    return float(output.stdout.strip().splitlines()[-1])


def _delivered(path: Path, predicate: pl.Expr) -> pl.Expr:
    """Get the predicate as Polars hands it to the scan."""
    captured: list[pl.Expr | None] = []
    translate = polars_lance.to_lance_sql

    def spy(expr: pl.Expr, schema: object = None) -> str | None:
        captured.append(expr)
        return translate(expr, schema)  # type: ignore[arg-type]

    polars_lance.to_lance_sql = spy
    try:
        scan_lance(path).filter(predicate).select(["id"]).head(1).collect()
    finally:
        polars_lance.to_lance_sql = translate

    assert captured and captured[-1] is not None, (
        "predicate was not pushed into the scan"
    )
    return captured[-1]


if __name__ == "__main__":
    main()
