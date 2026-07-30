"""`python -m plioc` -- run the suite and write a report.

    python -m plioc --html report.html --json report.json my.module:MyHarness

Harnesses are named `module:attr`, where `attr` is a callable taking a root directory and
returning an `IOHarness`. The reference harnesses (memory, plugin, parquet, ipc) are always
included and always come first, so the plugin under test is the last column -- and the one the
report's headline numbers are about.

Exit status is 1 when the plugin under test has an undeclared failure or a stale declaration, so
this is usable as a CI gate on its own without pytest.
"""

from __future__ import annotations

import argparse
import inspect
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory

from plioc import html, report, run
from plioc.harness import IOHarness


def _load(spec: str, root: Path) -> IOHarness:
    """Resolve `module:attr` to a harness.

    The root directory is passed only if the factory takes an argument, so a purely in-memory
    harness can be named here too.
    """
    module, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"expected module:attr, got {spec!r}")
    factory = getattr(import_module(module), attr)
    takes_root = bool(
        [
            p
            for p in inspect.signature(factory).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    )
    target = root / attr.lower()
    if takes_root:
        target.mkdir(parents=True, exist_ok=True)
        return factory(target)  # type: ignore[no-any-return]
    return factory()  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plioc", description=__doc__)
    parser.add_argument("harness", nargs="*", help="module:attr factories to test")
    parser.add_argument("--html", type=Path, default=Path("conformance-report.html"))
    parser.add_argument("--json", type=Path, default=None, help="also write the run as JSON")
    parser.add_argument(
        "--markdown", type=Path, default=None, help="also write the round-trip capability matrix"
    )
    parser.add_argument("--include-slow", action="store_true", help="run the heavyweight cases")
    parser.add_argument("--title", default="IO plugin conformance report")
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="omit memory/plugin/parquet/ipc; loses the is-it-me-or-the-format comparison",
    )
    args = parser.parse_args(argv)

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        harnesses: list[IOHarness] = [] if args.no_reference else _reference(root)
        harnesses += [_load(spec, root) for spec in args.harness]
        if not harnesses:
            parser.error("nothing to run: pass a module:attr harness or drop --no-reference")

        result = run.run_suite(harnesses, include_slow=args.include_slow)
        args.html.parent.mkdir(parents=True, exist_ok=True)
        args.html.write_text(html.document(result, title=args.title), encoding="utf-8")
        print(f"wrote {args.html}")
        if args.json:
            args.json.write_text(html.to_json(result), encoding="utf-8")
            print(f"wrote {args.json}")
        if args.markdown:
            args.markdown.write_text(report.render(harnesses), encoding="utf-8")
            print(f"wrote {args.markdown}")

    subject = result.runs[-1]
    for check in subject.failures:
        print(f"{subject.name}: {check.contract} {check.subject}: {check.detail}", file=sys.stderr)
    print(
        f"{subject.name}: "
        + ", ".join(f"{subject.count(s)} {s.value}" for s in run.Status if subject.count(s))
    )
    return 1 if subject.failures else 0


def _reference(root: Path) -> list[IOHarness]:
    from plioc.harnesses.files import IpcHarness, ParquetHarness
    from plioc.harnesses.memory import MemoryHarness
    from plioc.harnesses.plugin import PluginHarness

    return [
        MemoryHarness(),
        PluginHarness(),
        ParquetHarness(root / "parquet"),
        IpcHarness(root / "ipc"),
    ]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
