# polars-io-conformance

A generative conformance suite any Polars IO plugin can be tested against.

The corpus is a pure function of `(spec, seed)`. It is built entirely from Polars expressions over
a row index, materialised on demand, and never committed as bytes. A case is about ten lines of
data; a failing case shrunk out of Hypothesis is a ten-line JSON file, not a Parquet fixture.

## Using it

```python
import pytest
from plioc import ConformanceSuite, BaseHarness, Capabilities, Strictness

class MyHarness(BaseHarness):
    name = "myformat"

    def __init__(self, root): super().__init__(); self.root = root
    def target(self, name):   return self.root / f"{name.replace('/', '_')}.mine"
    def sink(self, lf, target): my_sink(lf, target)
    def scan(self, target):     return my_scan(target)

    def capabilities(self):
        return Capabilities(
            strictness=Strictness.ROW_ORDER,
            pushdown={"projection", "predicate", "limit"},
            known_failures={"nested/struct_empty": "myformat#1234"},
        )

class TestMine(ConformanceSuite):
    @pytest.fixture
    def harness(self, tmp_path):
        return MyHarness(tmp_path)
```

That is the whole integration. `pytest` then runs ~170 cases and ~95 queries against it.

Implement `probe()` as well if you can — see [Correctness is not engagement](#correctness-is-not-engagement).

## What it checks

| Contract | Question |
|---|---|
| **round-trip** | does everything written come back, at the strictness you claim? |
| **fixpoint** | does a declared loss happen *once*, or on every write? |
| **schema** | does `collect_schema()` agree with the data, without reading any? |
| **query correctness** | for every `(case, query)`, does the result match in-memory Polars? |
| **engagement** | did the plugin *use* the projection, predicate and limit it was pushed? |
| **metamorphic laws** | `count(p) + count(~p) + count(p is null) == n`, and four more |

### Correctness is not engagement

A plugin that ignores every predicate and filters the rows in Python is perfectly correct and
delivers none of the value. A plugin that pushes a predicate it translated wrongly is engaged and
returns the wrong rows. Those are different failures, so they are different assertions, and
neither is ever allowed to stand in for the other.

Engagement needs `probe()`. A harness that returns `None` from it gets a loud skip rather than a
pass, because a pass there would mean nothing.

### The mandate

`register_io_source` hands the predicate over as an **obligation**: Polars does not re-apply it.
A plugin that translates the part of a conjunction it understands, pushes that, and drops the
residual returns silently wrong answers — no exception, no warning, just the wrong rows. The
`mandate/*` queries span fully-translatable to entirely-opaque, in both operand orders and under
both connectives, and `harnesses/mutants.py::DropsResidual` exists to prove they catch it.

## Declaring limits honestly

- `Capabilities.strictness` — how exactly you claim to round-trip, on a six-rung ladder from
  `VALUES` to `METADATA`.
- `Capabilities.normalize` — a function describing a loss (`ns` → `us`, `Categorical` → `String`),
  applied to the expected frame so the suite checks *your declared* behaviour rather than identity.
  Asserted idempotent.
- `Capabilities.known_failures` — `{case_id: reason}`, applied as a **strict** xfail. When an
  upstream fix makes a declared failure pass, CI breaks until the declaration is updated. This is
  the one mechanism that stops a capability matrix from rotting into pessimism, and it works: it
  caught a wrong `Enum`-loss declaration on this suite's own Parquet harness.
- `Capabilities.dtypes` — dtypes you support. Cases outside the set skip rather than fail.

## The report

`python -m plioc` runs every contract and writes a standalone HTML page:

```
python -m plioc --html report.html --json report.json my.module:MyHarness
```

The reference harnesses are always included as columns, so a failure is attributable at a glance:
if Parquet and IPC lose it too, it is the format. Each failure carries the ~10-line spec that
reproduces it, because a conformance failure nobody can re-run gets triaged into a backlog and
forgotten. `--json` gives the same run as data, for diffing two commits.

Exit status is 1 on an undeclared failure or a stale declaration, so this works as a CI gate
without pytest. `plioc.report` alone still renders the round-trip matrix as markdown.

## How the corpus is generated

Six primitives in `gen/core.py`; everything else composes them.

- **splitmix64 in expressions.** Not `Expr.hash`: determinism is load-bearing, and nothing
  documents that hasher's output as fixed. The expression form is asserted bit-identical to a
  Python reference.
- **Palettes weighted toward pathology.** `-0.0`, subnormals, `MIN+1`, `""`, NFC *and* NFD forms of
  the same text, invalid UTF-8, DST-ambiguous instants, `';DROP TABLE t;--`. Roughly 60% of values
  come from a palette and 40% from ordinary bulk fill, because a corpus of nothing but edge cases
  lets bulk-path bugs hide.
- **Null placement as its own axis**, orthogonal to dtype: `ALTERNATING` and `BOUNDARY` are the two
  that find real bugs, so do not skip them.
- **Nesting by composition.** `ListGen(child=StructGen(...))` recurses, so `List<Struct<List<T>>>`
  is free, and the three-way distinction between a null list, an empty list, and a list of nulls
  is three independent knobs.

Four properties are asserted in CI, because each is a property fixture files cannot have:

1. **Determinism** — `build(spec)` is byte-identical across runs, processes and machines.
2. **Prefix stability** — `build(n=10) == build(n=10_000).head(10)`, so a fast CI run is a literal
   prefix of a nightly one. (`NullPattern.LAST` is the one documented exception.)
3. **Stream independence** — adding a column to a case does not perturb the others.
4. **Scale-freedom** — the same spec runs at 3 rows and at 50 million, fully lazily.

## Evidence that "any plugin" is not just a claim

`examples/third_party.py` wires up three plugins this project did not write -- `polars-avro`
(Rust), `polars-fastavro` (the same format through a Python bridge), and Delta Lake via
`deltalake` -- in about fifteen lines each. Running them found four bugs in *this suite* (see
PLAN-REVIEW.md, "What running it against somebody else's plugin changed") and a pile of real,
attributable limits in them, including a writer that silently mangles a column name to fit the
format's identifier rules and then cannot read the file back.

## The suite tests itself

`MemoryHarness` is the identity and must pass everything at `METADATA`; a case it fails is the
suite's bug, not a harness's. `ParquetHarness` and `IpcHarness` are real formats whose declared
losses tell a plugin author what is format-inherent.

`harnesses/mutants.py` holds harnesses with exactly one defect each — drops the predicate, drops
the residual, ignores the projection, reorders rows, loses the sign of zero, conflates an empty
list with a null one, stores a null struct as a struct of nulls. `tests/test_mutants.py` asserts
**every mutant is caught by at least one named case** and prints which one. That is this suite's
real coverage metric; line coverage of a generator says nothing about whether its corpus finds
bugs. It is not the converse: plenty of worthwhile cases catch no mutant, because a format that
cannot store `Decimal(38, 38)` is not a harness with a bug.

One mutant is asserted to *survive*. `LimitBeforeFilter` models a defect Polars cannot expose —
measured, `n_rows` is never delivered alongside a predicate — and pinning that down means a change
in Polars' pushdown shows up as a test failure instead of as a silent gap.

## Layout

```
src/plioc/
  spec.py        CaseSpec / ColumnSpec: the corpus, as data
  gen/           core (6 primitives), palettes, primitive, temporal, categorical, nested, layout
  corpus.py      curated cases over parameter grids
  query.py       predicate AST + QuerySpec
  queries.py     the curated adversarial query matrix
  harness.py     IOHarness, Capabilities, Probe -- what a plugin author implements
  harnesses/     memory (control), plugin (probeable control), files (parquet/ipc), mutants
  equality.py    the strictness ladder and the normalisers
  oracle.py      the runners: round-trip, fixpoint, differential, metamorphic
  strategies.py  Hypothesis, over specs -- never over data
  codec.py       specs to text and back
  regressions/   committed counterexamples
  selection.py   which cases and queries apply to a harness (shared by both surfaces below)
  run.py         running the suite outside pytest, as a value
  html.py        that value as a standalone HTML report
  report.py      the round-trip capability matrix, as markdown
  __main__.py    `python -m plioc`
  suite.py       ConformanceSuite
docs/api-findings.md   what Polars actually does, measured
PLAN-REVIEW.md         what changed from the original design, and why
```

## Development

```
uv sync
uv run pytest                    # the suite's own tests
uv run pytest tests/test_mutants.py -s   # which case catches which defect
```

`pytest -k "not slow"` is the default; `ConformanceSuite.include_slow = True` turns on the cases
that materialise real volume (1 MiB strings, a 1000-column schema, a 100k-element `is_in`).

Adding a case: put it on exactly one axis, tag it, and give it an id that reads as a path. If it
catches a mutant nothing else catches, say so in a comment. If it does not, that is fine — say
what it is for instead.
