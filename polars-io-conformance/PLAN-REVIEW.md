# Review of the `polars-io-conformance` build plan

The thesis holds up. A corpus that is a pure function of `(spec, seed)` is the right shape, the
five properties in §0 are the right ones to optimise for, and the mutation-testing discipline in
§4.1 is what separates this from a pile of round-trip assertions. What follows is what changed
and why; everything not listed was implemented as written.

Findings are split into **corrections** (the plan asserts something that is false, or specifies
something that cannot work) and **refinements** (it works, but there is a materially better
version). Measurements behind them are in `docs/api-findings.md`.

---

## Corrections

### C1. `stream = hash(name)` is not deterministic (§3.1)

Python randomises `hash()` of `str` per process unless `PYTHONHASHSEED` is set. A stream keyed
that way makes `build(spec)` produce different data on every run — which defeats §0's first
property, the digests in §3.6, and invariant 13.5, all at once. It would also have failed
intermittently rather than immediately, since a single process is self-consistent.

`gen/core.py` derives the stream from `blake2b(name.encode())`. `tests/test_determinism.py`
asserts a literal stream value for a fixed name, so a future change back to `hash()` fails.

### C2. The list construction in §5.3 is unnecessary, and its side effects are harmful

The plan builds variable-length lists by cross-join → filter → `group_by().agg()` → left-join.
That works, but:

- `group_by` output order is not deterministic without `maintain_order=True`, and the rejoin
  makes row order depend on the join implementation — both of which put §0's determinism and
  prefix-stability properties at the mercy of an optimiser decision.
- It makes list generation depend on `n` (the cross-join is over the whole frame), so
  `build(spec, n=100)` is no longer literally `build(spec, n=10_000).head(100)`. That is
  invariant 13.3, broken by the plan's own generator.

One expression does the whole job:

```python
pl.concat_arr([child(k) for k in range(max_len)]).cast(pl.List(t)).list.head(len_expr)
```

`list.head` accepts an expression. (`concat_arr` rather than the more obvious `concat_list` -- see
C9, which is why.) Verified for all three distinguished shapes — null list,
empty list, list of nulls — and for `List<Struct<List<...>>>`. The cost is that `max_len`
children are always evaluated; `max_len` is small and bounded by the spec, so this is cheap.

The plan's step 5 — "the three-way distinction must be independently controllable" — is the
part that mattered, and it survives intact. It is `ListGen(length=..., child_nulls=...,
nulls=...)`, three orthogonal knobs, with `nested/list_three_way` pinning all three.

One thing the plan does not say and should: each element position needs **its own substream**.
Packing `[child_expr] * k` from one expression makes every element of a row identical, which hides
element-ordering bugs. `ListGen` threads `ctx.at(k)`.

### C3. Three §7.4 pushdown assertions describe behaviour Polars does not have

Measured, not reasoned (table in `docs/api-findings.md` §4):

| Plan says | Actually |
|---|---|
| "Empty projection (`select(pl.len())` — zero columns read)" | The plugin receives a **one-column** projection. Zero is not reachable. |
| "Limit combined with a filter — limit applies *after* the filter" | Polars never sends both. `n_rows` is `None` whenever a predicate is pushed, so a plugin cannot get the interaction wrong. |
| "`collect_schema()` must not read data (probe: `rows_scanned == 0`)" | The plugin is not invoked at all. Assert *no scan call*, which is stronger. |

All three are still tested, against what actually happens. A fourth was added: `with_row_index`
placed before a filter suppresses predicate pushdown entirely, so the engagement assertions must
be skipped for that shape while the correctness assertion stays.

### C4. The strictness ratchet in §3.5 legislates false failures

> Tests run at the claimed level and **additionally assert that the next level up fails**.

A harness that declares `DTYPES` because one dtype round-trips lossily would, under this rule,
be required to fail `COLUMN_ORDER` on *every* case — including the ones where it is exactly
right. The rule turns one real loss into a suite-wide obligation to be wrong.

The honesty mechanism the plan already has is sufficient and precise: `known_failures` becomes
`xfail(strict=True)`, so a loss that gets fixed breaks CI until the declaration is updated. The
ratchet is implemented per-case and opt-in, as `Capabilities.exact_at`, for authors who want to
pin "this case is lossy and must stay lossy" — not as a blanket rule.

### C5. Invariant 13.6 is not an invariant

> Every new case must catch a mutant no existing case catches, or justify itself in review.

Most type-coverage cases catch no mutant — a `Decimal(38, 38)` case fails because a format cannot
store it, not because a harness dropped a predicate. Enforcing this would block exactly the cases
that make the corpus worth having.

What *is* enforceable, and is enforced, is the dual: **every mutant must be caught by at least one
case**. `tests/test_mutants.py` asserts it and prints the case that catches each. 13.6 is demoted
to a review heuristic in the README.

### C6. `Probe` as specified cannot be asserted against

`probe() -> Probe | None` returning scalar counters has no defined lifetime. Two queries against
one harness give cumulative numbers, and `assert probe.rows_scanned < case.n_rows` silently stops
meaning anything after the first query.

`Probe` records a list of `ScanCall`s (one per `io_source` invocation, with the arguments Polars
passed and the rows produced), and the oracle wraps each query in `harness.probing()`, which
clears and returns a fresh record. The plan's assertions are then well-defined, and "the plugin
was never called" — needed for C3's `collect_schema` case — becomes expressible.

### C7. `NullPattern.LAST` contradicts invariant 13.3

§3.3 defines `LAST` as `__i == n-1`, and invariant 13.3 says generation depends on `__i` and never
on `n`. Both cannot hold. `LAST` is worth keeping -- a trailing-null off-by-one is a real bug class
-- so it is kept as a *documented* exception: `NullPattern.depends_on_n` marks it, the
prefix-stability test skips the cases that use it, and a further test asserts it is the only such
pattern, so a second one cannot be added silently.

The check has to walk the whole spec, not just the column: a `LAST` on a struct field or on a
list's elements makes the case equally `n`-dependent, and a shallow check would have turned the
prefix-stability skip into a false pass.

### C8. A 1 MiB string in the default palette is 600 MB of corpus

§5.2 lists "a 1 MiB string" alongside `""` and `"\n\r\t"`. At the default `n_rows=1000` and a 60%
palette draw, that palette entry alone materialises well over half a gigabyte per case, in every
case that uses strings. It is a real thing to test and a terrible default. It lives in
`palettes.HUGE_STRINGS`, reachable only from `strings/huge`, `n_rows=8`, tagged `slow`.

### C9. `concat_list` builds the wrong dtype for a nested container

Found by the generative layer while implementing C2, and worth stating separately because it is a
trap rather than a design flaw in the plan.

`pl.concat_list` given list-typed operands **concatenates** them: `List<List<T>>` comes out as
`List<T>`. A spec declaring `List<List<Int64>>` would have built `List<Int64>`, the declared and
actual schemas would have disagreed, and every nested-list case would have been silently testing
one level less nesting than it claimed. `pl.concat_arr` nests instead -- but it flattens `Array`
operands the same way, so an `Array` child is packed through its list equivalent and cast back.
`gen/nested.py::_pack` is the one place that knows any of this.

Two further quirks are handled there: `concat_arr` with a single operand returns the operand
unwrapped rather than a one-element container, and `concat_list` erases the element dtype in a
zero-row frame.

### C10. Zero rows is not a smaller version of one row

Polars' broadcasting differs at zero rows: a length-1 literal expands to *one* row rather than
none, and `list.head` with a row-shaped length either raises or erases the element dtype. A
`n_rows=0` case built directly therefore ends up with a schema no other row count produces.

`CaseSpec.build()` builds zero-row cases at one row and truncates. There are no values to lose,
and it removes a whole class of edge behaviour from every generator rather than patching each one.

### C11. A harness failure is not always an `Exception`

A Rust panic surfaced through PyO3 is a `pyo3_runtime.PanicException`, which derives from
`BaseException` precisely so that `except Exception` does not swallow it. The plan's xfail
machinery, written the obvious way, therefore reports a panicking harness as an infrastructure
error rather than as the declared failure it is -- which is how the first real run of this suite
against a real plugin produced four undeclared failures that were in fact declared.

`harness.is_failure` decides: everything counts as a harness failure except genuine control flow
(`KeyboardInterrupt`, `SystemExit`, `GeneratorExit`) and pytest's own outcome exceptions.

### C12. The probe must record what was *read*, not what was requested

The plan's `Probe.columns_read` reads naturally as "the projection the plugin was handed". Recorded
that way it is worthless: a plugin that reads every column and subsets them in memory reports the
narrow projection it was given and looks perfectly engaged. `IgnoresProjection` passed the
engagement suite until the recording moved to what the scan actually read.

The same distinction applies to rows, so `ScanCall` carries both `rows_produced` and an optional
`rows_read`, and `Probe.rows_scanned` prefers the latter.

## Refinements

### R1. Digests need a canonical encoder, not `repr` (§3.6)

The plan correctly rejects `hash_rows()` and IPC bytes, then proposes "repr of each value with
fixed float formatting". `repr` of a nested Polars value is itself a Polars-version artefact, so
that trades one unstable serialisation for another.

`spec.digest()` walks `to_list()` output recursively and encodes leaves by type: floats as
`float.hex()` (exact, no formatting choice), decimals as sign/digits/exponent, temporals as
their integer physical value, bytes as hex. Nothing goes through `str()` of a Polars object. The
plan's good instinct — a mismatch prints a diff and prompts a re-bless rather than just failing —
is implemented as `plioc.digests --diff`.

### R2. The three "law" families are worth separating in the API (§3.5, §7.5)

Round-trip, normalisation-fixpoint, and metamorphic laws have different preconditions and fail
for different reasons, and the plan mixes them into the harness contract. They are three modules
of `oracle.py` with a common runner, so a plugin author can run just the metamorphic laws (which
need no oracle frame and are fast) in a pre-commit hook.

### R3. `order_key` should be a `Layout` concern, not a `CaseSpec` flag (§3.2)

It is only consulted when comparing, never when generating. Keeping it on `CaseSpec` invites the
reading that `__i` is data. It stays on `CaseSpec` because the *column* is emitted by `build()`,
but `equality.py` owns the sort, and `Strictness.ROW_ORDER` is what decides whether it is used —
one place, not two.

### R4. Regressions are JSON, not TOML (§8)

The plan asks for TOML. A predicate is a recursive tree containing nulls, floats including `NaN`,
and temporal literals -- TOML expresses that badly, and the standard library has no TOML *writer*,
so honouring the letter of the plan would mean adding a dependency to serialise a shape the format
is bad at. The property that mattered is "a committed, diffable, hand-editable spec instead of
committed bytes", and JSON has it.

### R5. One mutant is asserted to survive

`LimitBeforeFilter` models a defect Polars' plugin interface cannot expose (C3). Rather than delete
it, it lives in `UNREACHABLE_MUTANTS` with a test asserting no case catches it. That converts an
absence of coverage into a monitored fact: if Polars starts pushing a limit alongside a predicate,
the assertion flips and the pushdown suite is told it needs a new case.

### R6. `dataframely` — agreed, and the extra is not shipped

The plan's read is right: its sampler targets plausible-valid data and this suite needs
adversarial data. Shipping the optional schema-validation adapter would still couple the suite's
public API to a second schema model for one assertion (`scan(target).collect_schema() == expected`)
that is three lines without it. Not shipped, not stubbed.

### R7. Milestone order — S0's findings move work, not just risk

M3 (nested) gets much smaller after C2 and M5 (pushdown) gets bigger after C3 and C6. The
implementation followed the plan's order with those weights.

---

## What the generative layer found while being built

Evidence that the Hypothesis layer is worth its weight, since the plan justifies it on faith: C9,
C10 and C12 were all found by it against the reference harnesses, not by writing curated cases.
Every one of them was a bug in the suite's own generator or probe that the curated corpus passed
over -- which is exactly the failure mode a conformance suite cannot afford, because a generator
that builds the wrong dtype tests the wrong thing everywhere at once.

---

## What was kept exactly as written

The parts of the plan that are load-bearing and correct, listed so a future reader does not
"simplify" them:

- **Stream independence** (§0). Adding a column to a case must not perturb the others. It is why
  the stream is keyed by column *name* and not by position, and it is asserted in CI.
- **Generation depends on `__i`, never on `n`** (13.3). This is what makes a fast CI run a literal
  prefix of a nightly one; also asserted in CI.
- **§7.3, the mandate test.** This is the flagship. `register_io_source` hands the predicate over
  as an obligation, and a plugin that pushes a partial translation and drops the residual returns
  wrong answers with no error anywhere in the stack. `DropsResidualMutant` exists to prove the
  suite catches it.
- **§4.1's controls.** A conformance suite that is never mutation-tested is a suite that passes.
- **`known_failures` as strict xfail.** The single mechanism that keeps a capability matrix from
  rotting into pessimism.
- **No committed data files.**
