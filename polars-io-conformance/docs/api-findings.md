# S0 — API findings

Measured against **Polars 1.43.1**, CPython 3.10. Re-run `tests/test_api_assumptions.py` to
re-verify; every finding below is asserted there, so a Polars upgrade that invalidates one
fails the suite rather than silently changing what the corpus generates.

## 1. Pure-lazy row index — primary works

```python
pl.LazyFrame().select(pl.int_range(0, n, dtype=pl.UInt64).alias("__i"))
```

Works with no input columns. The eager fallback is not needed and is not implemented.

## 2. Deterministic per-row pseudo-randomness — fallback taken

`Expr.hash(seed=..., seed_1=..., seed_2=..., seed_3=...)` exists and runs, but its output is
not a documented, stabilised value. Determinism is load-bearing here (§0), so the suite takes
the arithmetic fallback: **splitmix64 evaluated in expressions**, verified bit-exact against a
Python reference in `tests/test_api_assumptions.py`.

Three operator details cost time if you rediscover them:

| Want | Not this | This |
|---|---|---|
| wrapping `u64` multiply | — | `a * pl.lit(K, pl.UInt64)` — Polars wraps, it does not raise |
| logical shift right | `expr >> k` — `TypeError`, no `__rshift__` on `Expr` | `expr // pl.lit(1 << k, pl.UInt64)` (exact for unsigned) |
| element-wise xor | `expr.bitwise_xor(other)` — that is the *reduction*, it takes no argument | `a ^ b` |

Casts are **checked, not wrapping**: `(x % 256).cast(pl.Int8)` raises. Narrow-width values are
produced by mapping into the target range before the cast.

## 3. Palette lookup — primary works, universally

```python
pl.lit(pl.Series(values, dtype=dt)).gather(idx_expr % len(values))
```

Verified to preserve dtype for `Enum`, `Categorical`, `Decimal(38, 2)`, tz-aware `Datetime`,
`Duration`, `Struct` (including null entries), `Array(Float32, 2)`, `List`, `Null`, `Int128`.
Neither `replace_strict` nor the palette-join fallback is needed; `gather` is the single path.

## 4. What `register_io_source` actually delivers

Measured with a probing plugin over a 5-row frame. These drive §7 of the plan, and several of
them contradict what a reading of the API would suggest.

| Query | `with_columns` | `predicate` | `n_rows` | `batch_size` |
|---|---|---|---|---|
| `.filter(a > 2).select("b")` | `['a', 'b']` | pushed | `None` | `100000` |
| `.select(pl.len())` | `['a']` | `None` | `None` | `None` |
| `.head(2)` | `None` | `None` | `2` | `None` |
| `.filter(a > 2).head(1)` | `None` | pushed | **`None`** | `100000` |
| `.with_row_index().filter(a > 3)` | `None` | **`None`** | `None` | `None` |
| `.collect_schema()` | *plugin not called at all* | | | |

Consequences the suite encodes:

- **A count is not a zero-column read.** `select(pl.len())` arrives as a one-column
  projection, so "empty projection" cannot be asserted; the contract is "exactly one column".
- **Limit and predicate never arrive together.** Polars withholds `n_rows` once a predicate is
  pushed, so "the limit applies after the filter" is not a thing the plugin can get wrong —
  the assertion is that `limit_received is None` whenever a predicate was pushed.
- **`with_row_index` before a filter suppresses predicate pushdown.** Engagement must not be
  asserted for that shape; correctness still must.
- **Planning reads nothing.** `collect_schema()` produces no scan call at all, which is a
  stronger and simpler assertion than `rows_scanned == 0`.
- A column used only by the predicate is included in the projection, so "predicate on a column
  outside the projection" is handled by Polars, not by the plugin.

## 5. Shapes Polars cannot represent

- **A frame with zero columns has zero rows.** `select([])` on an `n`-row frame yields `(0, 0)`.
  The "0 columns" corpus case is therefore also a 0-row case.
- **`pl.struct([])` raises.** An empty struct column is only constructible through a
  `pl.Series(..., dtype=pl.Struct({}))` palette, which is how `StructGen` emits it.
