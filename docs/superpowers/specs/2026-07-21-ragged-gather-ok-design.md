# Ragged fancy-indexing: O(k) gather instead of O(n) (issue #69)

**Date:** 2026-07-21
**Issue:** [ML4GLand/SeqPro#69](https://github.com/ML4GLand/SeqPro/issues/69)
**Status:** Approved — ready for implementation plan

## Problem

`Ragged._gather_indices` (`python/seqpro/rag/_core.py:1041`), integer-array branch:

```python
idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
idx = np.where(idx < 0, idx + n, idx)
```

`np.arange(n)[where]` materializes a full n-length array only to select k elements
from it — **O(n) time and memory per gather**. It is doing two jobs implicitly:

1. **negative-index normalization** (arange values are already in `[0, n-1]`), and
2. **bounds-checking** (raises `IndexError` on out-of-bounds).

Because arange-indexing already normalizes, line 1042's `np.where(idx < 0, ...)` is
**dead code** — every value is already non-negative by then.

### Measured impact (from the issue)

In a shuffle-buffer dataloader selecting k=256 rows from an n=65,536-row buffer, this
operation was ~23% of total training-pipeline time (~0.097 s per epoch-slice).

## Workload characterization

| dimension | typical | max | grows? | notes |
|-----------|---------|-----|--------|-------|
| n — ragged rows / buffer size | 65,536 | unbounded | **grows** | the `arange(n)` allocation — the bug |
| k — indices selected / batch | 256 | ~few thousand | ~fixed | target cost O(k) |

**Bound:** CPU-bound (allocate + fill an n-length arange, then fancy-index it).

**Design target:** `O(k) along n` via vectorized NumPy over the k indices — the first
rung of the perf ladder. The actual gather is *already* Rust (`ragged::select`, an O(k)
loop); no new Numba/Rust is warranted (see "Why not Rust/Numba" below).

## Downstream facts that constrain the fix

- `OFFSET_TYPE = np.int64` (`python/seqpro/rag/_utils.py:10`).
- `_starts_stops()` returns contiguous int64 views (`offsets[:-1]`/`offsets[1:]`, or
  rows of the `(2, N)` offsets array). Therefore `np.ascontiguousarray(starts, np.int64)`
  at `_core.py:1047-1049` is a **no-op view, not a hidden O(n) copy** in the common
  contiguous case. The `arange` is the sole O(n) cost. (The Phase-3 profile confirms
  this empirically before any change is made.)
- Rust `ragged::select` (`crates/seqpro-core/src/ragged.rs:288`) **rejects** negatives
  and out-of-bounds with a `String` error mapped to `PyValueError`; it does not
  normalize negatives. So negatives MUST be normalized in Python before the Rust call.
- The pure-NumPy `ImportError` fallback (`starts[idx]`) raises `IndexError`.

## The change

Replace the arange with direct, O(k) index resolution in the integer-array branch:

```python
else:
    idx = np.atleast_1d(np.asarray(where))
    if idx.dtype.kind not in "iu":
        raise IndexError(
            "only integers, slices (`:`), and integer arrays are valid indices"
        )
    idx = idx.astype(np.int64, copy=False)
    neg = idx < 0
    if neg.any():
        idx = np.where(neg, idx + n, idx)
    oob = (idx < 0) | (idx >= n)
    if oob.any():
        raise IndexError(
            f"index {int(idx[oob][0])} is out of bounds for axis 0 with size {n}"
        )
```

Everything after this branch (the Rust `select` call and the numpy fallback) is unchanged.

### Rationale

- **O(k), not O(n).** No full-range allocation. Normalization and bounds-check run over
  the k selected indices only.
- **Preserves `IndexError` semantics** callers see today, and — a bonus — unifies the
  Rust path and the numpy-fallback path onto the *same* error (previously `ValueError`
  from Rust vs `IndexError` from the fallback). Rust's own check becomes a redundant
  safety net (left in place).
- **Rejects float indices** via the `dtype.kind` guard, matching numpy's refusal to
  index with floats. The old arange rejected them implicitly; forcing `dtype=np.int64`
  in `asarray` would silently truncate them — a correctness regression, so the guard is
  explicit.
- **Same edge-case behavior** as today: scalar int → `atleast_1d` → 1-element; empty
  list → empty int64 array; lists/arrays via `asarray`.

Scope is this one branch. The bool-mask fast path (`np.flatnonzero`) and the slice fast
path are already O(k) and untouched.

### Why not Rust/Numba

Stop at vectorized NumPy:

- At k=256 the residual work is 2–3 numpy passes over 256 int64s — microseconds, below
  timing noise. Numba/Rust call overhead would dominate; there is no measurable rung
  above "vectorized" to climb to.
- The one architecturally tempting Rust consolidation — folding normalize+bounds into
  `select`'s existing single idx-loop — buys **zero** measured time (same O(k) gather)
  while adding surface: `select` returns a `String` → `PyValueError`, so raising a
  proper `IndexError` needs custom error plumbing, the numpy fallback must mirror the new
  semantics, and it requires a maturin rebuild.

**Arbiter:** the harness. If the Phase-3 profile surprises us and the k-sized numpy
passes actually register, escalate to the Rust consolidation. Not expected.

## Harness (Phase 3 — built and committed BEFORE the fix)

Standalone `benchmarks/bench_ragged_gather.py`, following the existing convention in
`benchmarks/bench_ragged_backends.py` (warmup, autoscale past a min batch, min-of-repeats).
It does three things:

1. **Correctness oracle.** The *current* arange-based logic is the reference. The
   candidate must return byte-identical `(sel_starts, sel_stops)` **and** identical error
   behavior across representative + edge cases:
   - empty index array
   - scalar int
   - negative indices (normalization parity vs positive equivalents)
   - out-of-bounds positive index → `IndexError`
   - out-of-bounds negative index (`< -n`) → `IndexError`
   - repeated indices (k > n)
   - float index array → `IndexError`
2. **Swept benchmark.** Sweep the dominating dim `n ∈ {4k, 16k, 65k, 262k, 1M}` at fixed
   k=256 (confirm the current curve rises ∝ n and the candidate is flat in n), plus a
   k-sweep at fixed n. **Record the baseline number** at (n=65536, k=256).
3. **Profile.** `pyinstrument` on the current path at n=65536 to empirically confirm
   `arange` is the hot spot and `ascontiguousarray` is not copying, before touching code.

The benchmark script is transitional (like `bench_ragged_backends.py`). The correctness
edge cases ALSO land as **permanent** `pytest` cases in `tests/test_ragged_core.py`
(these are the regression guard; the bench script may later be removed).

## Phase 4 — Optimize loop

Apply the one change → re-run oracle + sweep → confirm the flat-in-n curve and the
baseline improvement at (65536, 256) → stop. Stopping criterion: single hot spot removed,
Phase-0 target hit (gather O(k), independent of n).

## Testing

Permanent regression tests in `tests/test_ragged_core.py`, written first (TDD):

- OOB positive index in an int array → `IndexError`
- OOB negative index (`< -n`) → `IndexError`
- negative-index normalization returns the same rows as the positive equivalent
- empty index array → empty result
- scalar-int and list-of-ints parity
- float index array → `IndexError`

## Out of scope / no doc changes

- No public signature or behavior change (the visible OOB error type is preserved), so
  `skills/seqpro/SKILL.md` needs no update.
- Rust `select` unchanged; no maturin rebuild required for the fix.
