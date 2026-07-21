# O(k) Ragged Fancy-Index Gather Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the O(n) `np.arange(n)[where]` allocation in `Ragged._gather_indices` so an integer-array gather of k rows from an n-row ragged buffer runs in O(k), independent of n.

**Architecture:** Replace the arange-based index resolution (which implicitly did negative-normalization + bounds-checking over all n rows) with direct O(k) vectorized NumPy over the k selected indices: `np.asarray(where)` → integer-dtype guard → normalize negatives → explicit bounds-check raising `IndexError`. The Rust `ragged::select` gather kernel and the numpy `ImportError` fallback are unchanged. This is a **behavior-preserving performance refactor**: correctness tests stay green before and after; the benchmark's scaling curve is the red→green signal.

**Tech Stack:** Python 3, NumPy (vectorized), pytest, standalone `perf_counter` benchmark (matching `benchmarks/bench_ragged_backends.py`), pixi (`dev` env).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-21-ragged-gather-ok-design.md`.
- Change is confined to the integer-array `else` branch of `Ragged._gather_indices` (`python/seqpro/rag/_core.py`, currently lines 1040–1042). The slice fast-path and bool-mask path are already O(k) — do NOT touch them.
- `OFFSET_TYPE = np.int64`; `_starts_stops()` returns contiguous int64 views, so the `np.ascontiguousarray(starts, np.int64)` calls downstream are no-op views (no O(n) copy) — do not add copies.
- Preserve the visible error type: out-of-bounds and non-integer indices must raise `IndexError` (numpy indexing contract). Do NOT assert on exact error-message text in tests — only on `IndexError`.
- No public signature or behavior change → **no** `skills/seqpro/SKILL.md` update, **no** maturin rebuild.
- Run everything in the dev env: prefix commands with `pixi run -e dev`.
- Conventional commits enforced (`feat:`/`fix:`/`test:`/`perf:`/`docs:`). Pre-commit/prek hooks run on commit.

## File Structure

| File | Responsibility | Task |
|------|----------------|------|
| `benchmarks/bench_ragged_gather.py` (create) | Self-contained harness: correctness oracle (reference vs candidate index resolution) + n-sweep + end-to-end `rag[idx]` timing + baseline number. Transitional (like `bench_ragged_backends.py`). | 1 |
| `tests/test_ragged_core.py` (modify) | Permanent regression cases for the integer-array gather path (OOB, negative normalization, empty, scalar/list parity, float→IndexError). | 2 |
| `python/seqpro/rag/_core.py` (modify, ~lines 1040–1042) | The one-branch O(k) fix. | 3 |

**Parallelism:** Tasks 1 and 2 touch different files, need no shared state, and both verify against the *current* (unfixed) code — run them in parallel. Task 3 modifies `_core.py` and is verified by re-running Task 1's benchmark (flat curve + speedup) and Task 2's tests (still green), so it lands after 1 and 2. Use superpowers:dispatching-parallel-agents for Tasks 1+2, then subagent-driven-development for Task 3.

---

### Task 1: Benchmark harness (oracle + n-sweep + baseline)

**Files:**
- Create: `benchmarks/bench_ragged_gather.py`

**Interfaces:**
- Consumes: `seqpro.rag._core.Ragged.from_lengths(data, lengths)` for the end-to-end timing.
- Produces: two module-level pure functions used as the correctness oracle —
  - `reference_resolve(where, n) -> np.ndarray[int64]` (current arange logic)
  - `candidate_resolve(where, n) -> np.ndarray[int64]` (proposed O(k) logic)
  Both accept `where` (int scalar / list / np.ndarray) and buffer size `n`, return normalized int64 indices, and raise `IndexError` on out-of-bounds or non-integer input. Task 3 copies `candidate_resolve`'s body verbatim into `_core.py`.

- [ ] **Step 1: Write the benchmark script**

Create `benchmarks/bench_ragged_gather.py`:

```python
"""Benchmark + correctness oracle for Ragged integer-array fancy-indexing (issue #69).

Proves the O(k) index-resolution candidate (a) matches the current O(n)
arange-based reference on every representative + edge case and (b) is flat in n
while the reference grows ~proportionally to n. Transitional harness (cf.
benchmarks/bench_ragged_backends.py); the permanent guard is the pytest cases
in tests/test_ragged_core.py. See
docs/superpowers/specs/2026-07-21-ragged-gather-ok-design.md.

Run: pixi run -e dev python benchmarks/bench_ragged_gather.py
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from seqpro.rag._core import Ragged


def reference_resolve(where: Any, n: int) -> np.ndarray:
    """Current arange-based integer-array resolution — the O(n) reference/oracle."""
    idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
    idx = np.where(idx < 0, idx + n, idx)
    return idx


def candidate_resolve(where: Any, n: int) -> np.ndarray:
    """Proposed O(k) integer-array resolution. Body is copied verbatim into
    Ragged._gather_indices in Task 3."""
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
    return idx


# ── Correctness oracle ────────────────────────────────────────────────────────

def _check_oracle() -> None:
    n = 32
    # (where) cases that must produce identical normalized indices.
    equal_cases = [
        np.array([0, 2, 5, 31]),          # in-range positives
        3,                                 # scalar int
        np.array([-1, -2, -32]),           # negatives -> normalized
        np.array([], dtype=np.int64),      # empty
        [0, 2, 4],                         # python list
        np.array([1, 1, 1, 1]),            # repeated (k can exceed n)
        np.arange(64) % n,                 # k > n
    ]
    for where in equal_cases:
        r = reference_resolve(where, n)
        c = candidate_resolve(where, n)
        assert r.dtype == c.dtype == np.int64, (where, r.dtype, c.dtype)
        np.testing.assert_array_equal(r, c, err_msg=f"mismatch for {where!r}")

    # cases where BOTH must raise IndexError
    raise_cases = [
        np.array([0, n]),                  # OOB positive
        np.array([-n - 1]),                # OOB negative (< -n)
        np.array([0.0, 1.0]),              # float indices
    ]
    for where in raise_cases:
        for name, fn in (("reference", reference_resolve), ("candidate", candidate_resolve)):
            try:
                fn(where, n)
            except IndexError:
                continue
            raise AssertionError(f"{name} did not raise IndexError for {where!r}")
    print("oracle: OK (reference == candidate on all cases)")


# ── Timing helpers ────────────────────────────────────────────────────────────

def _time(fn: Callable[[], Any], *, repeats: int = 7, min_batch_s: float = 0.02) -> float:
    """Seconds per call: warm up, autoscale a batch past min_batch_s, min of repeats."""
    for _ in range(3):
        fn()
    iters = 1
    while True:
        t0 = perf_counter()
        for _ in range(iters):
            fn()
        if perf_counter() - t0 >= min_batch_s:
            break
        iters *= 2
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (perf_counter() - t0) / iters)
    return best


# ── Sweeps ────────────────────────────────────────────────────────────────────

K = 256
N_SWEEP = [4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304]


def _resolve_sweep() -> None:
    """Isolated index-resolution: reference (O(n)) vs candidate (O(k))."""
    print(f"\nindex resolution, k={K} (us/call):")
    print(f"{'n':>10} {'reference':>12} {'candidate':>12} {'speedup':>9}")
    rng = np.random.default_rng(0)
    for n in N_SWEEP:
        where = rng.integers(0, n, size=K)
        t_ref = _time(lambda: reference_resolve(where, n))
        t_cand = _time(lambda: candidate_resolve(where, n))
        print(f"{n:>10} {t_ref * 1e6:>12.3f} {t_cand * 1e6:>12.3f} {t_ref / t_cand:>8.1f}x")


def _end_to_end_sweep() -> None:
    """Full public-API gather Ragged.__getitem__(idx). Reflects whichever branch
    is currently compiled into _core.py (reference before Task 3, candidate after)."""
    print(f"\nend-to-end rag[idx], k={K} (us/call):")
    print(f"{'n':>10} {'rag[idx]':>12}")
    rng = np.random.default_rng(1)
    for n in N_SWEEP:
        rag = Ragged.from_lengths(np.arange(n, dtype=np.int32), np.ones(n, dtype=np.int64))
        idx = rng.integers(0, n, size=K)
        t = _time(lambda: rag[idx])
        print(f"{n:>10} {t * 1e6:>12.3f}")


if __name__ == "__main__":
    _check_oracle()
    _resolve_sweep()
    _end_to_end_sweep()
    # Baseline number for the spec (n=65536, k=256):
    rng = np.random.default_rng(2)
    where = rng.integers(0, 65_536, size=K)
    print(
        f"\nbaseline @ n=65536 k=256: "
        f"reference={_time(lambda: reference_resolve(where, 65_536)) * 1e6:.2f} us  "
        f"candidate={_time(lambda: candidate_resolve(where, 65_536)) * 1e6:.2f} us"
    )
```

- [ ] **Step 2: Run the harness against current code**

Run: `pixi run -e dev python benchmarks/bench_ragged_gather.py`

Expected:
- First line: `oracle: OK (reference == candidate on all cases)` — the oracle passes.
- `index resolution` table: `reference` us/call rises roughly proportionally with `n` (≈64× from n=4096 to n=4M), `candidate` stays flat (~a few us across all n), `speedup` grows large (tens to hundreds ×) at big n.
- `end-to-end` table: `rag[idx]` rises with n (current code still uses the arange branch).
- Final `baseline @ n=65536 k=256` line prints two microsecond numbers with candidate < reference.

If the oracle line does not print (an assertion fired), the candidate logic diverges from the reference — fix `candidate_resolve` before proceeding, because Task 3 copies it verbatim.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/bench_ragged_gather.py
git commit -m "perf(rag): benchmark harness for O(k) ragged fancy-index gather (issue #69)"
```

---

### Task 2: Permanent regression tests for the integer-array gather path

**Files:**
- Modify: `tests/test_ragged_core.py` (append new tests near the existing `test_getitem_*bool_mask*` tests, ~line 399)

**Interfaces:**
- Consumes: `seqpro.rag._core.Ragged.from_lengths(data, lengths)` (already imported at top of the file).
- Produces: nothing consumed by later tasks (pure test additions).

These are characterization tests for a behavior-preserving refactor: they **pass on the current code** (locking in current behavior, and filling a real coverage gap — integer-array OOB and float indices are currently untested) and must **stay green after Task 3**. They are not expected to go red first; the benchmark scaling curve is this change's red→green signal.

- [ ] **Step 1: Add the tests**

Append to `tests/test_ragged_core.py`:

```python
def test_getitem_int_array_selects_rows():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    got = rag[np.array([0, 2])]
    assert got.shape == (2, None)
    np.testing.assert_array_equal(got[0], np.array([0, 1, 2]))  # row 0 -> data 0:3
    np.testing.assert_array_equal(got[1], np.array([5, 6, 7, 8, 9]))  # row 2 -> data 5:10


def test_getitem_int_array_negative_normalization_parity():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    pos = rag[np.array([0, 2])]
    neg = rag[np.array([-3, -1])]  # same rows via negative indexing
    np.testing.assert_array_equal(neg[0], pos[0])
    np.testing.assert_array_equal(neg[1], pos[1])


def test_getitem_int_array_out_of_bounds_positive_raises():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    with pytest.raises(IndexError):
        rag[np.array([0, 3])]  # 3 rows, index 3 is OOB


def test_getitem_int_array_out_of_bounds_negative_raises():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    with pytest.raises(IndexError):
        rag[np.array([-4])]  # 3 rows, -4 normalizes to -1 -> OOB


def test_getitem_int_array_empty():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    got = rag[np.array([], dtype=np.int64)]
    assert got.shape == (0, None)


def test_getitem_scalar_and_list_parity():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    from_list = rag[[1]]
    np.testing.assert_array_equal(from_list[0], np.array([3, 4]))  # row 1 -> data 3:5


def test_getitem_float_index_raises():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    with pytest.raises(IndexError):
        rag[np.array([0.0, 1.0])]
```

- [ ] **Step 2: Run the new tests against current code**

Run:
```bash
pixi run -e dev pytest tests/test_ragged_core.py -k "int_array or scalar_and_list or float_index" -v
```
Expected: all 7 new tests **PASS** (they characterize existing behavior). If `test_getitem_scalar_and_list_parity` or `test_getitem_int_array_empty` fails on a shape assertion, adjust the expected `shape`/`data` to what the current code returns and note it — the goal is to lock *actual* current behavior, and Task 3 must preserve whatever that is.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ragged_core.py
git commit -m "test(rag): cover integer-array fancy-index gather (OOB, negatives, float, empty)"
```

---

### Task 3: Apply the O(k) fix in `_gather_indices`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (the integer-array `else` branch, currently lines 1040–1042)

**Interfaces:**
- Consumes: `candidate_resolve`'s body from Task 1 (copied inline; `n = len(starts)` is already in scope at the top of `_gather_indices`).
- Produces: no signature change; `_gather_indices` still returns `(sel_starts, sel_stops)`.

- [ ] **Step 1: Replace the arange branch**

In `python/seqpro/rag/_core.py`, find (inside `_gather_indices`):

```python
        else:
            idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
            idx = np.where(idx < 0, idx + n, idx)
```

Replace with:

```python
        else:
            # O(k) resolution: no full-range arange allocation. Normalize
            # negatives and bounds-check over just the k selected indices,
            # raising IndexError (numpy contract) — this also unifies the
            # error type across the Rust and numpy-fallback gather paths.
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

- [ ] **Step 2: Run the regression tests (must stay green)**

Run:
```bash
pixi run -e dev pytest tests/test_ragged_core.py -k "int_array or scalar_and_list or float_index" -v
```
Expected: all 7 tests from Task 2 **PASS**.

- [ ] **Step 3: Run the full ragged test file (no regression)**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -q`
Expected: PASS (no failures introduced elsewhere — bool-mask and slice paths untouched).

- [ ] **Step 4: Re-run the benchmark — confirm the win**

Run: `pixi run -e dev python benchmarks/bench_ragged_gather.py`
Expected:
- `oracle: OK` still prints.
- `end-to-end rag[idx]` table is now **flat in n** (the public-API gather no longer allocates `arange(n)`), whereas before Task 3 it rose with n. This flat curve is the red→green signal for the perf change.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py
git commit -m "perf(rag): O(k) integer-array gather in _gather_indices (closes #69)"
```

---

## Self-Review

**Spec coverage:**
- O(n)→O(k) fix in the integer-array branch → Task 3. ✔
- Correctness oracle (reference vs candidate, all listed edge cases) → Task 1 `_check_oracle`. ✔
- Swept benchmark over dominating dim n at fixed k, + baseline number → Task 1 `_resolve_sweep`/`_end_to_end_sweep` + baseline print. ✔
- Empirical hot-spot confirmation → the n-sweep curve (reference ∝ n) substitutes for pyinstrument, which is not in the pixi envs (noted in Global Constraints / plan intro). ✔
- Permanent regression tests (OOB±, negative normalization, empty, scalar/list parity, float→IndexError) → Task 2. ✔
- IndexError semantics preserved; no SKILL.md change; no Rust rebuild → Global Constraints + Task 3. ✔
- "Why not Rust/Numba" (stop at vectorized) → reflected in scope: only the one branch changes, Rust untouched. ✔

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — every step has full code or an exact command + expected output. ✔

**Type consistency:** `reference_resolve(where, n)` / `candidate_resolve(where, n)` return `np.ndarray[int64]` and are referenced identically in Tasks 1 and 3; `Ragged.from_lengths(data, lengths)` used consistently; the candidate body in Task 3 matches `candidate_resolve` in Task 1 verbatim. ✔
