# Flat-buffer `to_padded` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `seqpro.rag.to_padded(rag, pad_value, *, length=None)` — a parallel Numba densify-and-right-pad over a `Ragged`'s flat `(data, offsets)` buffer, byte-identical to the awkward `rpad`/`pad_none` idiom and generic over the stored dtype.

**Architecture:** One dtype-agnostic byte-copy kernel (view the flat buffer as `uint8`, copy `itemsize × elements` per row into a pre-filled output) in `seqpro/rag/_ops.py`, alongside `reverse_complement`. Guards and the contiguity-pack step mirror `reverse_complement`. Output shape is `(*leading_dims, out_len)`.

**Tech Stack:** Python, NumPy, Numba (`@nb.njit(parallel=True)`), awkward (only for `ak.to_packed` on non-contiguous input), pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-05-31-flat-buffer-to-padded-design.md`

**Conventions:**
- Run everything with `pixi run` (e.g. `pixi run pytest ...`, `pixi run python ...`).
- Tests live in `tests/test_ragged_to_padded.py`. Import the function under test from `seqpro.rag._ops` (matches `tests/test_ragged_rc.py`).
- Commit messages follow commitizen conventional commits (a `commitizen check` pre-commit hook enforces this).

---

### Task 1: Core `to_padded` — pad to batch max (`length=None`)

**Files:**
- Modify: `python/seqpro/rag/_ops.py` (add kernel + function, extend `__all__`)
- Modify: `python/seqpro/rag/__init__.py` (export `to_padded`)
- Test: `tests/test_ragged_to_padded.py` (create)

- [ ] **Step 1: Write the failing test file with the basic S1 pad-to-max case**

Create `tests/test_ragged_to_padded.py`:

```python
"""Tests for seqpro.rag.to_padded (flat-buffer ragged densify-and-pad)."""

from __future__ import annotations

import awkward as ak
import awkward.operations.str as ak_str
import numpy as np
import pytest

from seqpro.rag import Ragged, lengths_to_offsets
from seqpro.rag._ops import to_padded


# --------------------------------------------------------------------------- #
# Awkward references (mirror gvl's _ragged.to_padded)
# --------------------------------------------------------------------------- #


def _naive_pad_bytes(rag: Ragged, pad_value: bytes, length: int) -> np.ndarray:
    return Ragged(ak_str.rpad(rag, length, pad_value)).to_numpy()


def _naive_pad_numeric(rag: Ragged, pad_value, length: int) -> np.ndarray:
    orig_dtype = rag.dtype
    r = ak.pad_none(rag, length, axis=-1, clip=True)
    r = ak.fill_none(r, pad_value)
    return ak.to_numpy(r).astype(orig_dtype, copy=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_bytes_rag(seqs: list[str]) -> Ragged:
    lengths = np.array([len(s) for s in seqs], dtype=np.uint32)
    flat = "".join(seqs).encode()
    data = np.frombuffer(flat, dtype="S1").copy()
    return Ragged.from_lengths(data, lengths)


def _make_numeric_rag(rows: list[list], dtype) -> Ragged:
    lengths = np.array([len(r) for r in rows], dtype=np.uint32)
    flat = np.array([x for r in rows for x in r], dtype=dtype)
    return Ragged.from_lengths(flat, lengths)


# --------------------------------------------------------------------------- #
# Core: pad to batch max
# --------------------------------------------------------------------------- #


def test_pad_to_max_bytes_basic():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N")
    expected = np.array([[b"A", b"T", b"C", b"G"], [b"G", b"G", b"N", b"N"]], dtype="S1")
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run pytest tests/test_ragged_to_padded.py::test_pad_to_max_bytes_basic -v`
Expected: FAIL — `ImportError: cannot import name 'to_padded' from 'seqpro.rag._ops'`.

- [ ] **Step 3: Implement the kernel and function in `_ops.py`**

In `python/seqpro/rag/_ops.py`, extend `__all__` and append the kernel + function. The existing top of the file is:

```python
__all__ = ["reverse_complement"]
```

Change it to:

```python
__all__ = ["reverse_complement", "to_padded"]
```

Append at the end of the file:

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _to_padded_copy(
    data_u1: NDArray[np.uint8],
    offsets: NDArray[np.int64],
    out_u1: NDArray[np.uint8],
    itemsize: int,
    out_len: int,
) -> None:  # pragma: no cover - exercised via to_padded
    """Copy each ragged row's bytes into a pre-filled (n_rows, out_len) buffer.

    ``out_u1`` is the flat uint8 view of a C-contiguous ``(n_rows, out_len)`` array
    already filled with the pad value. For each row, the first
    ``min(row_len, out_len)`` elements are copied (longer rows are truncated);
    padded positions keep the pre-filled value. Parallel across rows.
    """
    n = offsets.shape[0] - 1
    row_stride = out_len * itemsize
    for i in nb.prange(n):
        row_len = offsets[i + 1] - offsets[i]
        ncopy = row_len if row_len < out_len else out_len
        nbytes = ncopy * itemsize
        src = offsets[i] * itemsize
        dst = i * row_stride
        for b in range(nbytes):
            out_u1[dst + b] = data_u1[src + b]


def to_padded(
    rag: Ragged,
    pad_value,
):
    """Densify a Ragged into a right-padded rectilinear array via a flat-buffer kernel.

    Flat-buffer alternative to the awkward idiom
    ``Ragged(ak_str.rpad(rag, L, v)).to_numpy()`` (bytes) /
    ``ak.to_numpy(ak.fill_none(ak.pad_none(rag, L, axis=-1, clip=True), v))`` (numeric):
    each row is copied once into a pre-filled output buffer in a single parallel pass.
    Pads the last axis to the batch maximum ``rag.lengths.max()``.

    Parameters
    ----------
    rag
        Ragged array with exactly one ragged dimension and no fixed trailing
        dimensions (the ragged axis is last). Any fixed-itemsize dtype.
    pad_value
        Fill value for positions past each row's length; must be castable to
        ``rag.data.dtype`` (e.g. ``b"N"`` for S1, ``-1`` for int32).

    Returns
    -------
    NDArray
        Dense array of dtype ``rag.data.dtype`` and shape
        ``(*rag.shape[:rag_dim], out_len)``.
    """
    rag_dim = rag.rag_dim

    offsets = np.ascontiguousarray(rag.offsets, dtype=np.int64)
    n_rows = offsets.shape[0] - 1

    out_len = int(rag.lengths.max()) if n_rows else 0

    dtype = rag.data.dtype
    itemsize = dtype.itemsize

    out = np.full((n_rows, out_len), pad_value, dtype=dtype)
    if n_rows and out_len:
        data_u1 = np.ascontiguousarray(rag.data).reshape(-1).view(np.uint8)
        out_u1 = out.reshape(-1).view(np.uint8)
        _to_padded_copy(data_u1, offsets, out_u1, itemsize, out_len)

    leading = rag.shape[:rag_dim]
    if leading:
        out = out.reshape(*leading, out_len)
    return out
```

- [ ] **Step 4: Export `to_padded` from the package**

In `python/seqpro/rag/__init__.py`, change:

```python
from ._ops import reverse_complement
```
to
```python
from ._ops import reverse_complement, to_padded
```

and add `"to_padded"` to `__all__`:

```python
__all__ = [
    "OFFSET_TYPE",
    "DTYPE_co",
    "RDTYPE_co",
    "Ragged",
    "is_rag_dtype",
    "lengths_to_offsets",
    "reverse_complement",
    "to_padded",
]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run pytest tests/test_ragged_to_padded.py::test_pad_to_max_bytes_basic -v`
Expected: PASS.

- [ ] **Step 6: Add numeric, float, and leading-dim coverage**

Append to `tests/test_ragged_to_padded.py`:

```python
def test_pad_to_max_int32_basic():
    rag = _make_numeric_rag([[0, 1, 2, 3], [4, 5]], np.int32)
    out = to_padded(rag, -1)
    expected = np.array([[0, 1, 2, 3], [4, 5, -1, -1]], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.int32


def test_pad_to_max_float32_basic():
    rag = _make_numeric_rag([[1.5], [2.5, 3.5, 4.5]], np.float32)
    out = to_padded(rag, 0.0)
    expected = np.array([[1.5, 0.0, 0.0], [2.5, 3.5, 4.5]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float32


def test_pad_to_max_matches_awkward_bytes():
    rng = np.random.default_rng(1)
    n = 12
    lengths = rng.integers(0, 20, size=n)
    seqs = ["".join(rng.choice(list("ACGT"), size=int(L))) for L in lengths]
    rag = _make_bytes_rag(seqs)
    out = to_padded(rag, b"N")
    expected = _naive_pad_bytes(rag, b"N", int(lengths.max()))
    np.testing.assert_array_equal(out, expected)


def test_pad_to_max_leading_dims():
    """(2, 3, None) input densifies to (2, 3, out_len) with rows in the right cells."""
    rows = ["AT", "G", "TTTT", "CC", "A", "GGG"]
    lengths = np.array([len(s) for s in rows], dtype=np.uint32)
    data = np.frombuffer("".join(rows).encode(), dtype="S1").copy()
    offsets = lengths_to_offsets(lengths)
    rag = Ragged.from_offsets(data, (2, 3, None), offsets)
    out = to_padded(rag, b"N")
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[0, 0], np.frombuffer(b"ATNN", dtype="S1"))
    np.testing.assert_array_equal(out[1, 2], np.frombuffer(b"GGGN", dtype="S1"))
```

- [ ] **Step 7: Run the full test file to verify all pass**

Run: `pixi run pytest tests/test_ragged_to_padded.py -v`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add python/seqpro/rag/_ops.py python/seqpro/rag/__init__.py tests/test_ragged_to_padded.py
git commit -m "feat(rag): flat-buffer to_padded (pad to batch max)"
```

---

### Task 2: Fixed `length` — pad and truncate

**Files:**
- Modify: `python/seqpro/rag/_ops.py` (add the `length` keyword to `to_padded`)
- Test: `tests/test_ragged_to_padded.py` (append)

The kernel `_to_padded_copy` already truncates via `min(row_len, out_len)`; this task wires an explicit target length into `out_len`.

- [ ] **Step 1: Write failing tests for explicit `length`**

Append to `tests/test_ragged_to_padded.py`:

```python
def test_length_pad_beyond_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=6)
    expected = np.array(
        [[b"A", b"T", b"C", b"G", b"N", b"N"], [b"G", b"G", b"N", b"N", b"N", b"N"]],
        dtype="S1",
    )
    np.testing.assert_array_equal(out, expected)


def test_length_truncate_below_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=3)
    expected = np.array([[b"A", b"T", b"C"], [b"G", b"G", b"N"]], dtype="S1")
    np.testing.assert_array_equal(out, expected)


def test_length_equal_to_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out_explicit = to_padded(rag, b"N", length=4)
    out_default = to_padded(rag, b"N")
    np.testing.assert_array_equal(out_explicit, out_default)


def test_length_truncate_numeric():
    rag = _make_numeric_rag([[0, 1, 2, 3], [4, 5]], np.int32)
    out = to_padded(rag, -1, length=2)
    expected = np.array([[0, 1], [4, 5]], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/test_ragged_to_padded.py -k length -v`
Expected: FAIL — `TypeError: to_padded() got an unexpected keyword argument 'length'`.

- [ ] **Step 3: Add the `length` keyword to `to_padded`**

In `python/seqpro/rag/_ops.py`, change the signature:

```python
def to_padded(
    rag: Ragged,
    pad_value,
):
```
to
```python
def to_padded(
    rag: Ragged,
    pad_value,
    *,
    length: int | None = None,
):
```

and change the `out_len` line:

```python
    out_len = int(rag.lengths.max()) if n_rows else 0
```
to
```python
    if length is not None:
        out_len = int(length)
    elif n_rows:
        out_len = int(rag.lengths.max())
    else:
        out_len = 0
```

Also add the `length` parameter to the docstring (after `pad_value`):

```
    length
        Target length of the last axis. ``None`` (default) uses the batch maximum
        ``rag.lengths.max()``. An explicit ``length`` right-pads shorter rows and
        truncates longer rows to exactly ``length``.
```

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_ragged_to_padded.py -k length -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_ops.py tests/test_ragged_to_padded.py
git commit -m "feat(rag): to_padded fixed-length pad and truncate"
```

---

### Task 3: Guards — record layout, trailing dims, non-contiguous input

**Files:**
- Modify: `python/seqpro/rag/_ops.py` (add three guards at the top of `to_padded`)
- Test: `tests/test_ragged_to_padded.py` (append)

- [ ] **Step 1: Write the failing guard tests**

Append to `tests/test_ragged_to_padded.py`:

```python
def test_record_layout_raises():
    seq = _make_bytes_rag(["ATCG", "GG"])
    score = _make_numeric_rag([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]], np.float32)
    rec = ak.zip({"seq": seq, "score": score})
    assert isinstance(rec, Ragged)
    with pytest.raises(NotImplementedError, match="record-layout"):
        to_padded(rec, b"N")


def test_trailing_fixed_dim_raises():
    data = np.zeros((6, 4), dtype="S1")
    rag = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="ragged axis to be last"):
        to_padded(rag, b"N")


def test_non_contiguous_input_correct():
    """A sliced Ragged (non-contiguous offsets) still densifies correctly."""
    rag = _make_bytes_rag(["ATCG", "GG", "TTTAA", "C"])
    sliced = rag[1:3]  # offsets become (2, N) starts/stops
    out = to_padded(sliced, b"N")
    expected = np.array(
        [[b"G", b"G", b"N", b"N", b"N"], [b"T", b"T", b"T", b"A", b"A"]], dtype="S1"
    )
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run pytest tests/test_ragged_to_padded.py -k "record_layout or trailing or non_contiguous" -v`
Expected: FAIL — record-layout raises a cryptic error (not `NotImplementedError`); trailing-dim produces a wrong-shaped array (no `ValueError`); non-contiguous produces wrong rows because `offsets.shape[0] - 1` is computed from a `(2, N)` array.

- [ ] **Step 3: Add the guards to `to_padded`**

In `python/seqpro/rag/_ops.py`, insert these guards at the very start of `to_padded`, before `rag_dim = rag.rag_dim`:

```python
    rag._ensure_parts()
    if isinstance(rag._parts, dict):
        raise NotImplementedError(
            "to_padded is not defined on record-layout Ragged arrays; "
            "convert fields individually."
        )

    rag_dim = rag.rag_dim
    if any(d is not None for d in rag.shape[rag_dim + 1 :]):
        raise ValueError(
            "to_padded requires the ragged axis to be last "
            f"(no fixed trailing dims), got shape {rag.shape}."
        )

    if not rag.is_contiguous:
        import awkward as ak

        rag = Ragged(ak.to_packed(rag))
```

Then delete the now-duplicate `rag_dim = rag.rag_dim` line that previously started the body (the guard block above now defines it).

- [ ] **Step 4: Run to verify they pass**

Run: `pixi run pytest tests/test_ragged_to_padded.py -k "record_layout or trailing or non_contiguous" -v`
Expected: PASS.

- [ ] **Step 5: Run the whole file (no regressions)**

Run: `pixi run pytest tests/test_ragged_to_padded.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_ops.py tests/test_ragged_to_padded.py
git commit -m "feat(rag): guard to_padded for record/trailing-dim/non-contiguous"
```

---

### Task 4: Edge cases + multi-dtype awkward baseline

**Files:**
- Test: `tests/test_ragged_to_padded.py` (append)

- [ ] **Step 1: Write the edge-case and baseline tests**

Append to `tests/test_ragged_to_padded.py`:

```python
def test_empty_batch():
    rag = Ragged.from_lengths(
        np.frombuffer(b"", dtype="S1").copy(), np.array([], dtype=np.uint32)
    )
    out = to_padded(rag, b"N")
    assert out.shape == (0, 0)


def test_all_empty_rows():
    rag = Ragged.from_lengths(
        np.frombuffer(b"", dtype="S1").copy(), np.array([0, 0, 0], dtype=np.uint32)
    )
    out = to_padded(rag, b"N")
    assert out.shape == (3, 0)


def test_length_zero_truncates_all():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=0)
    assert out.shape == (2, 0)


def test_matches_awkward_int32_iinfo_max():
    rng = np.random.default_rng(3)
    rows = [list(rng.integers(0, 100, size=int(L))) for L in rng.integers(0, 15, size=10)]
    rag = _make_numeric_rag(rows, np.int32)
    pad = int(np.iinfo(np.int32).max)
    length = max((len(r) for r in rows), default=0)
    out = to_padded(rag, pad)
    expected = _naive_pad_numeric(rag, pad, length)
    np.testing.assert_array_equal(out, expected)


def test_matches_awkward_float32():
    rng = np.random.default_rng(4)
    rows = [list(rng.random(int(L)).astype(np.float32)) for L in rng.integers(1, 12, size=9)]
    rag = _make_numeric_rag(rows, np.float32)
    length = max(len(r) for r in rows)
    out = to_padded(rag, 0.0)
    expected = _naive_pad_numeric(rag, 0.0, length)
    np.testing.assert_array_equal(out, expected)
```

- [ ] **Step 2: Run to verify**

Run: `pixi run pytest tests/test_ragged_to_padded.py -v`
Expected: PASS (all). If `test_empty_batch` fails on `rag.lengths.max()`, the `n_rows == 0` guard in `to_padded` is missing or wrong — fix `_ops.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ragged_to_padded.py
git commit -m "test(rag): to_padded edge cases and multi-dtype awkward parity"
```

---

### Task 5: Microbench + docs/skill update

**Files:**
- Create: `scratch_bench_to_padded.py` (repo root; matches the existing `scratch_bench_rc.py` convention)
- Modify: the seqpro skill (`SKILL.md` — locate via `git ls-files | grep -i skill` or the project's skills dir) and `docs/ragged.md` if it enumerates `rag` ops

- [ ] **Step 1: Write the microbench script**

Create `scratch_bench_to_padded.py`:

```python
"""Microbench: flat-buffer to_padded vs the awkward rpad/pad_none idiom."""

import time
import tracemalloc

import awkward as ak
import awkward.operations.str as ak_str
import numpy as np

from seqpro.rag import Ragged, to_padded


def _naive_pad_bytes(rag, pad_value, length):
    return Ragged(ak_str.rpad(rag, length, pad_value)).to_numpy()


def main():
    rng = np.random.default_rng(0)
    n, max_len = 1024, 4096
    lengths = rng.integers(max_len // 2, max_len, size=n).astype(np.uint32)
    total = int(lengths.sum())
    data = np.frombuffer(b"".join(rng.choice([b"A", b"C", b"G", b"T"], size=total)), dtype="S1")
    rag = Ragged.from_lengths(data, lengths)
    L = int(lengths.max())

    # warmup (jit)
    to_padded(rag, b"N")
    _naive_pad_bytes(rag, b"N", L)

    def bench(fn, *a, rep=20):
        ts = []
        for _ in range(rep):
            t = time.perf_counter()
            fn(*a)
            ts.append(time.perf_counter() - t)
        return min(ts) * 1e3

    def peak(fn, *a):
        tracemalloc.start()
        fn(*a)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p / 1e6

    t_old = bench(_naive_pad_bytes, rag, b"N", L)
    t_new = bench(to_padded, rag, b"N")
    print(f"batch {n} rows x ~{max_len} b ({total / 1e6:.1f} MB), pad to {L}")
    print(f"awkward (old):    {t_old:.3f} ms/call   peak +{peak(_naive_pad_bytes, rag, b'N', L):.2f} MB")
    print(f"flat numba (new): {t_new:.3f} ms/call   peak +{peak(to_padded, rag, b'N'):.2f} MB")
    print(f"speedup: {t_old / t_new:.1f}x")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the microbench and record numbers**

Run: `pixi run python scratch_bench_to_padded.py`
Expected: `to_padded` is several× faster with far lower peak allocation. Note the printed numbers for the PR description / REGRESSIONS update.

- [ ] **Step 3: Update the seqpro skill**

Per the seqpro `CLAUDE.md` ("Don't add a feature or change a public signature without updating this skill"), add `to_padded` to the skill's `rag` op reference. Locate the skill file:

Run: `git ls-files | grep -iE 'skill|SKILL' ; ls .claude/skills 2>/dev/null`

Add a one-line entry next to `reverse_complement` describing:
`seqpro.rag.to_padded(rag, pad_value, *, length=None)` — flat-buffer densify-and-right-pad of a Ragged to a rectilinear array; `length=None` pads to batch max, explicit `length` pads/truncates; ragged-axis-last, non-record only.

- [ ] **Step 4: Run the full suite + lint**

Run: `pixi run pytest tests/test_ragged_to_padded.py tests/test_ragged_rc.py tests/test_ragged.py -v`
Expected: PASS.
Run: `pixi run ruff check python/seqpro/rag/_ops.py python/seqpro/rag/__init__.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add scratch_bench_to_padded.py
git add -A  # skill / docs updates
git commit -m "docs(rag): document to_padded; add microbench"
```

---

## Self-Review

**Spec coverage:**
- API `to_padded(rag, pad_value, *, length=None)` → Task 1 (core + export), Task 2 (`length`).
- One generic byte-copy kernel (approach A) → Task 1 Step 3.
- `length=None` → batch max; explicit `length` pad/truncate → Tasks 1, 2.
- Output shape `(*leading, out_len)`, dtype-generic → Task 1 (leading-dim + numeric/float tests).
- Guards (record / trailing-dim / non-contiguous) → Task 3.
- Edge cases (empty, all-empty, length=0, pad_value cast) → Task 4 (pad_value-cast is numpy's `np.full` behavior, exercised implicitly; no separate test needed).
- Byte-identical to awkward across S1/int32/float32 → Tasks 1 & 4.
- Microbench → Task 5.
- Downstream gvl pass-through → explicitly out of scope (spec), not a task here.

**Placeholder scan:** No TBD/TODO; every code step shows full code; every run step shows the command and expected result. Task 5 Step 3 locates the skill file by command rather than hardcoding a path (path unknown at plan-write time) — acceptable, the action and content are fully specified.

**Type/name consistency:** `to_padded`, `_to_padded_copy`, `pad_value`, `length`, `out_len`, `itemsize`, `row_stride` used consistently across tasks. Helper names `_make_bytes_rag`, `_make_numeric_rag`, `_naive_pad_bytes`, `_naive_pad_numeric` defined in Task 1 / reused in Task 4.

**Red-green ordering:** Task 1 implements `to_padded(rag, pad_value)` with `out_len` fixed to the batch max — no `length` keyword. Task 2 adds `*, length=None`; its tests call `length=` and genuinely fail with `TypeError` before the Task 2 implementation step, then pass after. The kernel's `min(row_len, out_len)` truncation is written in Task 1 but only reachable once `length` can make `out_len < row_len` in Task 2, where the truncate tests exercise it. Guards (Task 3) likewise fail-first. Clean red-green throughout.
