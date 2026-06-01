# rag.to_packed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Numba-parallelized `to_packed()` to `seqpro.rag` (method + free function) that gathers each ragged row's bytes into a fresh contiguous, zero-based buffer, replacing `awkward.to_packed()` for `Ragged` arrays.

**Architecture:** A single generic `@nb.njit(parallel=True)` byte-view gather kernel does one contiguous read + write per row in a `prange` loop. A plain-NumPy driver in `rag/_ops.py` unboxes the array to `(data, offsets, shape)`, byte-scales offsets by `elem = prod(trailing_dims) * itemsize`, computes output offsets via cumsum, allocates the output, runs the kernel, and rebuilds a `Ragged` with 1-D offsets. Record layouts loop the kernel per field over shared offsets.

**Tech Stack:** Python, NumPy, Numba (`==0.58.1`), awkward (`==2.5.0`), pytest, marimo/seaborn (bench env).

---

## Background the implementer needs

`Ragged` wraps a single-ragged-dimension awkward array as `RagParts(data, shape, offsets)`:

- **`offsets` is 1-D** `(n+1,)` for a contiguous `ListOffsetArray` (row `i` = `data[offsets[i]:offsets[i+1]]`), **or 2-D** `(2, n)` for a `ListArray` (`starts = offsets[0]`, `stops = offsets[1]`), which arises after slicing/reordering (e.g. `rag[::-1]`).
- **`shape`** is e.g. `(n, None)` for `S1`/`float64`, or `(n, None, 4)` for OHE. `data` is already reshaped to `(total_elems, *trailing)` where `trailing = shape[rag_dim+1:]`.
- **Record layout**: `rag._parts` is a `dict[str, RagParts]`, all sharing one `offsets` ndarray. `rag.offsets` returns that shared array.

`elem` (bytes per ragged-axis element) = `prod(trailing) * data.dtype.itemsize`. For `S1` → 1, OHE `uint8 (.,4)` → 4, `float64` → 8.

Conventions to follow (see `rag/_ops.py::reverse_complement`): `@nb.njit(parallel=True, nogil=True, cache=True)`, `nb.prange`, kernel marked `# pragma: no cover`. Use `OFFSET_TYPE` (= `np.int64`) from `._utils` for offsets.

Run tests with: `pixi run -e dev pytest <path> -v`.

---

## Task 1: Kernel + `to_packed` free function (non-record layouts)

**Files:**
- Modify: `python/seqpro/rag/_ops.py`
- Test: `tests/test_rag_to_packed.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_rag_to_packed.py`:

```python
import awkward as ak
import numpy as np
import pytest

from seqpro.rag import Ragged
from seqpro.rag._ops import to_packed


def _unpacked(lengths, dtype):
    """A Ragged backed by a ListArray (2-D offsets) via row-reversal.

    ``lengths`` may include zeros to exercise empty rows.
    """
    lengths = np.asarray(lengths)
    data = np.arange(int(lengths.sum()), dtype=dtype)
    rev = Ragged.from_lengths(data, lengths)[::-1]  # reorder -> ListArray
    assert rev.offsets.ndim == 2
    return rev


class TestToPackedFlat:
    def test_2d_offsets_matches_awkward(self):
        rag = _unpacked([3, 0, 2, 4], np.dtype("float64"))
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert out.is_contiguous
        assert ak.to_list(out) == ak.to_list(rag)
        # matches awkward's packing exactly
        assert ak.to_list(out) == ak.to_list(ak.to_packed(rag))

    def test_bytes_dtype(self):
        seqs = ["ATG", "C", "GGGG"]
        data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").copy()
        lengths = np.array([len(s) for s in seqs])
        rag = Ragged.from_lengths(data, lengths)[::-1]
        out = to_packed(rag)
        assert out.dtype == np.dtype("S1")
        assert ak.to_list(out) == ak.to_list(rag)

    def test_trailing_fixed_dims(self):
        # OHE-like: (n, None, 4) uint8
        data = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        rag = Ragged.from_lengths(data, np.array([2, 1]))[::-1]
        out = to_packed(rag)
        assert out.shape[1:] == rag.shape[1:]
        assert ak.to_list(out) == ak.to_list(rag)

    def test_empty_array(self):
        rag = Ragged.empty((0, None), np.float64)
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert len(out) == 0

    def test_copy_true_returns_owned_buffer(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        out = to_packed(rag, copy=True)
        assert out.data.base is None  # freshly allocated
        out.data[0] = 999.0
        assert rag.data[0] == 0.0  # input untouched

    def test_copy_false_passthrough_when_packed(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        out = to_packed(rag, copy=False)
        assert out is rag  # zero-copy passthrough

    def test_copy_false_raises_when_unpacked(self):
        rag = _unpacked([3, 2], np.dtype("float64"))
        with pytest.raises(ValueError, match="already-packed"):
            to_packed(rag, copy=False)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py -v`
Expected: FAIL — `ImportError: cannot import name 'to_packed'`.

- [ ] **Step 3: Implement the kernel and driver**

In `python/seqpro/rag/_ops.py`, update the imports and `__all__`, and add the kernel + function. The current top of the file is:

```python
import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._array import Ragged, is_rag_dtype

__all__ = ["reverse_complement"]
```

Change `__all__` to:

```python
__all__ = ["reverse_complement", "to_packed"]
```

Add `OFFSET_TYPE` to the imports:

```python
from ._array import Ragged, is_rag_dtype
from ._utils import OFFSET_TYPE
```

Append to the end of the file:

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _pack(
    src_bytes: NDArray[np.uint8],
    in_starts: NDArray[np.int64],
    in_stops: NDArray[np.int64],
    out_bytes: NDArray[np.uint8],
    out_starts: NDArray[np.int64],
) -> None:  # pragma: no cover - exercised via to_packed
    """Gather each row's contiguous byte span into the packed output buffer.

    All offsets are in *byte* units. Row ``i`` copies
    ``src_bytes[in_starts[i]:in_stops[i]]`` to
    ``out_bytes[out_starts[i]:out_starts[i] + (in_stops[i] - in_starts[i])]``.
    One contiguous read + write per row, parallel across rows.
    """
    n = in_starts.shape[0]
    for i in nb.prange(n):
        length = in_stops[i] - in_starts[i]
        out_bytes[out_starts[i] : out_starts[i] + length] = src_bytes[
            in_starts[i] : in_stops[i]
        ]


def _pack_parts(
    data: NDArray, shape: tuple, offsets: NDArray, copy: bool
) -> tuple[NDArray, NDArray]:
    """Pack one flat (data, offsets) pair. Returns (packed_data, packed_offsets_1d).

    Raises ValueError if ``copy=False`` and the input is not already packed.
    """
    rag_dim = shape.index(None)
    trailing = shape[rag_dim + 1 :]
    elem = int(np.prod(trailing, dtype=np.int64)) * data.dtype.itemsize

    if offsets.ndim == 1:
        starts = offsets[:-1]
        stops = offsets[1:]
        zero_based = offsets.size > 0 and offsets[0] == 0
    else:
        starts = offsets[0]
        stops = offsets[1]
        zero_based = False  # ListArray -> treat as unpacked

    n_elems = data.shape[0]
    is_packed = (
        offsets.ndim == 1
        and zero_based
        and data.flags.c_contiguous
        and int(offsets[-1]) == n_elems
    )
    if is_packed and not copy:
        return data, offsets
    if not copy:
        raise ValueError(
            "to_packed(copy=False) requires already-packed input "
            "(contiguous, zero-based, 1-D offsets); got an unpacked array."
        )

    lengths = (stops - starts).astype(np.int64)
    out_offsets = np.empty(lengths.size + 1, dtype=OFFSET_TYPE)
    out_offsets[0] = 0
    np.cumsum(lengths, out=out_offsets[1:])

    if is_packed:  # copy=True on already-packed input
        new_data = np.ascontiguousarray(data).copy()
        return new_data, out_offsets

    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    src_bytes = data.view(np.uint8).reshape(-1)
    out_bytes = np.empty(int(out_offsets[-1]) * elem, dtype=np.uint8)
    _pack(
        src_bytes,
        (starts.astype(np.int64) * elem),
        (stops.astype(np.int64) * elem),
        out_bytes,
        (out_offsets[:-1] * elem),
    )
    out_data = out_bytes.view(data.dtype)
    if trailing:
        out_data = out_data.reshape(-1, *trailing)
    return out_data, out_offsets


def to_packed(rag: Ragged, *, copy: bool = True) -> Ragged:
    """Pack a Ragged array's data into a fresh contiguous, zero-based buffer.

    A Numba-parallelized replacement for ``Ragged(ak.to_packed(rag))``: it
    gathers each row's contiguous byte span into a new buffer with one
    parallel read+write per row, which is fast even when the source data is a
    ``np.memmap``. The result always has 1-D offsets starting at zero.

    Parameters
    ----------
    rag
        Ragged array (flat or record layout) with one ragged dimension.
    copy
        When ``True`` (default), always return a freshly allocated, owned
        packed array (safe to mutate in place afterwards). When ``False``,
        return the input zero-copy if it is already packed, and raise
        ``ValueError`` otherwise.

    Returns
    -------
    Ragged
        Contiguous, zero-based Ragged equal in value to ``rag``.
    """
    rag._ensure_parts()
    if isinstance(rag._parts, dict):
        import awkward as ak

        offsets = rag.offsets
        if offsets.ndim == 1 and (offsets.size == 0 or offsets[0] == 0) and not copy:
            # passthrough only if every field is already packed
            if all(
                p.data.flags.c_contiguous and int(offsets[-1]) == p.data.shape[0]
                for p in rag._parts.values()
            ):
                return rag
            raise ValueError(
                "to_packed(copy=False) requires already-packed input; "
                "got an unpacked record array."
            )
        if not copy:
            raise ValueError(
                "to_packed(copy=False) requires already-packed input; "
                "got an unpacked record array."
            )
        fields = {}
        for name, p in rag._parts.items():
            packed_data, packed_offsets = _pack_parts(p.data, p.shape, offsets, copy=True)
            fields[name] = Ragged.from_offsets(packed_data, p.shape, packed_offsets)
        return Ragged(ak.zip(fields, depth_limit=1))

    parts = rag._parts
    packed_data, packed_offsets = _pack_parts(parts.data, parts.shape, parts.offsets, copy)
    if packed_data is parts.data and packed_offsets is parts.offsets:
        return rag  # copy=False passthrough
    return Ragged.from_offsets(packed_data, parts.shape, packed_offsets)
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py::TestToPackedFlat -v`
Expected: PASS (all 7 tests). First run is slow (Numba compiles).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_ops.py tests/test_rag_to_packed.py
git commit -m "feat(rag): add Numba-parallelized to_packed for flat layouts"
```

---

## Task 2: Record-layout support

The record path was written in Task 1; this task adds its test coverage.

**Files:**
- Test: `tests/test_rag_to_packed.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_to_packed.py`:

```python
class TestToPackedRecord:
    def _record(self):
        import awkward as ak

        lengths = np.array([3, 2, 4])
        scores = np.arange(9, dtype=np.float64)
        flags = np.arange(9, dtype=np.int8)
        rec = ak.zip(
            {
                "score": Ragged.from_lengths(scores, lengths),
                "flag": Ragged.from_lengths(flags, lengths),
            }
        )
        return Ragged(rec)

    def test_record_unpacked_packs_all_fields(self):
        rec = self._record()[::-1]  # reorder -> ListArray-backed fields
        out = to_packed(rec)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert ak.to_list(out) == ak.to_list(rec)
        # fields share one offsets object (zero-copy SoA contract)
        assert out["score"].offsets is out["flag"].offsets

    def test_record_copy_false_passthrough(self):
        rec = self._record()
        out = to_packed(rec, copy=False)
        assert out is rec
```

- [ ] **Step 2: Run tests, verify pass**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py::TestToPackedRecord -v`
Expected: PASS. If `out["score"].offsets is out["flag"].offsets` fails, the `ak.zip` reassembly is not sharing offsets — verify `Ragged.__init__` extracts a single shared offsets for record layouts (it does, via `_extract_list_offsets`), so the assertion holds after construction.

- [ ] **Step 3: Commit**

```bash
git add tests/test_rag_to_packed.py
git commit -m "test(rag): cover to_packed record layouts"
```

---

## Task 3: `Ragged.to_packed()` method + exports

**Files:**
- Modify: `python/seqpro/rag/_array.py` (add method after `to_numpy`, ends at line 493)
- Modify: `python/seqpro/rag/__init__.py`
- Test: `tests/test_rag_to_packed.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_rag_to_packed.py`:

```python
class TestToPackedMethod:
    def test_method_delegates(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))[::-1]
        out = rag.to_packed()
        assert out.offsets.ndim == 1
        assert ak.to_list(out) == ak.to_list(rag)

    def test_method_copy_false(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        assert rag.to_packed(copy=False) is rag

    def test_exported_from_package(self):
        import seqpro.rag as rag_mod

        assert hasattr(rag_mod, "to_packed")
```

- [ ] **Step 2: Run, verify fail**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py::TestToPackedMethod -v`
Expected: FAIL — `AttributeError: 'Ragged' object has no attribute 'to_packed'`.

- [ ] **Step 3: Add the method**

In `python/seqpro/rag/_array.py`, immediately after the `to_numpy` method (after line 493, before `def __getitem__`), add:

```python
    def to_packed(self, copy: bool = True) -> Ragged[RDTYPE_co]:
        """Pack into a fresh contiguous, zero-based Ragged (1-D offsets).

        Numba-parallelized replacement for ``Ragged(ak.to_packed(self))``.
        See :func:`seqpro.rag.to_packed` for the ``copy`` semantics.

        Parameters
        ----------
        copy
            When ``True`` (default), return a freshly allocated owned array.
            When ``False``, return zero-copy if already packed, else raise.

        Returns
        -------
        Ragged[RDTYPE_co]
        """
        from ._ops import to_packed as _to_packed

        return _to_packed(self, copy=copy)
```

(The import is function-local to avoid a circular import: `_ops` imports from `_array`.)

- [ ] **Step 4: Update package exports**

In `python/seqpro/rag/__init__.py`, change the `_ops` import and `__all__`:

```python
from ._ops import reverse_complement, to_packed
```

and add `"to_packed"` to the `__all__` list (after `"reverse_complement"`).

- [ ] **Step 5: Run, verify pass**

Run: `pixi run -e dev pytest tests/test_rag_to_packed.py -v`
Expected: PASS (all classes).

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_array.py python/seqpro/rag/__init__.py tests/test_rag_to_packed.py
git commit -m "feat(rag): add Ragged.to_packed method and export to_packed"
```

---

## Task 4: Swap internal `ak.to_packed` call sites

**Files:**
- Modify: `python/seqpro/_encoders.py` (4 sites)
- Modify: `python/seqpro/alphabets/_alphabets.py` (1 site)
- Modify: `python/seqpro/rag/_ops.py` (`reverse_complement` fallback)

- [ ] **Step 1: Replace in `_encoders.py`**

There are four identical lines `seqs = Ragged(ak.to_packed(seqs))` inside `if isinstance(seqs, Ragged):` blocks. Replace each with:

```python
        seqs = seqs.to_packed()
```

- [ ] **Step 2: Replace in `alphabets/_alphabets.py`**

Replace the single line `seqs = Ragged(ak.to_packed(seqs))` with:

```python
        seqs = seqs.to_packed()
```

(The following `offsets = seqs.offsets  # 1D (n+1,) after to_packed` comment stays valid — `to_packed` always yields 1-D offsets.)

- [ ] **Step 3: Replace in `rag/_ops.py` (`reverse_complement`)**

Find the non-contiguous fallback in `reverse_complement`:

```python
    if not rag.is_contiguous:
        import awkward as ak

        rag = Ragged(ak.to_packed(rag))
```

Replace with:

```python
    if not rag.is_contiguous:
        rag = to_packed(rag)
```

(Remove the now-unused local `import awkward as ak` here. `to_packed` is defined in the same module.)

- [ ] **Step 4: Run the full suite, verify pass**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS — existing encoder/alphabet/RC tests guard correctness of the swap.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_encoders.py python/seqpro/alphabets/_alphabets.py python/seqpro/rag/_ops.py
git commit -m "perf(rag): use to_packed at internal ak.to_packed call sites"
```

---

## Task 5: Remove dead `as_contiguous` path in `unbox`

No caller in the package or tests passes `as_contiguous=True`; remove the parameter and its `ak.to_packed` branch.

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`unbox`, lines 731-753)

- [ ] **Step 1: Confirm there are no callers**

Run: `grep -rn "as_contiguous" python/ tests/`
Expected: only the definition in `_array.py` (the parameter, docstring, and `if as_contiguous:` branch). No call site passes it.

- [ ] **Step 2: Edit `unbox`**

Current signature and head:

```python
def unbox(
    arr: ak.Array | Ragged[DTYPE_co], as_contiguous: bool = False
) -> RagParts[DTYPE_co]:
    """Unbox an awkward array with a single ragged dimension into data, offsets, and shape.
    Is guaranteed to be zero-copy if as_contiguous is False, in which case the data is a view
    of the original array.

    Parameters
    ----------
    arr
        The awkward array to unbox.
    as_contiguous
        If True, the data will be returned as a contiguous array. May force a copy into memory.
        If the underlying data is memory-mapped, this could cause an out-of-memory error.

    Returns
    -------
    RagParts[DTYPE_co]
        Data, shape, and offsets extracted from the awkward array.
    """
    if as_contiguous:
        arr = ak.to_packed(arr)

    node = cast(Content, ak.to_layout(arr, allow_record=False))
```

Replace with:

```python
def unbox(arr: ak.Array | Ragged[DTYPE_co]) -> RagParts[DTYPE_co]:
    """Unbox an awkward array with a single ragged dimension into data, offsets, and shape.
    Always zero-copy: the returned data is a view of the original array.

    Parameters
    ----------
    arr
        The awkward array to unbox.

    Returns
    -------
    RagParts[DTYPE_co]
        Data, shape, and offsets extracted from the awkward array.
    """
    node = cast(Content, ak.to_layout(arr, allow_record=False))
```

- [ ] **Step 3: Run the full suite, verify pass**

Run: `pixi run -e dev pytest tests/ -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/seqpro/rag/_array.py
git commit -m "refactor(rag): drop unused as_contiguous packing path in unbox"
```

---

## Task 6: Microbenchmark script

**Files:**
- Create: `benchmarks/bench_to_packed.py`

- [ ] **Step 1: Write the benchmark script**

Create `benchmarks/bench_to_packed.py`:

```python
"""Microbenchmark: seqpro.rag.to_packed vs ak.to_packed vs NumPy gather.

Run in the bench env:
    pixi run -e bench python benchmarks/bench_to_packed.py --out bench_to_packed

Outputs <out>.csv and <out>_*.png. Sweeps n_rows, mean length, length
distribution, source (RAM vs memmap), dtype itemsize, and thread count, and
prints an effect-size ranking of which axes most affect the speedup ratio.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def make_lengths(rng, n_rows, mean_len, distribution):
    if distribution == "uniform":
        lo, hi = max(1, mean_len // 2), mean_len * 2
        return rng.integers(lo, hi + 1, size=n_rows)
    # long-tailed: most rows short, a few very long (load imbalance)
    lengths = rng.integers(1, max(2, mean_len // 4), size=n_rows)
    n_big = max(1, n_rows // 100)
    lengths[rng.choice(n_rows, n_big, replace=False)] = mean_len * 50
    return lengths


def build(rng, n_rows, mean_len, distribution, dtype, source, tmpdir):
    import seqpro.rag as spr

    lengths = make_lengths(rng, n_rows, mean_len, distribution).astype(np.int64)
    total = int(lengths.sum())
    if source == "memmap":
        path = Path(tmpdir) / "data.dat"
        data = np.memmap(path, dtype=dtype, mode="w+", shape=(total,))
        data[:] = rng.integers(0, 255, size=total).astype(dtype)
    else:
        data = rng.integers(0, 255, size=total).astype(dtype)
    rag = spr.Ragged.from_lengths(data, lengths)
    # reorder so the layout is an (unpacked) ListArray, forcing a real gather
    return rag[rng.permutation(n_rows)], total * np.dtype(dtype).itemsize


def numpy_gather(rag):
    import seqpro.rag as spr

    parts = rag._parts
    offs = parts.offsets
    starts, stops = (offs[0], offs[1]) if offs.ndim == 2 else (offs[:-1], offs[1:])
    lengths = stops - starts
    out_off = np.empty(lengths.size + 1, dtype=np.int64)
    out_off[0] = 0
    np.cumsum(lengths, out=out_off[1:])
    idx = np.repeat(starts - out_off[:-1], lengths) + np.arange(int(out_off[-1]))
    return spr.Ragged.from_offsets(parts.data[idx], parts.shape, out_off)


def timeit(fn, arg, repeats):
    fn(arg)  # warm up (JIT, caches)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="bench_to_packed")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n-rows", type=int, nargs="+", default=[1000, 10000, 100000, 1000000])
    p.add_argument("--mean-len", type=int, nargs="+", default=[16, 256, 4096])
    p.add_argument("--distributions", nargs="+", default=["uniform", "longtail"])
    p.add_argument("--sources", nargs="+", default=["ram", "memmap"])
    p.add_argument("--dtypes", nargs="+", default=["uint8", "float64"])
    p.add_argument("--threads", type=int, nargs="+", default=[int(os.cpu_count() or 1)])
    args = p.parse_args()

    import awkward as ak
    import pandas as pd
    import seqpro.rag as spr  # noqa: F401  (ensures kernel import)
    import tempfile

    rng = np.random.default_rng(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for nthreads in args.threads:
            import numba

            numba.set_num_threads(nthreads)
            impls = {
                "seqpro": lambda r: r.to_packed(),
                "ak": lambda r: spr.Ragged(ak.to_packed(r)),
                "numpy": numpy_gather,
            }
            for n_rows in args.n_rows:
                for mean_len in args.mean_len:
                    for dist in args.distributions:
                        for src in args.sources:
                            for dt in args.dtypes:
                                rag, nbytes = build(
                                    rng, n_rows, mean_len, dist, dt, src, tmp
                                )
                                for name, fn in impls.items():
                                    t = timeit(fn, rag, args.repeats)
                                    rows.append(
                                        dict(
                                            impl=name, n_rows=n_rows, mean_len=mean_len,
                                            distribution=dist, source=src, dtype=dt,
                                            threads=nthreads, seconds=t,
                                            gbps=nbytes / t / 1e9,
                                        )
                                    )

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False)
    _plots_and_summary(df, args.out)


def _plots_and_summary(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # throughput vs n_rows, faceted by mean_len x dtype, hue impl
    g = sns.relplot(
        data=df, x="n_rows", y="gbps", hue="impl", style="source",
        col="mean_len", row="dtype", kind="line", marker="o",
        facet_kws=dict(sharey=False),
    )
    g.set(xscale="log")
    g.savefig(f"{out}_throughput.png", dpi=120)

    # thread scaling for seqpro
    if df["threads"].nunique() > 1:
        sub = df[df.impl == "seqpro"]
        gt = sns.relplot(data=sub, x="threads", y="gbps", hue="mean_len", kind="line", marker="o")
        gt.savefig(f"{out}_threads.png", dpi=120)
    plt.close("all")

    # effect-size ranking on speedup (seqpro / ak) — goal 2
    wide = df.pivot_table(
        index=["n_rows", "mean_len", "distribution", "source", "dtype", "threads"],
        columns="impl", values="gbps",
    ).reset_index()
    wide["speedup"] = wide["seqpro"] / wide["ak"]
    print("\n=== Speedup (seqpro / ak) summary ===")
    print(wide["speedup"].describe())
    print("\n=== Mean speedup by axis (effect size) ===")
    axes = ["n_rows", "mean_len", "distribution", "source", "dtype", "threads"]
    spreads = {}
    for ax in axes:
        means = wide.groupby(ax)["speedup"].mean()
        spreads[ax] = means.max() - means.min()
        print(f"\n{ax}:")
        print(means)
    print("\n=== Axes ranked by speedup spread (most influential first) ===")
    for ax, s in sorted(spreads.items(), key=lambda kv: -kv[1]):
        print(f"  {ax:14s} spread={s:.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script (tiny sweep)**

Run: `pixi run -e bench python benchmarks/bench_to_packed.py --out /tmp/bench_smoke --n-rows 1000 --mean-len 64 --repeats 2 --sources ram --dtypes uint8`
Expected: completes without error; prints the speedup summary and ranking; writes `/tmp/bench_smoke.csv` and `/tmp/bench_smoke_throughput.png`.

- [ ] **Step 3: Run the full sweep and record findings**

Run: `pixi run -e bench python benchmarks/bench_to_packed.py --out benchmarks/results_to_packed`
Expected: `benchmarks/results_to_packed.csv` + PNGs. Confirm goal 1 (median `seqpro` speedup > 1 vs `ak`, including memmap rows) and note the top-ranked axes for goal 2.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_to_packed.py
git commit -m "bench(rag): microbenchmark to_packed throughput vs ak.to_packed"
```

(Do not commit large CSV/PNG result artifacts unless the maintainer wants them tracked.)

---

## Task 7: Update the seqpro skill

`CLAUDE.md` requires any PR adding a public feature to update `skills/seqpro/SKILL.md` in the same PR.

**Files:**
- Modify: `skills/seqpro/SKILL.md`

- [ ] **Step 1: Locate the Ragged section**

Run: `grep -n "Ragged" skills/seqpro/SKILL.md`
Find where `Ragged` methods (e.g. `to_numpy`, `view`) or the rag module API are documented.

- [ ] **Step 2: Add a `to_packed` entry**

Add an entry alongside the other `Ragged` methods, matching the file's existing style. Use this content:

> **`Ragged.to_packed(copy=True)`** / **`seqpro.rag.to_packed(rag, *, copy=True)`** — Pack a ragged array's data into a fresh contiguous, zero-based buffer (1-D offsets). Numba-parallelized replacement for `Ragged(ak.to_packed(rag))`; fast even when the source is a `np.memmap`. With `copy=True` (default) you always get an owned buffer (safe to mutate, e.g. before an in-place op); with `copy=False` you get a zero-copy passthrough if the array is already packed, or a `ValueError` if it is not.

- [ ] **Step 3: Commit**

```bash
git add skills/seqpro/SKILL.md
git commit -m "docs(skill): document Ragged.to_packed"
```

---

## Final verification

- [ ] Run the full test suite: `pixi run -e dev pytest tests/ -v` → all pass.
- [ ] Lint/format: `pixi run -e dev ruff check python/ tests/ benchmarks/ && pixi run -e dev ruff format --check python/ tests/ benchmarks/`.
- [ ] Confirm benchmark goal 1 (speedup > 1 incl. memmap) and goal 2 (axis ranking) are recorded from the Task 6 run.
```
