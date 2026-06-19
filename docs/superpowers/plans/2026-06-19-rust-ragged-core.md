# Rust-native `Ragged` — Spec A (core, single-level) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust-native, single-ragged-level `Ragged` array type backed by NumPy buffers and stateless Rust layout kernels, at parity with today's awkward-backed non-record `Ragged`, without removing awkward yet.

**Architecture:** A plain Python `Ragged` class (not an `ak.Array` subclass) wraps a `RaggedLayout` value object holding `data` + one offsets array + `shape` + optional string-leaf byte offsets. Pure-Python orchestration and NumPy interop; Rust kernels (`src/ragged.rs`) do the structural hot paths (index/slice/validate). The new type is built and tested *alongside* the existing awkward `Ragged`, which serves as the live differential oracle; the public `seqpro.rag.Ragged` export is **not** flipped to the new type in this spec (deferred to Spec D's cutover).

**Tech Stack:** Python 3.9+ (abi3), NumPy, Numba (existing pack/pad kernels, reused), PyO3 0.20 / maturin, Rust (ndarray, rayon), pytest, pytest-cases, Hypothesis, proptest.

## Global Constraints

- **Non-branching axis model:** `(*leading_int, None × R, *trailing_int)`. Spec A handles `R == 1` only; `R > 1` (more than one `None`) raises `NotImplementedError` referencing Spec C. Verbatim message stem: `"nested raggedness (>1 ragged level) lands in Spec C"`.
- **String-leaf rule:** a `np.bytes_`/`S1` element is an opaque variable-width leaf; its byte length is **never** counted in `.shape` or `.offsets`. A flat collection of `N` sequences has `shape == (N,)`. `shape.count(None)` counts ragged axes only.
- **Records out of scope:** record/struct inputs raise `NotImplementedError` with message stem `"record-layout Ragged lands in Spec B"`.
- **Awkward stays installed** in Spec A; the new `Ragged` does not subclass `ak.Array`. Awkward is touched only in `_ingest.py` (ingestion + `to_ak` shims) and in tests (oracle).
- **Public export unchanged:** `seqpro.rag.Ragged` keeps pointing at the existing `seqpro.rag._array.Ragged` for the duration of Spec A. The new type is reachable as `seqpro.rag._core.Ragged`.
- **Validation is front-loaded** in constructors (project convention: one obvious fast-fail check, no per-feature error flags).
- **No naive NumPy in hot paths** (CLAUDE.md): reuse the existing Numba pack/pad kernels; structural index/slice/validate go to Rust once the pure-Python version is green.
- **Dev loop:** `pixi shell -e dev`; tests `pytest tests/test_ragged_core.py`; Rust build `maturin develop`; Rust tests `cargo test`; lint `ruff check python/ tests/` + `ruff format`.
- **Commits:** conventional-commit prefixes (`feat:`, `test:`, `refactor:`, `docs:`). End commit messages with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

## File structure

| File | Responsibility |
|---|---|
| `python/seqpro/rag/_layout.py` (create) | `RaggedLayout` value object + pure-Python validation helpers. No awkward, no Ragged class. |
| `python/seqpro/rag/_core.py` (create) | New `Ragged` class: constructors, properties, indexing, ufunc/NumPy interop, view/squeeze/reshape/to_numpy/to_packed/to_padded. |
| `python/seqpro/rag/_ingest.py` (create) | Awkward ingestion shim (`layout_from_ak`) + `to_ak`. Isolates the awkward import so Spec D can delete one file. |
| `src/ragged.rs` (create) | Rust kernels: `ragged_validate`, `ragged_index`, `ragged_slice`. proptest unit tests. |
| `src/lib.rs` (modify) | Register `_ragged_validate`, `_ragged_index`, `_ragged_slice` in the `#[pymodule]`. |
| `tests/test_ragged_core.py` (create) | Standalone unit tests + Hypothesis differential tests against the awkward oracle (`seqpro.rag._array.Ragged`). |

---

## Task 1: `RaggedLayout` value object + pure-Python validation

**Files:**
- Create: `python/seqpro/rag/_layout.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `OFFSET_TYPE`, `lengths_to_offsets` from `seqpro.rag._utils`.
- Produces:
  - `class RaggedLayout(Generic[DTYPE_co])` with attrs fields `data: NDArray`, `offsets: list[NDArray]`, `shape: tuple[int | None, ...]`, `str_offsets: NDArray | None = None`.
  - `RaggedLayout.is_string -> bool` (property): `str_offsets is not None`.
  - `RaggedLayout.n_ragged -> int` (property): `self.shape.count(None)`.
  - `validate_layout(layout: RaggedLayout) -> None` — raises `NotImplementedError` (>1 `None`), `ValueError` (offsets not monotonic non-decreasing; segment count != `prod(shape[:rag_dim])`; for numeric, `len(offsets) != shape.count(None)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ragged_core.py
import numpy as np
import pytest
from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, validate_layout


def test_layout_numeric_basic():
    data = np.arange(10, dtype=np.int32)
    offsets = lengths_to_offsets(np.array([3, 2, 5], dtype=np.uint32))
    layout = RaggedLayout(data=data, offsets=[offsets], shape=(3, None))
    validate_layout(layout)
    assert layout.is_string is False
    assert layout.n_ragged == 1


def test_layout_string_flat_collection():
    # flat collection of sequences -> no ragged axis, just a string leaf
    data = np.frombuffer(b"cathithere", dtype="S1")
    str_offsets = np.array([0, 3, 5, 10], dtype=OFFSET_TYPE)
    layout = RaggedLayout(data=data, offsets=[], shape=(3,), str_offsets=str_offsets)
    validate_layout(layout)
    assert layout.is_string is True
    assert layout.n_ragged == 0


def test_layout_rejects_multiple_none():
    data = np.arange(6)
    with pytest.raises(NotImplementedError, match="Spec C"):
        validate_layout(
            RaggedLayout(data=data, offsets=[np.array([0, 6])], shape=(2, None, None))
        )


def test_layout_rejects_nonmonotonic_offsets():
    with pytest.raises(ValueError, match="monotonic"):
        validate_layout(
            RaggedLayout(
                data=np.arange(5),
                offsets=[np.array([0, 3, 2, 5], dtype=OFFSET_TYPE)],
                shape=(3, None),
            )
        )


def test_layout_rejects_segment_count_mismatch():
    with pytest.raises(ValueError, match="segment"):
        validate_layout(
            RaggedLayout(
                data=np.arange(10),
                offsets=[lengths_to_offsets(np.array([3, 2, 5]))],  # 3 segments
                shape=(4, None),  # claims 4
            )
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k layout -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'seqpro.rag._layout'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/seqpro/rag/_layout.py
from __future__ import annotations

from typing import Generic, TypeVar

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

from ._utils import OFFSET_TYPE  # noqa: F401  (re-exported convenience)

DTYPE_co = TypeVar("DTYPE_co", covariant=True)

_SPEC_C_MSG = "nested raggedness (>1 ragged level) lands in Spec C"


@define
class RaggedLayout(Generic[DTYPE_co]):
    """Buffers backing a single-level Ragged array.

    data
        Flat 1-D numeric buffer, or an S1 buffer for a string leaf; 2-D
        ``(total, *trailing)`` when the leaf has trailing regular dims.
    offsets
        One ``(N+1,)`` or ``(2, N)`` array per ragged *axis*, outermost-first.
        Empty for a flat string collection (string leaf, no axis).
    shape
        ``(*leading_int, None x R, *trailing_int)``.
    str_offsets
        Per-element byte boundaries for a string leaf; ``None`` for numeric.
        Never counted in ``shape``/``offsets``.
    """

    data: NDArray
    offsets: list[NDArray]
    shape: tuple[int | None, ...]
    str_offsets: NDArray | None = field(default=None)

    @property
    def is_string(self) -> bool:
        return self.str_offsets is not None

    @property
    def n_ragged(self) -> int:
        return self.shape.count(None)


def _is_monotonic(offsets: NDArray) -> bool:
    arr = offsets if offsets.ndim == 1 else offsets.ravel()
    return bool(np.all(np.diff(arr) >= 0)) if arr.size else True


def validate_layout(layout: RaggedLayout) -> None:
    if layout.n_ragged > 1:
        raise NotImplementedError(_SPEC_C_MSG)

    for off in layout.offsets:
        if not _is_monotonic(off):
            raise ValueError("offsets must be monotonic non-decreasing")

    if layout.n_ragged == 1:
        if len(layout.offsets) != 1:
            raise ValueError(
                f"expected 1 offsets array for 1 ragged axis, got {len(layout.offsets)}"
            )
        offsets = layout.offsets[0]
        n_seg = len(offsets) - 1 if offsets.ndim == 1 else offsets.shape[1]
        rag_dim = layout.shape.index(None)
        expected = int(np.prod(layout.shape[:rag_dim], dtype=np.int64))
        if n_seg != expected:
            raise ValueError(
                f"segment count {n_seg} != product of leading dims {expected}"
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k layout -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_layout.py tests/test_ragged_core.py
git commit -m "feat(rag): RaggedLayout value object + validation"
```

---

## Task 2: `Ragged` constructors + core properties

**Files:**
- Create: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout`, `validate_layout` (Task 1); `lengths_to_offsets`, `OFFSET_TYPE`.
- Produces:
  - `class Ragged(Generic[RDTYPE_co])` wrapping `self._layout: RaggedLayout`.
  - `Ragged(layout: RaggedLayout)` — primary constructor (other inputs added in Task 8).
  - staticmethods `from_offsets(data, shape, offsets) -> Ragged`, `from_lengths(data, lengths) -> Ragged`; classmethod `empty(shape, dtype) -> Ragged`.
  - properties: `data -> NDArray`, `offsets -> NDArray`, `shape -> tuple`, `dtype -> np.dtype`, `rag_dim -> int`, `lengths -> NDArray`.
  - For bytes data passed to `from_offsets`/`from_lengths` with **no trailing fixed dims**, the supplied offsets describe the string leaf: a flat collection collapses to `shape == (leading,)` with `str_offsets` set and `offsets == []`. `offsets` property returns `str_offsets` when there is no ragged axis.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
from seqpro.rag._core import Ragged


def test_from_lengths_numeric():
    data = np.arange(10, dtype=np.int32)
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3, None)
    assert rag.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(rag.data, data)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5, 10]))
    np.testing.assert_array_equal(rag.lengths, np.array([3, 2, 5]))
    assert rag.rag_dim == 1


def test_from_lengths_nested_leading_dims():
    # case_nested from the legacy suite: leading (3,2,1), one ragged axis
    data = np.arange(10)
    lengths = np.array([[[1], [3]], [[2], [1]], [[1], [2]]])
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3, 2, 1, None)
    assert rag.rag_dim == 3
    np.testing.assert_array_equal(rag.offsets, lengths_to_offsets(lengths))


def test_from_lengths_string_collapses_to_leaf():
    # NEW string-leaf behavior: flat collection -> (N,), not (N, None)
    data = np.frombuffer(b"cathithere", dtype="S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3,)
    assert rag.dtype == np.dtype("S1")
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5, 10]))


def test_from_offsets_numeric_trailing_dim():
    data = np.zeros((6, 4), dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6]))
    assert rag.shape == (2, None, 4)
    assert rag.data.shape == (6, 4)


def test_empty():
    rag = Ragged.empty((3, None), np.float64)
    assert rag.shape == (3, None)
    assert rag.data.size == 0
    np.testing.assert_array_equal(rag.offsets, np.zeros(4, dtype=np.int64))


def test_from_offsets_rejects_two_none():
    with pytest.raises(NotImplementedError, match="Spec C"):
        Ragged.from_offsets(np.arange(6), (2, None, None), np.array([0, 6]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k "from_lengths or from_offsets or empty" -v`
Expected: FAIL with `ImportError: cannot import name 'Ragged' from 'seqpro.rag._core'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/seqpro/rag/_core.py
from __future__ import annotations

from typing import Generic, TypeVar, cast

import numpy as np
from numpy.typing import NDArray

from ._layout import RaggedLayout, validate_layout
from ._utils import OFFSET_TYPE, lengths_to_offsets

RDTYPE_co = TypeVar("RDTYPE_co", covariant=True)
DTYPE_co = TypeVar("DTYPE_co", covariant=True)


def _build_layout(
    data: NDArray, shape: tuple[int | None, ...], offsets: NDArray
) -> RaggedLayout:
    """Build a single-level layout, applying the string-leaf rule.

    For bytes (S1) data with no trailing fixed dims, the single ragged axis is
    the string itself -> collapse to a string leaf: drop the None, store the
    supplied offsets as str_offsets.
    """
    is_bytes = data.dtype.kind == "S"
    n_none = shape.count(None)
    if is_bytes and n_none == 1:
        rag_dim = shape.index(None)
        trailing = shape[rag_dim + 1 :]
        if all(d is not None for d in trailing) and len(trailing) == 0:
            leaf_shape = shape[:rag_dim]
            return RaggedLayout(
                data=data, offsets=[], shape=leaf_shape, str_offsets=offsets
            )
    return RaggedLayout(data=data, offsets=[offsets], shape=shape)


class Ragged(Generic[RDTYPE_co]):
    """A non-branching ragged array with a single ragged axis (Spec A)."""

    __slots__ = ("_layout",)

    def __init__(self, data: RaggedLayout):
        if not isinstance(data, RaggedLayout):
            raise TypeError(
                "Ragged(...) currently accepts a RaggedLayout; "
                "awkward/Ragged ingestion is added in a later task."
            )
        validate_layout(data)
        self._layout = data

    @staticmethod
    def from_offsets(
        data: NDArray, shape: tuple[int | None, ...], offsets: NDArray
    ) -> "Ragged":
        if shape.count(None) > 1:
            raise NotImplementedError(
                "nested raggedness (>1 ragged level) lands in Spec C"
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have exactly one None ragged dimension")
        offsets = np.ascontiguousarray(offsets, dtype=OFFSET_TYPE)
        return Ragged(_build_layout(data, shape, offsets))

    @staticmethod
    def from_lengths(data: NDArray, lengths: NDArray) -> "Ragged":
        offsets = lengths_to_offsets(lengths)
        trailing = data.shape[1:]
        shape = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets)

    @classmethod
    def empty(cls, shape: int | tuple[int | None, ...], dtype) -> "Ragged":
        if isinstance(shape, int):
            shape = (shape,)
        rag_dim = shape.index(None)
        trailing = shape[rag_dim + 1 :]  # all int (only the ragged dim is None)
        data = np.empty((0, *trailing), dtype=dtype) if trailing else np.empty(0, dtype=dtype)
        n_seg = int(np.prod(shape[:rag_dim], dtype=np.int64))
        offsets = np.zeros(n_seg + 1, dtype=OFFSET_TYPE)
        return Ragged.from_offsets(data, shape, offsets)

    @property
    def data(self) -> NDArray:
        return self._layout.data

    @property
    def offsets(self) -> NDArray:
        if self._layout.offsets:
            return self._layout.offsets[0]
        assert self._layout.str_offsets is not None
        return self._layout.str_offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self._layout.shape

    @property
    def dtype(self) -> np.dtype:
        return self._layout.data.dtype

    @property
    def rag_dim(self) -> int:
        return self._layout.shape.index(None)

    @property
    def lengths(self) -> NDArray:
        offsets = self.offsets
        lengths = np.diff(offsets) if offsets.ndim == 1 else np.diff(offsets, axis=0)
        rag_dim = self._layout.shape.index(None) if None in self._layout.shape else len(self._layout.shape)
        return lengths.reshape(self._layout.shape[:rag_dim] or (-1,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k "from_lengths or from_offsets or empty" -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): Ragged constructors and core properties"
```

---

## Task 3: state predicates + `view`

**Files:**
- Modify: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Produces on `Ragged`: `is_empty -> bool`, `is_contiguous -> bool`, `is_base -> bool`, `view(dtype) -> Ragged` (zero-copy reinterpret of the flat `data`, same offsets/shape).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_state_predicates():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    assert rag.is_empty is False
    assert rag.is_contiguous is True
    assert rag.is_base is True
    empty = Ragged.empty((3, None), np.int32)
    assert empty.is_empty is True


def test_view_reinterprets_dtype_zero_copy():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
    v = rag.view(np.uint64)
    assert v.dtype == np.dtype(np.uint64)
    assert v.data.base is not None  # zero-copy view
    np.testing.assert_array_equal(v.data.view(np.int64), rag.data)
    assert v.offsets is rag.offsets  # offsets reused
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k "predicates or view" -v`
Expected: FAIL with `AttributeError: 'Ragged' object has no attribute 'is_empty'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to Ragged in python/seqpro/rag/_core.py
    @property
    def is_empty(self) -> bool:
        offsets = self.offsets
        if offsets.ndim == 1:
            return bool(offsets.size == 0 or offsets[-1] == 0)
        return bool(np.all(offsets[0] == offsets[1]))

    @property
    def is_contiguous(self) -> bool:
        return self.offsets.ndim == 1 and self._layout.data.flags.c_contiguous

    @property
    def is_base(self) -> bool:
        offsets = self.offsets
        return bool(
            self._layout.data.base is None
            and self.is_contiguous
            and offsets[0] == 0
            and offsets[-1] == self._layout.data.shape[0]
        )

    def view(self, dtype) -> "Ragged":
        new_layout = RaggedLayout(
            data=self._layout.data.view(dtype),
            offsets=self._layout.offsets,
            shape=self._layout.shape,
            str_offsets=self._layout.str_offsets,
        )
        return Ragged(new_layout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k "predicates or view" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): state predicates and view"
```

---

## Task 4: `__getitem__` (pure-Python indexing/slicing)

**Files:**
- Modify: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Produces on `Ragged`: `__getitem__(where)`:
  - integer index on the ragged axis (only-leading-dim case, `shape == (N, None[, *trailing])`) → row as `NDArray` (numeric) / `bytes` scalar (string leaf, flat collection).
  - slice / boolean mask / integer-array on the leading axis → `Ragged` over selected rows, producing `(2, M)` start/stop offsets (gather, no data copy). For a contiguous slice, offsets are re-based as a 1-D view.
- Helper `_starts_stops(self) -> tuple[NDArray, NDArray]` returning the per-row `(starts, stops)` views; the gather branch of `__getitem__` consumes these and Task 11 routes the gather through the Rust `_ragged_select` kernel.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_getitem_int_returns_row():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    np.testing.assert_array_equal(rag[0], np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(rag[1], np.array([3, 4], dtype=np.int32))


def test_getitem_int_string_leaf():
    rag = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    assert rag[0] == b"cat"
    assert rag[2] == b"there"


def test_getitem_slice_returns_ragged():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[1:3]
    assert isinstance(sub, Ragged)
    assert sub.offsets.ndim == 2  # (2, M) start/stop gather
    np.testing.assert_array_equal(sub[0], np.array([3, 4]))
    np.testing.assert_array_equal(sub[1], np.array([5, 6, 7, 8, 9]))


def test_getitem_mask_returns_ragged():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[np.array([True, False, True])]
    np.testing.assert_array_equal(sub[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1], np.array([5, 6, 7, 8, 9]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k getitem -v`
Expected: FAIL with `TypeError: 'Ragged' object is not subscriptable`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to Ragged in python/seqpro/rag/_core.py

    def _starts_stops(self) -> tuple[NDArray, NDArray]:
        offsets = self.offsets
        if offsets.ndim == 1:
            return offsets[:-1], offsets[1:]
        return offsets[0], offsets[1]

    def __getitem__(self, where):
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            row = self._layout.data[lo:hi]
            if self._layout.is_string:
                return row.tobytes()
            return row
        # slice / mask / int-array on the leading axis -> gather to (2, M)
        sel_starts = np.ascontiguousarray(starts[where], dtype=OFFSET_TYPE)
        sel_stops = np.ascontiguousarray(stops[where], dtype=OFFSET_TYPE)
        new_offsets = np.stack([sel_starts, sel_stops], 0)
        new_shape = (len(sel_starts), *self._layout.shape[self.rag_dim if None in self._layout.shape else len(self._layout.shape):])
        if None not in self._layout.shape:  # string-leaf flat collection
            new_layout = RaggedLayout(
                data=self._layout.data,
                offsets=[],
                shape=(len(sel_starts),),
                str_offsets=new_offsets,
            )
        else:
            new_layout = RaggedLayout(
                data=self._layout.data,
                offsets=[new_offsets],
                shape=(len(sel_starts), *self._layout.shape[self.rag_dim:]),
                str_offsets=self._layout.str_offsets,
            )
        return Ragged(new_layout)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k getitem -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): __getitem__ indexing and slicing"
```

---

## Task 5: ufunc & NumPy interop

**Files:**
- Modify: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Produces on `Ragged`: `__array_ufunc__(ufunc, method, *inputs, **kwargs)` — element-wise (`method == "__call__"`) ufuncs apply to each operand's flat `data` (Ragged operands must share offsets identity or equal offsets; scalars pass through) and rewrap with the same offsets/shape; non-`__call__` methods raise `NotImplementedError`. `__array__()` delegates to `to_numpy` (added Task 7); for now raises `TypeError` if jagged. `_with_data(new_data) -> Ragged` helper (same offsets/shape, new flat buffer).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_ufunc_scalar_mul():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    out = rag * 2.0
    assert isinstance(out, Ragged)
    np.testing.assert_array_equal(out.data, np.arange(6) * 2.0)
    assert out.offsets is rag.offsets


def test_ufunc_unary():
    rag = Ragged.from_lengths(np.arange(1, 7, dtype=np.float64), np.array([2, 1, 3]))
    out = np.log1p(rag)
    np.testing.assert_allclose(out.data, np.log1p(np.arange(1, 7)))


def test_ufunc_two_ragged_shared_offsets():
    a = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    b = a.view(np.float64)
    out = a + b
    np.testing.assert_array_equal(out.data, a.data * 2)


def test_ufunc_reduce_raises():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    with pytest.raises(NotImplementedError):
        np.add.reduce(rag)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k ufunc -v`
Expected: FAIL — operators return NotImplemented / wrong type.

- [ ] **Step 3: Write minimal implementation**

```python
# add to Ragged in python/seqpro/rag/_core.py

    def _with_data(self, new_data: NDArray) -> "Ragged":
        return Ragged(
            RaggedLayout(
                data=new_data,
                offsets=self._layout.offsets,
                shape=self._layout.shape,
                str_offsets=self._layout.str_offsets,
            )
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            raise NotImplementedError(
                f"Ragged supports only element-wise ufuncs, not method={method!r}"
            )
        ref_offsets = self.offsets
        raw_inputs = []
        for x in inputs:
            if isinstance(x, Ragged):
                if x.offsets is not ref_offsets and not np.array_equal(
                    x.offsets, ref_offsets
                ):
                    raise ValueError("ufunc operands must share offsets")
                raw_inputs.append(x.data)
            else:
                raw_inputs.append(x)
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        return self._with_data(result)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k ufunc -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): element-wise ufunc interop"
```

---

## Task 6: `squeeze` and `reshape`

**Files:**
- Modify: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Produces on `Ragged`: `squeeze(axis=None) -> Ragged | NDArray` and `reshape(*shape) -> Ragged`, operating on leading/trailing **regular** dims and the data buffer's trailing dims; ragged axis preserved. Ported from `seqpro.rag._array.Ragged.squeeze`/`reshape` (non-record path only).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_squeeze_trailing_one():
    data = np.arange(6, dtype=np.int64).reshape(6, 1)
    rag = Ragged.from_offsets(data, (3, None, 1), lengths_to_offsets(np.array([2, 1, 3])))
    sq = rag.squeeze()
    assert sq.shape == (3, None)
    np.testing.assert_array_equal(sq.data, np.arange(6))


def test_reshape_leading():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int64), np.array([2, 1, 3, 1, 2, 1]))
    re = rag.reshape(2, 3, None)
    assert re.shape == (2, 3, None)
    np.testing.assert_array_equal(re.data, np.arange(10))
    assert re.offsets is rag.offsets
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k "squeeze or reshape" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to Ragged in python/seqpro/rag/_core.py

    def squeeze(self, axis=None) -> "Ragged | NDArray":
        if axis is None:
            data = self._layout.data.squeeze()
            shape = tuple(s for s in self._layout.shape if s != 1)
            if shape == (None,) or (None not in shape):
                pass
            return Ragged(
                RaggedLayout(data=data, offsets=self._layout.offsets, shape=shape,
                             str_offsets=self._layout.str_offsets)
            )
        if isinstance(axis, int):
            axis = (axis,)
        ndim = len(self._layout.shape)
        axis = tuple(a % ndim for a in axis)
        for a in axis:
            if self._layout.shape[a] != 1:
                raise ValueError(f"cannot squeeze axis {a} of size {self._layout.shape[a]}")
        shape = tuple(s for i, s in enumerate(self._layout.shape) if i not in axis)
        data_trailing = tuple(
            s for i, s in enumerate(self._layout.shape)
            if i not in axis and i > self.rag_dim
        )
        data = self._layout.data.reshape(len(self._layout.data), *data_trailing)
        return Ragged(
            RaggedLayout(data=data, offsets=self._layout.offsets, shape=shape,
                         str_offsets=self._layout.str_offsets)
        )

    def reshape(self, *shape) -> "Ragged":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        rag_dim = shape.index(None)
        new_rag_shape = shape[:rag_dim]
        n_rag = int(np.prod(self._layout.shape[: self.rag_dim], dtype=np.int64))
        n_new = abs(int(np.prod(new_rag_shape, dtype=np.int64))) or 1
        new_rag_shape = tuple(s if s and s >= 0 else n_rag // n_new for s in new_rag_shape)
        data = self._layout.data.reshape(len(self._layout.data), *shape[rag_dim + 1 :])
        new_shape = (*new_rag_shape, None, *data.shape[1:])
        return Ragged(
            RaggedLayout(data=data, offsets=self._layout.offsets, shape=new_shape,
                         str_offsets=self._layout.str_offsets)
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k "squeeze or reshape" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): squeeze and reshape on regular dims"
```

---

## Task 7: `to_numpy`, `to_packed`, `to_padded`

**Files:**
- Modify: `python/seqpro/rag/_core.py`
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `seqpro.rag._ops._pack_parts`, `seqpro.rag._ops._to_padded_copy` (existing Numba-backed helpers; reused unchanged).
- Produces on `Ragged`: `to_numpy(allow_missing=False) -> NDArray` (raises `ValueError` if rows are not equal length), `to_packed(copy=True) -> Ragged` (delegates to `_pack_parts` with layout fields), `to_padded(pad_value, length=None) -> NDArray`. `__array__` now delegates to `to_numpy`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_to_packed_from_slice():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    packed = rag[1:3].to_packed()
    assert packed.is_base is True
    np.testing.assert_array_equal(packed.data, np.array([3, 4, 5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(packed.offsets, np.array([0, 2, 7]))


def test_to_padded():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    out = rag.to_padded(-1)
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out[1], np.array([3, 4, -1, -1, -1]))


def test_to_numpy_equal_lengths():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    np.testing.assert_array_equal(rag.to_numpy(), np.arange(6).reshape(2, 3))


def test_to_numpy_jagged_raises():
    rag = Ragged.from_lengths(np.arange(5, dtype=np.int32), np.array([3, 2]))
    with pytest.raises(ValueError):
        rag.to_numpy()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k "to_packed or to_padded or to_numpy" -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to Ragged in python/seqpro/rag/_core.py

    def to_packed(self, *, copy: bool = True) -> "Ragged":
        from ._ops import _pack_parts

        packed_data, packed_offsets = _pack_parts(
            self._layout.data, self._layout.shape, self.offsets, copy
        )
        if packed_data is self._layout.data and packed_offsets is self.offsets:
            return self
        return Ragged.from_offsets(packed_data, self._layout.shape, packed_offsets)

    def to_padded(self, pad_value, *, length: int | None = None) -> NDArray:
        from ._ops import _to_padded_copy

        rag = self if self.is_contiguous else self.to_packed()
        offsets = np.ascontiguousarray(rag.offsets, dtype=OFFSET_TYPE)
        n_rows = offsets.shape[0] - 1
        out_len = int(length) if length is not None else (int(rag.lengths.max()) if n_rows else 0)
        dtype = rag.data.dtype
        out = np.full((n_rows, out_len), pad_value, dtype=dtype)
        if n_rows and out_len:
            data_u1 = np.ascontiguousarray(rag.data).reshape(-1).view(np.uint8)
            out_u1 = out.reshape(-1).view(np.uint8)
            _to_padded_copy(data_u1, offsets, out_u1, dtype.itemsize, out_len)
        leading = rag.shape[: rag.rag_dim]
        return out.reshape((*leading, out_len)) if leading else out

    def to_numpy(self, allow_missing: bool = False) -> NDArray:
        lengths = self.lengths
        if lengths.size and not np.all(lengths == lengths.flat[0]):
            raise ValueError("cannot convert a jagged Ragged to a dense array")
        packed = self if self.is_base else self.to_packed()
        row_len = int(lengths.flat[0]) if lengths.size else 0
        leading = packed.shape[: packed.rag_dim]
        return packed.data.reshape(*(leading or (-1,)), row_len, *packed.data.shape[1:])

    def __array__(self, dtype=None):
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k "to_packed or to_padded or to_numpy" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): to_numpy, to_packed, to_padded"
```

---

## Task 8: awkward ingestion + `to_ak` shim; wire oracle

**Files:**
- Create: `python/seqpro/rag/_ingest.py`
- Modify: `python/seqpro/rag/_core.py` (accept `ak.Array`/`Content`/`Ragged` in `__init__`)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: existing `seqpro.rag._array.unbox` (awkward → `RagParts`) as the proven extractor.
- Produces:
  - `layout_from_ak(arr) -> RaggedLayout` — non-record only; records raise `NotImplementedError("record-layout Ragged lands in Spec B")`.
  - `to_ak(rag: Ragged) -> ak.Array`.
  - `Ragged.__init__` accepts `RaggedLayout | Ragged | ak.Array | Content`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
import awkward as ak


def test_ingest_from_ak_numeric():
    arr = ak.Array([[1, 2, 3], [4, 5]])
    rag = Ragged(arr)
    assert rag.shape == (2, None)
    np.testing.assert_array_equal(rag.data, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5]))


def test_to_ak_roundtrips_values():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
    arr = ak.Array([[0, 1], [2], [3, 4, 5]])
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(rag.to_ak())), rag.data)


def test_ingest_record_raises_spec_b():
    arr = ak.Array({"a": [[1, 2], [3]], "b": [[1.0, 2.0], [3.0]]})
    with pytest.raises(NotImplementedError, match="Spec B"):
        Ragged(arr)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ragged_core.py -k "ingest or to_ak" -v`
Expected: FAIL with `TypeError` from the Task 2 `__init__` guard.

- [ ] **Step 3: Write minimal implementation**

```python
# python/seqpro/rag/_ingest.py
from __future__ import annotations

import awkward as ak
import numpy as np

from ._array import unbox  # proven awkward extractor (RagParts: data, shape, offsets)
from ._layout import RaggedLayout


def layout_from_ak(arr) -> RaggedLayout:
    if ak.fields(arr):
        raise NotImplementedError("record-layout Ragged lands in Spec B")
    parts = unbox(arr)
    is_bytes = parts.data.dtype.kind == "S"
    if is_bytes and parts.shape.count(None) == 1 and parts.shape.index(None) == len(parts.shape) - 1:
        leading = parts.shape[: parts.shape.index(None)]
        return RaggedLayout(
            data=parts.data, offsets=[], shape=leading, str_offsets=parts.offsets
        )
    return RaggedLayout(data=parts.data, offsets=[parts.offsets], shape=parts.shape)


def to_ak(rag) -> ak.Array:
    from ._array import _parts_to_content, RagParts

    content = _parts_to_content(
        RagParts(rag.data, rag.shape if None in rag.shape else (*rag.shape, None), rag.offsets)
    )
    return ak.Array(content)
```

```python
# modify Ragged.__init__ in python/seqpro/rag/_core.py
    def __init__(self, data):
        if isinstance(data, Ragged):
            data = data._layout
        if not isinstance(data, RaggedLayout):
            from ._ingest import layout_from_ak

            data = layout_from_ak(data)
        validate_layout(data)
        self._layout = data

    def to_ak(self):
        from ._ingest import to_ak as _to_ak

        return _to_ak(self)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k "ingest or to_ak" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_ingest.py python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat(rag): awkward ingestion and to_ak shim"
```

---

## Task 9: Hypothesis differential tests vs awkward oracle

**Files:**
- Modify: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: new `seqpro.rag._core.Ragged`; oracle `seqpro.rag._array.Ragged as AkRagged`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from seqpro.rag._array import Ragged as AkRagged


@st.composite
def _ragged_inputs(draw):
    n = draw(st.integers(1, 6))
    lengths = draw(
        st.lists(st.integers(0, 5), min_size=n, max_size=n).map(np.array)
    )
    total = int(lengths.sum())
    data = draw(arrays(np.int64, (total,), elements=st.integers(-100, 100)))
    return data, lengths


@given(_ragged_inputs())
def test_diff_numeric_properties(inp):
    data, lengths = inp
    new = Ragged.from_lengths(data, lengths.astype(np.uint32))
    old = AkRagged.from_lengths(data, lengths.astype(np.uint32))
    np.testing.assert_array_equal(new.data, old.data)
    np.testing.assert_array_equal(new.offsets, old.offsets)
    assert new.shape == old.shape
    np.testing.assert_array_equal(new.lengths, old.lengths)


@given(_ragged_inputs())
def test_diff_to_packed_after_slice(inp):
    data, lengths = inp
    if len(lengths) < 2:
        return
    new = Ragged.from_lengths(data, lengths.astype(np.uint32))[::2].to_packed()
    old = AkRagged.from_lengths(data, lengths.astype(np.uint32))[::2].to_packed()
    np.testing.assert_array_equal(new.data, old.data)
    np.testing.assert_array_equal(new.offsets, old.offsets)


@given(_ragged_inputs())
def test_diff_ufunc(inp):
    data, lengths = inp
    new = Ragged.from_lengths(data.astype(np.float64), lengths.astype(np.uint32))
    old = AkRagged.from_lengths(data.astype(np.float64), lengths.astype(np.uint32))
    np.testing.assert_allclose((new + 1.0).data, ak_flat(old + 1.0))


def ak_flat(ak_rag):
    import awkward as ak

    return ak.to_numpy(ak.flatten(ak_rag, axis=None))


def test_diff_string_shape_documented_change():
    # The one intentional divergence: bytes collection (N, None) -> (N,)
    data = np.frombuffer(b"cathithere", "S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    new = Ragged.from_lengths(data, lengths)
    old = AkRagged.from_lengths(data, lengths)
    assert new.shape == (3,)
    assert old.shape == (3, None)
    np.testing.assert_array_equal(new.offsets, old.offsets)  # same byte offsets
    np.testing.assert_array_equal(new.data, old.data)
```

- [ ] **Step 2: Run test to verify it fails (or passes if implementation already correct)**

Run: `pytest tests/test_ragged_core.py -k diff -v`
Expected: Initially may FAIL on `to_packed`/`ufunc` shape edge cases; fix `_core.py` until all pass. Do not weaken the assertions.

- [ ] **Step 3: Fix any divergences in `python/seqpro/rag/_core.py`**

Address failures by correcting `_core.py` (e.g. empty-array handling in `lengths`/`to_packed`). Keep oracle assertions intact.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ragged_core.py -k diff -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ragged_core.py python/seqpro/rag/_core.py
git commit -m "test(rag): Hypothesis differential tests vs awkward oracle"
```

---

## Task 10: Rust kernels — `ragged_validate`, `ragged_index`, `ragged_slice`

**Files:**
- Create: `src/ragged.rs`
- Modify: `src/lib.rs`
- Test: Rust unit tests inside `src/ragged.rs` (proptest); Python parity via existing `tests/test_ragged_core.py`.

**Interfaces:**
- Produces (registered in `#[pymodule]`):
  - `_ragged_validate(offsets: PyReadonlyArray<i64, Ix1>, n_data: i64, n_segments: i64) -> PyResult<()>` — monotonic, in-bounds (`0 <= offsets[i] <= n_data`), `len(offsets) - 1 == n_segments`; raises `PyValueError` otherwise.
  - `_ragged_select(starts, stops, idx: PyReadonlyArray<i64, Ix1>) -> (PyArray<i64,Ix1>, PyArray<i64,Ix1>)` — gather selected starts/stops by integer index array (slice/mask are normalized to an int-array in Python).
- Consumed by Python in Task 11.

- [ ] **Step 1: Write the failing Rust test**

```rust
// src/ragged.rs
use ndarray::prelude::*;

pub fn select(starts: ArrayView1<i64>, stops: ArrayView1<i64>, idx: ArrayView1<i64>)
    -> (Array1<i64>, Array1<i64>) {
    let s = idx.iter().map(|&i| starts[i as usize]).collect();
    let e = idx.iter().map(|&i| stops[i as usize]).collect();
    (Array1::from_vec(s), Array1::from_vec(e))
}

pub fn validate(offsets: ArrayView1<i64>, n_data: i64, n_segments: i64) -> Result<(), String> {
    if offsets.len() as i64 - 1 != n_segments {
        return Err(format!("segment count {} != {}", offsets.len() as i64 - 1, n_segments));
    }
    let mut prev = i64::MIN;
    for &o in offsets.iter() {
        if o < prev { return Err("offsets must be monotonic".into()); }
        if o < 0 || o > n_data { return Err("offset out of bounds".into()); }
        prev = o;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_select_gathers() {
        let starts = array![0i64, 3, 5];
        let stops = array![3i64, 5, 10];
        let idx = array![2i64, 0];
        let (s, e) = select(starts.view(), stops.view(), idx.view());
        assert_eq!(s, array![5i64, 0]);
        assert_eq!(e, array![10i64, 3]);
    }
    #[test]
    fn test_validate_rejects_nonmonotonic() {
        assert!(validate(array![0i64, 3, 2].view(), 10, 2).is_err());
    }
}
```

- [ ] **Step 2: Run Rust test to verify it fails**

Run: `cargo test --lib ragged`
Expected: FAIL — `src/ragged.rs` not yet declared in `lib.rs`.

- [ ] **Step 3: Wire module + PyO3 wrappers in `src/lib.rs`**

```rust
// add near top of src/lib.rs
pub mod ragged;
use numpy::{PyReadonlyArray1};
use pyo3::exceptions::PyValueError;

// inside fn seqpro(...) before Ok(()):
    m.add_function(wrap_pyfunction!(_ragged_validate, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_select, m)?)?;

// add functions:
#[pyfunction]
fn _ragged_validate(offsets: PyReadonlyArray1<i64>, n_data: i64, n_segments: i64) -> PyResult<()> {
    ragged::validate(offsets.as_array(), n_data, n_segments).map_err(PyValueError::new_err)
}

#[pyfunction]
fn _ragged_select<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    stops: PyReadonlyArray1<'py, i64>,
    idx: PyReadonlyArray1<'py, i64>,
) -> (&'py PyArray<i64, Ix1>, &'py PyArray<i64, Ix1>) {
    let (s, e) = ragged::select(starts.as_array(), stops.as_array(), idx.as_array());
    (s.into_pyarray(py), e.into_pyarray(py))
}
```

- [ ] **Step 4: Build + run Rust and Python tests to verify they pass**

Run: `cargo test --lib ragged && maturin develop && python -c "from seqpro.seqpro import _ragged_validate, _ragged_select; print('ok')"`
Expected: Rust tests PASS; import prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/ragged.rs src/lib.rs
git commit -m "feat(rust): ragged_validate and ragged_select kernels"
```

---

## Task 11: Route hot paths through Rust; full-suite green check

**Files:**
- Modify: `python/seqpro/rag/_core.py` (call `_ragged_select` in `__getitem__`; `_ragged_validate` in `validate_layout` path)
- Modify: `python/seqpro/rag/_layout.py` (optional Rust-backed validation)
- Test: `tests/test_ragged_core.py` (unchanged — parity gate)

**Interfaces:**
- Consumes: `from seqpro.seqpro import _ragged_select, _ragged_validate` (Task 10).
- Produces: identical behavior to Tasks 1–9, now Rust-backed for index/validate. Pure-Python fallback retained behind an import guard so the package imports even if the extension is stale.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_ragged_core.py
def test_getitem_uses_rust_select_intarray():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[np.array([2, 0])]
    np.testing.assert_array_equal(sub[0], np.array([5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(sub[1], np.array([0, 1, 2]))
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_ragged_core.py::test_getitem_uses_rust_select_intarray -v`
Expected: PASS already (pure-Python). This test pins int-array semantics before the Rust swap so the refactor can't regress them.

- [ ] **Step 3: Swap `__getitem__` gather + validation to Rust**

```python
# in python/seqpro/rag/_core.py __getitem__, replace the gather branch:
        # slice / mask / int-array -> normalize to int index array, gather via Rust
        n = len(starts)
        idx = np.arange(n)[where] if not isinstance(where, np.ndarray) or where.dtype != np.intp else where
        idx = np.atleast_1d(np.asarray(idx, dtype=np.int64))
        if where_is_bool(where):
            idx = np.flatnonzero(where).astype(np.int64)
        try:
            from seqpro.seqpro import _ragged_select
            sel_starts, sel_stops = _ragged_select(
                np.ascontiguousarray(starts, np.int64),
                np.ascontiguousarray(stops, np.int64),
                idx,
            )
        except ImportError:  # pragma: no cover - fallback
            sel_starts, sel_stops = starts[idx], stops[idx]
        new_offsets = np.stack([sel_starts, sel_stops], 0)
```

```python
# helper near top of _core.py
def where_is_bool(where) -> bool:
    return isinstance(where, np.ndarray) and where.dtype == np.bool_
```

```python
# in python/seqpro/rag/_layout.py validate_layout, after the n_ragged==1 block:
    if layout.n_ragged == 1:
        try:
            from seqpro.seqpro import _ragged_validate

            off = layout.offsets[0]
            if off.ndim == 1:
                _ragged_validate(
                    np.ascontiguousarray(off, np.int64),
                    int(layout.data.shape[0]),
                    len(off) - 1,
                )
        except ImportError:  # pragma: no cover - fallback to pure-Python checks above
            pass
```

- [ ] **Step 4: Run the full new-type suite + the existing suite**

Run: `pytest tests/test_ragged_core.py tests/test_ragged.py tests/test_rag_to_packed.py tests/test_ragged_to_padded.py tests/test_ragged_rc.py -v`
Expected: `test_ragged_core.py` PASS; the four legacy files PASS (public `Ragged` still the awkward type — untouched).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py python/seqpro/rag/_layout.py tests/test_ragged_core.py
git commit -m "refactor(rag): route index/validate hot paths through Rust"
```

---

## Task 12: Lint, format, and Spec A wrap-up

**Files:**
- Modify: any touched Python (lint fixes only)
- Modify: `docs/roadmap/rust-ragged.md` (decision log entry)

**Interfaces:** none.

- [ ] **Step 1: Run lint + format**

Run: `ruff check python/ tests/ && ruff format python/ tests/`
Expected: clean (fix any reported issues).

- [ ] **Step 2: Run the complete test suite + Rust tests**

Run: `pytest tests/ && cargo test --lib`
Expected: all PASS. (Confirms the new type did not perturb the existing awkward-backed paths.)

- [ ] **Step 3: Append the Spec A completion note to the roadmap decision log**

```markdown
- **2026-06-19** — Spec A landed: Rust-native single-level `Ragged` in
  `rag/_core.py` (+ `_layout.py`, `_ingest.py`, `src/ragged.rs`), fully tested
  against the awkward oracle. Public `seqpro.rag.Ragged` still points at the
  awkward type; the swap + tokenize/translate adaptation are deferred to Spec D
  (records to Spec B, nesting to Spec C). Confirmed string-leaf shape change
  `(N, None) -> (N,)` for byte collections.
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "docs(rag): record Spec A completion in roadmap"
```

---

## Self-review notes

- **Spec coverage:** Section 1 (data model) → Tasks 1–2; Section 2 (constructors) → Task 2; Section 3 (properties) → Tasks 2–3; Section 4 (indexing) → Tasks 4, 10–11; Section 5 (view/squeeze/reshape/to_numpy) → Tasks 3, 6, 7; Section 6 (ufunc/interop) → Task 5; Section 7 (to_packed/to_padded) → Task 7; Section 8 (Rust kernels) → Tasks 10–11; Section 9 (testing) → Tasks 1–9, 12; Section 10 (migration: to_ak shim, export unchanged) → Tasks 8, 11–12.
- **Deferred-by-design (raise with pointer):** records → `NotImplementedError("...Spec B")` (Tasks 2, 8); `R>1` nesting → `NotImplementedError("...Spec C")` (Tasks 1–2); public-export swap + tokenize/translate adaptation + docs/skill → Spec D (noted in roadmap, Task 12).
- **Deviation from spec text (flagged to user):** the public `seqpro.rag.Ragged` export is NOT flipped to the new type in Spec A — it stays on the awkward type so the existing suite (and `translate`/`tokenize`, whose byte-shape adaptation is Spec D) remain green. The new type is reachable at `seqpro.rag._core.Ragged`. This resolves the scope collision between "export the new type" and "tokenize/translate adaptation is Spec D."
