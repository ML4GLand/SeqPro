# Ragged getitem fast paths — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `seqpro.rag.Ragged` cheap on the getitem hot path — a zero-copy contiguous-slice fast path across all layouts, a `validate=False` opt-out on `to_numpy()`, and a lean `from_offsets` — so GenVarLoader can retire its `_Flat` shadow layer.

**Architecture:** All changes are pure-Python in `python/seqpro/rag/_core.py`. A single gate in `__getitem__` intercepts plain step-1 slices of contiguous arrays and dispatches to a per-layout narrowing builder that rebases each contiguous offset level and narrows the data buffer (the already-packed slice, computed with views). Every other index falls through to today's unchanged code. `to_numpy` and `from_offsets` get keyword-gated cheap paths.

**Tech Stack:** Python, NumPy. Tests via `pixi run -e dev pytest`. No Rust, no rebuild.

## Global Constraints

- All implementation in `python/seqpro/rag/_core.py`; no Rust this round.
- `OFFSET_TYPE = np.int64` (from `seqpro.rag._utils`).
- The fast-path **gate** is exactly: `isinstance(where, slice)` AND `self.shape[0] is not None` AND `self.is_contiguous` AND `step == 1`. Anything else uses the existing path, byte-for-byte unchanged.
- `_slice_contiguous` returns `Ragged` when it handles the layout, or `None` to fall back — so partially-implemented tasks keep the suite green.
- Values must be byte-identical to the existing gather path. The parity oracle is `to_py(rag[sl]) == to_py(rag._getitem(sl))` (the latter bypasses the gate = old path).
- Test command: `pixi run -e dev pytest <path> -q`.
- Commit after each task with `git commit --no-verify` (pre-push hook runs `ruff format`/`prek`; run `pixi run -e dev lint` before the final push).

---

## Task 1: Gate, dispatcher, R=1 fast path + parity harness

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`__getitem__` at 434-447; add helpers after it)
- Test: `tests/test_ragged_slice_fastpath.py` (create)

**Interfaces:**
- Produces: `Ragged.__getitem__` gate; `Ragged._slice_contiguous(start, stop) -> Ragged | None`; `Ragged._outer_n_inner() -> int`; `Ragged._slice_contig_r1(start, stop) -> Ragged`. Later tasks add `_slice_contig_r2`, `_slice_contig_string`, `_slice_contig_record`.
- Test helper produced: `to_py(x)` and `assert_slice_parity(rag, sl)` in the test file, reused by Tasks 2-5.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ragged_slice_fastpath.py`:

```python
import numpy as np
import pytest
from seqpro.rag import Ragged
from seqpro.rag._utils import lengths_to_offsets


def to_py(x):
    """Universal, layout-agnostic materialization to nested python lists/bytes.
    Recurses by peeling the outer (always-int) axis; never calls to_packed, so it
    works on string and string-record results too."""
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, Ragged):
        return [to_py(x[i]) for i in range(len(x))]
    return np.asarray(x).tolist()


def assert_slice_parity(rag, sl):
    new = rag[sl]                 # fast path (gate fires)
    old = rag._getitem(sl)        # bypass gate -> original gather path
    if isinstance(new, Ragged):
        assert new.is_contiguous, "fast-path result must be contiguous"
    assert to_py(new) == to_py(old)


def _r1(lengths, dtype=np.int32, shape=None):
    lengths = np.asarray(lengths, np.int64)
    off = lengths_to_offsets(lengths)
    total = int(off[-1])
    data = np.arange(total, dtype=dtype)
    shp = shape if shape is not None else (len(lengths), None)
    return Ragged.from_offsets(data, shp, off)


@pytest.mark.parametrize("sl", [slice(1, 4), slice(0, 5), slice(2, 2),
                                slice(3, 1), slice(None), slice(-2, None)])
def test_r1_simple_parity(sl):
    assert_slice_parity(_r1([4, 2, 5, 3, 6]), sl)


def test_r1_multidim_parity():
    # shape (B=3, P=2, None): 6 segments
    rag = _r1([4, 2, 5, 3, 6, 1], shape=(3, 2, None))
    assert_slice_parity(rag, slice(1, 3))


def test_r1_trailing_dim_parity():
    # OHE-style trailing fixed dim: shape (N, None, 4)
    off = lengths_to_offsets(np.array([3, 1, 2], np.int64))
    data = np.arange(int(off[-1]) * 4, dtype=np.uint8).reshape(-1, 4)
    rag = Ragged.from_offsets(data, (3, None, 4), off)
    assert_slice_parity(rag, slice(0, 2))


def test_r1_result_is_narrowed_view():
    rag = _r1([4, 2, 5, 3, 6])
    out = rag[1:3]
    assert out.is_contiguous
    assert out.offsets[0] == 0
    assert np.shares_memory(out.data, rag.data)   # narrowed view, not a copy
    assert out.data.shape[0] == 2 + 5             # only rows 1,2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -q`
Expected: FAIL — `test_r1_result_is_narrowed_view` fails (old path returns `(2,N)` non-contiguous over the full buffer, so `offsets[0]` / `shares_memory size` differ), and `assert new.is_contiguous` fails.

- [ ] **Step 3: Add the gate to `__getitem__`**

Replace `__getitem__` (lines 434-447) with:

```python
    def __getitem__(
        self, where: Any
    ) -> "NDArray[Any] | bytes | dict[str, Any] | Ragged[Any]":
        # Fast path: a plain step-1 slice of a contiguous array becomes the
        # already-packed slice (narrowed buffer + rebased (N+1,) offsets) with no
        # gather, no (2,N) drift, no copy. Any other index uses the path below.
        if (
            isinstance(where, slice)
            and self._layout.shape and self._layout.shape[0] is not None
            and self.is_contiguous
        ):
            start, stop, step = where.indices(self._layout.shape[0])
            if step == 1:
                if stop < start:
                    stop = start  # numpy empty-slice semantics (e.g. a[5:2])
                fast = self._slice_contiguous(start, stop)
                if fast is not None:
                    return (
                        self._with_layout(fast._layout)
                        if type(self) is not Ragged
                        else fast
                    )
        result = self._getitem(where)
        if (
            type(self) is not Ragged
            and not isinstance(where, str)
            and isinstance(result, Ragged)
        ):
            return self._with_layout(result._layout)
        return result
```

- [ ] **Step 4: Add the dispatcher and R=1 builder**

Immediately after `__getitem__`, add:

```python
    def _outer_n_inner(self) -> int:
        """Product of the fixed dims between the outer axis and the first ragged
        axis (1 when the outer axis is immediately followed by the ragged axis)."""
        shape = self._layout.shape
        rag_dim = shape.index(None)
        inner = [d for d in shape[1:rag_dim] if d is not None]
        return int(np.prod(np.array(inner, dtype=np.int64))) if inner else 1

    def _slice_contiguous(self, start: int, stop: int) -> "Ragged[Any] | None":
        """Build the already-packed result of a contiguous step-1 outer slice, or
        None to fall back. Caller guarantees self.is_contiguous and shape[0] int."""
        layout = self._layout
        if isinstance(layout, RecordLayout):
            return None  # Task 4/5
        rl = self._rl
        if rl.is_string:
            return None  # Task 3
        if rl.n_ragged == 2:
            return None  # Task 2
        if rl.n_ragged == 1:
            return self._slice_contig_r1(start, stop)
        return None

    def _slice_contig_r1(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        n_inner = self._outer_n_inner()
        o0 = rl.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        base = int(o0[g0])
        new_off = o0[g0 : g1 + 1] - base                 # contiguous (M+1,) int64
        new_data = rl.data[base : int(o0[g1])]           # narrowed view
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(
            RaggedLayout(data=new_data, offsets=[new_off], shape=new_shape)
        )
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -q`
Expected: PASS (all R=1 tests).

- [ ] **Step 6: Run the existing ragged suite (no regressions)**

Run: `pixi run -e dev pytest tests/test_ragged.py tests/test_ragged_core.py -q`
Expected: PASS. (If any test asserts a `(2,N)` post-slice layout, update it to expect contiguous; note each such change in the commit message.)

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): contiguous-slice fast path for R=1 getitem"
```

---

## Task 2: R=2 fast path

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_slice_contiguous`; add `_slice_contig_r2`)
- Test: `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Consumes: `_outer_n_inner`, gate (Task 1).
- Produces: `Ragged._slice_contig_r2(start, stop) -> Ragged`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def _r2(group_counts, inner_lengths, dtype=np.int32):
    """R=2 array: outer groups -> middle segments -> data."""
    o0 = lengths_to_offsets(np.asarray(group_counts, np.int64))
    o1 = lengths_to_offsets(np.asarray(inner_lengths, np.int64))
    data = np.arange(int(o1[-1]), dtype=dtype)
    n_outer = len(group_counts)
    return Ragged.from_offsets(data, (n_outer, None, None), [o0, o1])


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_r2_parity(sl):
    # 3 groups with 2,1,2 middle segments; middles have these lengths
    rag = _r2([2, 1, 2], [4, 3, 5, 2, 6])
    assert_slice_parity(rag, sl)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k r2 -q`
Expected: FAIL — `_slice_contiguous` returns `None` for `n_ragged == 2`, so `rag[sl]` uses the old `(2,N)` path; `assert new.is_contiguous` fails (old R=2 slice keeps O1 global / `(2,L0')` O0).

- [ ] **Step 3: Implement**

In `_slice_contiguous`, replace `if rl.n_ragged == 2: return None  # Task 2` with `return self._slice_contig_r2(start, stop)`. Add the method:

```python
    def _slice_contig_r2(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        n_inner = self._outer_n_inner()
        o0, o1 = rl.offsets
        g0, g1 = start * n_inner, stop * n_inner
        m0, m1 = int(o0[g0]), int(o0[g1])      # middle-segment range
        new_o0 = o0[g0 : g1 + 1] - m0
        d0, d1 = int(o1[m0]), int(o1[m1])      # data range
        new_o1 = o1[m0 : m1 + 1] - d0
        new_data = rl.data[d0:d1]
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(
            RaggedLayout(data=new_data, offsets=[new_o0, new_o1], shape=new_shape)
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k r2 -q`
Expected: PASS.

- [ ] **Step 5: Regression check + commit**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_nested_diff.py -q`
Expected: PASS.

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): contiguous-slice fast path for R=2 getitem"
```

---

## Task 3: opaque-string fast path (flat + under-axis)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_slice_contiguous`; add `_slice_contig_string`)
- Test: `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Consumes: gate, `_outer_n_inner` (Task 1).
- Produces: `Ragged._slice_contig_string(start, stop) -> Ragged`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def _str_flat(strings):
    """Flat opaque-string collection: shape (N,)."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    so = lengths_to_offsets(np.array([len(s) for s in strings], np.int64))
    return Ragged.from_offsets(data, (len(strings),), so, str_offsets=so)


def _str_under_axis(rows):
    """String-under-axis: shape (N, None); each row is a list of byte strings."""
    flat = [s for row in rows for s in row]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    so = lengths_to_offsets(np.array([len(s) for s in flat], np.int64))
    o0 = lengths_to_offsets(np.array([len(r) for r in rows], np.int64))
    return Ragged.from_offsets(data, (len(rows), None), o0, str_offsets=so)


@pytest.mark.parametrize("sl", [slice(1, 3), slice(0, 4), slice(2, 2), slice(None)])
def test_string_flat_parity(sl):
    assert_slice_parity(_str_flat([b"AC", b"GGG", b"T", b"CCGT"]), sl)


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2)])
def test_string_under_axis_parity(sl):
    rows = [[b"AC", b"G"], [b"TT"], [b"CCG", b"A", b"T"]]
    assert_slice_parity(_str_under_axis(rows), sl)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k string -q`
Expected: FAIL — `_slice_contiguous` returns `None` for `is_string`; old path keeps `(2,M)` `str_offsets`/offsets, so `is_contiguous` is False on the result.

- [ ] **Step 3: Implement**

In `_slice_contiguous`, replace `if rl.is_string: return None  # Task 3` with `return self._slice_contig_string(start, stop)`. Add:

```python
    def _slice_contig_string(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        so = rl.str_offsets
        if not rl.offsets:
            # flat string collection: shape (N,), no axis offsets; slice str_offsets
            b0, b1 = int(so[start]), int(so[stop])
            new_so = so[start : stop + 1] - b0
            new_data = rl.data[b0:b1]
            return Ragged(
                RaggedLayout(
                    data=new_data, offsets=[], shape=(stop - start,), str_offsets=new_so
                )
            )
        # string-under-axis: O0 (outer -> variant) then str_offsets (variant -> byte)
        n_inner = self._outer_n_inner()
        o0 = rl.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        v0, v1 = int(o0[g0]), int(o0[g1])
        new_o0 = o0[g0 : g1 + 1] - v0
        b0, b1 = int(so[v0]), int(so[v1])
        new_so = so[v0 : v1 + 1] - b0
        new_data = rl.data[b0:b1]
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(
            RaggedLayout(
                data=new_data, offsets=[new_o0], shape=new_shape, str_offsets=new_so
            )
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k string -q`
Expected: PASS.

- [ ] **Step 5: Regression check + commit**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -k "str or string" -q`
Expected: PASS.

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): contiguous-slice fast path for opaque-string getitem"
```

---

## Task 4: record R=1 fast path (numeric + string fields)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_slice_contiguous`; add `_slice_contig_record`)
- Test: `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Consumes: gate, `_outer_n_inner` (Task 1).
- Produces: `Ragged._slice_contig_record(start, stop) -> Ragged | None` (dispatches R=1 here; returns None for R=2 until Task 5).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def _record_r1():
    """Record R=1 with a numeric field and a string-under-axis field sharing O0.
    shape (3, None); rows have 2,1,3 variants."""
    o0 = lengths_to_offsets(np.array([2, 1, 3], np.int64))
    n_var = int(o0[-1])
    start_field = Ragged.from_offsets(
        np.arange(n_var, dtype=np.int32), (3, None), o0
    )
    alts = [b"AC", b"G", b"T", b"CC", b"A", b"GG"]  # one per variant
    sdata = np.frombuffer(b"".join(alts), dtype="S1")
    sso = lengths_to_offsets(np.array([len(a) for a in alts], np.int64))
    alt_field = Ragged.from_offsets(
        sdata, (3, None), o0, str_offsets=sso
    )
    return Ragged.from_fields({"start": start_field, "alt": alt_field})


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_record_r1_parity(sl):
    assert_slice_parity(_record_r1(), sl)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k record_r1 -q`
Expected: FAIL — record dispatch returns `None`; old path yields `(2,N)` shared offsets, so `is_contiguous` is False.

- [ ] **Step 3: Implement**

In `_slice_contiguous`, replace `if isinstance(layout, RecordLayout): return None  # Task 4/5` with `return self._slice_contig_record(start, stop)`. Add:

```python
    def _slice_contig_record(self, start: int, stop: int) -> "Ragged[Any] | None":
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if len(rec.offsets) == 2:
            return self._slice_contig_record_r2(start, stop)  # Task 5
        n_inner = self._outer_n_inner()
        o0 = rec.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        v0, v1 = int(o0[g0]), int(o0[g1])
        shared = [o0[g0 : g1 + 1] - v0]            # one shared (M+1,) object for all
        out_rag_shape = rec.shape[rec.shape.index(None):]
        new_fields: dict[str, RaggedLayout[Any]] = {}
        for name, fl in rec.fields.items():
            fld_tail = fl.shape[fl.shape.index(None):]
            if fl.str_offsets is not None:
                so = fl.str_offsets
                b0, b1 = int(so[v0]), int(so[v1])
                new_fields[name] = RaggedLayout(
                    data=fl.data[b0:b1],
                    offsets=shared,
                    shape=(stop - start, *fld_tail),
                    str_offsets=so[v0 : v1 + 1] - b0,
                )
            else:
                new_fields[name] = RaggedLayout(
                    data=fl.data[v0:v1],
                    offsets=shared,
                    shape=(stop - start, *fld_tail),
                )
        return Ragged(
            RecordLayout(
                offsets=shared, shape=(stop - start, *out_rag_shape), fields=new_fields
            )
        )
```

> **Note:** every field and the record share the *same* `shared` list object (the
> zero-copy SoA contract `_validate_record_layout` enforces). String-under-axis
> fields are sliced via the 2-level narrow even though `to_packed` rejects them
> (Spec C) — the slice doesn't need the pack kernel.

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k record_r1 -q`
Expected: PASS.

- [ ] **Step 5: Regression check + commit**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py tests/test_ragged_record_indexing.py -q`
Expected: PASS.

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): contiguous-slice fast path for record R=1 getitem"
```

---

## Task 5: record R=2 fast path

**Files:**
- Modify: `python/seqpro/rag/_core.py` (add `_slice_contig_record_r2`)
- Test: `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Consumes: `_slice_contig_record` dispatch (Task 4).
- Produces: `Ragged._slice_contig_record_r2(start, stop) -> Ragged | None` (None when any field is string-under-axis — those fall back).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def _record_r2():
    """Record R=2: two numeric fields sharing [O0, O1]. 3 outer groups."""
    o0 = lengths_to_offsets(np.array([2, 1, 2], np.int64))   # middles per group
    o1 = lengths_to_offsets(np.array([3, 2, 4, 1, 5], np.int64))  # data per middle
    n = int(o1[-1])
    a = Ragged.from_offsets(np.arange(n, dtype=np.int32), (3, None, None), [o0, o1])
    b = Ragged.from_offsets(np.arange(n, dtype=np.float32), (3, None, None), [o0, o1])
    return Ragged.from_fields({"a": a, "b": b})


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_record_r2_parity(sl):
    assert_slice_parity(_record_r2(), sl)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k record_r2 -q`
Expected: FAIL — `_slice_contig_record_r2` is undefined (`AttributeError`).

- [ ] **Step 3: Implement**

Add the method:

```python
    def _slice_contig_record_r2(self, start: int, stop: int) -> "Ragged[Any] | None":
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if any(fl.str_offsets is not None for fl in rec.fields.values()):
            return None  # string-under-axis R=2 record: fall back to gather path
        n_inner = self._outer_n_inner()
        o0, o1 = rec.offsets
        g0, g1 = start * n_inner, stop * n_inner
        m0, m1 = int(o0[g0]), int(o0[g1])
        d0, d1 = int(o1[m0]), int(o1[m1])
        shared = [o0[g0 : g1 + 1] - m0, o1[m0 : m1 + 1] - d0]
        out_tail = rec.shape[rec.shape.index(None):]
        new_fields = {
            name: RaggedLayout(
                data=fl.data[d0:d1],
                offsets=shared,
                shape=(stop - start, *fl.shape[fl.shape.index(None):]),
            )
            for name, fl in rec.fields.items()
        }
        return Ragged(
            RecordLayout(
                offsets=shared, shape=(stop - start, *out_tail), fields=new_fields
            )
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -q`
Expected: PASS (all slice fast-path tests).

- [ ] **Step 5: Full ragged regression + commit**

Run: `pixi run -e dev pytest tests/test_ragged.py tests/test_ragged_core.py tests/test_ragged_core_records.py tests/test_ragged_nested_diff.py tests/test_rag_to_packed.py -q`
Expected: PASS.

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): contiguous-slice fast path for record R=2 getitem"
```

---

## Task 6: `to_numpy(..., *, validate=True)`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`to_numpy` at 1552-1600; R=2 branch 1566-1586; record branch 1555-1565)
- Test: `tests/test_ragged_core.py` (append) or `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Produces: `Ragged.to_numpy(allow_missing: bool = False, *, validate: bool = True)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def test_to_numpy_validate_false_matches_true():
    off = np.arange(4 + 1, dtype=np.int64) * 3        # 4 uniform rows of len 3
    data = np.arange(4 * 3, dtype=np.int32)
    rag = Ragged.from_offsets(data, (4, None), off)
    a = rag.to_numpy()                  # validate=True (default)
    b = rag.to_numpy(validate=False)    # trust-me
    np.testing.assert_array_equal(a, b)
    assert np.shares_memory(b, rag.data)   # zero-copy reshape


def test_to_numpy_validate_true_still_raises_on_jagged():
    rag = _r1([4, 2, 5])
    with pytest.raises(ValueError):
        rag.to_numpy()                  # jagged -> raise (unchanged default)


def test_to_numpy_validate_false_multidim():
    rag = _r1([3, 3, 3, 3], shape=(2, 2, None))   # (2,2,None) uniform len 3
    out = rag.to_numpy(validate=False)
    assert out.shape == (2, 2, 3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k to_numpy -q`
Expected: FAIL — `to_numpy() got an unexpected keyword argument 'validate'`.

- [ ] **Step 3: Implement**

Change the signature (line 1552-1554) to:

```python
    def to_numpy(
        self, allow_missing: bool = False, *, validate: bool = True
    ) -> "NDArray[Any] | dict[str, NDArray[Any]]":
```

In the **record branch** (1555-1565) thread `validate` through the per-field recursion:

```python
        if isinstance(self._layout, RecordLayout):
            return {
                field: cast(
                    "NDArray[Any]",
                    cast("Ragged[Any]", self[field]).to_numpy(
                        allow_missing=allow_missing, validate=validate
                    ),
                )
                for field in self._layout.fields
            }
```

In the **R=2 branch**, gate the two uniformity raises behind `validate` (lines 1576-1579):

```python
            if validate:
                if grp_lens.size and not np.all(grp_lens == grp_lens[0]):
                    raise ValueError("cannot convert a jagged outer axis to a dense array")
                if mid_lens.size and not np.all(mid_lens == mid_lens[0]):
                    raise ValueError("cannot convert a jagged inner axis to a dense array")
```

Replace the **single-level path** (lines 1592-1600) with:

```python
        if validate:
            lengths = self.lengths
            if lengths.size and not np.all(lengths == lengths.flat[0]):
                raise ValueError("cannot convert a jagged Ragged to a dense array")
            packed = self if self.is_base else self.to_packed()
            row_len = int(lengths.flat[0]) if lengths.size else 0
        else:
            # trust the caller: infer row_len from total // n_rows, no uniformity
            # scan. numpy's reshape still rejects a total-size mismatch for free.
            packed = self if self.is_base else self.to_packed()
            leading_dims = [d for d in packed.shape[: packed.rag_dim]]
            n_rows = (
                int(np.prod(np.array(leading_dims, dtype=np.int64)))
                if leading_dims
                else 1
            )
            total = packed._rl.data.shape[0]
            row_len = total // n_rows if n_rows else 0
        leading = packed.shape[: packed.rag_dim]
        return packed._rl.data.reshape(  # pyrefly: ignore[no-matching-overload]
            *(leading or (-1,)), row_len, *packed._rl.data.shape[1:]
        )
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k to_numpy -q`
Expected: PASS.

- [ ] **Step 5: Regression check + commit**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_to_padded.py -q`
Expected: PASS.

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "feat(rag): to_numpy(validate=False) skips the uniformity scan"
```

---

## Task 7: lean `from_offsets`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`from_offsets` at 108-145)
- Test: `tests/test_ragged_core.py` (append) or `tests/test_ragged_slice_fastpath.py`

**Interfaces:**
- Produces: same `from_offsets` signature; default `validate=False` no longer raises on size mismatch.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_slice_fastpath.py`:

```python
def test_from_offsets_validate_true_raises_on_size_mismatch():
    data = np.arange(10, dtype=np.int32)
    bad = np.array([0, 4, 20], np.int64)   # implies 20 elements, data has 10
    with pytest.raises(ValueError):
        Ragged.from_offsets(data, (2, None), bad, validate=True)


def test_from_offsets_default_skips_size_check():
    data = np.arange(10, dtype=np.int32)
    bad = np.array([0, 4, 20], np.int64)
    # default validate=False: constructs without raising (caller's contract)
    rag = Ragged.from_offsets(data, (2, None), bad)
    assert rag.shape == (2, None)


def test_from_offsets_preserves_already_canonical_offsets():
    data = np.arange(7, dtype=np.int32)
    off = np.array([0, 4, 7], np.int64)    # already C-contiguous int64
    rag = Ragged.from_offsets(data, (2, None), off)
    assert rag.offsets is off              # no ascontiguousarray copy
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k from_offsets -q`
Expected: FAIL — `test_from_offsets_default_skips_size_check` raises today (size check is unconditional); `test_from_offsets_preserves_already_canonical_offsets` fails (`ascontiguousarray` returns a new array, so `is` is False).

- [ ] **Step 3: Implement**

Replace `from_offsets` lines 118-145 with:

```python
        off_list = offsets if isinstance(offsets, list) else [offsets]
        off_list = [
            o
            if (
                isinstance(o, np.ndarray)
                and o.dtype == OFFSET_TYPE
                and o.flags.c_contiguous
            )
            else np.ascontiguousarray(o, dtype=OFFSET_TYPE)
            for o in off_list
        ]
        if str_offsets is not None:
            if not (
                isinstance(str_offsets, np.ndarray)
                and str_offsets.dtype == OFFSET_TYPE
                and str_offsets.flags.c_contiguous
            ):
                str_offsets = np.ascontiguousarray(str_offsets, dtype=OFFSET_TYPE)
            return Ragged(
                RaggedLayout(
                    data=data, offsets=off_list, shape=shape, str_offsets=str_offsets
                ),
                validate=validate,
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have a None ragged dimension")
        if validate:
            # eager data-size check (only when validating)
            n_none = shape.count(None)
            if n_none == 1 and len(off_list) == 1 and off_list[0].ndim == 1:
                rag_dim = shape.index(None)
                trailing = shape[rag_dim + 1 :]
                trailing_ints: list[int] = [d for d in trailing if d is not None]
                trailing_size = int(np.prod(trailing_ints)) if trailing_ints else 1
                if off_list[0].size > 0:
                    expected_size = int(off_list[0][-1]) * trailing_size
                    if data.size != expected_size:
                        raise ValueError(
                            f"Data size {data.size} does not match size implied by shape and contiguous offsets: {expected_size}"
                        )
        return Ragged(_build_layout(data, shape, off_list), validate=validate)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pixi run -e dev pytest tests/test_ragged_slice_fastpath.py -k from_offsets -q`
Expected: PASS.

- [ ] **Step 5: Find + fix tests relying on the old eager default**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -q`
Expected: any test that asserted `from_offsets(...)` raises on a size/shape mismatch *without* `validate=True` now fails. For each, add `validate=True` to that call (the test's intent is to exercise validation). Note each in the commit message.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_slice_fastpath.py
git commit --no-verify -m "perf(rag): lean from_offsets — elide ascontiguousarray, gate size-check behind validate"
```

---

## Task 8: bench, full suite, cross-repo parity, docs, tracking issue

**Files:**
- Create: `tests/bench_ragged_getitem.py` (standalone bench script, not collected by pytest)
- Modify: `/Users/david/.claude/skills/seqpro/SKILL.md` (skill doc)
- Modify: `docs/superpowers/specs/2026-06-24-ragged-getitem-fastpaths-design.md` (mark results)

- [ ] **Step 1: Add the bench script**

Create `tests/bench_ragged_getitem.py`:

```python
"""Standalone: re-run the getitem hot-path bench vs the design gates.
Run: pixi run -e dev python tests/bench_ragged_getitem.py
Gates: slice within ~1.2x of a bare numpy view-slice baseline;
       to_numpy(validate=False) within ~1.2x of a bare reshape;
       from_offsets within ~1.1x of a bare dataclass wrap."""
import time
import numpy as np
from seqpro.rag import Ragged
from seqpro.rag._utils import lengths_to_offsets


def bench(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


def make(B=128, P=2, L=2000):
    rng = np.random.default_rng(0)
    lengths = rng.integers(L // 2, L, size=B * P).astype(np.int64)
    off = lengths_to_offsets(lengths)
    data = rng.integers(0, 1000, size=int(off[-1])).astype(np.int32)
    return data, off, (B, P, None)


def make_uniform(B=128, P=2, L=2000):
    off = np.arange(B * P + 1, dtype=np.int64) * L
    data = np.arange(B * P * L, dtype=np.int32)
    return data, off, (B, P, None)


N = 20000
data, off, shape = make()
r = Ragged.from_offsets(data, shape, off)
print(f"from_offsets : {bench(lambda: Ragged.from_offsets(data, shape, off), N):.3f} us")
print(f"slice [16:112]: {bench(lambda: r[16:112], N):.3f} us  contiguous={r[16:112].is_contiguous}")
data, off, shape = make_uniform()
ru = Ragged.from_offsets(data, shape, off)
print(f"to_numpy(v=F): {bench(lambda: ru.to_numpy(validate=False), N):.3f} us")
print(f"to_numpy(v=T): {bench(lambda: ru.to_numpy(), N):.3f} us")
```

- [ ] **Step 2: Run the bench, confirm the wins**

Run: `pixi run -e dev python tests/bench_ragged_getitem.py`
Expected: slice result `contiguous=True`; `slice` ≈ 3 µs (was 8.2); `to_numpy(v=F)` ≈ 0.4–1 µs (was 4.8); `from_offsets` ≈ 0.4 µs (was 1.08). Record the numbers in the spec's testing section.

- [ ] **Step 3: Full seqpro suite**

Run: `pixi run -e dev test`
Expected: PASS. Fix any remaining `(2,N)`-assuming or eager-validate-assuming test.

- [ ] **Step 4: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 5: Cross-repo byte-identical parity (gvl)**

In GenVarLoader, point its seqpro at this branch (editable) and run the suite:

```bash
cd /Users/david/projects/GenVarLoader
# temporarily install the branch: pip install -e /Users/david/projects/SeqPro into the dev env
pixi run -e dev pytest tests -q
```
Expected: PASS, byte-identical (gvl's parity harness). This proves the slice representation change is transparent to consumers. Revert the editable pin after.

- [ ] **Step 6: Update the seqpro skill**

In `/Users/david/.claude/skills/seqpro/SKILL.md`:
- Document `to_numpy(..., *, validate=True)` and the `validate=False` trust-me path.
- Update the "Offsets layout drifts after slicing" pitfall: note that a **plain step-1 slice of a contiguous array** now stays contiguous (narrowed view); only non-step-1 / masked / int-array / already-`(2,N)` indices produce the `(2,N)` gather layout.
- Note `from_offsets`'s `validate=False` default no longer does the eager size check.

- [ ] **Step 7: File the deferred-Rust tracking issue**

Open a tracking issue in the SeqPro repo: "Evaluate porting Ragged slice/from_offsets glue to seqpro-core (Rust)", linking this spec, with the post-change bench numbers as the baseline a Rust port must beat (net of FFI overhead).

- [ ] **Step 8: Final commit**

```bash
git add tests/bench_ragged_getitem.py docs/superpowers/specs/2026-06-24-ragged-getitem-fastpaths-design.md
git commit --no-verify -m "test(rag): getitem hot-path bench + record fast-path results"
```

---

## Self-Review

**Spec coverage:**
- §1 contiguous-slice fast path (all layouts) → Tasks 1 (R=1), 2 (R=2), 3 (string), 4 (record R=1), 5 (record R=2). ✓
- §1 gate (`slice` + step 1 + `is_contiguous` + `shape[0] not None`) → Task 1. ✓
- §1 string-under-axis record fast-path (the `to_packed`-rejects asymmetry) → Task 4 (string fields) + note. ✓
- §2 `to_numpy(validate=)` across R=1/R=2/record → Task 6. ✓
- §3 lean `from_offsets` (ascontiguousarray elision + validate-gated size check) → Task 7. ✓
- Testing: parity oracle (Task 1 `to_py`/`assert_slice_parity`, reused 2-5), `to_numpy` (6), `from_offsets` validate (7), bench + full suite + cross-repo + docs + tracking issue (8). ✓

**Placeholder scan:** none — every step has concrete code/commands.

**Type consistency:** `_slice_contiguous -> Ragged|None`; `_slice_contig_r1/_r2/_string -> Ragged`; `_slice_contig_record/_record_r2 -> Ragged|None`; `_outer_n_inner -> int`; `to_py`/`assert_slice_parity` defined Task 1, reused later. Shared offsets list object passed identically to record + fields (validator contract). Consistent across tasks.
