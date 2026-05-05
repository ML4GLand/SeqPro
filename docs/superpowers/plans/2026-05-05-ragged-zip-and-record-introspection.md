# Ragged: ak.zip + Record-Layout Introspection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ak.zip` of Ragged inputs fully supported, replace state-conditional errors on record-layout `dtype`/`data`/`parts` with field-keyed dicts, extend `squeeze`/`reshape` to records, and remove the experimental disclaimer.

**Architecture:** All changes in `python/seqpro/rag/_array.py`. Centralize lazy `_parts` initialization in a single helper so behavior-dispatch-created `Ragged` instances (e.g., from `ak.zip`) work without an `__init__` call. Record-layout introspection delegates per-field via existing zero-copy `__getitem__`. `squeeze`/`reshape` on records dispatch field-wise then `ak.zip` back.

**Tech Stack:** Python, NumPy, awkward, pytest, pytest-cases.

**Spec:** `docs/superpowers/specs/2026-05-05-ragged-zip-and-record-introspection-design.md`.

---

### Task 1: Lazy `_parts` initialization helper

**Why:** `ak.zip` produces a `Ragged` via awkward behavior dispatch, bypassing `__init__`. Existing properties use `hasattr(self, "_parts")` + `unbox(self)` fallback, which raises on record layouts. Centralize the logic so any property/method gets correct init regardless of construction path.

**Files:**
- Modify: `python/seqpro/rag/_array.py` (add `_ensure_parts` method, replace `hasattr(self, "_parts")` blocks)

- [ ] **Step 1: Add probe test for behavior-dispatch construction**

Add to `tests/test_ragged.py` in `TestRecordRagged`:

```python
def test_zip_produces_initialized_ragged(self):
    r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2,1,3]))
    r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2,1,3]))
    z = ak.zip({"a": r1, "b": r2})
    assert isinstance(z, Ragged)
    # Must not raise even though __init__ was bypassed:
    np.testing.assert_array_equal(z.offsets, np.array([0, 2, 3, 6], dtype=z.offsets.dtype))
```

- [ ] **Step 2: Run test, expect failure**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_zip_produces_initialized_ragged -v`
Expected: FAIL — `unbox(self)` raises "Must extract a single field before unboxing a Ragged array of records." OR AttributeError on `_parts`.

- [ ] **Step 3: Add `_ensure_parts` helper and route all property fallbacks through it**

Add this method on `Ragged` (right after `__init__`):

```python
def _ensure_parts(self) -> None:
    """Idempotent lazy init for `_parts`. Handles Ragged instances created
    via awkward behavior dispatch (e.g. ak.zip) that bypass __init__."""
    if hasattr(self, "_parts"):
        return
    layout = cast(Content, ak.to_layout(self))
    if isinstance(layout, RecordArray) or _is_record_layout(layout):
        object.__setattr__(self, "_parts", None)
    else:
        object.__setattr__(self, "_parts", unbox(self))
```

Replace each `if not hasattr(self, "_parts"): self._parts = unbox(self)` block in `parts`, `data`, `offsets` with `self._ensure_parts()`.

- [ ] **Step 4: Run probe test, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_zip_produces_initialized_ragged -v`
Expected: PASS.

- [ ] **Step 5: Run full ragged suite to confirm no regressions**

Run: `pixi run pytest tests/test_ragged.py -v`
Expected: all previously-passing tests still pass; new test passes.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "fix: lazy _parts init for Ragged created via ak behavior dispatch

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: `dtype` returns dict for record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`dtype` property)
- Test: `tests/test_ragged.py` (`TestRecordRagged`)

- [ ] **Step 1: Write failing tests**

Add to `TestRecordRagged`:

```python
def test_dtype_dict(self, rag: Ragged):
    dt = rag.dtype
    assert isinstance(dt, dict)
    assert list(dt.keys()) == ["field0", "field1"]
    assert dt["field0"] == np.int64
    assert dt["field1"] == np.float64

def test_dtype_field_order_preserved(self):
    r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2,1,3]))
    r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2,1,3]))
    rag = Ragged(ak.zip({"zeta": r1, "alpha": r2}))
    assert list(rag.dtype.keys()) == ["zeta", "alpha"]
```

- [ ] **Step 2: Run, expect fail**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_dtype_dict tests/test_ragged.py::TestRecordRagged::test_dtype_field_order_preserved -v`
Expected: FAIL (currently `dtype` calls `self.data.dtype` which raises TypeError on records).

- [ ] **Step 3: Implement**

Replace the `dtype` property:

```python
@property
def dtype(self) -> np.dtype[RDTYPE] | dict[str, np.dtype]:
    """The dtype of the Ragged array. For record layouts, a dict of
    field name -> dtype, in awkward field order."""
    self._ensure_parts()
    if self._parts is None:
        return {f: self[f].dtype for f in ak.fields(self)}
    return self._parts.data.dtype
```

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged -v`
Expected: PASS for new tests + all previously-passing tests.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged.dtype returns field dict for record layouts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: `data` returns dict (zero-copy) for record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`data` property)
- Test: `tests/test_ragged.py`

- [ ] **Step 1: Write failing tests**

```python
def test_data_dict(self, rag: Ragged):
    d = rag.data
    assert isinstance(d, dict)
    assert list(d.keys()) == ["field0", "field1"]
    np.testing.assert_array_equal(d["field0"], np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(d["field1"], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))

def test_data_dict_zero_copy(self, rag: Ragged):
    d = rag.data
    # Each field's ndarray should be a view, not a fresh allocation.
    assert d["field0"].base is not None
    assert d["field1"].base is not None
```

- [ ] **Step 2: Run, expect fail**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_data_dict tests/test_ragged.py::TestRecordRagged::test_data_dict_zero_copy -v`
Expected: FAIL (currently raises TypeError).

The previously-existing `test_data_raises` will need to be removed/updated in this task since the contract is changing.

- [ ] **Step 3: Implement and remove old contract test**

Replace the `data` property:

```python
@property
def data(self) -> NDArray[RDTYPE] | dict[str, NDArray]:
    """The data of the Ragged array. For record layouts, a dict of
    field name -> zero-copy ndarray view, in awkward field order."""
    self._ensure_parts()
    if self._parts is None:
        return {f: self[f].data for f in ak.fields(self)}
    return self._parts.data
```

Remove `test_data_raises` from `TestRecordRagged` (contract changed).

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged.data returns zero-copy field dict for record layouts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: `parts` returns dict (shared offsets) for record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`parts` property)
- Test: `tests/test_ragged.py`

- [ ] **Step 1: Write failing tests**

```python
def test_parts_dict(self, rag: Ragged):
    p = rag.parts
    assert isinstance(p, dict)
    assert list(p.keys()) == ["field0", "field1"]
    from seqpro.rag._array import RagParts
    for v in p.values():
        assert isinstance(v, RagParts)

def test_parts_dict_shares_offsets(self, rag: Ragged):
    p = rag.parts
    assert p["field0"].offsets is rag.offsets
    assert p["field1"].offsets is rag.offsets
```

Remove the existing `test_data_raises`-style record-parts contract test if any exists for `parts` (none currently — `parts` raises today but no test pinned that).

- [ ] **Step 2: Run, expect fail**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_parts_dict tests/test_ragged.py::TestRecordRagged::test_parts_dict_shares_offsets -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Replace `parts` property:

```python
@property
def parts(self) -> RagParts[RDTYPE] | dict[str, RagParts]:
    """The parts of the Ragged array. For record layouts, a dict of
    field name -> RagParts; all share the same offsets ndarray."""
    self._ensure_parts()
    if self._parts is None:
        return {f: self[f].parts for f in ak.fields(self)}
    return self._parts
```

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged.parts returns offsets-sharing field dict for records

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: `view`, `apply`, `to_numpy` raise NotImplementedError on records

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`view`, `apply`, `to_numpy`)
- Test: `tests/test_ragged.py`

- [ ] **Step 1: Write failing tests**

```python
def test_view_raises_on_record(self, rag: Ragged):
    with pytest.raises(NotImplementedError, match="record"):
        _ = rag.view(np.float32)

def test_apply_raises_on_record(self, rag: Ragged):
    with pytest.raises(NotImplementedError, match="record"):
        _ = rag.apply(lambda x: x + 1)

def test_to_numpy_raises_on_record(self, rag: Ragged):
    with pytest.raises(NotImplementedError, match="record"):
        _ = rag.to_numpy()

def test_field_setitem_roundtrip(self, rag: Ragged):
    # The supported alternative to a record-level view: field-wise update.
    original = rag["field0"].data.copy()
    rag["field0"] = rag["field0"].view(np.uint64)
    np.testing.assert_array_equal(rag["field0"].data.view(np.int64), original)
```

- [ ] **Step 2: Run, expect fail (or wrong error)**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged -k "raises_on_record or setitem_roundtrip" -v`
Expected: FAIL — current behavior is various other errors (AttributeError, TypeError) not `NotImplementedError`.

- [ ] **Step 3: Implement record guards**

In `view`:

```python
def view(self, dtype: type[DTYPE] | str) -> Ragged[DTYPE]:
    """Return a view of the data with the given dtype."""
    self._ensure_parts()
    if self._parts is None:
        raise NotImplementedError(
            "view is not defined on record-layout Ragged arrays; "
            "update fields individually, e.g. rag['f'] = rag['f'].view(dtype)."
        )
    # ... existing body unchanged
```

In `apply`:

```python
def apply(self, gufunc, *args, **kwargs) -> Ragged[DTYPE]:
    self._ensure_parts()
    if self._parts is None:
        raise NotImplementedError(
            "apply is not defined on record-layout Ragged arrays; "
            "apply per field instead."
        )
    # ... existing body unchanged
```

In `to_numpy`:

```python
def to_numpy(self, allow_missing: bool = False) -> NDArray[RDTYPE]:
    """Note: not zero-copy if offsets or data are non-contiguous."""
    self._ensure_parts()
    if self._parts is None:
        raise NotImplementedError(
            "to_numpy is not defined on record-layout Ragged arrays; "
            "convert fields individually."
        )
    # ... existing body unchanged
```

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: clear NotImplementedError for view/apply/to_numpy on record Ragged

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: `squeeze` works on record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`squeeze`)
- Test: `tests/test_ragged.py`

- [ ] **Step 1: Write failing test**

```python
def test_squeeze_record(self):
    # Build a record Ragged with a non-ragged size-1 axis.
    f0 = Ragged.from_lengths(np.arange(6, dtype=np.int64).reshape(6, 1), np.array([2, 1, 3]))
    f1 = Ragged.from_lengths(np.arange(6, dtype=np.float64).reshape(6, 1), np.array([2, 1, 3]))
    rag = Ragged(ak.zip({"a": f0, "b": f1}, depth_limit=1))
    sq = rag.squeeze()
    assert isinstance(sq, Ragged)
    assert sq.shape == (3, None)
    np.testing.assert_array_equal(sq["a"].data, np.arange(6))
    # Offsets should still be shared zero-copy across fields.
    assert sq["a"].offsets is sq.offsets
```

- [ ] **Step 2: Run, expect fail**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_squeeze_record -v`
Expected: FAIL — current squeeze uses `self._parts.data` which is None.

- [ ] **Step 3: Implement**

Add a record dispatch at the top of `squeeze`:

```python
def squeeze(self, axis=None):
    self._ensure_parts()
    if self._parts is None:
        squeezed = {f: self[f].squeeze(axis) for f in ak.fields(self)}
        first = next(iter(squeezed.values()))
        if isinstance(first, np.ndarray):
            return squeezed
        return type(self)(ak.zip(squeezed, depth_limit=1))
    # ... existing body unchanged
```

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_squeeze_record -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged.squeeze supports record layouts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: `reshape` works on record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`reshape`)
- Test: `tests/test_ragged.py`

- [ ] **Step 1: Write failing test**

```python
def test_reshape_record(self):
    # 6 ragged groups, reshapeable to (2, 3, None).
    lengths = np.array([2, 1, 3, 1, 2, 1])
    data_a = np.arange(10, dtype=np.int64)
    data_b = np.arange(10, dtype=np.float64)
    f0 = Ragged.from_lengths(data_a, lengths)
    f1 = Ragged.from_lengths(data_b, lengths)
    rag = Ragged(ak.zip({"a": f0, "b": f1}, depth_limit=1))
    re = rag.reshape(2, 3, None)
    assert isinstance(re, Ragged)
    assert re.shape == (2, 3, None)
    assert re["a"].offsets is re.offsets
    np.testing.assert_array_equal(re["a"].data, data_a)
```

- [ ] **Step 2: Run, expect fail**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_reshape_record -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Add record dispatch at top of `reshape`:

```python
def reshape(self, *shape):
    self._ensure_parts()
    if self._parts is None:
        reshaped = {f: self[f].reshape(*shape) for f in ak.fields(self)}
        return type(self)(ak.zip(reshaped, depth_limit=1))
    # ... existing body unchanged
```

- [ ] **Step 4: Run, expect pass**

Run: `pixi run pytest tests/test_ragged.py::TestRecordRagged::test_reshape_record -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged.reshape supports record layouts

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: `TestZip` — coverage for ak.zip paths

**Files:**
- Test: `tests/test_ragged.py` (new `TestZip` class)

- [ ] **Step 1: Write the test class**

Append to `tests/test_ragged.py`:

```python
class TestZip:
    def _mk(self, dtype):
        return Ragged.from_lengths(np.arange(6, dtype=dtype), np.array([2, 1, 3]))

    def test_zip_auto_returns_ragged(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = ak.zip({"a": r1, "b": r2})
        assert isinstance(z, Ragged)

    def test_zip_explicit_wrap_returns_ragged(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = Ragged(ak.zip({"a": r1, "b": r2}))
        assert isinstance(z, Ragged)

    def test_zip_three_fields(self):
        r1, r2, r3 = self._mk(np.int64), self._mk(np.float64), self._mk(np.int32)
        z = ak.zip({"a": r1, "b": r2, "c": r3})
        assert list(z.dtype.keys()) == ["a", "b", "c"]
        np.testing.assert_array_equal(z["c"].data, np.arange(6, dtype=np.int32))

    def test_zip_field_order_preserved(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = ak.zip({"zeta": r1, "alpha": r2})
        assert list(z.dtype.keys()) == ["zeta", "alpha"]

    def test_zip_offsets_shared_across_fields(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = ak.zip({"a": r1, "b": r2})
        assert z["a"].offsets is z["b"].offsets

    def test_zip_depth_limit_with_extra_dim(self):
        # Ragged with non-ragged trailing dim -> depth_limit=1 keeps inner dims intact.
        data_a = np.arange(12, dtype=np.int64).reshape(6, 2)
        data_b = np.arange(12, dtype=np.float64).reshape(6, 2)
        r1 = Ragged.from_lengths(data_a, np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(data_b, np.array([2, 1, 3]))
        z = ak.zip({"a": r1, "b": r2}, depth_limit=1)
        assert isinstance(z, Ragged)
        np.testing.assert_array_equal(z["a"].data, data_a)
```

- [ ] **Step 2: Run all new tests**

Run: `pixi run pytest tests/test_ragged.py::TestZip -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ragged.py
git commit -m "test: ak.zip auto and explicit-wrap paths for Ragged

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: Update class docstring

**Files:**
- Modify: `python/seqpro/rag/_array.py` (`Ragged` docstring)

- [ ] **Step 1: Replace the disclaimer**

In the `Ragged` class docstring, replace:

```
- Strings are not supported since ASCII is sufficient for the bioinformatics domain.
- Bytestrings count as a ragged dimension, and we break from the Awkward convention to not include a "var" in the type string.
- Ragged arrays are not tested with support for Awkward records/fields or union types. Functionality that appears
to work with these features may be experimental. Recommended to use depth_limit=1 when using ak.zip with one or more
Ragged arrays as input.
```

with:

```
- Strings are not supported since ASCII is sufficient for the bioinformatics domain.
- Bytestrings count as a ragged dimension, and we break from the Awkward convention to not include a "var" in the type string.
- Record-layout Ragged arrays (produced by ak.zip of Ragged inputs or by passing a record-layout ak.Array) return
  field-keyed dicts from `dtype`, `data`, and `parts`. Use `rag["field"]` for zero-copy single-field access.
  `view`, `apply`, and `to_numpy` are not defined on record layouts; access individual fields. Union types remain unsupported.
```

- [ ] **Step 2: Run full ragged test suite**

Run: `pixi run pytest tests/test_ragged.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add python/seqpro/rag/_array.py
git commit -m "docs: update Ragged docstring for record-layout support

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 10: Final regression sweep

- [ ] **Step 1: Run the full project test suite**

Run: `pixi run test`
Expected: all PASS.

- [ ] **Step 2: Lint**

Run: `pixi run -e dev ruff check python/seqpro/rag tests/test_ragged.py`
Expected: clean.

- [ ] **Step 3: No commit (verification only)** — fix any issues inline and amend the most recent topical commit.
