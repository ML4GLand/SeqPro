# Ragged[np.void] Record Array Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable `Ragged` to wrap awkward `RecordArray` content so that `rag.offsets`, `rag['field']`, and `rag.field` work, with zero-copy shared offsets across all field views.

**Architecture:** Two private helpers (`_is_record_layout`, `_extract_list_offsets`) detect and unpack the record case; `Ragged.__init__`, `.offsets`, `.data`, and `.__getitem__` each get a targeted guard that routes record arrays through the new path. All changes live in `_array.py`; tests go in the existing `test_ragged.py`.

**Tech Stack:** Python, NumPy, awkward-array (ak), pytest, pytest-cases

---

## File Map

| File | Change |
|---|---|
| `python/seqpro/rag/_array.py` | Add 2 helpers; modify `__init__`, `offsets`, `data`, `__getitem__`; update `_parts` annotation |
| `tests/test_ragged.py` | Add `class TestRecordRagged` with 7 tests |

---

### Task 1: Write failing tests for `Ragged[np.void]`

**Files:**
- Modify: `tests/test_ragged.py`

- [ ] **Step 1: Add the test class to `tests/test_ragged.py`**

Append after the existing tests:

```python
class TestRecordRagged:
    @pytest.fixture
    def rag(self):
        return Ragged(
            ak.Array(
                {
                    "field0": [[1, 2], [3], [4, 5, 6]],
                    "field1": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
                }
            )
        )

    def test_offsets(self, rag):
        expected = np.array([0, 2, 3, 6], dtype=OFFSET_TYPE)
        np.testing.assert_array_equal(rag.offsets, expected)

    def test_data_raises(self, rag):
        with pytest.raises(TypeError):
            _ = rag.data

    def test_field_access_by_key(self, rag):
        np.testing.assert_array_equal(
            rag["field0"].data, np.array([1, 2, 3, 4, 5, 6])
        )
        np.testing.assert_array_equal(
            rag["field1"].data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_field_access_by_attr(self, rag):
        np.testing.assert_array_equal(rag.field0.data, rag["field0"].data)
        np.testing.assert_array_equal(rag.field1.data, rag["field1"].data)

    def test_offsets_shared_with_field(self, rag):
        assert rag["field0"].offsets is rag.offsets

    def test_offsets_shared_across_fields(self, rag):
        assert rag["field0"].offsets is rag["field1"].offsets

    def test_field_returns_ragged(self, rag):
        assert isinstance(rag["field0"], Ragged)
        assert isinstance(rag["field1"], Ragged)
```

Also add `import pytest` at the top of the file if not already present.

- [ ] **Step 2: Run the tests to confirm they all fail**

```bash
pixi run pytest tests/test_ragged.py::TestRecordRagged -v
```

Expected: all 7 tests FAIL. The likely error is `ValueError: Must extract a single field before unboxing` or similar from `unbox()` being called during `Ragged.__init__`.

---

### Task 2: Add `_is_record_layout` and `_extract_list_offsets` helpers

**Files:**
- Modify: `python/seqpro/rag/_array.py`

These two helpers are called in Tasks 3 and 4. Add them just above the `Ragged` class definition.

- [ ] **Step 1: Add the helpers**

In `_array.py`, find the line `def is_rag_dtype(...)` and add the two helpers immediately before the `class Ragged` definition (after `is_rag_dtype`):

```python
def _is_record_layout(layout: Content) -> bool:
    """Return True if the innermost content (past list/regular wrappers) is a RecordArray."""
    node = layout
    while isinstance(node, (ListOffsetArray, ListArray, RegularArray)):
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    return isinstance(node, RecordArray)


def _extract_list_offsets(layout: Content) -> NDArray[OFFSET_TYPE]:
    """Extract the offsets from the outermost list layer of a layout."""
    node = layout
    while not isinstance(node, (ListOffsetArray, ListArray)):
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    if isinstance(node, ListOffsetArray):
        return cast(NDArray, node.offsets.data)
    else:
        return np.stack([node.starts.data, node.stops.data], 0)  # type: ignore
```

No import changes needed — `ListOffsetArray`, `ListArray`, `RegularArray`, `RecordArray`, `cast`, and `NDArray` are already imported at the top of the file.

- [ ] **Step 2: Run existing tests to confirm no regression**

```bash
pixi run pytest tests/test_ragged.py -v -k "not TestRecordRagged"
```

Expected: all existing tests PASS.

---

### Task 3: Fix `Ragged.__init__` and `_parts` annotation for record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py`

`unbox()` raises when the layout contains a `RecordArray`. Skip it for record layouts; fix the type annotation so `None` is valid.

- [ ] **Step 1: Update the `_parts` class annotation**

In the `Ragged` class body, change:

```python
_parts: RagParts[RDTYPE]
```

to:

```python
_parts: RagParts[RDTYPE] | None
```

- [ ] **Step 2: Guard `unbox()` in `__init__`**

Change the end of `__init__` from:

```python
super().__init__(content, behavior=deepcopy(ak.behavior))
self._parts = unbox(self)
```

to:

```python
super().__init__(content, behavior=deepcopy(ak.behavior))
if _is_record_layout(content):
    self._parts = None
else:
    self._parts = unbox(self)
```

- [ ] **Step 3: Smoke-test that construction no longer raises**

```bash
pixi run python -c "
import awkward as ak
from seqpro.rag import Ragged
rag = Ragged(ak.Array({'field0': [[1, 2], [3], [4, 5, 6]], 'field1': [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]}))
print('OK:', type(rag))
"
```

Expected output: `OK: <class 'seqpro.rag._array.Ragged'>`

- [ ] **Step 4: Run existing tests to confirm no regression**

```bash
pixi run pytest tests/test_ragged.py -k "not TestRecordRagged" -v
```

Expected: all PASS.

---

### Task 4: Fix `Ragged.offsets` and `Ragged.data` for record layouts

**Files:**
- Modify: `python/seqpro/rag/_array.py`

- [ ] **Step 1: Update the `offsets` property**

Find the current `offsets` property:

```python
@property
def offsets(self) -> NDArray[OFFSET_TYPE]:
    """The offsets of the Ragged array. May be 1- or 2-dimensional."""
    return self.parts.offsets
```

Replace it with:

```python
@property
def offsets(self) -> NDArray[OFFSET_TYPE]:
    """The offsets of the Ragged array. May be 1- or 2-dimensional."""
    if self._parts is None:
        # Record layout — extract from list layer once and cache.
        # object.__setattr__ used in case ak.Array intercepts __setattr__.
        if not hasattr(self, "_offsets_cache"):
            offsets = _extract_list_offsets(ak.to_layout(self, allow_record=False))
            object.__setattr__(self, "_offsets_cache", offsets)
        return self._offsets_cache  # type: ignore[return-value]
    return self._parts.offsets
```

- [ ] **Step 2: Update the `data` property**

Find the current `data` property:

```python
@property
def data(self) -> NDArray[RDTYPE]:
    """The data of the Ragged array."""
    return self.parts.data
```

Replace it with:

```python
@property
def data(self) -> NDArray[RDTYPE]:
    """The data of the Ragged array."""
    if self._parts is None:
        raise TypeError(
            "use rag['field'] to access the data of a record Ragged array"
        )
    return self._parts.data
```

- [ ] **Step 3: Run `test_offsets` and `test_data_raises`**

```bash
pixi run pytest tests/test_ragged.py::TestRecordRagged::test_offsets tests/test_ragged.py::TestRecordRagged::test_data_raises -v
```

Expected: both PASS.

- [ ] **Step 4: Run existing tests to confirm no regression**

```bash
pixi run pytest tests/test_ragged.py -k "not TestRecordRagged" -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_array.py
git commit -m "feat: support Ragged[np.void] record layout — offsets and data"
```

---

### Task 5: Fix `Ragged.__getitem__` to share offsets on field access

**Files:**
- Modify: `python/seqpro/rag/_array.py`

When a string key is used on a record Ragged, the returned field Ragged must share the exact same offsets object as the parent.

- [ ] **Step 1: Update `__getitem__`**

Find the current `__getitem__`:

```python
def __getitem__(self, where):
    arr = super().__getitem__(where)
    if isinstance(arr, ak.Array):
        if _n_var(arr) == 1:
            return type(self)(arr)
        else:
            return _as_ak(arr)
    else:
        return arr
```

Replace it with:

```python
def __getitem__(self, where):
    arr = super().__getitem__(where)
    if isinstance(arr, ak.Array):
        if _n_var(arr) == 1:
            result = type(self)(arr)
            # For record field access, share the parent's offsets object (zero-copy).
            if isinstance(where, str) and self._parts is None:
                result._parts = RagParts(
                    result._parts.data, result._parts.shape, self.offsets
                )
            return result
        else:
            return _as_ak(arr)
    else:
        return arr
```

- [ ] **Step 2: Run all `TestRecordRagged` tests**

```bash
pixi run pytest tests/test_ragged.py::TestRecordRagged -v
```

Expected: all 7 tests PASS.

- [ ] **Step 3: Run the full test suite**

```bash
pixi run pytest tests/test_ragged.py -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add python/seqpro/rag/_array.py tests/test_ragged.py
git commit -m "feat: Ragged[np.void] record array support with zero-copy field offsets"
```
