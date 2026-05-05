# Ragged[np.void] — Record Array Support

**Date:** 2026-05-04
**Status:** Approved

## Goal

Enable `Ragged` to wrap awkward `RecordArray` content (i.e. `Ragged[np.void]`), with zero-copy shared offsets across field views.

The target API:

```python
rag = Ragged(ak.Array({'field0': [[1, 2], [3], [4, 5, 6]], 'field1': [[...], ...]}))

rag.offsets                        # shared offsets array
rag['field0'].data                 # int32 numpy array
rag['field0'].offsets is rag.offsets  # True — zero-copy
rag.field0.data == rag['field0'].data  # True — attr and key access equivalent
rag.data                           # raises TypeError
```

## Implementation — `python/seqpro/rag/_array.py`

Five targeted edits, no new files:

### 1. `_is_record_layout(layout: Content) -> bool`

Helper that walks the layout past `ListOffsetArray`/`ListArray` nodes and returns `True` if the first inner node is a `RecordArray`.

### 2. `Ragged.__init__`

Add a guard before `unbox()`:

```python
if isinstance(data, RagParts):
    content = _parts_to_content(data)
else:
    content = _as_ragged(data, highlevel=False)
super().__init__(content, behavior=deepcopy(ak.behavior))
# Skip unbox for record layouts — offsets extracted lazily in .offsets property
if not _is_record_layout(content):
    self._parts = unbox(self)
```

### 3. `Ragged.offsets` property

```python
@property
def offsets(self):
    if self._parts is None:
        # Record layout — extract from list layer and cache
        if not hasattr(self, '_offsets_cache'):
            offsets = _extract_list_offsets(ak.to_layout(self))
            object.__setattr__(self, '_offsets_cache', offsets)
        return self._offsets_cache
    return self._parts.offsets
```

`_extract_list_offsets(layout)` walks down to the first `ListOffsetArray` or `ListArray` and returns `.offsets.data` / `np.stack([starts, stops])` respectively.

### 4. `Ragged.data` property

Add a guard:

```python
@property
def data(self):
    if self._parts is None:
        raise TypeError("use rag['field'] to access fields of a record Ragged")
    return self._parts.data
```

### 5. `Ragged.__getitem__`

For string keys on a record Ragged, share offsets explicitly:

```python
def __getitem__(self, where):
    arr = super().__getitem__(where)
    if isinstance(arr, ak.Array):
        if _n_var(arr) == 1:
            result = type(self)(arr)
            # Share the exact offsets object so field.offsets is rag.offsets
            if isinstance(where, str) and self._parts is None:
                result._parts = RagParts(result._parts.data, result._parts.shape, self.offsets)
            return result
        else:
            return _as_ak(arr)
    else:
        return arr
```

## Tests — `tests/test_ragged.py`

New `class TestRecordRagged` (not parametrized with existing cases):

**Fixture** (`case_record` or `@pytest.fixture`):
- `field0`: `np.int32`, values `[1, 2, 3, 4, 5, 6]`
- `field1`: `np.float64`, values `[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`
- Lengths `[2, 1, 3]` → offsets `[0, 2, 3, 6]`
- Built via `Ragged(ak.Array({'field0': [[1,2],[3],[4,5,6]], 'field1': [[1.,2.],[3.],[4.,5.,6.]]}))`

| Test | Assertion |
|---|---|
| `test_offsets` | `np.testing.assert_array_equal(rag.offsets, [0, 2, 3, 6])` |
| `test_data_raises` | `pytest.raises(TypeError, lambda: rag.data)` |
| `test_field_access_by_key` | `.data` matches source arrays for `field0` and `field1` |
| `test_field_access_by_attr` | `rag.field0.data` equals `rag['field0'].data`, same for `field1` |
| `test_offsets_shared_with_field` | `rag['field0'].offsets is rag.offsets` |
| `test_offsets_shared_across_fields` | `rag['field0'].offsets is rag['field1'].offsets` |
| `test_field_returns_ragged` | `isinstance(rag['field0'], Ragged)` and `isinstance(rag['field1'], Ragged)` |

## Type annotation change

`_parts: RagParts[RDTYPE]` on the `Ragged` class becomes `_parts: RagParts[RDTYPE] | None`. Methods that go through `self.parts` (e.g. `apply`, `squeeze`, `reshape`) remain unsupported for record Ragged arrays — they will raise naturally via `_parts is None` → `NoneType` attribute error, or can be given an explicit guard if desired. This is a non-goal for this work.

## Non-goals

- `rag.data` returning a structured numpy array
- Nested record arrays (only one level of Record supported)
- Mutation of field data through the record Ragged
