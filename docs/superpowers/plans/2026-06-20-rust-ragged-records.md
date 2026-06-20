# Rust-native `Ragged` — String/Char Duality + Records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the opaque-string/ascii-char duality to the Rust-native `Ragged` core, then native record (struct-of-arrays) support, all on the new `rag/_core.py` path and differential-tested against the awkward oracle.

**Architecture:** Pure-Python layout algebra over the Spec A `RaggedLayout` value object and existing Rust/Numba kernels — no new Rust. String vs char is a dtype-as-descriptor distinction with zero-copy offset retags; a record is a new `RecordLayout` composing per-field `RaggedLayout`s over one shared offsets object.

**Tech Stack:** Python 3, NumPy (`>=1.26`), attrs, awkward (oracle only), Numba (existing pack/pad kernels), PyO3 Rust extension (existing kernels only), pytest + Hypothesis, pixi.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-06-20-rust-ragged-records-design.md`. **SSoT:** `docs/roadmap/rust-ragged.md` — read it; update its status when this work lands (same PR).
- **numpy floor stays `>=1.26`** — use `np.dtype('S')` / `np.dtype('S1')`, never `StringDType`.
- **Internal only.** Work lands on `python/seqpro/rag/_core.py` & friends. Do **not** touch the awkward `_array.py` public `Ragged`. Public `seqpro.rag.Ragged` stays awkward (swap is Spec D).
- **Awkward stays installed** as the differential oracle (`from seqpro.rag._array import Ragged as AkRagged`).
- **No new Rust kernels.** Reuse `_ragged_select`, the Numba pack (`_pack_parts`), `_ragged_validate`.
- **Opaque `'S'` strings are standalone only** — not record fields (alleles → Spec C).
- **Conventions:** validation front-loaded (fast-fail); no naive Python loops in hot paths (records iterate over *fields*, not elements — fine); conventional commits (`feat:`/`test:`/`refactor:`); `skills/seqpro/SKILL.md` is **not** updated in Spec B.
- **Env:** run everything under pixi: `pixi run -e dev pytest ...` (or `pixi shell -e dev`).
- **Build:** these tasks touch no Rust, so `maturin develop` is not required; the existing compiled `seqpro.abi3.so` is used as-is.

---

## Task Group 1 — Core string/char duality

### Task 1: `.dtype` reports `'S'` for opaque strings

**Files:**
- Modify: `python/seqpro/rag/_core.py:114-116` (`dtype` property)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout.is_string` (existing, `str_offsets is not None`), `RaggedLayout.data`.
- Produces: `Ragged.dtype` → `np.dtype('S')` (itemsize 0) when opaque string, else `data.dtype`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ragged_core.py`:

```python
def test_opaque_string_dtype_is_flexible_bytes():
    rag = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    assert rag.dtype == np.dtype("S")
    assert rag.dtype.itemsize == 0
    assert rag.is_string is True  # see Task 3 for is_string property
    # storage is still S1 bytes
    assert rag.data.dtype == np.dtype("S1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_core.py::test_opaque_string_dtype_is_flexible_bytes -v`
Expected: FAIL — `rag.dtype` is `dtype('S1')`, not `dtype('S')` (and `is_string` may be missing until Task 3; if so, split the assert out — it is re-added in Task 3).

- [ ] **Step 3: Implement the dtype change**

Replace the `dtype` property body (`_core.py:114-116`):

```python
    @property
    def dtype(self) -> np.dtype[Any]:
        if self._layout.is_string:
            return np.dtype("S")  # opaque variable-width string: descriptor, not S1 storage
        return self._layout.data.dtype
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/test_ragged_core.py::test_opaque_string_dtype_is_flexible_bytes -v`
Expected: PASS (the `is_string` assert may need Task 3; if so temporarily drop that line and restore in Task 3).

- [ ] **Step 5: Update the one Spec A test that asserted S1 for a collection**

In `tests/test_ragged_core.py::test_from_lengths_string_collapses_to_leaf`, change:

```python
    assert rag.dtype == np.dtype("S1")
```
to
```python
    assert rag.dtype == np.dtype("S")  # opaque string descriptor (string/char duality)
```

- [ ] **Step 6: Run the full core suite**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: report np.dtype('S') for opaque-string Ragged (string/char duality)"
```

---

### Task 2: `is_string` property + constructor disambiguation (None ⇒ chars, else opaque)

**Files:**
- Modify: `python/seqpro/rag/_core.py` — `_build_layout` (21-40), `from_offsets` (63-74), `from_lengths` (76-81); add `is_string` property.
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout`, `OFFSET_TYPE`, `lengths_to_offsets`.
- Produces:
  - `Ragged.is_string -> bool` (opaque string ⇔ `True`).
  - Constructor rule: `from_offsets(data, shape, offsets)` — `None in shape` ⇒ counted ragged axis (numeric **or** `S1` chars); no `None` + `S1` ⇒ opaque leaf (`offsets` stored as `str_offsets`). `from_lengths(S1, lengths)` ⇒ opaque `(N,)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_is_string_predicate():
    s = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), np.array([3, 3]))
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    assert s.is_string is True
    assert n.is_string is False

def test_from_offsets_S1_with_none_is_chars_not_opaque():
    data = np.frombuffer(b"cathithere", "S1")
    offsets = np.array([0, 3, 5, 10])
    chars = Ragged.from_offsets(data, (3, None), offsets)
    assert chars.is_string is False          # counted axis => chars
    assert chars.dtype == np.dtype("S1")
    assert chars.shape == (3, None)

def test_from_offsets_S1_without_none_is_opaque():
    data = np.frombuffer(b"cathithere", "S1")
    str_offsets = np.array([0, 3, 5, 10])
    opaque = Ragged.from_offsets(data, (3,), str_offsets)
    assert opaque.is_string is True
    assert opaque.dtype == np.dtype("S")
    assert opaque.shape == (3,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -k "is_string or from_offsets_S1" -v`
Expected: FAIL — `is_string` missing; `from_offsets((3,None))` currently collapses S1 to opaque.

- [ ] **Step 3: Add the `is_string` property**

After the `dtype` property in `_core.py`:

```python
    @property
    def is_string(self) -> bool:
        """True for an opaque variable-width string Ragged (dtype 'S', shape (N,))."""
        return self._layout.is_string
```

- [ ] **Step 4: Rewrite `_build_layout` for the new disambiguation rule**

Replace `_build_layout` (`_core.py:21-40`):

```python
def _build_layout(
    data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any]
) -> RaggedLayout[Any]:
    """Build a single-level layout under the string/char rule.

    - ``None`` present in ``shape``  -> counted ragged axis: numeric, or S1 *chars*
      (length is an axis). ``offsets`` is the ragged axis; ``str_offsets`` is None.
    - no ``None`` + S1 data          -> opaque *string* leaf: ``offsets`` is stored
      as ``str_offsets``; the byte-length is not an axis.
    """
    if None in shape:
        return RaggedLayout(data=data, offsets=[offsets], shape=shape)
    if data.dtype.kind == "S":
        return RaggedLayout(data=data, offsets=[], shape=shape, str_offsets=offsets)
    raise ValueError("shape must have exactly one None ragged dimension for numeric data")
```

- [ ] **Step 5: Make `from_offsets` accept the no-None S1 (opaque) case**

Replace the guard in `from_offsets` (`_core.py:67-74`) so a no-`None` shape is allowed only for `S1` data:

```python
    @staticmethod
    def from_offsets(
        data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any]
    ) -> "Ragged[Any]":
        if shape.count(None) > 1:
            raise NotImplementedError(
                "nested raggedness (>1 ragged level) lands in Spec C"
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have exactly one None ragged dimension")
        offsets = np.ascontiguousarray(offsets, dtype=OFFSET_TYPE)
        return Ragged(_build_layout(data, shape, offsets))
```

(This body is unchanged from Spec A except it now routes through the rewritten `_build_layout`; keep it verbatim to be safe.)

- [ ] **Step 6: Make `from_lengths(S1, ...)` build the opaque leaf directly**

Replace `from_lengths` (`_core.py:76-81`):

```python
    @staticmethod
    def from_lengths(data: NDArray[Any], lengths: NDArray[Any]) -> "Ragged[Any]":
        offsets = lengths_to_offsets(lengths)
        if data.dtype.kind == "S" and data.ndim == 1:
            # opaque string collection: (N,), byte-length is not an axis
            shape: tuple[int | None, ...] = tuple(lengths.shape)
            return Ragged.from_offsets(data, shape, offsets)
        trailing = data.shape[1:]
        shape = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets)
```

- [ ] **Step 7: Run the affected tests**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -k "is_string or from_offsets_S1 or from_lengths or string" -v`
Expected: PASS.

- [ ] **Step 8: Run the full core suite (catch collapse-behavior regressions)**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -v`
Expected: PASS. If `test_from_offsets_numeric_trailing_dim` or string tests fail, reconcile against the rule above.

- [ ] **Step 9: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: disambiguate opaque-string vs char layout by presence of None in shape"
```

---

### Task 3: zero-copy `to_chars()` / `to_strings()`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (add two methods)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout`, `is_string`.
- Produces:
  - `Ragged.to_chars() -> Ragged` — opaque `'S'` `(N,)` → `'S1'` `(N, None)`; promotes `str_offsets` → `offsets`; same data buffer; raises if not opaque.
  - `Ragged.to_strings() -> Ragged` — `'S1'` `(N, None)` (no trailing) → opaque `'S'` `(N,)`; demotes `offsets` → `str_offsets`; raises if not a 1-D `S1` char leaf.

- [ ] **Step 1: Write the failing tests**

```python
def test_to_chars_zero_copy_and_shape():
    s = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    c = s.to_chars()
    assert c.dtype == np.dtype("S1")
    assert c.shape == (3, None)
    assert c.is_string is False
    assert c.data is s.data                      # zero-copy buffer
    np.testing.assert_array_equal(c.offsets, s.offsets)  # str_offsets promoted
    np.testing.assert_array_equal(c[0], np.frombuffer(b"cat", "S1"))

def test_to_strings_roundtrip():
    s = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    back = s.to_chars().to_strings()
    assert back.dtype == np.dtype("S")
    assert back.shape == (3,)
    assert back[0] == b"cat"
    assert back.data is s.data

def test_to_chars_raises_on_non_opaque():
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    with pytest.raises(ValueError, match="opaque"):
        n.to_chars()

def test_to_strings_raises_on_trailing_dims():
    data = np.zeros((6, 4), dtype="S1")
    chars_with_trailing = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6]))
    with pytest.raises(ValueError, match="1-D|trailing"):
        chars_with_trailing.to_strings()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -k "to_chars or to_strings" -v`
Expected: FAIL — methods not defined.

- [ ] **Step 3: Implement the conversions**

Add to `Ragged` in `_core.py` (e.g. after `view`):

```python
    def to_chars(self) -> "Ragged[Any]":
        """Zero-copy view of an opaque string ('S', (N,)) as ascii chars
        ('S1', (N, None)); the byte-length becomes a counted ragged axis."""
        if not self._layout.is_string:
            raise ValueError("to_chars() requires an opaque string Ragged (dtype 'S')")
        assert self._layout.str_offsets is not None
        new_shape = (*self._layout.shape, None)
        return Ragged(
            RaggedLayout(
                data=self._layout.data,
                offsets=[self._layout.str_offsets],
                shape=new_shape,
            )
        )

    def to_strings(self) -> "Ragged[Any]":
        """Zero-copy view of a 1-D ascii-char leaf ('S1', (N, None)) as an opaque
        string ('S', (N,)); the length axis becomes an uncounted byte leaf."""
        if self._layout.is_string:
            return self
        if self._layout.data.dtype.kind != "S":
            raise ValueError("to_strings() requires an S1 char Ragged")
        if self._layout.data.ndim != 1 or self._layout.shape[self.rag_dim + 1 :]:
            raise ValueError("to_strings() requires a 1-D S1 char leaf (no trailing dims)")
        return Ragged(
            RaggedLayout(
                data=self._layout.data,
                offsets=[],
                shape=self._layout.shape[: self.rag_dim],
                str_offsets=self.offsets,
            )
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -k "to_chars or to_strings" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: add zero-copy to_chars/to_strings between opaque-string and char Ragged"
```

---

## Task Group 2 — Records

### Task 4: `RecordLayout` value object + validation arm

**Files:**
- Modify: `python/seqpro/rag/_layout.py` (add `RecordLayout`, extend `validate_layout`)
- Test: `tests/test_ragged_core_records.py` (new)

**Interfaces:**
- Consumes: `RaggedLayout`, `OFFSET_TYPE`.
- Produces:
  - `RecordLayout(offsets: list[NDArray], shape: tuple[int|None,...], fields: dict[str, RaggedLayout])`.
  - `validate_layout(RaggedLayout | RecordLayout)` — record arm: non-empty fields; every field numeric/char (not opaque, not nested); every field's `offsets[0] is offsets[0]` (shared identity); ragged-shape agreement; per-field single-level validation.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ragged_core_records.py`:

```python
import numpy as np
import pytest
from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, RecordLayout, validate_layout


def _two_field_record():
    shared = lengths_to_offsets(np.array([2, 1, 3], dtype=np.uint32))
    f0 = RaggedLayout(data=np.arange(6, dtype=np.int32), offsets=[shared], shape=(3, None))
    f1 = RaggedLayout(data=np.arange(6, dtype=np.float64), offsets=[shared], shape=(3, None))
    return RecordLayout(offsets=[shared], shape=(3, None), fields={"a": f0, "b": f1})


def test_record_layout_validates():
    validate_layout(_two_field_record())


def test_record_layout_rejects_empty_fields():
    with pytest.raises(ValueError, match="empty|at least one"):
        validate_layout(RecordLayout(offsets=[np.array([0])], shape=(0, None), fields={}))


def test_record_layout_rejects_unshared_offsets():
    a = lengths_to_offsets(np.array([2, 1, 3], dtype=np.uint32))
    b = a.copy()  # equal values, different object
    f0 = RaggedLayout(data=np.arange(6, dtype=np.int32), offsets=[a], shape=(3, None))
    f1 = RaggedLayout(data=np.arange(6, dtype=np.int32), offsets=[b], shape=(3, None))
    with pytest.raises(ValueError, match="shared|same offsets"):
        validate_layout(RecordLayout(offsets=[a], shape=(3, None), fields={"a": f0, "b": f1}))


def test_record_layout_rejects_opaque_field():
    shared = lengths_to_offsets(np.array([3, 3], dtype=np.uint32))
    opaque = RaggedLayout(data=np.frombuffer(b"catdog", "S1"), offsets=[], shape=(2,), str_offsets=shared)
    with pytest.raises(NotImplementedError, match="Spec C|opaque"):
        validate_layout(RecordLayout(offsets=[shared], shape=(2, None), fields={"s": opaque}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -v`
Expected: FAIL — `RecordLayout` not importable.

- [ ] **Step 3: Add `RecordLayout` and validation**

In `_layout.py`, add after `RaggedLayout`:

```python
@define
class RecordLayout:
    """Struct-of-arrays: named numeric/char fields sharing one ragged offsets object.

    offsets
        The single shared ragged offsets object (Spec B: ``len == 1``); identical
        object to every field's ``offsets[0]``.
    shape
        Canonical ragged shape ``(*leading, None, *trailing?)`` (the first field's).
    fields
        Insertion-ordered field name -> single-level ``RaggedLayout`` (numeric or
        S1 chars). Opaque-string fields are out of scope (Spec C).
    """

    offsets: list[NDArray[Any]]
    shape: tuple[int | None, ...]
    fields: dict[str, RaggedLayout[Any]]
```

Add `from typing import Any` import if missing (it is already imported). Then extend `validate_layout` to dispatch:

```python
def validate_layout(layout: RaggedLayout[Any] | RecordLayout) -> None:
    if isinstance(layout, RecordLayout):
        _validate_record_layout(layout)
        return
    # ... existing RaggedLayout body unchanged ...
```

And add the record validator:

```python
def _validate_record_layout(layout: RecordLayout) -> None:
    if not layout.fields:
        raise ValueError("record layout must have at least one field (got empty)")
    if not layout.offsets:
        raise ValueError("record layout must have a shared offsets array")
    shared = layout.offsets[0]
    rag_dim = layout.shape.index(None)
    ragged_shape = layout.shape[: rag_dim + 1]
    for name, field in layout.fields.items():
        if field.is_string:
            raise NotImplementedError(
                f"opaque-string record field {name!r} (S-under-axis) lands in Spec C"
            )
        if not field.offsets or field.offsets[0] is not shared:
            raise ValueError(
                f"field {name!r} must share the record's offsets object (zero-copy SoA)"
            )
        if field.shape[: field.shape.index(None) + 1] != ragged_shape:
            raise ValueError(
                f"field {name!r} ragged shape {field.shape} disagrees with record {layout.shape}"
            )
        validate_layout(field)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_layout.py tests/test_ragged_core_records.py
git commit -m "feat: add RecordLayout value object and validation arm"
```

---

### Task 5: `Ragged.from_fields` + `_layout` plumbing + `rag.zip` export

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`__init__` validate call, add `from_fields`, `_is_record` helper)
- Modify: `python/seqpro/rag/__init__.py` (export `zip`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `RecordLayout`, `RaggedLayout`, `validate_layout`.
- Produces:
  - `Ragged.from_fields(fields: dict[str, Ragged]) -> Ragged` (record).
  - `Ragged._is_record -> bool` property.
  - `seqpro.rag.zip(fields) -> Ragged` (alias → `from_fields`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ragged_core_records.py`:

```python
from seqpro.rag._core import Ragged


def _record_ragged():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    return Ragged.from_fields({"a": a, "b": b})


def test_from_fields_builds_record():
    rag = _record_ragged()
    assert rag._is_record is True
    assert rag.fields == ["a", "b"]            # Task 6 adds .fields; if missing, skip


def test_from_fields_canonicalizes_shared_offsets():
    rag = _record_ragged()
    assert rag["a"].offsets is rag["b"].offsets  # Task 7 adds field access


def test_from_fields_rejects_empty():
    with pytest.raises(ValueError, match="empty|at least one"):
        Ragged.from_fields({})


def test_from_fields_rejects_offset_mismatch():
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32))
    b = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 1, 2], np.uint32))
    with pytest.raises(ValueError, match="offset|equal"):
        Ragged.from_fields({"a": a, "b": b})


def test_from_fields_rejects_opaque_field():
    lens = np.array([3, 3], dtype=np.uint32)
    s = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), lens)  # opaque
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    with pytest.raises(NotImplementedError, match="Spec C|opaque|chars"):
        Ragged.from_fields({"s": s, "n": n})


def test_zip_alias():
    import seqpro.rag as rag_mod
    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    rec = rag_mod.zip({"a": a, "b": b})
    assert rec._is_record is True
```

*(Some asserts depend on Tasks 6/7; if running this task in isolation, comment the `.fields`/field-access lines and restore them when those tasks land — they are covered by their own tasks too.)*

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "from_fields or zip_alias" -v`
Expected: FAIL — `from_fields` not defined.

- [ ] **Step 3: Accept `RecordLayout` in `__init__` and add `_is_record`**

In `_core.py`, update the imports to include `RecordLayout`:

```python
from ._layout import RaggedLayout, RecordLayout, validate_layout
```

`Ragged.__init__` already calls `validate_layout(data)` and stores `self._layout = data`; since `validate_layout` now dispatches on `RecordLayout`, the only change is the `isinstance` guard so a `RecordLayout` isn't mistaken for raw data. Update the `__init__` head:

```python
    def __init__(self, data: Any):
        if isinstance(data, Ragged):
            data = data._layout
        if not isinstance(data, (RaggedLayout, RecordLayout)):
            from ._ingest import layout_from_ak

            data = layout_from_ak(data)
        validate_layout(data)
        self._layout = data
```

Add the predicate property:

```python
    @property
    def _is_record(self) -> bool:
        return isinstance(self._layout, RecordLayout)
```

- [ ] **Step 4: Implement `from_fields`**

Add to `Ragged` (near `from_offsets`):

```python
    @staticmethod
    def from_fields(fields: "dict[str, Ragged[Any]]") -> "Ragged[Any]":
        """Build a record (struct-of-arrays) from named single-field Ragged inputs
        that share one ragged axis. Sequence fields must be chars (see to_chars)."""
        if not fields:
            raise ValueError("from_fields requires at least one field (got empty)")
        items = list(fields.items())
        for name, f in items:
            if f._is_record:
                raise NotImplementedError(
                    f"record-of-record field {name!r} lands in Spec C"
                )
            if f.is_string:
                raise NotImplementedError(
                    f"opaque-string field {name!r} is Spec C; pass chars via .to_chars()"
                )
        shared = items[0][1].offsets
        for name, f in items[1:]:
            if not np.array_equal(f.offsets, shared):
                raise ValueError(f"field {name!r} offsets are not equal to the first field's")
        rec_shape = items[0][1].shape
        rebound: dict[str, RaggedLayout[Any]] = {}
        for name, f in items:
            rebound[name] = RaggedLayout(
                data=f._layout.data, offsets=[shared], shape=f._layout.shape
            )
        return Ragged(RecordLayout(offsets=[shared], shape=rec_shape, fields=rebound))
```

- [ ] **Step 5: Export `zip` from the package**

In `python/seqpro/rag/__init__.py`, add (after the existing imports):

```python
from ._core import Ragged as _CoreRagged


def zip(fields):  # noqa: A001  (intentional ak.zip-compatible name)
    """Build a record Ragged from a dict of single-field Ragged inputs.

    Alias for ``Ragged.from_fields``; operates on the Rust-native core path.
    """
    return _CoreRagged.from_fields(fields)
```

And add `"zip"` to `__all__`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "from_fields or zip_alias" -v`
Expected: PASS (lines depending on Tasks 6/7 still commented if those aren't landed yet).

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_core.py python/seqpro/rag/__init__.py tests/test_ragged_core_records.py
git commit -m "feat: add Ragged.from_fields record constructor and rag.zip alias"
```

---

### Task 6: Record-branch properties (`data`, `dtype`, `offsets`, `shape`, `fields`, `lengths`, state)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (properties branch on `_is_record`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `RecordLayout.fields`, `.offsets`, `.shape`.
- Produces (record branch):
  - `data -> dict[str, NDArray]`; `dtype -> np.dtype` structured `[(name, fdtype), …]`;
  - `offsets -> NDArray` (shared); `shape -> tuple`; `fields -> list[str]`;
  - `lengths`, `is_empty`, `is_contiguous`, `is_base` over the shared offsets / all fields.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_data_dict_zero_copy():
    rag = _record_ragged()
    d = rag.data
    assert list(d.keys()) == ["a", "b"]
    np.testing.assert_array_equal(d["a"], np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(d["b"], np.arange(6, dtype=np.float64))
    assert d["a"].base is not None

def test_record_dtype_structured():
    rag = _record_ragged()
    assert rag.dtype == np.dtype([("a", np.int32), ("b", np.float64)])

def test_record_offsets_shape_fields_lengths():
    rag = _record_ragged()
    np.testing.assert_array_equal(rag.offsets, np.array([0, 2, 3, 6]))
    assert rag.shape == (3, None)
    assert rag.fields == ["a", "b"]
    np.testing.assert_array_equal(rag.lengths, np.array([2, 1, 3]))

def test_record_state_predicates():
    rag = _record_ragged()
    assert rag.is_empty is False
    assert rag.is_contiguous is True
    assert rag.is_base is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "record_data or record_dtype or offsets_shape or state_predicates" -v`
Expected: FAIL.

- [ ] **Step 3: Branch the properties on `_is_record`**

In `_core.py`, update each property. `data`:

```python
    @property
    def data(self) -> "NDArray[Any] | dict[str, NDArray[Any]]":
        if isinstance(self._layout, RecordLayout):
            return {f: fl.data for f, fl in self._layout.fields.items()}
        return self._layout.data
```

`dtype`:

```python
    @property
    def dtype(self) -> np.dtype[Any]:
        if isinstance(self._layout, RecordLayout):
            return np.dtype([(f, fl.data.dtype) for f, fl in self._layout.fields.items()])
        if self._layout.is_string:
            return np.dtype("S")
        return self._layout.data.dtype
```

`offsets` (record returns the shared array):

```python
    @property
    def offsets(self) -> NDArray[Any]:
        if isinstance(self._layout, RecordLayout):
            return self._layout.offsets[0]
        if self._layout.offsets:
            return self._layout.offsets[0]
        assert self._layout.str_offsets is not None
        return self._layout.str_offsets
```

`shape` and `rag_dim` already read `self._layout.shape` / `.index(None)`; `RecordLayout` has both, so they work unchanged. Add a `fields` property:

```python
    @property
    def fields(self) -> list[str]:
        if isinstance(self._layout, RecordLayout):
            return list(self._layout.fields)
        raise TypeError("fields is only defined on record Ragged arrays")
```

`lengths` already derives from `self.offsets` + `self._layout.shape` — works for records (shared offsets, canonical shape). Verify the `None in self._layout.shape` branch holds (record shape has a `None`).

`is_empty` uses `self.offsets` — works. Update `is_contiguous` / `is_base` to fold over fields:

```python
    @property
    def is_contiguous(self) -> bool:
        if isinstance(self._layout, RecordLayout):
            return self.offsets.ndim == 1 and all(
                fl.data.flags.c_contiguous for fl in self._layout.fields.values()
            )
        return self.offsets.ndim == 1 and self._layout.data.flags.c_contiguous

    @property
    def is_base(self) -> bool:
        offsets = self.offsets
        if isinstance(self._layout, RecordLayout):
            fields = self._layout.fields.values()
            owns = all(fl.data.base is None for fl in fields)
            size0 = next(iter(fields)).data.shape[0]
            return bool(owns and self.is_contiguous and offsets[0] == 0 and offsets[-1] == size0)
        data = self._layout.data
        owns_memory = data.base is None or (data.base is not None and data.base.base is None)
        return bool(
            owns_memory and self.is_contiguous and offsets[0] == 0 and offsets[-1] == data.shape[0]
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full core suite (no non-record regressions)**

Run: `pixi run -e dev pytest tests/test_ragged_core.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: record-branch properties (data/dtype/offsets/shape/fields/state)"
```

---

### Task 7: Field access (`rag['f']` / `rag.f`) + mutation (`rag['f'] = ...`)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`__getitem__` str key, add `__getattr__`, add `__setitem__`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `RecordLayout.fields`, `from_fields` validation helpers.
- Produces:
  - `rag['f'] -> Ragged` (zero-copy single-field; `offsets is rag.offsets`).
  - `rag.f -> Ragged` (same).
  - `rag['f'] = new_ragged` — replace/add a numeric/char field whose offsets equal the record's (copy-on-write of `fields`).

- [ ] **Step 1: Write the failing tests**

```python
def test_field_access_by_key_and_attr():
    rag = _record_ragged()
    np.testing.assert_array_equal(rag["a"].data, np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(rag.a.data, rag["a"].data)
    assert rag["a"].offsets is rag.offsets
    assert rag["a"].offsets is rag["b"].offsets

def test_field_access_unknown_raises():
    rag = _record_ragged()
    with pytest.raises(KeyError):
        rag["nope"]

def test_setitem_replace_field():
    rag = _record_ragged()
    rag["a"] = rag["a"].view(np.uint32)
    assert rag["a"].dtype == np.dtype(np.uint32)
    assert rag["a"].offsets is rag.offsets

def test_setitem_add_field():
    rag = _record_ragged()
    new = Ragged.from_offsets(np.arange(6, dtype=np.int16), (3, None), rag.offsets)
    rag["c"] = new
    assert rag.fields == ["a", "b", "c"]
    assert rag["c"].offsets is rag.offsets

def test_setitem_offset_mismatch_raises():
    rag = _record_ragged()
    bad = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 1, 2], np.uint32))
    with pytest.raises(ValueError, match="offset|equal"):
        rag["d"] = bad
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "field_access or setitem" -v`
Expected: FAIL.

- [ ] **Step 3: Field access in `__getitem__`**

At the top of `__getitem__` in `_core.py`, add the record-string-key branch before the existing single-level logic:

```python
    def __getitem__(self, where: Any) -> "NDArray[Any] | bytes | dict[str, Any] | Ragged[Any]":
        if isinstance(self._layout, RecordLayout):
            return self._getitem_record(where)
        # ... existing single-level body unchanged ...
```

Add the record getitem helper (row-axis indexing for non-str `where` lands in Task 8; for now handle str keys and delegate others to a stub that Task 8 fills):

```python
    def _getitem_record(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if isinstance(where, str):
            try:
                field = rec.fields[where]
            except KeyError:
                raise KeyError(where)
            return Ragged(field)  # field.offsets[0] is the shared object (zero-copy)
        return self._getitem_record_rows(where)  # Task 8
```

- [ ] **Step 4: Attribute access**

Add `__getattr__` (only called when normal attribute lookup fails; safe with `__slots__`):

```python
    def __getattr__(self, name: str) -> "Ragged[Any]":
        # only reached when `name` is not a real attribute/slot
        layout = self.__dict__.get("_layout") if hasattr(self, "__dict__") else None
        layout = layout if layout is not None else object.__getattribute__(self, "_layout")
        if isinstance(layout, RecordLayout) and name in layout.fields:
            return Ragged(layout.fields[name])
        raise AttributeError(name)
```

*(Note: `Ragged` uses `__slots__ = ("_layout",)`; `object.__getattribute__(self, "_layout")` retrieves it without recursing through `__getattr__`.)*

- [ ] **Step 5: Field mutation `__setitem__`**

```python
    def __setitem__(self, key: str, value: "Ragged[Any]") -> None:
        if not isinstance(self._layout, RecordLayout):
            raise TypeError("item assignment is only supported on record Ragged arrays")
        if not isinstance(key, str):
            raise TypeError("record field assignment requires a string field name")
        if value._is_record or value.is_string:
            raise NotImplementedError("record fields must be numeric/char single fields")
        shared = self._layout.offsets[0]
        if not np.array_equal(value.offsets, shared):
            raise ValueError("assigned field offsets must equal the record's offsets")
        new_field = RaggedLayout(data=value._layout.data, offsets=[shared], shape=value._layout.shape)
        new_fields = dict(self._layout.fields)
        new_fields[key] = new_field
        self._layout = RecordLayout(offsets=[shared], shape=self._layout.shape, fields=new_fields)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "field_access or setitem" -v`
Expected: PASS.

- [ ] **Step 7: Run full core suite (ensure `__getattr__` didn't break normal attrs)**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: record field access (key/attr) and __setitem__ mutation"
```

---

### Task 8: Row-axis indexing on records (`slice`/`mask`/`int-array` → record; `int` → dict)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (refactor single-level gather into a helper; add `_getitem_record_rows`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `_ragged_select` (Rust), the existing single-level gather math.
- Produces:
  - `rag[slice|mask|int_array] -> Ragged` (record; one shared gather across fields).
  - `rag[int] -> dict[str, NDArray | bytes]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_row_slice_returns_record():
    rag = _record_ragged()
    sub = rag[1:3]
    assert sub._is_record is True
    np.testing.assert_array_equal(sub["a"][0], np.array([2]))         # row 1 of a
    np.testing.assert_array_equal(sub["b"][1], np.array([3.0, 4.0, 5.0]))  # row 2 of b
    assert sub["a"].offsets is sub["b"].offsets                       # shared gather

def test_record_row_mask_returns_record():
    rag = _record_ragged()
    sub = rag[np.array([True, False, True])]
    np.testing.assert_array_equal(sub["a"][0], np.array([0, 1]))
    np.testing.assert_array_equal(sub["a"][1], np.array([3, 4, 5]))

def test_record_row_int_returns_dict():
    rag = _record_ragged()
    row = rag[0]
    assert set(row.keys()) == {"a", "b"}
    np.testing.assert_array_equal(row["a"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(row["b"], np.array([0.0, 1.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "row_slice or row_mask or row_int" -v`
Expected: FAIL — `_getitem_record_rows` is a stub.

- [ ] **Step 3: Refactor the single-level gather into a reusable helper**

In `_core.py`, extract the index-normalization + Rust select from the existing `__getitem__` into:

```python
    def _row_gather(self, where: Any) -> tuple[NDArray[Any], NDArray[Any]]:
        """Given a slice/mask/int-array `where`, return (sel_starts, sel_stops)
        as contiguous OFFSET_TYPE arrays for the shared ragged axis."""
        starts, stops = self._starts_stops()
        n = len(starts)
        if _where_is_bool(where):
            if where.shape[0] != n:
                raise IndexError(
                    f"boolean index did not match indexed array along axis 0; "
                    f"size of axis is {n} but size of corresponding boolean axis is {where.shape[0]}"
                )
            idx = np.flatnonzero(where).astype(np.int64)
        else:
            idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
            idx = np.where(idx < 0, idx + n, idx)
        try:
            from seqpro.seqpro import _ragged_select  # type: ignore[missing-import]

            sel_starts, sel_stops = _ragged_select(
                np.ascontiguousarray(starts, np.int64),
                np.ascontiguousarray(stops, np.int64),
                idx,
            )
        except ImportError:  # pragma: no cover
            sel_starts, sel_stops = starts[idx], stops[idx]
        return (
            np.ascontiguousarray(sel_starts, dtype=OFFSET_TYPE),
            np.ascontiguousarray(sel_stops, dtype=OFFSET_TYPE),
        )
```

Then refactor the existing non-record `__getitem__` slice/mask/array branch to call `self._row_gather(where)` instead of the inline logic (behavior identical; this keeps DRY for records). Re-run `pixi run -e dev pytest tests/test_ragged_core.py -k getitem -v` to confirm parity before continuing.

- [ ] **Step 4: Implement record row indexing**

Replace the `_getitem_record_rows` stub:

```python
    def _getitem_record_rows(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            out: dict[str, Any] = {}
            for name, fl in rec.fields.items():
                row = fl.data[lo:hi]
                out[name] = row
            return out
        sel_starts, sel_stops = self._row_gather(where)
        new_offsets = np.stack([sel_starts, sel_stops], 0)
        new_shape = (len(sel_starts), *rec.shape[rec.shape.index(None) :])
        new_fields = {
            name: RaggedLayout(
                data=fl.data,
                offsets=[new_offsets],
                shape=(len(sel_starts), *fl.shape[fl.shape.index(None) :]),
            )
            for name, fl in rec.fields.items()
        }
        return Ragged(RecordLayout(offsets=[new_offsets], shape=new_shape, fields=new_fields))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "row_slice or row_mask or row_int" -v`
Expected: PASS.

- [ ] **Step 6: Run full core suite**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: record row-axis indexing (slice/mask -> record, int -> dict)"
```

---

### Task 9: `squeeze` / `reshape` on records (per-field)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`squeeze`, `reshape` record branch)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: per-field `squeeze`/`reshape` (existing single-level), `from_fields`.
- Produces: `rag.squeeze(...) -> Ragged` (record), `rag.reshape(...) -> Ragged` (record).

- [ ] **Step 1: Write the failing tests**

```python
def _record_with_leading_two():
    lens = np.array([2, 1, 3, 1, 2, 1], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(10, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(10, dtype=np.float64), lens)
    return Ragged.from_fields({"a": a, "b": b})

def test_record_reshape_leading():
    rag = _record_with_leading_two()
    re = rag.reshape(2, 3, None)
    assert re._is_record is True
    assert re.shape == (2, 3, None)
    assert re["a"].offsets is re["b"].offsets

def test_record_squeeze_trailing_one():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_offsets(np.arange(6, dtype=np.int64).reshape(6, 1), (3, None, 1),
                            lengths_to_offsets(lens))
    b = Ragged.from_offsets(np.arange(6, dtype=np.float64).reshape(6, 1), (3, None, 1),
                            lengths_to_offsets(lens))
    rag = Ragged.from_fields({"a": a, "b": b})
    sq = rag.squeeze()
    assert sq.shape == (3, None)
    assert sq["a"].offsets is sq["b"].offsets
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "record_reshape or record_squeeze" -v`
Expected: FAIL (record passed through single-level path, which assumes a data buffer).

- [ ] **Step 3: Add record branches**

At the top of `squeeze`:

```python
    def squeeze(self, axis: int | tuple[int, ...] | None = None):
        if isinstance(self._layout, RecordLayout):
            return Ragged.from_fields({f: self[f].squeeze(axis) for f in self._layout.fields})
        # ... existing single-level body ...
```

At the top of `reshape`:

```python
    def reshape(self, *shape: int | None) -> "Ragged[Any]":
        if isinstance(self._layout, RecordLayout):
            return Ragged.from_fields({f: self[f].reshape(*shape) for f in self._layout.fields})
        # ... existing single-level body ...
```

*(Each per-field result shares the same offsets values; `from_fields` re-canonicalizes them to one object, preserving the zero-copy contract.)*

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "record_reshape or record_squeeze" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: per-field squeeze/reshape on record Ragged"
```

---

### Task 10: Record-aware `to_packed`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`to_packed` record branch)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `_ops._pack_parts` (existing Numba pack), `RecordLayout`.
- Produces: `rag.to_packed(copy=True) -> Ragged` (record; one shared packed offsets across fields). `copy=False` passes through iff already packed, else raises.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_to_packed_after_slice():
    rag = _record_ragged()[::-1]  # gather -> (2, M) offsets
    packed = rag.to_packed()
    assert packed._is_record is True
    assert packed.is_base is True
    assert packed["a"].offsets is packed["b"].offsets
    np.testing.assert_array_equal(packed["a"].data, np.array([3, 4, 5, 2, 0, 1], dtype=np.int32))
    np.testing.assert_array_equal(packed.offsets, np.array([0, 3, 4, 6]))

def test_record_to_packed_copy_false_passthrough():
    rag = _record_ragged()
    assert rag.to_packed(copy=False) is rag

def test_record_to_packed_copy_false_unpacked_raises():
    rag = _record_ragged()[::-1]
    with pytest.raises(ValueError, match="already-packed"):
        rag.to_packed(copy=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "to_packed" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the record branch in `to_packed`**

At the top of `to_packed`:

```python
    def to_packed(self, *, copy: bool = True) -> "Ragged[Any]":
        from ._ops import _pack_parts

        if isinstance(self._layout, RecordLayout):
            rec = self._layout
            shared = self.offsets
            if not copy:
                already = (
                    shared.ndim == 1
                    and (shared.size == 0 or shared[0] == 0)
                    and all(
                        fl.data.flags.c_contiguous and int(shared[-1]) == fl.data.shape[0]
                        for fl in rec.fields.values()
                    )
                )
                if already:
                    return self
                raise ValueError(
                    "to_packed(copy=False) requires already-packed input; got an unpacked record."
                )
            packed_offsets: NDArray[Any] | None = None
            new_fields: dict[str, RaggedLayout[Any]] = {}
            for name, fl in rec.fields.items():
                pdata, poff = _pack_parts(fl.data, fl.shape, shared, copy=True)
                if packed_offsets is None:
                    packed_offsets = poff
                new_fields[name] = RaggedLayout(data=pdata, offsets=[packed_offsets], shape=fl.shape)
            assert packed_offsets is not None
            return Ragged(
                RecordLayout(offsets=[packed_offsets], shape=rec.shape, fields=new_fields)
            )
        # ... existing single-level body ...
```

*(All fields pack from the same `shared` offsets, so every `poff` is value-identical; reuse the first as the single shared object.)*

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "to_packed" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: record-aware to_packed (one shared packed offsets across fields)"
```

---

### Task 11: `to_numpy` / `to_padded` (per-field dict) + `view`/ufunc raise on records

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`to_numpy`, `to_padded`, `view`, `__array_ufunc__` record branches)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: per-field `to_numpy`/`to_padded` (existing single-level).
- Produces:
  - `rag.to_numpy() -> dict[str, NDArray]`; `rag.to_padded(pad_value, length=None) -> dict[str, NDArray]`.
  - `rag.view(dtype)` / ufunc on a record → `NotImplementedError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_record_to_numpy_dict():
    lens = np.array([3, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    rag = Ragged.from_fields({"a": a, "b": b})
    out = rag.to_numpy()
    np.testing.assert_array_equal(out["a"], np.arange(6, dtype=np.int32).reshape(2, 3))
    np.testing.assert_array_equal(out["b"], np.arange(6, dtype=np.float64).reshape(2, 3))

def test_record_to_padded_dict():
    rag = _record_ragged()
    out = rag.to_padded(-1)
    assert set(out.keys()) == {"a", "b"}
    np.testing.assert_array_equal(out["a"][1], np.array([2, -1, -1], dtype=np.int32))

def test_record_view_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError):
        rag.view(np.uint32)

def test_record_ufunc_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError):
        rag + 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "to_numpy or to_padded or view_raises or ufunc_raises" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the branches**

`to_numpy`:

```python
    def to_numpy(self, allow_missing: bool = False):
        if isinstance(self._layout, RecordLayout):
            return {f: self[f].to_numpy(allow_missing) for f in self._layout.fields}
        # ... existing single-level body ...
```

`to_padded`:

```python
    def to_padded(self, pad_value: Any, *, length: int | None = None):
        if isinstance(self._layout, RecordLayout):
            return {f: self[f].to_padded(pad_value, length=length) for f in self._layout.fields}
        # ... existing single-level body ...
```

`view`:

```python
    def view(self, dtype: Any) -> "Ragged[Any]":
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError(
                "view is not defined on record Ragged arrays; view a field, "
                "e.g. rag['f'] = rag['f'].view(dtype)."
            )
        # ... existing single-level body ...
```

`__array_ufunc__` — add the guard at the top:

```python
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(x, Ragged) and x._is_record for x in inputs):
            raise NotImplementedError(
                "element-wise ufuncs are not defined on record Ragged arrays; "
                "operate on individual fields."
            )
        # ... existing body ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "to_numpy or to_padded or view_raises or ufunc_raises" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: record to_numpy/to_padded dicts; raise view/ufunc on records"
```

---

### Task 12: `_ingest` bridge — record `layout_from_ak` + `to_ak`

**Files:**
- Modify: `python/seqpro/rag/_ingest.py`
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `_array.unbox`, `_array._extract_list_offsets`, `ak.fields`, `RecordLayout`, `RaggedLayout`.
- Produces:
  - `layout_from_ak(ak_record) -> RecordLayout` (fields rebound onto one shared offsets; opaque S1 field → char field).
  - `to_ak(record_ragged) -> ak.Array` (record layout).

- [ ] **Step 1: Write the failing tests**

```python
import awkward as ak

def test_ingest_record_from_ak():
    arr = ak.Array({"a": [[1, 2], [3], [4, 5, 6]], "b": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]})
    rag = Ragged(arr)
    assert rag._is_record is True
    assert rag.fields == ["a", "b"]
    np.testing.assert_array_equal(rag["a"].data, np.array([1, 2, 3, 4, 5, 6]))
    assert rag["a"].offsets is rag["b"].offsets

def test_record_to_ak_roundtrips():
    rag = _record_ragged()
    out = rag.to_ak()
    assert set(ak.fields(out)) == {"a", "b"}
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(out["a"])), rag["a"].data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "ingest_record or to_ak" -v`
Expected: FAIL — `layout_from_ak` raises "record-layout Ragged lands in Spec B".

- [ ] **Step 3: Implement record ingest**

Replace `layout_from_ak` in `_ingest.py`:

```python
def layout_from_ak(arr: Any) -> "RaggedLayout[Any] | RecordLayout":
    if ak.fields(arr):
        from ._array import _extract_list_offsets, unbox
        from ._layout import RecordLayout

        shared = np.ascontiguousarray(_extract_list_offsets(ak.to_layout(arr)), dtype=OFFSET_TYPE)
        fields: dict[str, RaggedLayout[Any]] = {}
        rec_shape: tuple[int | None, ...] | None = None
        for f in ak.fields(arr):
            parts = unbox(arr[f])  # data, shape (with None), offsets (ignored; use shared)
            # records hold chars, not opaque strings: keep the None axis as-is
            fields[f] = RaggedLayout(data=parts.data, offsets=[shared], shape=parts.shape)
            if rec_shape is None:
                rec_shape = parts.shape
        assert rec_shape is not None
        return RecordLayout(offsets=[shared], shape=rec_shape, fields=fields)

    parts = unbox(arr)
    is_bytes = parts.data.dtype.kind == "S"
    if (
        is_bytes
        and parts.shape.count(None) == 1
        and parts.shape.index(None) == len(parts.shape) - 1
    ):
        # single list-axis S1 ak input -> opaque string by default (matches from_lengths)
        leading = parts.shape[: parts.shape.index(None)]
        return RaggedLayout(
            data=parts.data, offsets=[], shape=leading, str_offsets=parts.offsets
        )
    return RaggedLayout(data=parts.data, offsets=[parts.offsets], shape=parts.shape)
```

Add `from ._array import unbox` at module top (already imported) and the `_extract_list_offsets`/`RecordLayout`/`RaggedLayout` imports as shown.

- [ ] **Step 4: Implement record `to_ak`**

Replace `to_ak` in `_ingest.py`:

```python
def to_ak(rag: Any) -> ak.Array:
    from ._array import _parts_to_content, RagParts
    from ._layout import RecordLayout

    if isinstance(rag._layout, RecordLayout):
        return ak.zip({f: to_ak(rag[f]) for f in rag.fields}, depth_limit=1)

    content = _parts_to_content(
        RagParts(
            rag.data,
            rag.shape if None in rag.shape else (*rag.shape, None),
            rag.offsets,
        )
    )
    return ak.Array(content)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "ingest_record or to_ak" -v`
Expected: PASS.

- [ ] **Step 6: Run full core + records suite**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_ingest.py tests/test_ragged_core_records.py
git commit -m "feat: ingest/emit record layouts via awkward bridge (oracle interop)"
```

---

### Task 13: Differential Hypothesis suite + port legacy record tests

**Files:**
- Modify: `tests/test_ragged_core_records.py` (add differential + ported cases)
- Test: same file

**Interfaces:**
- Consumes: `Ragged` (core), `AkRagged` (`from seqpro.rag._array import Ragged as AkRagged`), `ak.zip`.
- Produces: parity coverage of record properties/ops vs the awkward oracle; numeric+char mixed record coverage.

- [ ] **Step 1: Write the differential + char-record tests**

Add to `tests/test_ragged_core_records.py`:

```python
import awkward as ak
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from seqpro.rag._array import Ragged as AkRagged


@st.composite
def _record_inputs(draw):
    n = draw(st.integers(1, 6))
    lengths = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n).map(np.array))
    total = int(lengths.sum())
    a = draw(arrays(np.int64, (total,), elements=st.integers(-50, 50)))
    b = draw(arrays(np.float64, (total,), elements=st.floats(-50, 50, allow_nan=False)))
    return lengths.astype(np.uint32), a, b


@given(_record_inputs())
def test_diff_record_properties(inp):
    lengths, a, b = inp
    new = Ragged.from_fields(
        {"a": Ragged.from_lengths(a, lengths), "b": Ragged.from_lengths(b, lengths)}
    )
    old = AkRagged(ak.zip({"a": ak.unflatten(a, lengths), "b": ak.unflatten(b, lengths)}))
    np.testing.assert_array_equal(new.offsets, old.offsets)
    assert new.shape == old.shape
    assert new.fields == list(ak.fields(old))
    np.testing.assert_array_equal(new["a"].data, old["a"].data)
    np.testing.assert_array_equal(new["b"].data, old["b"].data)


@given(_record_inputs())
def test_diff_record_to_packed_after_slice(inp):
    lengths, a, b = inp
    if len(lengths) < 2:
        return
    new = Ragged.from_fields(
        {"a": Ragged.from_lengths(a, lengths), "b": Ragged.from_lengths(b, lengths)}
    )[::-1].to_packed()
    old = AkRagged(
        ak.zip({"a": ak.unflatten(a, lengths), "b": ak.unflatten(b, lengths)})
    )[::-1].to_packed()
    np.testing.assert_array_equal(new["a"].data, old["a"].data)
    np.testing.assert_array_equal(new.offsets, old.offsets)


def test_char_field_record_aligns_on_length():
    # annotated-haplotypes shape: chars + per-base numeric, one shared offsets
    lens = np.array([3, 2], dtype=np.uint32)
    hap = Ragged.from_lengths(np.frombuffer(b"ATGCG", "S1"), lens).to_chars()
    annot = Ragged.from_lengths(np.arange(5, dtype=np.float32), lens)
    rec = Ragged.from_fields({"hap": hap, "annot": annot})
    assert rec.dtype == np.dtype([("hap", "S1"), ("annot", np.float32)])
    assert rec["hap"].offsets is rec["annot"].offsets
    np.testing.assert_array_equal(rec["hap"][0], np.frombuffer(b"ATG", "S1"))
```

- [ ] **Step 2: Run the differential tests**

Run: `pixi run -e dev pytest tests/test_ragged_core_records.py -k "diff_record or char_field" -v`
Expected: PASS (fix any oracle-construction mismatches by adjusting the awkward side, not the core).

- [ ] **Step 3: Port the legacy record assertions onto the core path**

Mirror the value-level assertions from `tests/test_ragged.py::TestRecordRagged` (field access, shared-offsets identity, dtype dict→structured) and `tests/test_rag_to_packed.py::TestToPackedRecord` (pack all fields, shared offsets) as core-path tests in `tests/test_ragged_core_records.py`. Use `Ragged.from_fields` for construction and `np.testing.assert_array_equal` for values; assert `rag["a"].offsets is rag["b"].offsets`.

- [ ] **Step 4: Run the full record + core suite**

Run: `pixi run -e dev pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole test suite (no regressions to awkward path)**

Run: `pixi run -e dev pytest -q`
Expected: PASS (or only pre-existing unrelated failures; investigate any new ones).

- [ ] **Step 6: Commit**

```bash
git add tests/test_ragged_core_records.py
git commit -m "test: differential record suite vs awkward oracle + char-record alignment"
```

---

### Task 14: Update the SSoT roadmap status

**Files:**
- Modify: `docs/roadmap/rust-ragged.md` (Spec B status + decision log)

**Interfaces:** none (docs).

- [ ] **Step 1: Mark Spec B landed in the roadmap**

In `docs/roadmap/rust-ragged.md`, change the Spec B entry heading from
`*(Design approved 2026-06-20 …)*` to note it has **landed** (date), and add a
decision-log line: "Spec B landed: string/char duality + native records in
`_core.py` (`RecordLayout`, `from_fields`/`rag.zip`, field access/mutation, row
indexing, record `to_packed`/`to_numpy`/`to_padded`, awkward bridge), oracle-tested.
Public `Ragged` still awkward; swap + tokenize/translate adaptation remain Spec D."

- [ ] **Step 2: Commit**

```bash
git add docs/roadmap/rust-ragged.md
git commit -m "docs: mark Spec B (string/char duality + records) landed in roadmap SSoT"
```

---

## Self-Review

**Spec coverage:**
- §1 string/char model → Tasks 1–3 (dtype 'S', disambiguation, to_chars/to_strings). ✓
- §2 RecordLayout + invariant → Task 4. ✓
- §3 constructors (from_fields/zip) → Task 5. ✓
- §4 properties → Task 6. ✓
- §5 field access + mutation → Task 7. ✓
- §6 row indexing → Task 8. ✓
- §7 squeeze/reshape/to_packed/to_numpy/to_padded/view/ufunc → Tasks 9–11. ✓
- §8 no new Rust → honored (no Rust tasks). ✓
- §9 ingest bridge → Task 12. ✓
- §10 testing (differential + ports) → Tasks 4–13 inline + Task 13. ✓
- §11 migration (zip export, internal-only, skill not updated) → Task 5 + Global Constraints. ✓
- SSoT update obligation → Task 14. ✓

**Placeholder scan:** No TBD/TODO; every code step shows code; every test step shows assertions and the exact `pixi run` command + expected result.

**Type consistency:** `from_fields(dict[str, Ragged]) -> Ragged`, `RecordLayout(offsets, shape, fields)`, `_is_record`, `_row_gather -> (starts, stops)`, `_getitem_record`/`_getitem_record_rows`, `to_chars`/`to_strings` used consistently across tasks. `.data`/`.dtype` widen to dict/structured only on the record branch. `offsets` returns the shared array for records and `str_offsets` for opaque strings (unchanged Spec A behavior for the latter).

**Note for the implementer:** several Task 5 test asserts reference `.fields` / field access from Tasks 6–7. When executing strictly task-by-task, those asserts are also covered by their own tasks; keep them and let Task 5 fail only on the `from_fields`-specific asserts, or temporarily comment the cross-task lines and restore them when Tasks 6–7 land.
