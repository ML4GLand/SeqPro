# Ragged `__getitem__` numpy-consistency + subclass-preserving transforms — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two independent seqpro PRs — (PR1) make `Ragged.__getitem__` numpy-consistent for multi-leading-axis records, then (PR2) make structural transforms preserve a `Ragged` subclass — so downstream consumers stop re-implementing structural ops.

**Architecture:** PR1 routes a non-tuple key on a multi-leading-axis record through the already-correct tuple peel path (`A[x] ≡ A[(x,)]`). PR2 adds a `_with_layout(layout)` constructor that rebuilds via `object.__new__(type(self))`, and turns each structural public method (`__getitem__`, `reshape`, `squeeze`, `to_packed`) into a thin wrapper that rewraps its result, leaving the existing method bodies untouched as private `_impl`s. Field extraction (string key) stays base `Ragged`.

**Tech Stack:** Python, `seqpro.rag._core`, pytest, pixi (`pixi run -e dev`).

## Global Constraints

- seqpro dev env: run everything via `pixi run -e dev <cmd>` (platform linux-64/osx-arm64).
- These are **Python-only** edits to `python/seqpro/rag/_core.py` — no Rust changes, so no `PYO3_PYTHON` needed for commits.
- Tests are flat: `tests/test_*.py` (no `tests/rag/` subdir).
- Pre-commit runs ruff + pyrefly; keep both clean (`pixi run -e dev lint`, `pixi run -e dev typecheck`).
- The two PRs are independent and both branch off `main`. PR1 first, then PR2.
- numpy indexing contract: a non-tuple key is treated as a 1-tuple — `A[x]` must equal `A[(x,)]`.
- Subclassing contract (PR2): a `Ragged` subclass must declare `__slots__ = ()` and hold **no instance state beyond `_layout`**. Structural transforms reconstruct via `object.__new__(type(self))`, bypassing `__init__`.

---

# PR1 — `__getitem__` numpy-consistency for record layouts

**Branch:** `fix/ragged-getitem-numpy-record` off `main`.

### Task 1: Non-tuple record indexing equals the 1-tuple form

**Files:**
- Modify: `python/seqpro/rag/_core.py` — `_getitem_record` (currently lines 807–816)
- Test: `tests/test_ragged_record_indexing.py` (create)

**Interfaces:**
- Consumes: `Ragged.from_lengths`, `Ragged.reshape`, `Ragged.from_fields`, `Ragged.__getitem__`, `Ragged.rag_dim`, `Ragged.shape`.
- Produces: no new public API; fixes behavior of `Ragged.__getitem__` on record layouts with `rag_dim > 1`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ragged_record_indexing.py`:

```python
import numpy as np
import pytest

from seqpro.rag import Ragged


def _multidim_record():
    """A (3, 2, None) numeric record: 6 ragged segments over a (batch=3, ploidy=2)
    leading grid."""
    lengths = np.array([[1, 2], [3, 1], [2, 2]], np.uint32)  # (3, 2)
    data = np.arange(int(lengths.sum()), dtype=np.int32)
    start = Ragged.from_lengths(data, lengths.reshape(-1)).reshape(3, 2, None)
    return Ragged.from_fields({"start": start})


@pytest.mark.parametrize(
    "key",
    [
        0,
        slice(1, 3),
        slice(None),
        np.array([True, False, True]),
        np.array([0, 2]),
    ],
    ids=["int", "slice", "full_slice", "mask", "int_array"],
)
def test_nontuple_equals_one_tuple_multidim_record(key):
    """numpy contract: rec[k] must equal rec[(k,)] for a multi-leading-axis record."""
    rec = _multidim_record()
    got = rec[key]
    want = rec[(key,)]
    assert type(got) is type(want), (type(got), type(want))
    if isinstance(want, Ragged):
        assert got.shape == want.shape, (got.shape, want.shape)
        np.testing.assert_array_equal(
            np.asarray(got["start"].data), np.asarray(want["start"].data)
        )
        np.testing.assert_array_equal(
            np.asarray(got["start"].offsets[0]), np.asarray(want["start"].offsets[0])
        )


def test_slice_preserves_ploidy_axis():
    """Regression for the specific bug: rec[1:3] kept the ploidy axis."""
    rec = _multidim_record()
    assert rec[1:3].shape == (2, 2, None)
    assert rec[0].shape == (2, None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_record_indexing.py -v`
Expected: FAIL — `rec[1:3]` returns shape `(2, None)` (not `(2, 2, None)`) and `rec[0]` returns a `dict` (not a `(2, None)` Ragged), so the `type`/`shape` assertions fail.

- [ ] **Step 3: Implement the fix**

In `python/seqpro/rag/_core.py`, change `_getitem_record` (lines 807–816) from:

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

to:

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
        # numpy contract: a non-tuple key is treated as a 1-tuple (A[x] == A[(x,)]).
        # For a single-ragged-axis record with >1 leading fixed axis, route through
        # the multidim peel path directly (under the SAME guard it requires, so we
        # never re-dispatch through __getitem__ and risk recursion). Records that are
        # not single-None fall through to the flat record-rows path unchanged.
        if (
            not isinstance(where, tuple)
            and self.rag_dim > 1
            and rec.shape.count(None) == 1
        ):
            return self._getitem_tuple_multidim((where,))
        return self._getitem_record_rows(where)
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `pixi run -e dev pytest tests/test_ragged_record_indexing.py -v`
Expected: PASS (all parametrizations + the ploidy regression).

- [ ] **Step 5: Run the full ragged test suite to verify no regression**

Run: `pixi run -e dev pytest tests/test_ragged.py tests/test_ragged_core.py tests/test_ragged_core_records.py tests/test_ragged_to_padded.py -q`
Expected: PASS (no behavior change for `rag_dim == 1` records or non-record arrays).

- [ ] **Step 6: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: clean (ruff "All checks passed", pyrefly 0 errors).

- [ ] **Step 7: Commit, branch, push, open PR**

```bash
git checkout -b fix/ragged-getitem-numpy-record
git add python/seqpro/rag/_core.py tests/test_ragged_record_indexing.py
git commit -m "fix(rag): non-tuple record indexing matches numpy (A[x] == A[(x,)])

Multi-leading-axis records (e.g. (batch, ploidy, ~variants)) flattened the
leading fixed axes on rec[1:3]/rec[0] because the non-tuple path went through
_getitem_record_rows while the 1-tuple path peeled the first axis correctly.
Route non-tuple keys through _getitem_tuple_multidim under the same guard."
git push -u origin fix/ragged-getitem-numpy-record
gh pr create --base main --title "fix(rag): non-tuple record indexing matches numpy" \
  --body "Restores A[x] == A[(x,)] for multi-leading-axis records. See docs/superpowers/specs/2026-06-23-ragged-subclass-getitem-design.md (PR1)."
```

---

# PR2 — subclass-preserving structural transforms (Design B)

**Branch:** `feat/ragged-subclass-preserving-transforms` off `main` (independent of PR1).

### Task 2: `_with_layout` subclass-preserving constructor

**Files:**
- Modify: `python/seqpro/rag/_core.py` — add a method to `class Ragged` (near `__init__`, ~line 67)
- Test: `tests/test_ragged_subclass.py` (create)

**Interfaces:**
- Produces: `Ragged._with_layout(self, layout) -> Ragged` — returns an instance of `type(self)` sharing `layout`, bypassing `__init__`. Later tasks call this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_ragged_subclass.py`:

```python
import numpy as np

from seqpro.rag import Ragged


class _Sub(Ragged):
    __slots__ = ()


def _ragged():
    return Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32))


def test_with_layout_preserves_subclass():
    sub = _Sub(_ragged()._layout)
    out = sub._with_layout(_ragged()._layout)
    assert type(out) is _Sub


def test_with_layout_base_returns_base():
    base = _ragged()
    out = base._with_layout(_ragged()._layout)
    assert type(out) is Ragged
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -v`
Expected: FAIL with `AttributeError: 'Ragged' object has no attribute '_with_layout'`.

- [ ] **Step 3: Implement `_with_layout`**

In `python/seqpro/rag/_core.py`, inside `class Ragged`, immediately after `__init__` (after line 77 region), add:

```python
    def _with_layout(self, layout: Any) -> "Ragged[Any]":
        """Reconstruct a same-kind container around ``layout``, preserving the
        concrete subclass. Bypasses ``__init__`` (subclasses carry no state beyond
        ``_layout``; see the subclassing contract). Used by structural transforms
        so a ``Ragged`` subclass survives slicing/reshape/squeeze/to_packed."""
        obj = object.__new__(type(self))
        obj._layout = layout
        return obj
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/ragged-subclass-preserving-transforms
git add python/seqpro/rag/_core.py tests/test_ragged_subclass.py
git commit -m "feat(rag): add _with_layout subclass-preserving constructor"
```

### Task 3: `__getitem__` preserves subclass on positional indexing

**Files:**
- Modify: `python/seqpro/rag/_core.py` — `__getitem__` (line 425); rename its body to `_getitem`, add a thin `__getitem__` wrapper.
- Test: `tests/test_ragged_subclass.py` (extend)

**Interfaces:**
- Consumes: `Ragged._with_layout` (Task 2).
- Produces: `Ragged._getitem(self, where)` (renamed existing body) + new `__getitem__` wrapper.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_subclass.py`:

```python
def _record():
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32))
    b = Ragged.from_lengths(np.arange(6, dtype=np.int32) * 10, np.array([2, 1, 3], np.uint32))
    return Ragged.from_fields({"a": a, "b": b})


def test_getitem_positional_preserves_subclass():
    sub = _Sub(_record()._layout)
    assert type(sub[0:2]) is _Sub          # positional row slice -> subclass
    assert type(sub[np.array([0, 2])]) is _Sub


def test_getitem_field_extraction_stays_base():
    sub = _Sub(_record()._layout)
    assert type(sub["a"]) is Ragged        # string key -> bare field, base Ragged
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -k getitem -v`
Expected: FAIL — `type(sub[0:2])` is `Ragged`, not `_Sub`.

- [ ] **Step 3: Rename the body and add the wrapper**

In `python/seqpro/rag/_core.py`, rename the existing method `def __getitem__(self, where) -> ...:` (line 425) to `def _getitem(self, where: Any) -> Any:` (change ONLY the `def` line; leave the entire body unchanged, including its internal `self[k]` / `result[k]` recursion — re-dispatching through the wrapper is correct and only redundantly rewraps intermediates).

Immediately above `_getitem`, add the public wrapper:

```python
    def __getitem__(
        self, where: Any
    ) -> "NDArray[Any] | bytes | dict[str, Any] | Ragged[Any]":
        result = self._getitem(where)
        # Preserve the concrete subclass for positional (structural) results.
        # A string key is field extraction -> keep the bare field as base Ragged.
        # Non-Ragged results (dict / bytes / ndarray / scalar) pass through.
        if (
            type(self) is not Ragged
            and not isinstance(where, str)
            and isinstance(result, Ragged)
        ):
            return self._with_layout(result._layout)
        return result
```

- [ ] **Step 4: Run subclass + full ragged tests**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -v && pixi run -e dev pytest tests/test_ragged.py tests/test_ragged_core.py tests/test_ragged_core_records.py -q`
Expected: PASS (subclass preserved on positional, base on field extraction; no regressions for base `Ragged`).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_subclass.py
git commit -m "feat(rag): __getitem__ preserves subclass on positional indexing"
```

### Task 4: `reshape`, `squeeze`, `to_packed` preserve subclass

**Files:**
- Modify: `python/seqpro/rag/_core.py` — `squeeze` (1162), `reshape` (1212), `to_packed` (1272); rename each body to `_impl`, add thin wrappers.
- Test: `tests/test_ragged_subclass.py` (extend)

**Interfaces:**
- Consumes: `Ragged._with_layout` (Task 2).
- Produces: `_squeeze_impl`, `_reshape_impl`, `_to_packed_impl` (renamed bodies) + wrappers `squeeze`, `reshape`, `to_packed` with identical signatures to today.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ragged_subclass.py`:

```python
def test_reshape_squeeze_to_packed_preserve_subclass():
    sub = _Sub(_record()._layout)            # shape (3, None) record
    assert type(sub.reshape(1, 3, None)) is _Sub
    assert type(sub.reshape(1, 3, None).squeeze(0)) is _Sub
    assert type(sub.to_packed()) is _Sub


def test_reshape_squeeze_to_packed_base_unchanged():
    base = _record()
    assert type(base.reshape(1, 3, None)) is Ragged
    assert type(base.to_packed()) is Ragged
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -k "reshape_squeeze" -v`
Expected: FAIL — `type(sub.reshape(...))` is `Ragged`, not `_Sub`.

- [ ] **Step 3: Rename bodies and add wrappers**

For each of `squeeze`, `reshape`, `to_packed` in `python/seqpro/rag/_core.py`: rename the existing `def NAME(self, ...) -> ...:` to `def _NAME_impl(self, ...) -> ...:` (change only the `def` line; leave bodies unchanged — their internal `Ragged(fl).reshape(...)`/`Ragged(fl).squeeze(...)` calls operate on fresh base `Ragged`s and stay base, which is correct), then add the public wrapper directly above it.

`squeeze` wrapper (matches signature `axis: int | tuple[int, ...] | None = None`):

```python
    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> "Ragged[Any] | NDArray[Any]":
        result = self._squeeze_impl(axis)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result
```

`reshape` wrapper (matches signature `*shape: int | None`):

```python
    def reshape(self, *shape: int | None) -> "Ragged[Any]":
        result = self._reshape_impl(*shape)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result
```

`to_packed` wrapper (matches signature `*, copy: bool = True`):

```python
    def to_packed(self, *, copy: bool = True) -> "Ragged[Any]":
        result = self._to_packed_impl(copy=copy)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result
```

- [ ] **Step 4: Run subclass + full ragged tests**

Run: `pixi run -e dev pytest tests/test_ragged_subclass.py -v && pixi run -e dev pytest tests/test_ragged.py tests/test_ragged_core.py tests/test_ragged_core_records.py tests/test_ragged_to_padded.py tests/test_rag_to_packed.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + typecheck**

Run: `pixi run -e dev lint && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 6: Commit, push, open PR**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_subclass.py
git commit -m "feat(rag): reshape/squeeze/to_packed preserve Ragged subclass

Boundary-wrap the structural transforms through _with_layout so a Ragged
subclass survives them; bodies unchanged. Field extraction stays base Ragged."
git push -u origin feat/ragged-subclass-preserving-transforms
gh pr create --base main --title "feat(rag): subclass-preserving structural transforms" \
  --body "Design B seam: _with_layout + boundary wrappers on __getitem__/reshape/squeeze/to_packed. See docs/superpowers/specs/2026-06-23-ragged-subclass-getitem-design.md (PR2)."
```

---

## Deferred: GenVarLoader consumer cleanup (separate plan, after both PRs release)

Not implementable until both seqpro PRs merge and a seqpro version including them is released (GVL must pin it). When that lands, write a separate GVL plan to:
- Make `RaggedVariants` subclass `Ragged` (`__slots__ = ()`), dropping the `_rag` composition field.
- Keep domain methods/properties (`alt`/`ref`/`start`/`dosage`, derived `ilen`/`end`, `rc_`, `_alt_chars`, `__init__` invariants) and the allele-aware `concatenate` override.
- Delete the structural overrides: `reshape`, `to_packed`, `squeeze`, and the entire `__getitem__` override (base now numpy-correct + subclass-preserving; string-key field access returns a bare field `Ragged` from base `_getitem_record`).
- Bump the seqpro pin to the version including both PRs; run the full GVL suite **including torch + slow tiers** (`KMP_DUPLICATE_LIB_OK=TRUE pixi run -e default pytest tests -m "slow or not slow"`).

## Self-Review

- **Spec coverage:** PR1 §→ Task 1; PR2 `_with_layout` → Task 2; `__getitem__` seam → Task 3; reshape/squeeze/to_packed seam → Task 4; concatenate explicitly out-of-seam (module function, GVL override) — noted; GVL cleanup → Deferred section. Recursion hazard from spec → handled in Task 1 Step 3 (direct `_getitem_tuple_multidim` call under guard).
- **Placeholders:** none — all steps carry concrete code/commands/expected output.
- **Type consistency:** `_with_layout(layout)` defined in Task 2 and consumed verbatim in Tasks 3–4; renamed bodies `_getitem`/`_squeeze_impl`/`_reshape_impl`/`_to_packed_impl` referenced consistently by their wrappers.
