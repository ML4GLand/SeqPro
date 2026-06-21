# Rust-native `Ragged` — Spec C (nested R=2 + string-under-axis) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the landed single-level Rust-native `Ragged` core to two nested ragged levels (R=2) plus the opaque-string-under-a-ragged-axis leaf, so the three doubly-ragged consumer cases (alleles, flat variant windows, codon annotations) work natively, differential-tested against the awkward oracle.

**Architecture:** Python holds the NumPy buffers and orchestrates layout algebra; Rust does the two new hot-loop kernels (nested-pack, nested-gather). Nested offsets are stored as a length-R list `[O0, O1]` (outermost-first); slicing stays zero-copy via Spec A's `(2, ·)` lazy gather forms (inner offsets stay global) and packs only on irregular middle gathers. Records compose per-field `RaggedLayout`s over one shared nested offsets list, each field keeping its own `str_offsets`.

**Tech Stack:** Python 3, NumPy (`>=1.26`), Numba (existing pad/pack kernels), Rust + PyO3/maturin (`src/ragged.rs`, bound in `src/lib.rs`), awkward (oracle only, removed in Spec D), Hypothesis (differential tests), pixi (env/build).

## Global Constraints

- **numpy floor `>=1.26`** — drives the `'S'`/`'S1'` dtype-as-descriptor choice; do not use `StringDType`/`np.str_`.
- **Internal-only build.** Public `seqpro.rag.Ragged` (the awkward subclass in `_array.py`) stays untouched; all work lands on the `rag/_core.py` path. No user-facing behavior ships (public swap is Spec D).
- **Awkward stays installed as the differential oracle.** Tests build a native `Ragged` (`from seqpro.rag._core import Ragged`) and an oracle (`from seqpro.rag._array import Ragged as AkRagged`, or raw `ak.Array`) and assert parity.
- **R = 2 cap.** `shape.count(None) >= 3` → `NotImplementedError`. Record-of-record (a field that is itself a record) → `NotImplementedError`.
- **No naive copies in hot paths.** Per-segment Python loops / per-segment `np.concatenate` are disallowed; use the Numba/Rust kernels. Zero-copy where the lazy form allows.
- **Rust hygiene:** strict lint/format gates must pass (`cargo fmt`, `cargo clippy`, `cargo check`); prefer the `wide` crate over `std::arch` intrinsics if SIMD is introduced (none required here). Fix any existing violations you touch.
- **Conventional commits** (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`). Pre-commit hooks run ruff, pyrefly, cargo fmt/clippy, commitizen.
- **Build after `src/` changes:** `maturin develop` before running Python tests that hit a new kernel.
- **SSoT:** the roadmap (`docs/roadmap/rust-ragged.md`) and Spec C design (`docs/superpowers/specs/2026-06-20-rust-ragged-nested-design.md`) are binding; update the roadmap status/decision log in the final task when Spec C lands.

## Canonical data model (read before any task)

`RaggedLayout(data, offsets: list[NDArray], shape, str_offsets)`:

- **Canonical R=2:** `offsets == [O0, O1]`, both 1-D. `O0` (len `L0+1`, `L0 = ∏ leading_int`) maps outer row → **middle-segment** index range. `O1` (len `M_total+1`, `M_total = O0[-1]`) maps middle segment → **data** range. `data` has `O1[-1]` rows. `shape == (*leading_int, None, None, *trailing_int)`.
- **Lazy outer-sliced R=2:** `offsets == [O0g, O1]` where `O0g` is `(2, L0')` giving each selected outer a *contiguous* `[mid_start, mid_stop)` range into the **global** 1-D `O1`. No data/`O1` movement.
- **Lazy inner-gathered R=2:** `offsets == [O0', O1g]` where `O0'` is 1-D per-group selected counts and `O1g` is `(2, n_sel)` data ranges (after `rag[:, mask]`).
- **String-under-axis:** `offsets == [O0]` (one real level) **and** `str_offsets` set; `shape == (*leading, None)`; `dtype 'S'`; `is_string True`. (Spec B standalone string `offsets == []` remains the zero-real-level case.)
- **Chars from a string-under-axis:** `offsets == [O0, str_off]`, `str_offsets None`, `shape == (*leading, None, None)`, `dtype 'S1'`.

Internal helper used throughout (define in Task 1): `_level_bounds(entry)` → `(starts, stops)`: 1-D entry → `(entry[:-1], entry[1:])`; `(2, n)` entry → `(entry[0], entry[1])`.

---

## Task 1: Nested layout validation (R=2 arm) + `_level_bounds` helper

**Files:**
- Modify: `python/seqpro/rag/_layout.py` (replace the `n_ragged > 1` early raise; add the R=2 arm and `_level_bounds`)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout`, `validate_layout` (Spec A), `_is_monotonic` (Spec A), Rust `_ragged_validate`.
- Produces: `_level_bounds(entry: NDArray) -> tuple[NDArray, NDArray]`; `validate_layout` now accepts `n_ragged == 2` and raises `NotImplementedError` only for `n_ragged >= 3`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_ragged_core.py
def test_layout_nested_r2_valid():
    data = np.arange(10, dtype=np.int32)
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)        # 2 outer rows -> 2,1 middles
    o1 = np.array([0, 3, 5, 10], dtype=OFFSET_TYPE)    # 3 middles -> data
    layout = RaggedLayout(data=data, offsets=[o0, o1], shape=(2, None, None))
    validate_layout(layout)
    assert layout.n_ragged == 2

def test_layout_nested_rejects_r3():
    with pytest.raises(NotImplementedError, match="3 or more|R >= 3|three"):
        validate_layout(
            RaggedLayout(
                data=np.arange(6),
                offsets=[np.array([0, 1]), np.array([0, 2]), np.array([0, 6])],
                shape=(1, None, None, None),
            )
        )

def test_layout_nested_rejects_inner_segment_mismatch():
    with pytest.raises(ValueError, match="segment|middle"):
        validate_layout(
            RaggedLayout(
                data=np.arange(10),
                offsets=[np.array([0, 2, 3], dtype=OFFSET_TYPE),
                         np.array([0, 3, 5], dtype=OFFSET_TYPE)],  # only 2 middles, O0 needs 3
                shape=(2, None, None),
            )
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "nested_r2_valid or nested_rejects_r3 or nested_rejects_inner" -v`
Expected: FAIL (current `validate_layout` raises `NotImplementedError("Spec C")` for `n_ragged > 1`).

- [ ] **Step 3: Implement the R=2 arm**

In `_layout.py`, add the helper near `_is_monotonic`:

```python
def _level_bounds(entry: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return (starts, stops) for one offsets entry (1-D canonical or (2, n) gather)."""
    if entry.ndim == 2:
        return entry[0], entry[1]
    return entry[:-1], entry[1:]
```

Replace the `n_ragged > 1` guard in `validate_layout` with:

```python
    if layout.n_ragged > 2:
        raise NotImplementedError("nested raggedness with 3 or more levels (R >= 3) is unsupported")

    for off in layout.offsets:
        if not _is_monotonic(off):
            raise ValueError("offsets must be monotonic non-decreasing")

    if layout.n_ragged == 2:
        if len(layout.offsets) != 2:
            raise ValueError(
                f"expected 2 offsets arrays for 2 ragged axes, got {len(layout.offsets)}"
            )
        o0, o1 = layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        rag_dim = layout.shape.index(None)
        leading = [d for d in layout.shape[:rag_dim] if d is not None]
        expected_l0 = int(np.prod(np.array(leading, dtype=np.int64))) if leading else 1
        if len(o0_starts) != expected_l0:
            raise ValueError(
                f"outer segment count {len(o0_starts)} != product of leading dims {expected_l0}"
            )
        n_middle = len(o1) - 1 if o1.ndim == 1 else o1.shape[1]
        max_mid = int(o0_stops.max()) if len(o0_stops) else 0
        if o0.ndim == 1 and int(o0[-1]) != n_middle:
            raise ValueError(
                f"O0 references {int(o0[-1])} middle segments but O1 has {n_middle}"
            )
        if max_mid > n_middle:
            raise ValueError(f"O0 middle index {max_mid} exceeds O1 segment count {n_middle}")
        return
```

Keep the existing `n_ragged == 1` block (Spec A) unchanged below this, and delete the now-unused `_SPEC_C_MSG` constant.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "nested or layout" -v`
Expected: PASS (new nested tests pass; Spec A layout tests still green). Note `test_layout_rejects_multiple_none` matches `"Spec C"` today — update its `match=` to `"R >= 3|3 or more"` and shape to `(2, None, None, None)` in the same edit.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_layout.py tests/test_ragged_core.py
git commit -m "feat: validate R=2 nested ragged layouts; cap at R<=2"
```

---

## Task 2: Nested constructors — `from_offsets` (list) and `from_lengths` (nested)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`from_offsets`, `from_lengths`, `_build_layout`)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `_build_layout`, `validate_layout`, `lengths_to_offsets`, `OFFSET_TYPE`.
- Produces:
  - `Ragged.from_offsets(data, shape, offsets)` where `offsets` is `NDArray` (R≤1, back-compat) **or** `list[NDArray]` (R=2: `[O0, O1]`). `count(None) >= 3` → `NotImplementedError`.
  - `Ragged.from_lengths(data, lengths)` extended: `lengths` may be a tuple `(outer_counts, inner_lengths)` for R=2 → builds `[O0, O1]`, `shape (*outer_counts.shape, None, None, *trailing)`. Single-array form unchanged.

- [ ] **Step 1: Write failing tests**

```python
def test_from_offsets_nested_list():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (2, None, None),
        [np.array([0, 2, 3]), np.array([0, 3, 5, 10])],
    )
    assert rag.shape == (2, None, None)
    assert len(rag.offsets) == 2
    np.testing.assert_array_equal(rag[0][0], np.arange(3))   # peel (Task 5/6 verifies fully)

def test_from_offsets_rejects_three_none():
    with pytest.raises(NotImplementedError):
        Ragged.from_offsets(np.arange(6), (1, None, None, None),
                            [np.array([0, 1]), np.array([0, 2]), np.array([0, 6])])

def test_from_lengths_nested_tuple():
    data = np.arange(10, dtype=np.int32)
    outer = np.array([2, 1])            # row0 has 2 middles, row1 has 1
    inner = np.array([3, 2, 5])         # the 3 middles' leaf lengths
    rag = Ragged.from_lengths(data, (outer, inner))
    assert rag.shape == (2, None, None)
    oracle = ak.Array(ak.contents.ListOffsetArray(
        ak.index.Index64(np.array([0, 2, 3])),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.array([0, 3, 5, 10])),
            ak.contents.NumpyArray(data)),
    ))
    assert rag.to_ak().to_list() == oracle.to_list()
```

Note: `to_ak()` parity needs Task 14 (bridge). If executing strictly in order, assert structural equality (`rag.offsets`, `rag.data`) here and add the `to_ak` assertion when Task 14 lands; the executor may reorder the bridge earlier if convenient.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "from_offsets_nested or rejects_three or from_lengths_nested" -v`
Expected: FAIL (`from_offsets` rejects 2 `None`s today; `from_lengths` doesn't accept a tuple).

- [ ] **Step 3: Implement**

In `_core.py`, update `_build_layout` to accept a list and dispatch on real-level count:

```python
def _build_layout(data, shape, offsets):
    n_none = shape.count(None)
    off_list = offsets if isinstance(offsets, list) else [offsets]
    if n_none == 0:
        if data.dtype.kind == "S":
            (off,) = off_list
            return RaggedLayout(data=data, offsets=[], shape=shape, str_offsets=off)
        raise ValueError("shape must have a None ragged dimension for numeric data")
    if len(off_list) != n_none:
        raise ValueError(f"expected {n_none} offsets arrays, got {len(off_list)}")
    return RaggedLayout(data=data, offsets=off_list, shape=shape)
```

Update `from_offsets`:

```python
    @staticmethod
    def from_offsets(data, shape, offsets):
        if shape.count(None) >= 3:
            raise NotImplementedError("nested raggedness with R >= 3 is unsupported")
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have a None ragged dimension")
        off_list = offsets if isinstance(offsets, list) else [offsets]
        off_list = [np.ascontiguousarray(o, dtype=OFFSET_TYPE) for o in off_list]
        return Ragged(_build_layout(data, shape, off_list))
```

Update `from_lengths` to accept the nested tuple:

```python
    @staticmethod
    def from_lengths(data, lengths):
        if isinstance(lengths, tuple):
            outer_counts, inner_lengths = lengths
            o0 = lengths_to_offsets(np.asarray(outer_counts).reshape(-1))
            o1 = lengths_to_offsets(np.asarray(inner_lengths).reshape(-1))
            trailing = data.shape[1:]
            shape = (*np.asarray(outer_counts).shape, None, None, *trailing)
            return Ragged.from_offsets(data, shape, [o0, o1])
        offsets = lengths_to_offsets(lengths)
        if data.dtype.kind == "S" and data.ndim == 1:
            return Ragged.from_offsets(data, tuple(lengths.shape), offsets)
        trailing = data.shape[1:]
        return Ragged.from_offsets(data, (*lengths.shape, None, *trailing), offsets)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "from_offsets or from_lengths" -v`
Expected: PASS (new + existing constructor tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: nested constructors from_offsets(list)/from_lengths(tuple) for R=2"
```

---

## Task 3: String-under-axis leaf + nested `to_chars`/`to_strings`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`to_chars`, `to_strings`, `from_offsets` string-under-axis path, `is_string`/`dtype` already work)
- Modify: `python/seqpro/rag/_layout.py` (`validate_layout`: allow `is_string` with a real `offsets` level)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `RaggedLayout` (`offsets`, `str_offsets`, `is_string`), `validate_layout`.
- Produces:
  - String-under-axis layout: `offsets=[O0]`, `str_offsets` set, `shape=(*leading, None)`, `dtype 'S'`.
  - `Ragged.from_offsets(S1_data, (*leading, None), offsets, str_offsets=...)` — new optional `str_offsets` kw building the string-under-axis leaf.
  - `to_chars()` promotes `str_offsets` → innermost real level: `(…, ~var) 'S'` → `(…, ~var, ~len) 'S1'`.
  - `to_strings()` demotes innermost real level → `str_offsets`: `(…, ~var, ~len) 'S1'` → `(…, ~var) 'S'`.

- [ ] **Step 1: Write failing tests**

```python
def test_string_under_axis_build_and_dtype():
    data = np.frombuffer(b"ACGTAC", dtype="S1")
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)            # 2 rows: 2 + 1 variants
    str_off = np.array([0, 1, 3, 6], dtype=OFFSET_TYPE)    # 3 variants' byte lens
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    assert rag.is_string is True
    assert rag.dtype == np.dtype("S")
    assert rag.shape == (2, None)

def test_string_under_axis_to_chars_to_strings_roundtrip():
    data = np.frombuffer(b"ACGTAC", dtype="S1")
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)
    str_off = np.array([0, 1, 3, 6], dtype=OFFSET_TYPE)
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    chars = rag.to_chars()
    assert chars.dtype == np.dtype("S1")
    assert chars.shape == (2, None, None)
    assert len(chars.offsets) == 2
    assert chars.offsets[1] is str_off            # zero-copy: str_offsets became inner level
    assert chars.data is data
    back = chars.to_strings()
    assert back.dtype == np.dtype("S")
    assert back.shape == (2, None)
    assert back.offsets[0] is o0
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "string_under_axis" -v`
Expected: FAIL (`from_offsets` has no `str_offsets` kw; `to_chars`/`to_strings` only handle the zero-real-level case).

- [ ] **Step 3: Implement**

In `_layout.py` `validate_layout`, the `n_ragged == 1` arm currently assumes numeric/char. Allow a string-under-axis (1 real level + `str_offsets`): no extra check needed beyond the existing monotonic/segment checks plus validating `str_offsets` is monotonic and ends at `data.shape[0]`. Add, inside the `n_ragged == 1` block after the existing checks:

```python
        if layout.str_offsets is not None:
            if not _is_monotonic(layout.str_offsets):
                raise ValueError("str_offsets must be monotonic non-decreasing")
            if layout.str_offsets.ndim == 1 and int(layout.str_offsets[-1]) != int(layout.data.shape[0]):
                raise ValueError("str_offsets must end at the data length")
```

In `_core.py` `from_offsets`, add the `str_offsets` kw:

```python
    @staticmethod
    def from_offsets(data, shape, offsets, *, str_offsets=None):
        if shape.count(None) >= 3:
            raise NotImplementedError("nested raggedness with R >= 3 is unsupported")
        off_list = offsets if isinstance(offsets, list) else [offsets]
        off_list = [np.ascontiguousarray(o, dtype=OFFSET_TYPE) for o in off_list]
        if str_offsets is not None:
            str_offsets = np.ascontiguousarray(str_offsets, dtype=OFFSET_TYPE)
            return Ragged(RaggedLayout(data=data, offsets=off_list, shape=shape, str_offsets=str_offsets))
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have a None ragged dimension")
        return Ragged(_build_layout(data, shape, off_list))
```

Replace `to_chars`/`to_strings` with the generalized promote/demote (handles both the zero-real-level Spec B case and the string-under-axis case):

```python
    def to_chars(self):
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError("to_chars() is not defined on record Ragged arrays; convert fields.")
        if not self._rl.is_string:
            raise ValueError("to_chars() requires an opaque string Ragged (dtype 'S')")
        assert self._rl.str_offsets is not None
        new_offsets = [*self._layout.offsets, self._rl.str_offsets]   # str_offsets -> innermost real level
        new_shape = (*self._layout.shape, None)
        return Ragged(RaggedLayout(data=self._rl.data, offsets=new_offsets, shape=new_shape))

    def to_strings(self):
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError("to_strings() is not defined on record Ragged arrays; convert fields.")
        if self._rl.is_string:
            return self
        if self._rl.data.dtype.kind != "S":
            raise ValueError("to_strings() requires an S1 char Ragged")
        if self._rl.data.ndim != 1 or self._layout.shape[self.rag_dim + 1:]:
            raise ValueError("to_strings() requires a 1-D S1 char leaf (no trailing dims)")
        *outer_offsets, inner = self._layout.offsets   # innermost real level -> str_offsets
        new_shape = self._layout.shape[:-1]            # drop the inner None
        return Ragged(RaggedLayout(data=self._rl.data, offsets=outer_offsets, shape=new_shape, str_offsets=inner))
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "string or to_chars or to_strings" -v`
Expected: PASS (new string-under-axis + existing Spec B standalone-string tests).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py python/seqpro/rag/_layout.py tests/test_ragged_core.py
git commit -m "feat: string-under-axis leaf + nested to_chars/to_strings"
```

---

## Task 4: Outer-row indexing on R=2 (slice/mask/int-array, lazy)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`__getitem__`, `_row_gather` — generalize to carry the inner level)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `_row_gather` (Spec A), `_level_bounds` (import from `_layout`), Rust `_ragged_select`.
- Produces: `rag[slice|mask|int_array]` over axis 0 on an R=2 array → R=2 `Ragged` with `offsets = [(2, L0'), O1_global]`, data shared.

- [ ] **Step 1: Write failing test**

```python
def test_r2_outer_slice_preserves_nesting():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (3, None, None),
                              [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])])
    sub = rag[1:3]                       # outer rows 1,2
    assert sub.shape == (2, None, None)
    assert len(sub.offsets) == 2
    # row1 had 1 middle (data 5:8), row2 had 1 middle (data 8:10)
    np.testing.assert_array_equal(sub[0][0], np.array([5, 6, 7]))
    np.testing.assert_array_equal(sub[1][0], np.array([8, 9]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "r2_outer_slice" -v`
Expected: FAIL (current `__getitem__` only builds single-level results from `_row_gather`).

- [ ] **Step 3: Implement**

In `__getitem__`, after the record dispatch and the existing single-ragged handling, branch on `n_ragged`. Refactor so `_row_gather` returns the gathered **outer** middle-index ranges, then build the result preserving the inner level. Replace the non-record body with:

```python
        if self._layout.n_ragged == 2:
            return self._getitem_r2(where)
        # ... existing single-level body unchanged ...
```

Add:

```python
    def _getitem_r2(self, where):
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        if isinstance(where, (int, np.integer)):           # peel one outer row -> 1-level Ragged
            a, b = int(o0_starts[where]), int(o0_stops[where])
            if o1.ndim == 1:
                inner = o1[a:b + 1]                          # contiguous slice, zero-copy
            else:
                inner = np.stack([o1[0][a:b], o1[1][a:b]], 0)
            trailing = self._layout.shape[self.rag_dim + 2:]
            return Ragged(RaggedLayout(data=self._rl.data, offsets=[inner],
                                       shape=(b - a, None, *trailing)))
        # slice / mask / int-array on the outer axis: gather O0 ranges, keep O1 global
        from seqpro.rag._core import _where_is_bool  # already module-local
        sel_starts, sel_stops = self._gather_indices(where, o0_starts, o0_stops)
        new_o0 = np.stack([sel_starts, sel_stops], 0)
        trailing = self._layout.shape[self.rag_dim + 2:]
        return Ragged(RaggedLayout(
            data=self._rl.data, offsets=[new_o0, o1],
            shape=(len(sel_starts), None, None, *trailing)))
```

Factor the index-resolution half of Spec A's `_row_gather` into a reusable `_gather_indices(where, starts, stops)` that resolves `where` (bool mask / slice / int-array) against `len(starts)` and returns `(sel_starts, sel_stops)` via `_ragged_select` (lift the body of the existing `_row_gather`, parameterized on the starts/stops passed in). Keep `_row_gather` calling `_gather_indices(where, *self._starts_stops())` for the Spec A path.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "r2_outer or outer_slice or getitem" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: R=2 outer-row indexing (lazy gather, peel to 1-level)"
```

---

## Task 5: Peel chain + leaf access on R=2 (`rag[i][j]`, `rag[i, j]`)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`__getitem__` tuple handling)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `_getitem_r2` (Task 4).
- Produces: tuple indexing `rag[i, j, ...]` == left-to-right chaining (`rag[i][j]...`); `rag[i][j]` on R=2 returns the flat data row (`NDArray`, or `bytes` for a string leaf).

- [ ] **Step 1: Write failing test**

```python
def test_r2_tuple_indexing_and_leaf():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (3, None, None),
                              [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])])
    np.testing.assert_array_equal(rag[0, 1], np.array([3, 4]))   # row0, middle1 -> data 3:5
    np.testing.assert_array_equal(rag[0][1], np.array([3, 4]))   # chaining equivalence
    assert isinstance(rag[2, 0], np.ndarray)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "r2_tuple" -v`
Expected: FAIL (`__getitem__` does not handle tuple keys).

- [ ] **Step 3: Implement**

At the top of `__getitem__`, before the record/single dispatch, add tuple handling (skip when the key is a 2-row `(2, n)`-style ndarray, which never appears as a user key):

```python
        if isinstance(where, tuple):
            result: Any = self
            for k in where:
                result = result[k]
            return result
```

Peeling on R=2 with an integer already returns a 1-level `Ragged` (Task 4); indexing that with an int hits the Spec A integer-leaf path and returns the flat row. No further change needed.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "r2_tuple or r2_outer or peel" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: R=2 tuple indexing + leaf access via peel chaining"
```

---

## Task 6: Per-group inner int / slice (`rag[:, k]`, `rag[:, a:b]`)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_getitem_r2` tuple-with-`slice(None)`-leading path)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `_getitem_r2`, `_level_bounds`, Rust `_ragged_select`.
- Produces:
  - `rag[:, k]` (int on the first `None`) → `(L0, ~K)` **R=1** (data-level `(2, L0)` gather); out-of-range `k` for any group → `IndexError`.
  - `rag[:, a:b]` (slice) → `(L0, None, None)` **R=2** preserved, lazy `O0` shrunk per group.

- [ ] **Step 1: Write failing tests**

```python
def test_r2_per_group_inner_int():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])])
    got = rag[:, 0]                       # 0th middle of each group
    assert got.shape == (2, None)
    np.testing.assert_array_equal(got[0], np.array([0, 1, 2]))   # group0 middle0 -> 0:3
    np.testing.assert_array_equal(got[1], np.array([5, 6, 7]))   # group1 middle0 -> 5:8

def test_r2_per_group_inner_int_out_of_range():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 1, 4]), np.array([0, 3, 5, 8, 10])])
    with pytest.raises(IndexError):
        rag[:, 2]                         # group0 has only 1 middle

def test_r2_per_group_inner_slice():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])])
    sub = rag[:, 0:1]                      # first middle of each group, keep nesting
    assert sub.shape == (2, None, None)
    np.testing.assert_array_equal(sub[0, 0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1, 0], np.array([5, 6, 7]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "per_group_inner" -v`
Expected: FAIL (tuple path chains, treating `rag[:, k]` as `rag[:][k]`, which is wrong).

- [ ] **Step 3: Implement**

The generic tuple loop in Task 5 is wrong for a leading `slice(None)` + inner selector. Special-case a 2-tuple `(slice(None)-or-outer-selector, inner_selector)` on an R=2 array **before** the generic loop in `__getitem__`:

```python
        if (isinstance(where, tuple) and len(where) == 2
                and self._layout.n_ragged == 2 and self._is_full_slice(where[0])):
            return self._getitem_inner(where[1])
```

with:

```python
    @staticmethod
    def _is_full_slice(k):
        return isinstance(k, slice) and k == slice(None)

    def _getitem_inner(self, sel):
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        o1_starts, o1_stops = _level_bounds(o1)
        if isinstance(sel, (int, np.integer)):                  # k-th middle of each group -> R=1
            counts = o0_stops - o0_starts
            if np.any(sel >= counts) or np.any(-sel > counts):
                raise IndexError(f"inner index {sel} out of range for some group")
            mid_idx = (o0_starts + (sel if sel >= 0 else counts + sel)).astype(np.int64)
            ds = np.ascontiguousarray(o1_starts[mid_idx], dtype=OFFSET_TYPE)
            de = np.ascontiguousarray(o1_stops[mid_idx], dtype=OFFSET_TYPE)
            trailing = self._layout.shape[self.rag_dim + 2:]
            return Ragged(RaggedLayout(data=self._rl.data, offsets=[np.stack([ds, de], 0)],
                                       shape=(len(mid_idx), None, *trailing)))
        if isinstance(sel, slice):                              # local per-group slice -> R=2
            start, stop, step = sel.indices(1 << 62)
            if step != 1:
                raise NotImplementedError("step != 1 inner slices are unsupported")
            new_starts = np.minimum(o0_starts + start, o0_stops)
            new_stops = np.minimum(o0_starts + stop, o0_stops)
            new_o0 = np.stack([new_starts.astype(OFFSET_TYPE), new_stops.astype(OFFSET_TYPE)], 0)
            trailing = self._layout.shape[self.rag_dim + 2:]
            return Ragged(RaggedLayout(data=self._rl.data, offsets=[new_o0, o1],
                                       shape=(len(new_starts), None, None, *trailing)))
        return self._getitem_inner_gather(sel)                  # mask / int-array -> Task 8
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "per_group_inner" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: per-group inner int/slice indexing (rag[:, k], rag[:, a:b])"
```

---

## Task 7: Rust `nested_gather` kernel + binding

**Files:**
- Modify: `src/ragged.rs` (add `nested_gather` + unit tests)
- Modify: `src/lib.rs` (add `_ragged_nested_gather` pyfunction + register)
- Test: Rust unit tests in `src/ragged.rs`

**Interfaces:**
- Produces (Rust): `nested_gather(o0_starts: ArrayView1<i64>, o0_stops: ArrayView1<i64>, mask: ArrayView1<bool>) -> Result<(Array1<i64>, Array1<i64>), String>` returning `(counts, sel_mid_idx)`: per-group selected middle counts (`len L0`) and the selected global middle indices in row-major group order.
- Produces (Python binding): `seqpro.seqpro._ragged_nested_gather(o0_starts, o0_stops, mask) -> (counts, sel_idx)`.

- [ ] **Step 1: Write failing Rust unit test**

```rust
// src/ragged.rs  (in the tests module)
#[test]
fn test_nested_gather_selects_per_group() {
    let o0_starts = array![0i64, 2];
    let o0_stops = array![2i64, 4];               // group0: middles 0,1 ; group1: middles 2,3
    let mask = array![true, false, true, true];   // keep middle 0 (g0), 2,3 (g1)
    let (counts, idx) = nested_gather(o0_starts.view(), o0_stops.view(), mask.view()).unwrap();
    assert_eq!(counts, array![1i64, 2]);
    assert_eq!(idx, array![0i64, 2, 3]);
}

#[test]
fn test_nested_gather_rejects_short_mask() {
    let o0_starts = array![0i64];
    let o0_stops = array![3i64];
    let mask = array![true, false];               // mask shorter than middle count
    assert!(nested_gather(o0_starts.view(), o0_stops.view(), mask.view()).is_err());
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib nested_gather`
Expected: FAIL (function not defined).

- [ ] **Step 3: Implement the kernel + binding**

In `src/ragged.rs`:

```rust
pub fn nested_gather(
    o0_starts: ArrayView1<i64>,
    o0_stops: ArrayView1<i64>,
    mask: ArrayView1<bool>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    let n_groups = o0_starts.len();
    if o0_stops.len() != n_groups {
        return Err("o0_starts and o0_stops length mismatch".into());
    }
    let mut counts = Vec::with_capacity(n_groups);
    let mut sel = Vec::new();
    for g in 0..n_groups {
        let (a, b) = (o0_starts[g], o0_stops[g]);
        if a < 0 || b < a {
            return Err("invalid o0 range".into());
        }
        let mut c = 0i64;
        for m in a..b {
            let mi = m as usize;
            if mi >= mask.len() {
                return Err("mask shorter than middle segment count".into());
            }
            if mask[mi] {
                sel.push(m);
                c += 1;
            }
        }
        counts.push(c);
    }
    Ok((Array1::from_vec(counts), Array1::from_vec(sel)))
}
```

In `src/lib.rs`, register and wrap (use `PyReadonlyArray1<bool>`):

```rust
    m.add_function(wrap_pyfunction!(_ragged_nested_gather, m)?)?;
```

```rust
#[pyfunction]
fn _ragged_nested_gather<'py>(
    py: Python<'py>,
    o0_starts: PyReadonlyArray1<'py, i64>,
    o0_stops: PyReadonlyArray1<'py, i64>,
    mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<(&'py PyArray<i64, Ix1>, &'py PyArray<i64, Ix1>)> {
    let (counts, idx) =
        ragged::nested_gather(o0_starts.as_array(), o0_stops.as_array(), mask.as_array())
            .map_err(PyValueError::new_err)?;
    Ok((counts.into_pyarray(py), idx.into_pyarray(py)))
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib nested_gather && cargo clippy --all-targets -- -D warnings && cargo fmt --check`
Expected: PASS (tests green, no clippy/fmt issues). Then `maturin develop`.

- [ ] **Step 5: Commit**

```bash
git add src/ragged.rs src/lib.rs
git commit -m "feat: nested_gather Rust kernel + binding for per-group middle selection"
```

---

## Task 8: Per-group inner mask / int-array (`rag[:, mask]`, `rag[:, idx]`)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_getitem_inner_gather`)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: `seqpro.seqpro._ragged_nested_gather` (Task 7), `_ragged_select`, `_level_bounds`.
- Produces:
  - `rag[:, mask]` where `mask` is a 1-D bool of length `M_total` → `(L0, None, None)` **R=2** with recomputed `O0` (1-D counts) and `O1` gathered to `(2, n_sel)`.
  - `rag[:, idx]` where `idx` is a 1-D int array of *uniform* per-group positions → `(L0, len(idx), None)` (a new regular middle axis), built by stacking the per-group-int path; requires every group to have `> max(idx)` middles.

- [ ] **Step 1: Write failing tests**

```python
def test_r2_inner_mask():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])])
    mask = np.array([True, False, True, True])     # over the 4 middles
    sub = rag[:, mask]
    assert sub.shape == (2, None, None)
    np.testing.assert_array_equal(sub.lengths.tolist(), [1, 2])  # group counts after mask
    np.testing.assert_array_equal(sub[0, 0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1, 0], np.array([5, 6, 7]))

def test_r2_inner_uniform_int_array():
    data = np.arange(12, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 4]), np.array([0, 3, 6, 9, 12])])
    sub = rag[:, np.array([0, 1])]                  # middles 0 and 1 of each group
    assert sub.shape == (2, 2, None)
    np.testing.assert_array_equal(sub[0, 1], np.array([3, 4, 5]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "r2_inner_mask or r2_inner_uniform" -v`
Expected: FAIL (`_getitem_inner_gather` not implemented).

- [ ] **Step 3: Implement**

```python
    def _getitem_inner_gather(self, sel):
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        o1_starts, o1_stops = _level_bounds(o1)
        trailing = self._layout.shape[self.rag_dim + 2:]
        if _where_is_bool(sel):                         # mask over the global middle axis -> R=2
            from seqpro.seqpro import _ragged_nested_gather
            counts, sel_idx = _ragged_nested_gather(
                np.ascontiguousarray(o0_starts, np.int64),
                np.ascontiguousarray(o0_stops, np.int64),
                np.ascontiguousarray(sel, np.bool_),
            )
            new_o0 = lengths_to_offsets(counts.astype(np.uint32))
            ds = np.ascontiguousarray(o1_starts[sel_idx], dtype=OFFSET_TYPE)
            de = np.ascontiguousarray(o1_stops[sel_idx], dtype=OFFSET_TYPE)
            return Ragged(RaggedLayout(data=self._rl.data, offsets=[new_o0, np.stack([ds, de], 0)],
                                       shape=(len(o0_starts), None, None, *trailing)))
        idx = np.atleast_1d(np.asarray(sel, dtype=np.int64))      # uniform per-group int array
        counts = o0_stops - o0_starts
        if np.any(idx.max() >= counts) or np.any(idx.min() < -counts.min()):
            raise IndexError("uniform inner index out of range for some group")
        cols = [self._getitem_inner(int(k)) for k in idx]        # each is (L0, ~K) R=1
        ds = np.stack([c.offsets[0][0] for c in cols], 1).reshape(-1)  # interleave per group
        de = np.stack([c.offsets[0][1] for c in cols], 1).reshape(-1)
        return Ragged(RaggedLayout(data=self._rl.data, offsets=[np.stack([ds, de], 0)],
                                   shape=(len(counts), len(idx), None, *trailing)))
```

(The uniform-int-array path stacks the per-group-int results so column order is `(group, idx)` row-major, matching the `(L0, len(idx), ~K)` shape.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "r2_inner" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: per-group inner mask/int-array indexing via nested_gather"
```

---

## Task 9: Rust `nested_pack` kernel + binding

**Files:**
- Modify: `src/ragged.rs` (`nested_pack` + unit tests)
- Modify: `src/lib.rs` (`_ragged_nested_pack` pyfunction + register)
- Test: Rust unit tests in `src/ragged.rs`

**Interfaces:**
- Produces (Rust): `nested_pack(o0_starts, o0_stops, o1_starts, o1_stops, src_bytes: ArrayView1<u8>, elem: i64) -> Result<(Array1<i64>, Array1<i64>, Array1<u8>), String>` returning canonical zero-based `(o0_out [L0+1], o1_out [M_total+1], out_bytes)`. `o0_*` index the middle axis; `o1_*` index data (in elements); `elem` is bytes per leaf element.
- Produces (Python binding): `seqpro.seqpro._ragged_nested_pack(...)`.

- [ ] **Step 1: Write failing Rust unit test**

```rust
#[test]
fn test_nested_pack_two_level() {
    // group0: middles 0,1 ; group1: middle 2.  middle lens (elem=1): 2,1,3
    let o0_starts = array![0i64, 2];
    let o0_stops = array![2i64, 3];
    let o1_starts = array![0i64, 2, 3];
    let o1_stops = array![2i64, 3, 6];
    let src: Array1<u8> = array![10, 11, 20, 30, 31, 32];
    let (o0, o1, out) =
        nested_pack(o0_starts.view(), o0_stops.view(), o1_starts.view(), o1_stops.view(), src.view(), 1).unwrap();
    assert_eq!(o0, array![0i64, 2, 3]);
    assert_eq!(o1, array![0i64, 2, 3, 6]);
    assert_eq!(out, array![10, 11, 20, 30, 31, 32]);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib nested_pack`
Expected: FAIL (not defined).

- [ ] **Step 3: Implement kernel + binding**

In `src/ragged.rs`:

```rust
pub fn nested_pack(
    o0_starts: ArrayView1<i64>,
    o0_stops: ArrayView1<i64>,
    o1_starts: ArrayView1<i64>,
    o1_stops: ArrayView1<i64>,
    src: ArrayView1<u8>,
    elem: i64,
) -> Result<(Array1<i64>, Array1<i64>, Array1<u8>), String> {
    let n_groups = o0_starts.len();
    let mut o0 = Vec::with_capacity(n_groups + 1);
    o0.push(0i64);
    let mut o1 = vec![0i64];
    let mut out: Vec<u8> = Vec::new();
    let mut mid_count = 0i64;
    for g in 0..n_groups {
        for m in o0_starts[g]..o0_stops[g] {
            let mi = m as usize;
            let (a, b) = (o1_starts[mi] * elem, o1_stops[mi] * elem);
            if a < 0 || b < a || b as usize > src.len() {
                return Err("data span out of bounds".into());
            }
            out.extend_from_slice(&src.as_slice().unwrap()[a as usize..b as usize]);
            o1.push(out.len() as i64 / elem);
            mid_count += 1;
        }
        o0.push(mid_count);
    }
    Ok((Array1::from_vec(o0), Array1::from_vec(o1), Array1::from_vec(out)))
}
```

In `src/lib.rs` register `_ragged_nested_pack` and wrap it returning `(o0, o1, out_bytes)` as three pyarrays (`PyReadonlyArray1<u8>` for `src`).

- [ ] **Step 4: Run to verify pass**

Run: `cargo test --lib nested_pack && cargo clippy --all-targets -- -D warnings && cargo fmt --check`
Expected: PASS. Then `maturin develop`.

- [ ] **Step 5: Commit**

```bash
git add src/ragged.rs src/lib.rs
git commit -m "feat: nested_pack Rust kernel + binding for two-level pack"
```

---

## Task 10: Nested `to_packed` (non-record)

**Files:**
- Modify: `python/seqpro/rag/_ops.py` (`_nested_pack_parts`)
- Modify: `python/seqpro/rag/_core.py` (`to_packed` R=2 branch)
- Test: `tests/test_rag_to_packed.py`

**Interfaces:**
- Consumes: `seqpro.seqpro._ragged_nested_pack`, `_level_bounds`, `OFFSET_TYPE`.
- Produces: `_nested_pack_parts(data, shape, offsets_list, copy) -> (packed_data, [o0, o1])` in `_ops.py`; `Ragged.to_packed(copy=True)` handles R=2 → canonical `[o0, o1]`, zero-based, contiguous data; `copy=False` passes through iff already packed else raises.

- [ ] **Step 1: Write failing test**

```python
# tests/test_rag_to_packed.py
def test_to_packed_nested_after_outer_slice():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (3, None, None),
                              [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])])
    packed = rag[1:3].to_packed()
    assert packed.offsets[0].ndim == 1 and packed.offsets[0][0] == 0
    assert packed.offsets[1].ndim == 1 and packed.offsets[1][0] == 0
    assert packed.data.flags.c_contiguous
    np.testing.assert_array_equal(packed[0, 0], np.array([5, 6, 7]))
    np.testing.assert_array_equal(packed[1, 0], np.array([8, 9]))
    np.testing.assert_array_equal(packed.data, np.array([5, 6, 7, 8, 9]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_rag_to_packed.py -k "nested_after_outer" -v`
Expected: FAIL (current `to_packed` calls `_pack_parts` which only handles a single offsets array).

- [ ] **Step 3: Implement**

In `_ops.py`:

```python
def _nested_pack_parts(data, shape, offsets_list, copy):
    from ._layout import _level_bounds
    if len(offsets_list) != 2:
        raise ValueError("_nested_pack_parts expects exactly 2 offset levels")
    o0, o1 = offsets_list
    o0_starts, o0_stops = _level_bounds(o0)
    o1_starts, o1_stops = _level_bounds(o1)
    rag_dim = shape.index(None)
    trailing = shape[rag_dim + 2:]
    elem = int(np.prod([d for d in trailing if d is not None], dtype=np.int64)) * data.dtype.itemsize
    already = (o0.ndim == 1 and o1.ndim == 1 and (o0.size == 0 or o0[0] == 0)
               and (o1.size == 0 or o1[0] == 0) and data.flags.c_contiguous
               and int(o1[-1]) == data.shape[0])
    if already and not copy:
        return data, [o0, o1]
    if not copy:
        raise ValueError("to_packed(copy=False) requires already-packed input; got an unpacked nested array.")
    from seqpro.seqpro import _ragged_nested_pack
    src = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    out_o0, out_o1, out_bytes = _ragged_nested_pack(
        np.ascontiguousarray(o0_starts, np.int64), np.ascontiguousarray(o0_stops, np.int64),
        np.ascontiguousarray(o1_starts, np.int64), np.ascontiguousarray(o1_stops, np.int64),
        src, elem)
    out_data = out_bytes.view(data.dtype)
    if trailing:
        out_data = out_data.reshape(-1, *trailing)
    return out_data, [out_o0.astype(OFFSET_TYPE), out_o1.astype(OFFSET_TYPE)]
```

In `_core.py` `to_packed`, before the single-level call, add (non-record path):

```python
        if self._layout.n_ragged == 2:
            from ._ops import _nested_pack_parts
            packed_data, packed_offsets = _nested_pack_parts(
                self._rl.data, self._layout.shape, self._layout.offsets, copy)
            if packed_data is self._rl.data and packed_offsets == self._layout.offsets:
                return self
            return Ragged.from_offsets(packed_data, self._layout.shape, packed_offsets)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_rag_to_packed.py -v`
Expected: PASS (nested + existing single-level).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_ops.py python/seqpro/rag/_core.py tests/test_rag_to_packed.py
git commit -m "feat: nested to_packed via nested_pack kernel"
```

---

## Task 11: Per-axis `to_padded` and `to_numpy` (non-record, R=2)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`to_padded`, `to_numpy` R=2 branches)
- Test: `tests/test_ragged_to_padded.py`

**Interfaces:**
- Consumes: single-level `to_padded` (Spec A), `_nested_pack_parts`, `_level_bounds`.
- Produces:
  - `to_padded(pad_value, *, length=None, axis=None)` on R=2: `axis=-1` pads inner → `(*leading, ~M, K)` (R=1, trailing dim); `axis=-2` pads outer → `(*leading, M, ~K)` (R=1, empty inner slots for short groups); `axis=None` pads both → dense `(*leading, M, K)`. `length` is a scalar for one axis or a 2-tuple `(M, K)` for both.
  - `to_numpy(allow_missing=False)` on R=2: requires both ragged axes already rectangular; raises otherwise; returns `(*leading, M, K)`.

- [ ] **Step 1: Write failing tests**

```python
def test_r2_to_padded_inner():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    out = rag.to_padded(-1, axis=-1)             # pad inner ~K
    assert out.shape == (2, None, 5)             # R=1 trailing dim K=max(3,2,5)=5
    np.testing.assert_array_equal(out[0, 0], np.array([0, 1, 2, -1, -1]))

def test_r2_to_padded_both_dense():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    dense = rag.to_padded(-1)                     # both axes
    assert dense.shape == (2, 2, 5)              # M=max(2,1)=2, K=5
    np.testing.assert_array_equal(dense[1, 1], np.full(5, -1))   # padded middle slot

def test_r2_to_numpy_rectangular():
    data = np.arange(12, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 4]), np.array([0, 3, 6, 9, 12])])
    arr = rag.to_numpy()
    assert arr.shape == (2, 2, 3)
    np.testing.assert_array_equal(arr[1, 1], np.array([9, 10, 11]))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_to_padded.py -k "r2_to_padded or r2_to_numpy" -v`
Expected: FAIL (no R=2 branch).

- [ ] **Step 3: Implement**

In `to_padded`, before the single-level body, add:

```python
        if self._layout.n_ragged == 2:
            return self._to_padded_nested(pad_value, length=length, axis=axis)
```

```python
    def _to_padded_nested(self, pad_value, *, length, axis):
        o0, o1 = self.to_packed().offsets if not self.is_contiguous else self._layout.offsets
        rag = self if self.is_contiguous else self.to_packed()
        len_m, len_k = (length if isinstance(length, tuple) else (length, length))
        # middle-as-rows 1-level view over O1 (data partitioned by inner segments)
        inner_view = Ragged(RaggedLayout(data=rag._rl.data, offsets=[rag._layout.offsets[1]],
                                         shape=(len(rag._layout.offsets[1]) - 1, None,
                                                *rag._layout.shape[self.rag_dim + 2:])))
        if axis == -1:                              # pad inner only -> (*leading, ~M, K)
            padded = inner_view.to_padded(pad_value, length=len_k)   # (M_total, K)
            return Ragged(RaggedLayout(data=padded, offsets=[rag._layout.offsets[0]],
                                       shape=(*rag.shape[:self.rag_dim], None, padded.shape[1])))
        # pad inner first (-> (M_total, K)), then pad the outer groups of K-wide elements
        padded = inner_view.to_padded(pad_value, length=len_k)        # (M_total, K)
        outer_view = Ragged(RaggedLayout(data=padded, offsets=[rag._layout.offsets[0]],
                                         shape=(*rag.shape[:self.rag_dim], None, padded.shape[1])))
        if axis == -2:                              # pad outer only -> (*leading, M, ~K): structural
            raise NotImplementedError("axis=-2 (outer-only, inner ragged) — implement structural empty-slot pad")
        dense = outer_view.to_padded(pad_value, length=len_m)         # (*leading, M, K)
        return dense
```

For `axis=-2` (outer-only while inner stays ragged), implement the structural empty-slot pad: build a new `O1'` of length `L0 * M + 1` where each group's selected middle segments keep their `O1` spans and padding positions are zero-length (`stop == start`), data unchanged; return `shape (*leading, M, None, *trailing)`. Add a focused test for it:

```python
def test_r2_to_padded_outer_structural():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    out = rag.to_padded(0, axis=-2)              # M=2 ; group1 gets one empty middle slot
    assert out.shape == (2, 2, None)
    assert out[1, 1].size == 0
```

In `to_numpy`, add before the single-level body:

```python
        if self._layout.n_ragged == 2:
            o0, o1 = self.to_packed().offsets
            mid_lens = np.diff(o1)
            grp_lens = np.diff(o0)
            if grp_lens.size and not np.all(grp_lens == grp_lens[0]):
                raise ValueError("cannot convert a jagged outer axis to a dense array")
            if mid_lens.size and not np.all(mid_lens == mid_lens[0]):
                raise ValueError("cannot convert a jagged inner axis to a dense array")
            return self.to_padded(np.zeros((), self.dtype)[()])   # rectangular -> pad is identity
```

(When already rectangular, padding to the max equals the dense array; reuse the `axis=None` path.)

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_to_padded.py -v`
Expected: PASS (nested inner/both/outer/to_numpy + existing single-level).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_to_padded.py
git commit -m "feat: per-axis nested to_padded + rectangular to_numpy (R=2)"
```

---

## Task 12: Nested `squeeze` / `reshape` / `lengths` (non-record)

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`squeeze`, `reshape`, `lengths` R=2 handling)
- Test: `tests/test_ragged_core.py`

**Interfaces:**
- Consumes: existing `squeeze`/`reshape`/`lengths`.
- Produces: `lengths` on R=2 returns the **outer** group counts reshaped to leading dims (inner lengths via `rag[i].lengths`); `squeeze` drops size-1 leading/trailing dims preserving both ragged levels; `reshape` adjusts leading int dims of the outer level only.

- [ ] **Step 1: Write failing test**

```python
def test_r2_lengths_outer_counts():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    np.testing.assert_array_equal(rag.lengths, np.array([2, 1]))   # outer middle counts

def test_r2_squeeze_leading_one():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (1, 2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    sq = rag.squeeze(0)
    assert sq.shape == (2, None, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "r2_lengths or r2_squeeze" -v`
Expected: FAIL (`lengths` uses `self.offsets` = `offsets[0]` but the reshape/diff assumes single level; `squeeze` reshapes data against a single ragged dim).

- [ ] **Step 3: Implement**

`lengths`: `self.offsets` already returns `offsets[0]`; for R=2 `np.diff(O0)` gives outer counts. Confirm the `leading` reshape uses `shape[:rag_dim]` (the first `None`'s position) — already correct. Add a guard so the `(2, ·)` outer form uses `_level_bounds`:

```python
    @property
    def lengths(self):
        o0 = self._layout.offsets[0] if self._layout.offsets else self._rl.str_offsets
        starts, stops = _level_bounds(o0)
        raw = (stops - starts)
        rag_dim = self._layout.shape.index(None) if None in self._layout.shape else len(self._layout.shape)
        leading = self._layout.shape[:rag_dim]
        reshape_arg = [d for d in leading if d is not None] if leading else [-1]
        return raw.reshape(reshape_arg)
```

`squeeze(axis)`: for R=2, when squeezing a leading int dim, reshape `data` against **trailing** dims after the *second* `None` and keep both offset levels. Update the data-trailing computation to use `i > rag_dim + 1` (trailing starts after the inner `None`) and pass both offset levels through unchanged. `reshape`: only the leading int dims (before the first `None`) are recomputed; data trailing uses `shape[rag_dim + 2:]`; offsets pass through. Apply the analogous `+2` offset to the existing single-level slicing in both methods, guarded by `n_ragged == 2`.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py -k "r2_lengths or r2_squeeze or reshape or squeeze" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "feat: R=2 lengths/squeeze/reshape"
```

---

## Task 13: Nested records — validation + `from_fields`/`zip` + string-under-axis fields

**Files:**
- Modify: `python/seqpro/rag/_layout.py` (`_validate_record_layout`: allow R=2 fields + per-field `str_offsets`; share full offsets list)
- Modify: `python/seqpro/rag/_core.py` (`from_fields`: nested + string-under-axis fields; share `[O0, O1]`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `RecordLayout`, `RaggedLayout`, `validate_layout`, `from_fields` (Spec B).
- Produces:
  - `RecordLayout.offsets` is the full shared list (`[O0]` or `[O0, O1]`); each field's `offsets` **is** that list (identity); a field may carry its own `str_offsets` (string-under-axis); record-of-record / R≥3 fields → `NotImplementedError`.
  - `Ragged.from_fields(fields)` / `seqpro.rag.zip(fields)` accept R=2 and string-under-axis fields; canonicalize all fields onto one shared offsets list.

- [ ] **Step 1: Write failing tests**

```python
def test_record_nested_from_fields():
    data_a = np.arange(10, dtype=np.int32)
    data_b = (np.arange(10, dtype=np.float64) / 2)
    offs = [np.array([0, 2, 3]), np.array([0, 3, 5, 10])]
    a = Ragged.from_offsets(data_a, (2, None, None), offs)
    b = Ragged.from_offsets(data_b, (2, None, None), offs)
    rec = Ragged.from_fields({"a": a, "b": b})
    assert rec.fields == ["a", "b"]
    assert rec.shape == (2, None, None)
    assert rec["a"].offsets[0] is rec["b"].offsets[0]      # shared O0
    assert rec["a"].offsets[1] is rec["b"].offsets[1]      # shared O1
    np.testing.assert_array_equal(rec["a"][0, 0], np.arange(3))

def test_record_string_under_axis_field():
    ref = Ragged.from_offsets(np.frombuffer(b"ACG", "S1"), (2, None),
                              np.array([0, 1, 2]), str_offsets=np.array([0, 1, 3]))
    alt = Ragged.from_offsets(np.frombuffer(b"TTGG", "S1"), (2, None),
                              np.array([0, 1, 2]), str_offsets=np.array([0, 2, 4]))
    rec = Ragged.from_fields({"ref": ref, "alt": alt})
    assert rec["ref"].is_string and rec["alt"].is_string
    assert rec["ref"].offsets[0] is rec["alt"].offsets[0]  # shared ~variants offsets
    assert rec["ref"][0] == b"A"
    assert rec["alt"][0] == b"TT"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core_records.py -k "nested_from_fields or string_under_axis_field" -v`
Expected: FAIL (record validation rejects `is_string` fields and assumes a single shared offsets array).

- [ ] **Step 3: Implement**

In `_layout.py` `_validate_record_layout`: drop the unconditional `is_string` rejection; instead require every field to share the **full** offsets list (identity per level) and agree on ragged shape. A field may set `str_offsets` (string-under-axis) — validate it like Task 3. Record-of-record / R≥3 → `NotImplementedError`. Replace the loop body:

```python
    shared = layout.offsets
    rag_dim = layout.shape.index(None)
    ragged_shape = layout.shape[: rag_dim + 1]
    for name, fld in layout.fields.items():
        if len(fld.offsets) != len(shared) or any(fo is not so for fo, so in zip(fld.offsets, shared)):
            raise ValueError(f"field {name!r} must use the shared offsets list (zero-copy SoA)")
        if fld.shape[: fld.shape.index(None) + 1] != ragged_shape:
            raise ValueError(f"field {name!r} ragged shape {fld.shape} disagrees with record {layout.shape}")
        validate_layout(fld)
```

In `_core.py` `from_fields`: allow `is_string` fields; share the whole list; preserve per-field `str_offsets`:

```python
    @staticmethod
    def from_fields(fields):
        if not fields:
            raise ValueError("from_fields requires at least one field (got empty)")
        items = list(fields.items())
        for name, f in items:
            if f._is_record:
                raise NotImplementedError(f"record-of-record field {name!r} is unsupported")
            if f._layout.n_ragged >= 3:
                raise NotImplementedError(f"R>=3 field {name!r} is unsupported")
        shared = items[0][1]._layout.offsets
        for name, f in items[1:]:
            fo = f._layout.offsets
            if len(fo) != len(shared) or any(not np.array_equal(a, b) for a, b in zip(fo, shared)):
                raise ValueError(f"field {name!r} offsets are not equal to the first field's")
        rec_shape = items[0][1].shape
        rebound = {
            name: RaggedLayout(data=f._rl.data, offsets=shared,
                               shape=f._layout.shape, str_offsets=f._rl.str_offsets)
            for name, f in items
        }
        return Ragged(RecordLayout(offsets=shared, shape=rec_shape, fields=rebound))
```

Update `RecordLayout.offsets` everywhere it's assumed length-1: `Ragged.offsets` returns `offsets[0]` (still the outer level — fine); add an internal use of the full list where pack/index need it (Tasks 14–15).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core_records.py -v`
Expected: PASS (nested + string-under-axis records + existing Spec B records).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_layout.py python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: nested + string-under-axis record fields sharing full offsets list"
```

---

## Task 14: Nested record indexing + `to_packed`/`to_padded`/`to_numpy`

**Files:**
- Modify: `python/seqpro/rag/_core.py` (`_getitem_record_rows`, record `to_packed`/`to_padded`/`to_numpy`)
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `_getitem_r2`, `_nested_pack_parts`, per-field methods from Tasks 10–11.
- Produces: record-aware R=2 row indexing (outer slice/mask → record `Ragged`; full-leading-int → dict of per-field rows); record `to_packed` emits one shared `[O0, O1]` across fields; `to_padded`/`to_numpy` → per-field dicts (string-under-axis fields raise on numeric densify, mirroring Spec B).

- [ ] **Step 1: Write failing test**

```python
def test_record_nested_to_packed_shared_offsets():
    offs = [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])]
    a = Ragged.from_offsets(np.arange(10, np.int32), (3, None, None), offs)
    b = Ragged.from_offsets(np.arange(10, np.float64), (3, None, None), offs)
    rec = Ragged.from_fields({"a": a, "b": b})[1:3].to_packed()
    assert rec["a"].offsets[0] is rec["b"].offsets[0]
    assert rec["a"].offsets[1] is rec["b"].offsets[1]
    np.testing.assert_array_equal(rec["a"][0, 0], np.array([5, 6, 7]))

def test_record_nested_row_slice():
    offs = [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])]
    a = Ragged.from_offsets(np.arange(10, np.int32), (3, None, None), offs)
    rec = Ragged.from_fields({"a": a})
    sub = rec[1:3]
    assert sub._is_record and sub.shape == (2, None, None)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core_records.py -k "nested_to_packed or nested_row_slice" -v`
Expected: FAIL (`_getitem_record_rows` and record `to_packed` assume a single offsets level).

- [ ] **Step 3: Implement**

`_getitem_record_rows`: when `n_ragged == 2`, compute the shared gather once (reuse `_getitem_r2`'s outer-gather math on the shared `[O0, O1]`) and rebuild each field's `RaggedLayout` onto the new shared offsets list; `rag[int]`/full-leading peel returns a dict of per-field rows.

Record `to_packed`: when `n_ragged == 2`, call `_nested_pack_parts` **once** using the shared offsets, then reuse the resulting `[o0, o1]` for every field's data pack (each field packs its own data buffer with the same selection). Emit one shared list referenced by all fields.

Record `to_padded`/`to_numpy`: per-field dicts as in Spec B, passing `axis`/`length` through; a string-under-axis field raises `TypeError` on numeric densify (matching Spec B's per-field string raise).

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core_records.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "feat: nested record indexing + record-aware to_packed/to_padded/to_numpy"
```

---

## Task 15: `_ingest` bridge — R=2 + string-under-axis (oracle interop)

**Files:**
- Modify: `python/seqpro/rag/_ingest.py` (`layout_from_ak`, `to_ak`)
- Test: `tests/test_ragged_core.py`, `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `unbox`/`_extract_list_offsets`/`_parts_to_content` (`_array.py`), `RaggedLayout`, `RecordLayout`.
- Produces: `layout_from_ak` builds `[O0, O1]` for a two-list-axis input, string-under-axis for a list-over-`S1` input, and nested records sharing `[O0, O1]`; `to_ak` assembles nested `ListOffsetArray`s (+ `RecordArray`), emitting the bytestring-under-list form for string-under-axis.

- [ ] **Step 1: Write failing test**

```python
def test_bridge_r2_roundtrip():
    arr = ak.Array([[[1, 2, 3], [4, 5]], [[6]]])     # (2, ~M, ~K) numeric
    rag = Ragged(arr)
    assert rag.shape == (2, None, None)
    assert rag.to_ak().to_list() == arr.to_list()

def test_bridge_string_under_axis_roundtrip():
    arr = ak.Array([[b"AC", b"G"], [b"TTT"]])        # (2, ~var) bytestrings
    rag = Ragged(arr)
    assert rag.is_string and rag.shape == (2, None)
    assert rag.to_ak().to_list() == arr.to_list()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_ragged_core.py -k "bridge_r2 or bridge_string_under" -v`
Expected: FAIL (`layout_from_ak` only handles single-level + record; `to_ak` only single-level + record).

- [ ] **Step 3: Implement**

In `layout_from_ak`, detect nesting depth via `unbox`/awkward layout walking: a two-list-axis content → `[O0, O1]` `RaggedLayout` with `shape (L0, None, None, *trailing)`; a list-axis whose content is an `S1`/bytestring leaf → string-under-axis (`offsets=[O0]`, `str_offsets=<inner byte offsets>`, `shape (L0, None)`); records extract the shared `[O0, O1]` (or `[O0]`) once and unbox each field onto it. In `to_ak`, wrap data in nested `ak.contents.ListOffsetArray`s (outer over inner over `NumpyArray`); for string-under-axis emit the awkward bytestring form (`ListOffsetArray` of `char` with `__array__="bytestring"` parameter) over the `O0` list.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_ragged_core.py tests/test_ragged_core_records.py -k "bridge" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/rag/_ingest.py tests/test_ragged_core.py tests/test_ragged_core_records.py
git commit -m "feat: _ingest bridge for R=2 + string-under-axis (oracle interop)"
```

---

## Task 16: Differential property tests (Hypothesis) for nested core + records

**Files:**
- Create: `tests/test_ragged_nested_diff.py`
- Test: the new file

**Interfaces:**
- Consumes: native `Ragged` (`_core`), awkward oracle (`ak.Array` built from the same buffers), all methods from Tasks 1–15.

- [ ] **Step 1: Write the property tests**

```python
import awkward as ak
import numpy as np
from hypothesis import given, settings, strategies as st
from seqpro.rag._core import Ragged

@st.composite
def nested_r2(draw):
    n_outer = draw(st.integers(1, 4))
    outer_counts = draw(st.lists(st.integers(0, 3), min_size=n_outer, max_size=n_outer))
    n_mid = sum(outer_counts)
    inner_lens = draw(st.lists(st.integers(0, 4), min_size=n_mid, max_size=n_mid))
    total = sum(inner_lens)
    data = np.arange(total, dtype=np.int64)
    return np.array(outer_counts), np.array(inner_lens), data

def _oracle(outer, inner, data):
    o0 = np.concatenate([[0], np.cumsum(outer)]).astype(np.int64)
    o1 = np.concatenate([[0], np.cumsum(inner)]).astype(np.int64)
    return ak.Array(ak.contents.ListOffsetArray(
        ak.index.Index64(o0),
        ak.contents.ListOffsetArray(ak.index.Index64(o1), ak.contents.NumpyArray(data))))

@settings(max_examples=200)
@given(nested_r2())
def test_diff_construct_index_pack(t):
    outer, inner, data = t
    rag = Ragged.from_lengths(data, (outer, inner))
    ora = _oracle(outer, inner, data)
    assert rag.to_ak().to_list() == ora.to_list()
    assert rag.lengths.tolist() == ak.num(ora, axis=1).to_list()
    for i in range(len(outer)):
        assert rag[i].to_ak().to_list() == ora[i].to_list()
    assert rag.to_packed().to_ak().to_list() == ak.to_packed(ora).to_list()

@settings(max_examples=200)
@given(nested_r2(), st.integers(0, 2))
def test_diff_inner_int(t, k):
    outer, inner, data = t
    if np.any(outer <= k):
        return
    rag = Ragged.from_lengths(data, (outer, inner))
    ora = _oracle(outer, inner, data)
    assert rag[:, k].to_ak().to_list() == ora[:, k].to_list()
```

Add analogous property tests for `rag[:, a:b]`, `rag[:, mask]`, outer slice/mask, and a record variant (`ak.zip` two numeric fields over the same nesting). Cover empty groups, empty inner segments, and a string-under-axis variant (bytestrings) round-trip + `to_chars`/`to_strings`.

- [ ] **Step 2: Run to verify failure / iterate**

Run: `pytest tests/test_ragged_nested_diff.py -v`
Expected: Initially may surface gaps — fix the implementation tasks they trace to (this is the integration safety net), then green.

- [ ] **Step 3..4: Make green**

Iterate until all property tests pass against the oracle.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ragged_nested_diff.py
git commit -m "test: differential Hypothesis suite for nested R=2 + records vs awkward oracle"
```

---

## Task 17: Consumer-case exit tests + roadmap SSoT update

**Files:**
- Create: `tests/test_ragged_nested_consumers.py`
- Modify: `docs/roadmap/rust-ragged.md` (Spec C status + decision-log "landed" entry)
- Test: the new file

**Interfaces:**
- Consumes: the full Spec C surface.

- [ ] **Step 1: Write the three consumer-case tests**

```python
import awkward as ak
import numpy as np
from seqpro.rag._core import Ragged

def test_consumer_alleles_string_under_axis():
    # (batch=1, ploidy=2, ~variants) opaque allele strings -> chars (R=2)
    data = np.frombuffer(b"ACGTT", "S1")
    o0 = np.array([0, 2, 3], dtype=np.int64)            # 2 "rows" (ploidy flattened): 2,1 variants
    str_off = np.array([0, 1, 2, 5], dtype=np.int64)    # 3 variants
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    chars = rag.to_chars()
    assert chars.shape == (2, None, None) and chars.dtype == np.dtype("S1")
    assert chars[1, 0].tobytes() == b"GTT"

def test_consumer_flat_variant_windows():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, None),
                              [np.array([0, 2, 3]), np.array([0, 3, 5, 10])])
    dense = rag.to_padded(-1)                            # model-input tensor
    assert dense.shape == (2, 2, 5)

def test_consumer_codon_annotations_record():
    offs = [np.array([0, 2, 3]), np.array([0, 3, 5, 10])]
    pos = Ragged.from_offsets(np.arange(10, np.int32), (2, None, None), offs)
    strand = Ragged.from_offsets(np.ones(10, np.int8), (2, None, None), offs)
    rec = Ragged.from_fields({"codon_pos": pos, "strand": strand})
    assert rec.fields == ["codon_pos", "strand"]
    assert rec[0, 1, 0] == {"codon_pos": pos[0, 1][0], "strand": 1} or rec["codon_pos"][0, 1].size  # field access works
    packed = rec.to_packed()
    assert packed["codon_pos"].offsets[1] is packed["strand"].offsets[1]
```

- [ ] **Step 2: Run to verify**

Run: `pytest tests/test_ragged_nested_consumers.py -v`
Expected: PASS (exit criteria for Spec C).

- [ ] **Step 3: Run the full rag suite + lints**

Run:
```bash
pytest tests/test_ragged_core.py tests/test_ragged_core_records.py tests/test_rag_to_packed.py tests/test_ragged_to_padded.py tests/test_ragged_nested_diff.py tests/test_ragged_nested_consumers.py -q
cargo test --lib && cargo clippy --all-targets -- -D warnings && cargo fmt --check
ruff check python/ tests/ && ruff format --check python/ tests/
```
Expected: all green.

- [ ] **Step 4: Update the roadmap SSoT**

In `docs/roadmap/rust-ragged.md`, change the Spec C entry status from "Design approved" to **landed**, and append a decision-log entry dated 2026-06-20 summarizing what shipped (R=2 cap, full nested indexing incl. `rag[:, k]`/`rag[:, a:b]`/`rag[:, mask]`/uniform int-array, per-axis `to_padded`, string-under-axis standalone+record, nested records, `nested_pack`+`nested_gather` kernels, bridge; public swap + tokenize/translate still Spec D; SKILL.md untouched per internal-only rule). Note the implementation refinement: `to_padded(both)` composes two single-level pads (no third kernel); `to_padded(axis=-2)` does a structural empty-slot pad.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ragged_nested_consumers.py docs/roadmap/rust-ragged.md
git commit -m "test: Spec C consumer-case exit tests; docs: mark Spec C landed in roadmap SSoT"
```

---

## Self-Review

**Spec coverage** (Spec C §1–§10):
- §1 nested data model → Tasks 1, 2. §2 string-under-axis + conversions → Task 3. §3 constructors → Tasks 2, 3, 13. §4 indexing (all forms) → Tasks 4, 5, 6, 8. §5 records → Tasks 13, 14. §6 to_packed/to_padded/to_numpy → Tasks 10, 11, 14. §7 Rust kernels → Tasks 7, 9. §8 bridge → Task 15. §9 testing → Tasks 16, 17. §10 migration/SSoT → Task 17.
- **Refinement vs §7:** `to_padded` reuses the single-level Numba pad kernel (compose for "both"; structural offset edit for outer-only) rather than a dedicated nested pad kernel; only `nested_pack` + `nested_gather` are new Rust kernels. Recorded in Task 17's roadmap update.

**Placeholder scan:** Task 11 (`axis=-2`) and Task 16 (analogous property tests) intentionally describe additional cases in prose alongside concrete code/tests; each names exact shapes, return types, and an executable focused test — no "TBD"/"handle edge cases" left abstract. All code steps show code.

**Type consistency:** `_level_bounds` (Task 1) used identically in Tasks 4, 6, 8, 10, 12. `_nested_pack_parts(data, shape, offsets_list, copy) -> (data, [o0, o1])` defined Task 10, reused Task 14. `nested_gather -> (counts, sel_idx)` / `nested_pack -> (o0, o1, out_bytes)` (Tasks 7, 9) match their Python call sites (Tasks 8, 10). `from_offsets(..., str_offsets=None)` (Task 3) used in Tasks 13, 17. Record `offsets` is the shared **list** (Task 13) consumed by Task 14.
