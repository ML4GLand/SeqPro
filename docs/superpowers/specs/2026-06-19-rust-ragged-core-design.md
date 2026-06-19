# Design: Rust-native `Ragged` — Spec A (core, single-level)

**Date:** 2026-06-19
**Status:** Approved design, pending implementation plan
**Epic:** [Rust-native Ragged](../../roadmap/rust-ragged.md) — Spec A of 4
**Branch:** `feat/rust-ragged`

## Goal

Replace the `awkward.Array` base class of `seqpro.rag.Ragged` with a plain
Python class backed by NumPy buffers and stateless Rust layout kernels — at
**single ragged-level** parity with today's non-record API. This is the
foundation spec: it proves the data model and removes awkward from the core
single-level path. Records (Spec B), nesting depth ≥2 (Spec C), and the final
awkward-dependency removal + downstream cutover (Spec D) build on it.

See the [roadmap](../../roadmap/rust-ragged.md) for the locked epic-wide
decisions (full removal / drop-in, Python-buffers + Rust-algebra, ragged-run +
regular-ends axis model, string-leaf). This spec applies them to the single-level
case.

## Scope

**In scope (single ragged level, non-record):**
- New `Ragged` class (not an `ak.Array` subclass) and its internal layout struct.
- Constructors: `from_lengths`, `from_offsets`, `empty`.
- Properties: `data`, `offsets`, `shape`, `dtype`, `lengths`, `rag_dim`,
  `is_empty`, `is_contiguous`, `is_base`.
- Methods: `__getitem__`, `view`, `squeeze`, `reshape`, `to_numpy`,
  `to_packed`, `to_padded`.
- ufunc + NumPy-function interop (`__array_ufunc__` / `__array_function__`).
- String-leaf semantics for `Ragged[np.bytes_]`.
- Rust kernels: single-level row-slice/index, single-level pack, layout
  validation.

**Out of scope (later specs):**
- Record / struct-of-arrays layouts and `zip` (Spec B). In Spec A, record inputs
  raise a clear `NotImplementedError` pointing at Spec B.
- Two or more nested ragged levels (Spec C). In Spec A, constructing `>1` ragged
  level raises `NotImplementedError` pointing at Spec C.
- Removing the `awkward` dependency from `pyproject`/`Cargo` and adapting
  `tokenize`/`translate` and docs/skill (Spec D). Awkward stays installed and
  importable during Spec A; the new `Ragged` simply does not use it. `to_ak`
  becomes a thin shim retained for Spec D-era interop.

## Approach

Mirror the existing `kshuffle`/`tokenize`/`translate` boundary: **Python owns
orchestration and buffers; Rust owns only the inner structural compute.** The
new `Ragged` is a small Python class wrapping a `RaggedLayout` value object;
hot/structural ops delegate to Rust kernels in a new `src/ragged.rs` registered
in the `#[pymodule]` in `src/lib.rs`.

We build the Python data model and pure-Python interop first (it can run on the
existing NumPy pack kernels), then move the row-slice/index and validation hot
paths into Rust once parity tests are green — keeping each step reviewable and
the data model honest before optimizing.

## Section 1 — Data model

### `RaggedLayout` (replaces `RagParts`)

A value object holding the buffers. For Spec A it carries exactly one ragged
level, but its fields are shaped to generalize in Spec C without a rewrite.

```
RaggedLayout:
    data:    NDArray            # flat 1-D numeric buffer, or S1 buffer for a string leaf;
                                # 2-D (total, *trailing) when the leaf has trailing regular dims
    offsets: list[NDArray]      # one (N+1,) or (2, N) array per ragged level, outermost-first.
                                # Spec A: len == 1.
    shape:   tuple[int|None,..] # (*leading_int, None×R, *trailing_int)
    str_offsets: NDArray | None # per-element byte boundaries for a string leaf; None for numeric.
                                # NOT counted in shape/offsets.
```

- **Numeric leaf:** `str_offsets is None`. `len(offsets) == R == shape.count(None)`.
  `data` is 1-D, or 2-D `(total, *trailing)` when there are trailing regular dims.
- **String leaf (`np.bytes_`):** `data` is the flat `S1` buffer, `str_offsets`
  is the per-element byte boundaries, and the byte-length is **not** an axis.
  The number of ragged *axes* (`len(offsets)`) depends on whether the strings
  sit under a ragged axis — see "String leaf at a single level" below.

#### String leaf at a single level

Spec A's single ragged level can itself be the string axis or sit above a string
leaf. Two cases, kept explicit to avoid ambiguity:

1. **Flat collection of sequences** — `["ATG", "CG"]`. Shape `(2,)`, dtype bytes.
   The variation *is* the string. `offsets == []` (no ragged *axis*),
   `str_offsets == [0, 3, 5]`, `data` = `S1` buffer `[A,T,G,C,G]`. This replaces
   today's `(2, None)` representation (roadmap: string-leaf decision).
2. **Strings under one ragged axis** — e.g. per-region variable count of allele
   strings, `(regions, ~)` dtype bytes. `offsets == [region_offsets]` (the `~`
   axis), `str_offsets` = per-string byte boundaries, `data` = `S1` buffer.
   `shape == (n_regions, None)`.

`shape.count(None)` therefore counts ragged **axes only**; the string leaf is
never among them. `len(offsets) == shape.count(None)` holds for both numeric and
string leaves.

### `Ragged`

A `Generic[RDTYPE_co]` Python class wrapping one `RaggedLayout`. No awkward base,
no behavior patching, no `__typestr__` hook. Properties read straight off the
layout; ops return new `Ragged`s. `is_rag_dtype` keeps its current signature and
semantics (now a plain isinstance + dtype check, no awkward layout walking).

## Section 2 — Constructors

- **`from_offsets(data, shape, offsets)`** — unchanged signature. Validates
  exactly one `None` in `shape` (Spec A; `>1` → `NotImplementedError` → Spec C),
  segment count vs `prod(shape[:rag_dim])`, and data size vs offsets (contiguous
  case). For bytes dtype, interprets the supplied offsets as the **string leaf**
  when `shape` has no `None` (case 1 above) or as the ragged axis with a derived
  string leaf otherwise — the rule is documented and validated in one place.
- **`from_lengths(data, lengths)`** — unchanged signature; builds offsets via
  `lengths_to_offsets` (existing util).
- **`empty(shape, dtype)`** — unchanged; zero-length data + zero offsets.

All three return a `Ragged` wrapping a `RaggedLayout`. Validation is front-loaded
in the constructor (project convention: fast-fail, single obvious check).

## Section 3 — Properties

Direct reads off the layout, matching today's semantics and return types:

- `data` → flat `NDArray` (S1 for bytes). `offsets` → the single ragged-level
  `NDArray` (`(N+1,)` or `(2,N)`); raises if there is no ragged axis (string
  case 1) — or returns `str_offsets`? **Decision:** `offsets` returns the ragged
  *axis* offsets; for a flat string collection (no axis) it returns `str_offsets`
  for backward continuity, documented explicitly. `shape`, `dtype`, `rag_dim`
  (`shape.index(None)`), `lengths` (`np.diff` of offsets, reshaped to leading
  dims), `is_empty`, `is_contiguous`, `is_base` — all ported from today's
  implementation, reading the layout instead of walking an awkward `Content`.

## Section 4 — Indexing & slicing

`__getitem__` replaces awkward's `super().__getitem__` + `_n_var` re-boxing:

- **Integer index on the ragged axis** (`rag[i]`) → returns the i-th row as a
  plain `NDArray` (numeric) or `bytes` scalar / `S1` row (string). Matches the
  awkward behavior of returning a non-`Ragged` for a fully-indexed row.
- **Slice / fancy index on a leading axis** (`rag[a:b]`, `rag[mask]`,
  `rag[idx_array]`) → returns a `Ragged` over the selected rows. Implemented by
  selecting offset entries and either viewing (contiguous slice) or producing a
  `(2, N)` start/stop offsets view (gather) — no data copy, mirroring awkward's
  `project()`-style non-copy indexing. Callers needing contiguous data pack
  afterwards (`to_packed`), exactly as today.
- The row-slice/gather math is the first Rust kernel
  (`ragged_index` / `ragged_slice`): given offsets + an index/slice/mask, emit
  the selected `(starts, stops)` and (for integer index) the flat `(lo, hi)`
  span. Pure-Python fallback first, Rust once parity holds.

## Section 5 — `view`, `squeeze`, `reshape`, `to_numpy`

Ported from today's `_array.py`, rewritten against `RaggedLayout` instead of
awkward:

- **`view(dtype)`** — reinterpret the flat `data` buffer
  (`data.view(dtype)`), same offsets/shape. Zero-copy.
- **`squeeze(axis)`** / **`reshape(*shape)`** — operate on the leading/trailing
  *regular* dims and the data buffer's trailing dims exactly as today (the
  ragged-axis logic is unchanged; the awkward round-trip is removed). Return a
  `Ragged`, or an `NDArray` when squeezing collapses to a dense array
  (preserved behavior).
- **`to_numpy(allow_missing=False)`** — densify via `to_padded`-style copy when
  rows are equal length, else raise the same error awkward raised for jagged
  `to_numpy`. For bytes, restore the `S1` trailing view as today.

## Section 6 — ufunc & NumPy interop

This is what awkward's behavior dispatch gave us for free; we implement it
directly:

- **`__array_ufunc__`** — for element-wise ufuncs (`np.log1p`, `+`, `*`, …),
  apply the ufunc to each operand's flat `data` and rewrap with the **same
  offsets/shape**. Mixed `Ragged`/scalar and `Ragged`/`Ragged` (matching
  offsets) operands supported; mismatched offsets raise. Reductions and other
  methods (`method != "__call__"`) raise `NotImplementedError` unless trivially
  supported. This reproduces today's `apply_ufunc` semantics (new `Ragged`, same
  offsets, no offset copies).
- **`__array_function__`** — minimal: enough for the functions the current test
  suite and `_ops` exercise; everything else raises a clear `TypeError`.
- **`__array__`** — delegates to `to_numpy` (so `np.asarray` works on
  equal-length ragged), else raises.

## Section 7 — `to_packed` / `to_padded`

`_ops.to_packed` / `to_padded` already operate on the flat `(data, offsets)`
representation with Numba kernels and only touch `Ragged` through
`_ensure_parts` / `.offsets` / `.data` / `from_offsets`. **Adapt them to read
`RaggedLayout` instead of awkward parts**; the Numba pack/pad kernels themselves
are unchanged. The record branch in `to_packed` is deferred to Spec B (raises
`NotImplementedError` in Spec A). `Ragged.to_packed` stays a thin delegator.

The single-level row-slice produced by `__getitem__` may yield `(2, N)`
start/stop offsets; `to_packed` already handles that path (`_pack_parts`), so
gather-then-pack works unchanged.

## Section 8 — Rust kernels (`src/ragged.rs`)

New module registered alongside `_k_shuffle` / `_tokenize` / `_translate`:

- **`ragged_index` / `ragged_slice`** — given a 1-D offsets array and an
  index/slice/mask/int-array, return selected `(starts, stops)` (and flat span
  for a scalar index). Shape-agnostic over leading dims.
- **`ragged_pack`** — single-level gather of `(2, N)` or `(N+1,)` offsets +
  flat buffer into a contiguous zero-based buffer. (Initially we may keep the
  existing Numba `_pack`; the Rust version is added if benchmarks justify it —
  the data model does not depend on which runs.)
- **`ragged_validate`** — monotonic offsets, in-bounds, segment-count vs
  leading-dim product, contiguous ragged run. Powers constructor fast-fail.

Contract mirrors `kshuffle`: contiguous arrays in, `IxDyn` where shape-agnostic,
no Python objects crossing the boundary. Pure-Python equivalents land first so
the Python layer and tests are independent of the Rust build; Rust replaces the
hot paths once parity tests pass.

## Section 9 — Testing (TDD, differential against awkward)

Awkward is still installed in Spec A, so the **old awkward-backed `Ragged` is
the correctness oracle**:

- **Differential property tests (Hypothesis):** generate single-level inputs
  (numeric + bytes, with/without trailing regular dims, empty, contiguous and
  sliced/gathered) and assert the new `Ragged` matches the old awkward `Ragged`
  on every in-scope property and op (`data`, `offsets`, `shape`, `lengths`,
  indexing results, ufunc results, `to_packed`, `to_padded`, `to_numpy`).
- **Existing `tests/` suite** for the single-level paths must pass against the
  new implementation.
- **String-leaf tests:** the `(N,)`-vs-`(N, None)` shape change is asserted
  explicitly (flat collection → `(N,)`; strings-under-one-axis → `(n, None)`),
  with the byte-offsets round-tripping through `from_offsets`/`.data`.
- **Rust unit tests** per kernel: index/slice/mask math, pack, validate.
- **Edge cases:** empty input; single sliced/gathered (non-contiguous) array →
  pack; record input → `NotImplementedError` (Spec B); `>1` ragged level →
  `NotImplementedError` (Spec C).

## Section 10 — Migration & compatibility within Spec A

- Public constructor/property/method signatures are **unchanged** except the
  documented byte-collection `.shape` change (`(N, None)` → `(N,)`).
- `to_ak` is retained as a shim (builds an awkward array from the layout) so
  any internal/test interop keeps working until Spec D.
- `_ops.py`, `rag/__init__.py` exports unchanged in surface.
- `skills/seqpro/SKILL.md` / `docs/ragged.md`: **not** updated in Spec A (the
  byte-collection shape change is documented in Spec D's cutover, when the
  awkward dependency is actually removed and the user-facing story stabilizes).
  Spec A is an internal re-backing; the public sequence/record API behavior is
  otherwise preserved. *(If Spec A ships the `.shape` change in a way users can
  observe before Spec D, revisit this and update the skill in the same PR per
  the repo's skill-update rule.)*

## Open questions deferred to later specs

- Exact native `zip`/record constructor surface (Spec B).
- Nested-offset indexing semantics for `R ≥ 2` (Spec C).
- Whether `ragged_pack` moves to Rust or stays Numba (decided by Spec A/D
  benchmarks).
