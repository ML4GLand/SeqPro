# Design: Rust-native `Ragged` — Spec B (records / struct-of-arrays)

**Date:** 2026-06-20
**Status:** Approved design, pending implementation plan
**Epic:** [Rust-native Ragged](../../roadmap/rust-ragged.md) — Spec B of 4
**Branch:** `feat/rust-ragged`

## Goal

Add native record (struct-of-arrays) support to the Rust-native `Ragged` built in
[Spec A](2026-06-19-rust-ragged-core-design.md), at parity with today's
awkward-backed record API — without awkward. A record `Ragged` is a set of named
fields that **share one ragged offsets structure**, each field carrying its own
flat data buffer (and, for string fields, its own `str_offsets`). This replaces
the `ak.zip` / `RecordArray` machinery with a small Python value object plus the
Spec A per-field kernels.

Like Spec A, this is an **internal** build: record support lands on the new
`rag/_core.py` path and is **differential-tested against the still-installed
awkward `Ragged` as the correctness oracle**. The public `seqpro.rag.Ragged`
keeps pointing at the awkward type; the public swap, awkward removal, and
`tokenize`/`translate` adaptation are all deferred to Spec D.

See the [roadmap](../../roadmap/rust-ragged.md) for the locked epic-wide
decisions. This spec applies them to single-level records (R = 1).

## Scope

**In scope (single ragged level, records):**
- New `RecordLayout` value object; `Ragged._layout` becomes
  `RaggedLayout | RecordLayout`.
- Constructors: `Ragged.from_fields` (canonical) + `seqpro.rag.zip` (alias).
- Record-branch properties: `data` (dict), `dtype` (structured descriptor),
  `offsets`/`shape` (shared), `fields` (names), `lengths`, `is_empty`,
  `is_contiguous`, `is_base`.
- Field access (`rag['f']` / `rag.f`, zero-copy shared offsets) and mutation
  (`rag['f'] = ...`).
- Row-axis indexing: `slice`/`mask`/`int-array` → record `Ragged`; `int` → dict.
- Methods: `squeeze`, `reshape`, record-aware `to_packed`, and `to_numpy` /
  `to_padded` returning per-field dicts (SoA).
- `_ingest` bridge: awkward record layout ↔ `RecordLayout` (oracle interop).

**Out of scope:**
- Nested records, fields that are themselves records, and R ≥ 2 raggedness
  (Spec C). These raise `NotImplementedError` pointing at Spec C.
- Removing awkward from dependencies, swapping the public `Ragged`, and adapting
  `tokenize`/`translate` + docs/skill (Spec D). Awkward stays installed as the
  oracle.
- `view` and element-wise ufuncs on records (ambiguous across heterogeneous
  fields) — `NotImplementedError` pointing to per-field access.

## Section 1 — Data model

A new value object holds a record's buffers; it composes Spec A `RaggedLayout`s
rather than re-implementing leaf storage.

```
RecordLayout:
    shape:   tuple[int|None,...]      # the shared ragged structure, e.g. (N, None)
    offsets: list[NDArray]            # ONE shared offsets object. Spec B: len == 1.
    fields:  dict[str, RaggedLayout]  # insertion-ordered; every field's
                                      # RaggedLayout.offsets[0] IS the same shared object.
```

- **Each field is a single-level `RaggedLayout`.** Numeric fields:
  `str_offsets is None`, flat (or `(total, *trailing)`) `data`. String fields:
  `data` is the `S1` buffer, `str_offsets` is the per-element byte boundaries.
  Mixed string + numeric fields at one level share identical axes and the same
  ragged `offsets` — this falls out of the model with no special casing.
- **`Ragged._layout` is `RaggedLayout | RecordLayout`.** Methods branch on
  `isinstance(self._layout, RecordLayout)`; the non-record path is exactly Spec A.
- **Invariant (validated once, at construction):** every field's
  `offsets[0]` is the *same object* as `RecordLayout.offsets[0]`, and every
  field's `shape` equals `RecordLayout.shape`.
- **Field order** is dict insertion order, preserved everywhere a record is
  enumerated (`.data`, `.dtype`, `.fields`, `to_*` dicts).

`validate_layout` gains a `RecordLayout` arm: non-empty fields, shared-offsets
identity, per-field shape agreement, and each field individually validated via
the existing single-level checks. Nested/record-of-record fields → Spec C
`NotImplementedError`.

## Section 2 — Constructors

- **`Ragged.from_fields(fields: dict[str, Ragged]) -> Ragged`** — canonical.
  - Rejects an empty dict.
  - Requires each value to be a **single-field (non-record) `Ragged`**
    (record-of-record → Spec C `NotImplementedError`).
  - Requires all fields' offsets to be **array-equal and same shape**; raises a
    clear error otherwise.
  - **Canonicalizes to one shared offsets object** — the first field's — so the
    zero-copy `rag['a'].offsets is rag['b'].offsets` contract holds. Field data
    buffers are referenced, not copied.
  - Returns a `Ragged` wrapping a `RecordLayout`.
- **`seqpro.rag.zip(fields)`** — thin module-level alias for `from_fields`,
  exported from `rag/__init__.py`. Eases the downstream `ak.zip(...)` →
  `sp.rag.zip(...)` migration (one code path; no separate behavior).

Both front-load validation per project convention (fast-fail, single obvious
check).

## Section 3 — Properties (record branch)

Direct reads off `RecordLayout`; non-record behavior is unchanged from Spec A.

| Property | Record return |
|---|---|
| `data` | `dict[str, NDArray]` — zero-copy per-field buffers, insertion order. |
| `dtype` | numpy **structured** dtype `[(name, field_dtype), …]`. |
| `offsets` | the shared offsets `NDArray`. |
| `shape` | the shared `shape` tuple. |
| `fields` | `list[str]` of field names (insertion order). |
| `lengths` | `np.diff` of shared offsets, reshaped to leading dims (as Spec A). |
| `is_empty` / `is_contiguous` / `is_base` | computed off shared offsets; the data-contiguity / ownership checks fold over **all** fields. |

**`dtype` is a descriptor, not a layout.** The structured dtype is used purely as
a concise, serializable, numpy-compatible carrier of field→dtype information. The
**actual memory layout is SoA (a dict of independent buffers), not AoS.** This is
documented loudly on the property; callers must not assume packed-struct memory.

**`.parts` is dropped.** Spec A's `_core.py` already omits it; the old
`dict[str, RagParts]` form has no analog now that `RagParts` is gone, and
`.data` + `.offsets` + `.dtype` cover its uses. *This diverges from the roadmap's
Spec B line, which lists `.parts`; the roadmap should get a one-line edit
recording the drop.*

## Section 4 — Field access & mutation

- **`rag['field']` / `rag.field`** → a zero-copy single-field `Ragged` whose
  `offsets is rag.offsets` (the shared object) and whose layout is the field's
  `RaggedLayout`. Unknown field → `KeyError` / `AttributeError`.
- **`rag['field'] = new_ragged`** → replace an existing field or add a new one.
  Validates `new_ragged` is a single-field `Ragged` whose offsets are array-equal
  to the record's, canonicalizes it onto the shared offsets object, and returns
  via copy-on-write of the `RecordLayout.fields` dict (underlying buffers shared,
  not copied). This keeps the documented update idiom
  (`rag['f'] = rag['f'].view(...)`) working.

## Section 5 — Row-axis indexing

`__getitem__` distinguishes a string key (field access, §4) from a row selector
on the leading/ragged axis:

- **`rag[slice]` / `rag[mask]` / `rag[int_array]`** → a record `Ragged`. The
  start/stop gather (`_ragged_select`, Spec A) is computed **once** from the
  shared offsets and applied to every field, so fields stay aligned and
  zero-copy; the result shares one new `(2, N)` offsets object across fields.
- **`rag[int]`** → `dict[str, value]`: the struct at that row, each value being
  the field's row (an `NDArray` for numeric, `bytes` for a string field) via the
  field's Spec A integer-index path.

## Section 6 — Methods on records

- **`squeeze` / `reshape`** → dispatch per field with the shared offsets/shape
  adjusted once; zero-copy; return a record `Ragged`. (When a squeeze collapses a
  field to a bare 1-D array, the record still returns a record `Ragged` over the
  adjusted fields — record structure is preserved.)
- **`to_packed(copy=True)`** → record-aware. Compute the pack gather once from
  the shared offsets, pack each field's data buffer (reusing the Spec A
  single-level `_pack_parts` per field), and emit **one** new shared, zero-based
  1-D offsets object referenced by every field. Python orchestrates; no new Rust.
- **`to_numpy` / `to_padded`** → `dict[str, NDArray]`, one densified / padded
  array per field, **preserving SoA** (not a structured array). **Raise if any
  field is a string leaf** (densifying/padding a variable-width string leaf into
  a per-field dict element is out of scope); the message points to per-field
  handling. Equal-length / padding semantics per field match Spec A.
- **`view`** and **`__array_ufunc__`** → `NotImplementedError` on records,
  message pointing to per-field access (a single dtype reinterpretation or
  element-wise op across heterogeneous fields is ill-defined).

## Section 7 — Rust kernels

**No new kernels.** Record operations reuse Spec A's `_ragged_select` (row
gather), the single-level pack, and `_ragged_validate`, invoked per field with
the shared offsets. `src/ragged.rs` is unchanged. This mirrors the epic's
Python-orchestration / Rust-leaf-compute boundary: the record layer is pure
Python layout algebra over Spec A primitives.

## Section 8 — `_ingest` bridge (oracle interop)

Awkward stays installed in Spec B as the differential oracle, so the bridge must
round-trip records:

- **`layout_from_ak`** learns to detect a record layout
  (`ak.fields(arr)` non-empty) and build a `RecordLayout`: extract the shared
  list offsets once, unbox each field to a `RaggedLayout` pointed at that shared
  offsets object, preserving field order. (Today this raises "Spec B".)
- **`to_ak`** learns the reverse: assemble a `RecordArray` of per-field contents
  over the shared offsets.

These exist only for testing/interop and are removed/retired in Spec D.

## Section 9 — Testing (TDD, differential against awkward)

The old awkward-backed `Ragged` remains the correctness oracle.

- **Differential property tests (Hypothesis):** generate single-level record
  inputs (2–3 fields; numeric-only, and mixed string + numeric; with/without
  trailing regular dims; empty, contiguous, and sliced/gathered). Build the same
  record both natively (`from_fields`) and via `ak.zip`, and assert parity on
  `.data`, `.dtype`, `.offsets`, `.shape`, `.fields`, `.lengths`, field access,
  row `slice`/`mask`/`int-array`/`int`, `squeeze`, `reshape`, `to_packed`, and
  the per-field `to_numpy`/`to_padded` dicts.
- **Zero-copy contracts:** `rag['a'].offsets is rag['b'].offsets is rag.offsets`;
  field `.data` buffers are views (`base is not None`); `to_packed` produces one
  shared offsets across fields.
- **Port existing cases:** `TestRecordRagged` and `TestToPackedRecord` (incl. the
  indexed-record / sliced-then-fielded cases) run against the native path.
- **Edge cases / raises:** empty field dict → raise; offset-mismatched fields →
  raise; `to_numpy`/`to_padded` on a record with a string field → raise; `view` /
  ufunc on a record → `NotImplementedError`; nested / record-of-record → Spec C
  `NotImplementedError`.
- **Rust:** no new kernels, so no new Rust unit tests; existing Spec A kernel
  tests cover the per-field calls.

**Exit:** genoray + single-level GVL / GVF record cases work on the native path;
the record differential suite is green with awkward as oracle.

## Section 10 — Migration & compatibility within Spec B

- Public `seqpro.rag.Ragged` still points at the awkward type; this work is
  internal to the `_core.py` path. No user-facing behavior changes ship in
  Spec B (the public swap is Spec D).
- `seqpro.rag.zip` is added to `rag/__init__.py` exports (new public name, but it
  operates on the new core path; downstream adoption happens at the Spec D
  cutover).
- `skills/seqpro/SKILL.md` / `docs/ragged.md`: **not** updated in Spec B (no
  observable public surface change yet; the record story is documented at the
  Spec D cutover when awkward is removed and the public type swaps). *If `rag.zip`
  becomes observably usable by users before Spec D, revisit and update the skill
  in the same PR per the repo's skill-update rule.*

## Decision log (this spec)

- **2026-06-20** — Record model: separate `RecordLayout` composing per-field
  `RaggedLayout`s with a shared offsets object (vs. extending `RaggedLayout` or a
  dict-of-`Ragged` wrapper).
- **2026-06-20** — Constructor: `from_fields` canonical + `rag.zip` alias;
  fields must be offset-equal, canonicalized to one shared object.
- **2026-06-20** — `.parts` dropped (diverges from roadmap wording; roadmap to be
  amended). `.dtype` returns a structured dtype as a *descriptor only*,
  documented as SoA-not-AoS.
- **2026-06-20** — `__setitem__` field mutation kept. Row `int` index → dict;
  `slice`/`mask` → record `Ragged`. `to_numpy`/`to_padded` → per-field dict (SoA),
  raising on string fields. `view`/ufunc raise on records.
- **2026-06-20** — No new Rust; records reuse Spec A kernels per field.

## Open questions deferred to later specs

- Nested-record and R ≥ 2 record indexing/pack semantics (Spec C).
- Whether `rag.zip` / `from_fields` should also accept raw `(lengths, buffers)`
  inputs (deferred; Ragged-input form is sufficient for the consumer cases).
- Final public record surface and docs/skill updates at the awkward cutover
  (Spec D).
