# Design: Rust-native `Ragged` — Spec B (string/char duality + records)

**Date:** 2026-06-20
**Status:** Approved design, pending implementation plan
**Epic:** [Rust-native Ragged](../../roadmap/rust-ragged.md) — Spec B of 4
**Branch:** `feat/rust-ragged`

## Goal

Two things, delivered together because the second depends on the first:

1. **Core string/char duality** (a prerequisite refinement to the landed Spec A
   core): distinguish an **opaque string** (`'S'`, length not an axis) from
   **ascii chars** (`'S1'`, length is a ragged axis), with zero-copy conversions.
2. **Native record / struct-of-arrays `Ragged`** at parity with today's
   awkward-backed record API — without awkward. A record is a set of named fields
   that **share one ragged offsets object and the same shape**, each field a
   numeric or `S1`-char leaf with its own data buffer. Replaces `ak.zip` /
   `RecordArray` with a small Python value object over the Spec A kernels.

Like Spec A, this is an **internal** build: it lands on the new `rag/_core.py`
path and is **differential-tested against the still-installed awkward `Ragged` as
the correctness oracle**. The public `seqpro.rag.Ragged` keeps pointing at the
awkward type; the public swap, awkward removal, and `tokenize`/`translate`
adaptation are all deferred to Spec D.

See the [roadmap](../../roadmap/rust-ragged.md) for the locked epic-wide
decisions. This spec applies them to single ragged level (R = 1).

**SSoT compliance.** The roadmap is the single source of truth for this epic; all
Rust-`Ragged` work must read it and update it in the same PR. This spec obeys that
directive: the roadmap's locked decisions, Spec B/C entries, and decision log were
amended (2026-06-20) for the string/char duality, the `.parts` drop, and the
record-scope tightening. Any PR implementing this spec must likewise update the
roadmap's status when Spec B lands.

## Scope

**In scope:**
- **Core string/char model:** opaque `'S'` vs `'S1'` chars; `.dtype`-as-descriptor;
  zero-copy `to_chars()` / `to_strings()`; constructor defaults.
- **Records:** new `RecordLayout`; `Ragged._layout` becomes
  `RaggedLayout | RecordLayout`.
- Constructors `Ragged.from_fields` (canonical) + `seqpro.rag.zip` (alias).
- Record-branch properties: `data` (dict), `dtype` (structured descriptor),
  `offsets`/`shape` (shared), `fields` (names), `lengths`, `is_empty`,
  `is_contiguous`, `is_base`.
- Field access (`rag['f']` / `rag.f`, zero-copy shared offsets) and mutation
  (`rag['f'] = ...`).
- Row-axis indexing: `slice`/`mask`/`int-array` → record `Ragged`; `int` → dict.
- Methods: `squeeze`, `reshape`, record-aware `to_packed`, per-field
  `to_numpy`/`to_padded` dicts.
- `_ingest` bridge: awkward record layout ↔ `RecordLayout` (oracle interop).

**Out of scope:**
- **Opaque `'S'` strings as record fields.** In Spec B, `'S'` strings are a
  standalone single-field `Ragged` only. A record field that is an opaque string
  **under** a shared ragged axis (the alleles `(…, ~variants)` + per-allele
  `str_offsets`, a 2-offset-array leaf) is **Spec C**.
- Nested records, fields that are themselves records, R ≥ 2 (Spec C) →
  `NotImplementedError` pointing at Spec C.
- Removing awkward, swapping the public `Ragged`, adapting `tokenize`/`translate`
  + docs/skill (Spec D). Awkward stays installed as the oracle.
- `view` and element-wise ufuncs on records → `NotImplementedError`.

## Section 1 — Core string/char model (prerequisite, modifies Spec A)

Physical storage is unchanged: a flat single-byte (`S1`) buffer + offsets. The
interpretation of variable-length ASCII is chosen by **dtype as a semantic
descriptor** (decoupled from the storage buffer — the same descriptor-not-storage
pattern records use for their structured dtype).

| | opaque **string** | ascii **chars** |
|---|---|---|
| `.dtype` | `np.dtype('S')` (itemsize 0) | `np.dtype('S1')` (itemsize 1) |
| `.shape` | `(N,)` — length **not** an axis | `(N, ~length)` — length **is** a ragged axis |
| length boundaries in | `str_offsets` (uncounted leaf) | `offsets` (counted axis) |
| layout fields | `offsets == []`, `str_offsets` set | `offsets == [real]`, `str_offsets is None` |
| internal `is_string` | `True` | `False` (an `S1` numeric-like leaf) |
| `rag[i]` | `bytes` scalar (`b'ATG'`) | `S1` row array (`[b'A', b'T', b'G']`) |

- **`.dtype` special-case:** when `is_string` (opaque), `.dtype` returns
  `np.dtype('S')` regardless of the `S1` storage buffer; `.data` still returns the
  `S1` bytes. Otherwise `.dtype` returns `data.dtype` (so chars report `S1`).
- **Zero-copy conversions** (no data movement — only an offsets/dtype retag):
  - **`to_chars() -> Ragged`** — opaque `'S'` `(N,)` → `'S1'` `(N, None)`. Promotes
    `str_offsets` to the ragged `offsets`; same `S1` data buffer; `str_offsets`
    cleared. Raises if called on a non-opaque (`is_string is False`) Ragged.
  - **`to_strings() -> Ragged`** — `'S1'` `(N, None)` → opaque `'S'` `(N,)`. Demotes
    `offsets` to `str_offsets`; same data buffer. Requires a **1-D `S1` leaf** (no
    trailing regular dims, single ragged level); raises otherwise.
- **Constructor default:** `from_lengths(S1_data, lengths)` → **opaque string**
  `(N,)` (matches current Spec A behavior, now reporting `np.dtype('S')`).
  `.to_chars()` is the switch to the `(N, None)` char view. `from_offsets` with an
  explicit `None` in the shape and `S1` data builds the **char** layout (length is
  an axis); `from_offsets` with a no-`None` shape and `S1` data builds the opaque
  leaf. The disambiguating rule (presence of `None` ⇒ chars; absence ⇒ opaque) is
  documented and validated in one place (replaces Spec A's unconditional collapse).
- numpy floor stays `>=1.26` (drives the `'S'`/`'S1'` choice over `StringDType`).

This refinement touches `_layout.py` (dtype reporting helper, `is_string` already
present), `_core.py` (`dtype` property, `to_chars`/`to_strings`, constructor
disambiguation, `__getitem__` int return), and `_ingest.layout_from_ak` (a single
list-axis `S1` ak input maps to **opaque** by default, matching `from_lengths`).

## Section 2 — Record data model

A new value object holds a record's buffers; it composes Spec A `RaggedLayout`s
rather than re-implementing leaf storage.

```
RecordLayout:
    offsets: list[NDArray]            # ONE shared offsets object. Spec B: len == 1.
    shape:   tuple[int|None,...]      # canonical ragged shape, e.g. (N, None)
    fields:  dict[str, RaggedLayout]  # insertion-ordered; every field's
                                      # offsets[0] IS the same shared object.
```

- **Each field is a single-level `RaggedLayout`** that is **numeric or `S1` chars**
  — i.e. every field has real `offsets` (`str_offsets is None`) and shape
  `(*leading, None, *trailing?)`. (Opaque `'S'` fields are out of scope, §Scope.)
- **Invariant (validated once, at construction):** every field's `offsets[0]`
  **is the same object** as `RecordLayout.offsets[0]`, and every field's
  ragged shape (`shape[:rag_dim+1]`) agrees with `RecordLayout.shape`. Fields may
  differ only in **trailing regular dims** (e.g. one field `(N, None)` scalar,
  another `(N, None, 4)` OHE). `RecordLayout.shape` is the first field's full shape.
- **`Ragged._layout` is `RaggedLayout | RecordLayout`.** Methods branch on
  `isinstance(self._layout, RecordLayout)`; the non-record path is exactly Spec A.
- **Field order** is dict insertion order, preserved everywhere a record is
  enumerated (`.data`, `.dtype`, `.fields`, `to_*` dicts).

`validate_layout` gains a `RecordLayout` arm: non-empty fields, every field
numeric/char (not opaque `'S'`, not a nested record → else Spec C
`NotImplementedError`), shared-offsets identity, ragged-shape agreement, and each
field individually validated via the existing single-level checks.

## Section 3 — Constructors

- **`Ragged.from_fields(fields: dict[str, Ragged]) -> Ragged`** — canonical.
  - Rejects an empty dict.
  - Requires each value to be a **single-field, non-record, non-opaque** `Ragged`
    (record-of-record / opaque `'S'` field → `NotImplementedError` pointing at the
    relevant scope note). Sequence fields must be passed as chars (`.to_chars()`).
  - Requires all fields' **`.offsets` to be array-equal** and ragged shapes to
    agree (trailing regular dims may differ); raises a clear error otherwise.
  - **Canonicalizes to one shared offsets object** — the first field's `.offsets` —
    rebinding every field's `RaggedLayout.offsets` onto it, so the zero-copy
    `rag['a'].offsets is rag['b'].offsets` contract holds. Data buffers are
    referenced, not copied.
  - Sets `RecordLayout.shape` to the first field's shape. Returns the record `Ragged`.
- **`seqpro.rag.zip(fields)`** — thin module-level alias for `from_fields`,
  exported from `rag/__init__.py`. Eases the downstream `ak.zip(...)` →
  `sp.rag.zip(...)` migration (one code path).

Both front-load validation per project convention (fast-fail, single obvious check).

## Section 4 — Properties (record branch)

Direct reads off `RecordLayout`; non-record behavior is unchanged from Spec A.

| Property | Record return |
|---|---|
| `data` | `dict[str, NDArray]` — zero-copy per-field buffers (`S1` for char fields), insertion order. |
| `dtype` | numpy **structured** dtype `[(name, field_dtype), …]` (char fields appear as `S1`). |
| `offsets` | the shared offsets `NDArray`. |
| `shape` | the canonical ragged `shape` (first field's). |
| `fields` | `list[str]` of field names (insertion order). |
| `lengths` | `np.diff` of shared offsets, reshaped to leading dims (as Spec A). |
| `is_empty` / `is_contiguous` / `is_base` | computed off shared offsets; data-contiguity / ownership checks fold over **all** fields. |

**`dtype` is a descriptor, not a layout.** The structured dtype is a concise,
serializable, numpy-compatible carrier of field→dtype info; the **actual memory
layout is SoA (a dict of independent buffers), not AoS** — documented loudly.

**`.parts` is dropped.** Spec A's `_core.py` already omits it; the old
`dict[str, RagParts]` has no analog (`RagParts` is gone), and `.data` + `.offsets`
+ `.dtype` cover its uses. (Roadmap amended 2026-06-20 to record the drop.)

## Section 5 — Field access & mutation

- **`rag['field']` / `rag.field`** → a zero-copy single-field `Ragged` whose
  `offsets is rag.offsets` and whose layout is the field's `RaggedLayout`. Unknown
  field → `KeyError` / `AttributeError`.
- **`rag['field'] = new_ragged`** → replace an existing field or add a new one.
  Validates `new_ragged` is a single-field numeric/char `Ragged` whose offsets are
  array-equal to the record's, canonicalizes it onto the shared offsets object, and
  returns via copy-on-write of `RecordLayout.fields` (buffers shared, not copied).
  Keeps the documented update idiom (`rag['f'] = rag['f'].view(...)`) working.

## Section 6 — Row-axis indexing

`__getitem__` distinguishes a string key (field access, §5) from a row selector:

- **`rag[slice]` / `rag[mask]` / `rag[int_array]`** → a record `Ragged`. The
  start/stop gather (`_ragged_select`, Spec A) is computed **once** from the shared
  offsets and applied to every field, so fields stay aligned and zero-copy; the
  result shares one new `(2, N)` offsets object across fields.
- **`rag[int]`** → `dict[str, value]`: the struct at that row, each value the
  field's row (`NDArray` for numeric, `S1` row array for a char field) via the
  field's Spec A integer-index path.

## Section 7 — Methods on records

- **`squeeze` / `reshape`** → dispatch per field with the shared offsets/shape
  adjusted once; zero-copy; return a record `Ragged`.
- **`to_packed(copy=True)`** → record-aware. Compute the pack once from the shared
  offsets, pack each field's data buffer (reusing Spec A `_pack_parts` per field),
  and emit **one** new shared, zero-based 1-D offsets object referenced by every
  field. Python orchestrates; no new Rust. `copy=False` passes through iff already
  packed (all fields contiguous, 1-D zero-based offsets), else raises.
- **`to_numpy` / `to_padded`** → `dict[str, NDArray]`, one densified / padded array
  per field, **preserving SoA** (not a structured array). All fields are
  numeric/char (`S1`), so per-field densify/pad follows Spec A directly — **no
  string-raise needed** (there are no opaque `'S'` fields in a Spec B record).
  `to_padded(pad_value, length=None)` applies one `pad_value`/`length` across fields.
- **`view`** and **`__array_ufunc__`** → `NotImplementedError` on records (a single
  dtype reinterpretation / element-wise op across heterogeneous fields is
  ill-defined); message points to per-field access.

## Section 8 — Rust kernels

**No new kernels.** Record operations reuse Spec A's `_ragged_select` (row gather),
the single-level pack, and `_ragged_validate`, invoked per field with the shared
offsets. The string/char conversions are pure offset/dtype retags (no kernel).
`src/ragged.rs` is unchanged.

## Section 9 — `_ingest` bridge (oracle interop)

Awkward stays installed in Spec B as the differential oracle, so the bridge must
round-trip both the string/char distinction and records:

- **`layout_from_ak`** — for a single list-axis `S1` ak input, default to the
  **opaque** layout (matching `from_lengths`); for a record layout
  (`ak.fields(arr)` non-empty), build a `RecordLayout`: extract the shared list
  offsets once, unbox each field to a `RaggedLayout` rebound onto that shared
  offsets object, preserving field order. A field that unboxes to an opaque `S1`
  leaf is re-expressed as a char field for the record (records hold chars).
  (Today this raises "Spec B".)
- **`to_ak`** — assemble a `RecordArray` of per-field contents over the shared
  offsets (e.g. `ak.zip({f: self[f].to_ak() ...}, depth_limit=1)`).

These exist only for testing/interop and are retired in Spec D.

## Section 10 — Testing (TDD, differential against awkward)

The awkward-backed `Ragged` remains the correctness oracle.

**Core string/char:**
- `from_lengths(S1, lengths)` → `.dtype == np.dtype('S')`, `.shape == (N,)`,
  `rag[i]` is `bytes`.
- `to_chars()` → `.dtype == np.dtype('S1')`, `.shape == (N, None)`,
  `rag[i]` is an `S1` array; data buffer identity preserved (zero-copy); offsets is
  the former `str_offsets`.
- `to_strings()` round-trips (`to_chars().to_strings()` ≡ original; offsets/data
  identity); raises on trailing dims / non-char input; `to_chars()` raises on
  non-opaque input.
- Differential: opaque `'S'` collection vs awkward `(N, None)` bytes oracle —
  `.data`/`.offsets` equal, `.shape` is the documented `(N,)` divergence.

**Records (Hypothesis differential):**
- Generate single-level record inputs (2–3 fields; numeric-only and numeric+chars;
  with/without trailing regular dims; empty, contiguous, and sliced/gathered).
  Build natively (`from_fields`, sequences via `.to_chars()`) and via `ak.zip`;
  assert parity on `.data`, `.dtype`, `.offsets`, `.shape`, `.fields`, `.lengths`,
  field access, row `slice`/`mask`/`int-array`/`int`, `squeeze`, `reshape`,
  `to_packed`, and per-field `to_numpy`/`to_padded`.
- **Zero-copy contracts:** `rag['a'].offsets is rag['b'].offsets is rag.offsets`;
  field `.data` are views (`base is not None`); `to_packed` produces one shared
  offsets across fields.
- **Port existing cases:** `TestRecordRagged` and `TestToPackedRecord` (incl. the
  indexed-record / sliced-then-fielded cases) against the native path.
- **Edge cases / raises:** empty field dict → raise; offset-mismatched fields →
  raise; opaque `'S'` field passed to `from_fields` → `NotImplementedError`
  (Spec C pointer); `view`/ufunc on a record → `NotImplementedError`; nested /
  record-of-record → Spec C `NotImplementedError`.
- **Rust:** no new kernels, no new Rust unit tests; Spec A kernel tests cover the
  per-field calls.

**Exit:** genoray + single-level GVL / GVF record cases work on the native path
(sequences expressed as chars); the string/char + record differential suites are
green with awkward as oracle.

## Section 11 — Migration & compatibility within Spec B

- Public `seqpro.rag.Ragged` still points at the awkward type; this work is
  internal to the `_core.py` path. No user-facing behavior changes ship (the public
  swap is Spec D).
- `seqpro.rag.zip` is added to `rag/__init__.py` exports (operates on the new core
  path; downstream adoption happens at the Spec D cutover).
- The string/char `.dtype`/`.shape` reporting and `to_chars`/`to_strings` live on
  the new `_core.Ragged` only; the awkward public type is untouched.
- `skills/seqpro/SKILL.md` / `docs/ragged.md`: **not** updated in Spec B (no
  observable public surface change yet; documented at the Spec D cutover). *If
  `rag.zip` / the string-char API becomes observably usable by users before Spec D,
  revisit and update the skill in the same PR per the repo's skill-update rule.*

## Decision log (this spec)

- **2026-06-20** — Record model: separate `RecordLayout` composing per-field
  `RaggedLayout`s with a shared offsets object (vs. extending `RaggedLayout` or a
  dict-of-`Ragged` wrapper).
- **2026-06-20** — Constructor: `from_fields` canonical + `rag.zip` alias; fields
  offset-equal, canonicalized to one shared object.
- **2026-06-20** — `.parts` dropped; `.dtype` (record) is a structured descriptor,
  SoA-not-AoS.
- **2026-06-20** — `__setitem__` mutation kept. Row `int` → dict; `slice`/`mask` →
  record `Ragged`. `to_numpy`/`to_padded` → per-field dict (SoA). `view`/ufunc raise.
- **2026-06-20** — No new Rust; records reuse Spec A kernels per field.
- **2026-06-20** — **String/char duality** (supersedes the original record
  invariant correction). ASCII has two interpretations: opaque `'S'` (`(N,)`,
  `str_offsets`) and `'S1'` chars (`(N, ~length)`, real `offsets`), distinguished by
  `.dtype` as a descriptor. Zero-copy `to_chars`/`to_strings`. `from_lengths(S1)`
  defaults to opaque. `np.dtype('S')` chosen over `StringDType`/`np.str_` to keep
  the numpy `>=1.26` floor. **Consequence:** record fields that length-align with
  numeric siblings are chars (same shape, one shared offsets), so the record
  invariant is **shared offsets + same shape** and the prior
  "shared-offsets-not-shape" correction is withdrawn. **Opaque `'S'` strings are
  standalone-only in Spec B**; opaque-string-under-an-axis (alleles) → Spec C.

## Open questions deferred to later specs

- Opaque-string-under-a-ragged-axis leaf (2-offset-array field) for records;
  nested-record and R ≥ 2 indexing/pack semantics (Spec C).
- Whether `rag.zip` / `from_fields` should also accept raw `(lengths, buffers)`
  inputs (deferred; Ragged-input form suffices for the consumer cases).
- Final public string/char + record surface and docs/skill updates at the awkward
  cutover (Spec D).
