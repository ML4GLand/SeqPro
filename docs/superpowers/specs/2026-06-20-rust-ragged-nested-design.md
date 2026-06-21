# Design: Rust-native `Ragged` — Spec C (nested raggedness + string-under-axis)

**Date:** 2026-06-20
**Status:** Approved design, pending implementation plan
**Epic:** [Rust-native Ragged](../../roadmap/rust-ragged.md) — Spec C of 4
**Branch:** `feat/rust-ragged`

## Goal

Generalize the landed single-level core (Spec A/B) to **two nested ragged levels
(R = 2)** plus the **opaque-string-under-a-ragged-axis** leaf, so the three
doubly-ragged consumer cases work natively without awkward:

| Case | Shape | Form | Where |
|---|---|---|---|
| Alt/ref alleles | `(batch, ploidy, ~variants)` `'S'` | 1 ragged + string-under-axis | GVL `_haps.py` / `_flat_variants.py` |
| Flat variant windows | `(batch, ploidy, ~variants, ~window)` | R = 2 | GVL `_flat_variants.py` |
| Codon annotations | `(regions, ~genes, ~codons)` | R = 2 record | GVF `annotations.py` |

Like Spec A/B this is an **internal** build on the `rag/_core.py` path,
**differential-tested against the still-installed awkward `Ragged` as the
correctness oracle**. The public `seqpro.rag.Ragged` keeps pointing at the
awkward type; the public swap, awkward removal, and `tokenize`/`translate`
adaptation remain Spec D.

See the [roadmap](../../roadmap/rust-ragged.md) for the locked epic-wide
decisions. This spec applies them at R = 2.

**SSoT compliance.** The roadmap is the single source of truth for this epic; all
Rust-`Ragged` work must read it and update it in the same PR. This spec obeys that
directive: the roadmap's decision log was amended (2026-06-20) for the four Spec C
forks decided during brainstorming — the **R = 2 cap** (divergence from the
roadmap's "arbitrary depth" framing), **full numpy-style nested indexing**,
**per-axis `to_padded`**, and **string-under-axis as a record field**. Any PR
implementing this spec must likewise update the roadmap's status when Spec C lands.

## Scope

**In scope:**
- **Nested raggedness at R = 2**: a 2-element nested offsets list `[O0, O1]`;
  nested `validate_layout`; nested constructors (`from_offsets` list form,
  `from_lengths` nested form).
- **Full numpy-style nested indexing**: positional logical-axis selectors,
  including per-group inner selection (`rag[:, k]`, `rag[:, a:b]`, `rag[:, mask]`).
- **Opaque-string-under-a-ragged-axis** leaf (non-empty `offsets` **and**
  `str_offsets`), standalone **and** as a record field; nested `to_chars()` /
  `to_strings()` that promote/demote `str_offsets ↔ innermost offset level`.
- **Nested records**: `RecordLayout` fields may be R = 2 and/or string-under-axis,
  sharing the full offsets list, each field keeping its own `str_offsets`.
- **Nested methods**: `to_packed` (record-aware), per-axis `to_padded`,
  `to_numpy`, `squeeze`, `reshape`.
- **Rust**: two new kernels — nested-pack and nested-gather. Reuse `select` /
  `validate` for outer/lazy paths.
- `_ingest` bridge round-tripping R = 2 + string-under-axis (incl. records).

**Out of scope:**
- **R ≥ 3.** Three or more nested ragged levels → `NotImplementedError`. Decided
  against the roadmap's "arbitrary depth" framing by YAGNI: every surveyed
  consumer tops out at R = 2 (see roadmap Shape survey). Algorithms are written so
  generalizing later is mechanical, but the contract is R ≤ 2.
- **Record-of-record** (a field that is itself a record) → `NotImplementedError`.
  Spec C generalizes *ragged depth*, not record nesting.
- Removing awkward, swapping the public `Ragged`, adapting `tokenize`/`translate`
  + docs/skill (Spec D). Awkward stays installed as the oracle.
- `view` and element-wise ufuncs on records (already raise from Spec B).

## Section 1 — Nested data model

`RaggedLayout` is unchanged in shape; the generalization is that `offsets` is now
a **length-`R` list**, outermost-first (already typed `list[NDArray]` since Spec
A). For R = 2:

```
RaggedLayout:
    data:        NDArray            # O1[-1] elements (1-D, or 2-D with trailing)
    offsets:     [O0, O1]           # nested, outermost-first
    shape:       (*leading_int, None, None, *trailing_int)   # count(None) == 2
    str_offsets: NDArray | None     # per-element byte boundaries (string leaf)
```

- **`O0`** (level 0): length `L0 + 1` where `L0 = ∏ leading_int` (1 if no leading
  dims). `O0[i]:O0[i+1]` is the range of **middle segments** for outer row `i`.
- **`O1`** (level 1): length `total_middle + 1` where `total_middle = O0[-1]`.
  `O1[j]:O1[j+1]` is the range in `data` for middle segment `j`.
- **`data`**: `O1[-1]` elements.

**Logical-axis order** is `(leading_int…, None, None, trailing…)`. The first
`None` is the **middle** ragged axis (`~M`), the second is the **inner** ragged
axis (`~K`). A leading int is the count of top-level rows (`L0`).

### Lazy gather forms (approach A)

Mirroring Spec A's single-level `(2, N)` gather, nested slicing stays zero-copy by
keeping inner offsets **global** and using gather forms only where needed:

- **Outer slice/mask/int-array** → `O0` becomes `(2, L0')`: each selected group
  carries a *contiguous* `[mid_start, mid_stop)` range into the **global** `O1`.
  No data or `O1` movement.
- **Per-group inner int** (`rag[:, k]`) → a data-level `(2, N)` gather (the result
  is R = 1; see §4).
- **Per-group middle mask/int-array** (`rag[:, mask]`) → may break contiguity of
  the middle-index ranges; this path computes a fresh `O0` (and packs when the
  lazy form is not representable) via the nested-gather kernel.

`is_contiguous` / `is_base` require **both** offset levels 1-D, zero-based, and
data C-contiguous (folded over all record fields).

### `validate_layout` (R = 2 arm)

`n_ragged == 2`:
- `len(offsets) == 2`; each level monotonic non-decreasing (handles 1-D and
  `(2, ·)` forms, as Spec A).
- `O0`'s segment count == `∏ leading_int`.
- `O0[-1]` (or max stop of the `(2,·)` form) == `O1`'s segment count.
- `O1[-1]` == `data.shape[0]`.
- Rust `_ragged_validate` is called per level for the 1-D canonical case
  (`(2,·)` lazy forms validated in Python, as Spec A).

`n_ragged >= 3` → `NotImplementedError` (the existing `_SPEC_C_MSG` is replaced
with an R ≥ 3 message).

## Section 2 — String-under-axis leaf + conversions

A leaf with **non-empty `offsets` AND `str_offsets` set** is an opaque string
under a ragged axis: shape `(…, ~var)`, `dtype 'S'`, `is_string == True`. The
standalone opaque string of Spec A/B (`offsets == []`, `str_offsets` set) remains
valid as the zero-real-level special case.

| | standalone string (Spec B) | string-under-axis (Spec C) | chars |
|---|---|---|---|
| `offsets` | `[]` | `[O0]` (one real level) | `[O0, O1]` |
| `str_offsets` | set | set | `None` |
| `.dtype` | `'S'` | `'S'` | `'S1'` |
| `.shape` | `(N,)` | `(*leading, None)` | `(*leading, None, None)` |
| `is_string` | `True` | `True` | `False` |

- **`to_chars()`** promotes `str_offsets` to the **innermost real offset level**:
  appends `None` to `shape`, moves `str_offsets` into `offsets` as the new last
  level, clears `str_offsets`. `(…, ~var)` `'S'` → `(…, ~var, ~len)` `'S1'`
  (R grows by 1; from a standalone string this is the Spec B `(N,) → (N, None)`).
- **`to_strings()`** demotes the **innermost real offset level** to `str_offsets`:
  pops the last `offsets` entry into `str_offsets`, drops the last `None` from
  `shape`. Requires a 1-D `S1` leaf (no trailing regular dims). `(…, ~var, ~len)`
  `'S1'` → `(…, ~var)` `'S'`.
- Both are zero-copy (data buffer identity preserved). Round-trip
  `to_chars().to_strings()` ≡ original.
- After `to_chars()`, a string-under-axis becomes an R = 2 `'S1'` char array
  (the alleles case: `(batch, ploidy, ~variants, ~allele_len)`).

`_build_layout` / `from_offsets` disambiguation generalizes the Spec B rule:
the count of `None` in `shape` is the number of **real** offset levels; an `S1`
buffer whose `shape` has one fewer `None` than offset boundaries available carries
the remainder in `str_offsets` (string-under-axis). Documented and validated in
one place.

## Section 3 — Constructors

- **`from_offsets(data, shape, offsets)`** — `offsets` accepts a **list** of
  arrays (len == number of real ragged levels). R = 2 builds `[O0, O1]`; a single
  array is still accepted for R = 1 (back-compat). `count(None) >= 3` →
  `NotImplementedError`. String-under-axis is built by passing `offsets=[O0]` plus
  the per-element boundaries as `str_offsets` (via the dedicated path / `to_chars`
  inverse), keeping one obvious construction route.
- **`from_lengths`** — nested form: `from_lengths(data, lengths)` where `lengths`
  may be a **nested** structure (outer counts + inner lengths) producing `[O0,
  O1]`. The single-level form is unchanged. Exact accepted `lengths` forms
  (a list-of-arrays vs an awkward-style nested length array) are enumerated in the
  implementation plan; the canonical input is `(outer_counts, inner_lengths)`.
- **`from_fields(fields)` / `rag.zip(fields)`** — fields may now be R = 2 numeric/
  char and/or string-under-axis. The shared object is the **full offsets list**
  (`O0` and `O1` both shared/identity-canonicalized); each field keeps its **own
  `str_offsets`**. Record-of-record / R ≥ 3 fields → `NotImplementedError`.

All constructors front-load validation (fast-fail, single obvious check), per repo
convention.

## Section 4 — Indexing (full nested semantics)

`__getitem__` maps a tuple positionally onto logical axes
`(leading_int…, None, None, trailing…)`. A bare non-tuple key indexes axis 0.
String keys are field access (records, §5).

**Leading-dim and peel:**
- `rag[i]` on a single leading dim → selects that top row → R = 2 sub-array with
  the leading dim removed. With multiple leading dims, peels the outermost.
- Fixing **all** leading dims (e.g. `rag[i]` when `L0` has one leading dim, or
  `rag[i, j]` for two) down to one `L0` segment → a **1-level** Ragged
  `(M_g, ~K)`: the group's middle segments become the new level-0, via a
  zero-copy contiguous slice of `O1` (`O1[O0[g] : O0[g+1] + 1]`). Data shared.
- `rag[g][j]` / `rag[g, j]` → the flat data row (`NDArray`, or `bytes` for a
  string leaf), reusing the Spec A leaf path.

**Per-group inner selection** (selector on the first `None`, `:` on leading):
- `rag[:, k]` (int) → k-th middle segment of every group → `(N, ~K)`, **R = 1**.
  Computed as a data-level `(2, N)` gather: for group `g`, segment index
  `O0[g] + k`, data span `O1[O0[g]+k] : O1[O0[g]+k+1]`. Out-of-range `k` for any
  group → `IndexError`.
- `rag[:, a:b]` (slice) → per-group local slice of the middle axis → **R = 2**
  preserved, lazy (`O0` becomes `(2, L0)` with shrunk ranges; `O1` global).
- `rag[:, mask]` / `rag[:, int_array]` (over the middle axis) → **R = 2** with a
  recomputed `O0` (per-group counts change); uses the **nested-gather kernel**,
  packing when the lazy `(2,·)` form cannot represent the result.

**Outer (top-row) selection:**
- `rag[slice]` / `rag[mask]` / `rag[int_array]` over axis 0 → **R = 2** preserved,
  lazy `O0` as `(2, L0')`, `O1` global, data shared (Spec A's `_row_gather`
  generalized to carry the inner level).

## Section 5 — Records (nested + string-under-axis fields)

- `RecordLayout` fields may be R = 2 numeric/char and/or string-under-axis.
- **Invariant (validated at construction):** every field shares the **full
  offsets list** (`O0` and `O1` identity-equal to the record's), and ragged shapes
  agree across fields; fields may differ in trailing regular dims and in their
  **own `str_offsets`** (uncounted, so it doesn't break the shared-offsets rule).
  This is the natural extension of Spec B's "shared offsets + same shape" —
  string-under-axis fields share the counted axes and carry private byte
  boundaries.
- **Field access / mutation, row indexing** extend per-field over the shared
  nested offsets: field access is zero-copy (shared `[O0, O1]`); `rag[int]`/all-
  leading-fixed → dict of per-field rows; `slice`/`mask`/`int-array` (outer or
  per-group) → record `Ragged` with one shared recomputed offsets list.
- **Record-of-record** and **R ≥ 3 fields** → `NotImplementedError`.

## Section 6 — `to_packed` / `to_padded` / `to_numpy`

- **`to_packed(copy=True)`** → nested pack to canonical zero-based `[O0, O1]` via
  the **nested-pack kernel** (one two-level gather). Record-aware: compute the
  pack once from the shared offsets, pack each field's data, emit **one** shared
  `[O0, O1]` referenced by every field. `copy=False` passes through iff already
  packed (both levels 1-D zero-based, all data C-contiguous), else raises.
  String-under-axis: `str_offsets` is re-based alongside the data pack.
- **`to_padded(pad_value, *, length=None, axis=None)`** → **per-axis** padding.
  - `axis` selects which ragged axis (or axes) to densify; the rest stay ragged.
  - Padding the **inner** axis (`~K`) → a trailing regular dim:
    `(…, ~M, ~K) → (…, ~M, K)` (R drops to 1; reuses the Spec A pad kernel per
    middle segment).
  - Padding the **outer** axis (`~M`) → uniform middle count per group: short
    groups get empty (zero-length) inner slots; result `(…, M, ~K)`.
  - Padding **both** → fully dense `(…, M, K)` (nested pad).
  - `length` accepts a per-padded-axis spec (scalar for one axis, tuple aligned to
    the padded axes); `None` uses each axis's batch maximum.
  - Records → per-field dict, same `axis`/`length`/`pad_value` across fields.
- **`to_numpy(allow_missing=False)`** → requires the targeted (all, by default)
  ragged axes already rectangular; raises otherwise. Records → per-field dict.

## Section 7 — Rust kernels (`src/ragged.rs`)

Two new kernels; Python orchestrates buffers (roadmap boundary: Python holds
NumPy buffers, Rust does layout algebra).

- **`nested_pack`** — inputs: `O0` (1-D or `(2,·)`), global `O1`, element byte
  span, data bytes. Output: canonical zero-based `O0'`, `O1'`, and packed data
  bytes via a two-level gather (one contiguous read+write per leaf segment).
  Drives `to_packed` and `to_padded(both)`.
- **`nested_gather`** — inputs: `O0`, `O1`, and a per-group middle selection
  (`mask` or `int-array`). Output: new `O0'` (recomputed group counts) plus the
  selected middle/data ranges (gather form or packed). Drives
  `rag[:, mask]` / `rag[:, int_array]`.

The outer-row gather (`rag[slice|mask|int_array]`) and per-group inner int
(`rag[:, k]`) reuse the existing `select` plus index arithmetic in Python; the
existing `validate` is called per level. Kernels follow the repo's `wide`/strict
hygiene conventions; no naive per-segment Python/NumPy copies on the hot path.

## Section 8 — `_ingest` bridge (oracle interop)

Awkward stays installed as the differential oracle; the bridge round-trips R = 2
and string-under-axis (incl. records):

- **`layout_from_ak`** — a two-list-axis awkward input → `[O0, O1]`
  `RaggedLayout`; a list-axis-over-`S1` input → string-under-axis (real `O0` +
  `str_offsets`) by default, matching `from_lengths`; record layouts build a
  nested `RecordLayout` sharing the extracted `[O0, O1]`, each field unboxed onto
  it (a field unboxing to an opaque `S1`-under-axis leaf is kept as
  string-under-axis).
- **`to_ak`** — assemble nested `ListOffsetArray`s (and `RecordArray` for records)
  from `[O0, O1]`; string-under-axis emits the awkward bytestring-under-list form.

Retired in Spec D.

## Section 9 — Testing (TDD, differential against awkward)

The awkward-backed `Ragged` remains the correctness oracle.

**Nested core (Hypothesis differential):**
- Generate R = 2 inputs (with/without leading dims, with/without trailing regular
  dims; empty, contiguous, sliced/gathered). Build natively (`from_offsets` list /
  `from_lengths` nested) and via awkward; assert parity on `.data`, `.offsets`
  (both levels), `.shape`, `.lengths`, `.dtype`, and every indexing form in §4
  (`rag[i]`, all-leading-fixed peel, `rag[:, k]`, `rag[:, a:b]`, `rag[:, mask]`,
  `rag[:, int_array]`, outer `slice`/`mask`/`int_array`), `to_packed`, per-axis
  `to_padded`, `to_numpy`, `squeeze`, `reshape`.
- **Zero-copy contracts:** outer slice and per-group local slice produce views
  (`base is not None`); inner offsets stay identity-shared where the lazy form
  applies; `to_packed` yields one shared `[O0, O1]` across record fields.

**String-under-axis:**
- Construct `(…, ~var)` `'S'`; `to_chars()` → `(…, ~var, ~len)` `'S1'` (R = 2),
  data identity preserved, `str_offsets` becomes the innermost offset level;
  `to_strings()` round-trips and raises on trailing dims. Differential vs the
  awkward bytestring-under-list oracle.

**Records:**
- Nested records (numeric + char + string-under-axis fields, e.g. alleles
  `{ref, alt}`): parity on field access, row indexing, `to_packed`, per-field
  `to_numpy`/`to_padded`; each field's own `str_offsets` preserved; shared
  `[O0, O1]` identity across fields.

**Consumer cases (exit criteria):** alleles `(batch, ploidy, ~variants)` `'S'`
(+ `to_chars`), flat variant windows `(batch, ploidy, ~variants, ~window)`, codon
annotations `(regions, ~genes, ~codons)` record — all work natively, differential
against awkward.

**Edge cases / raises:** R ≥ 3 → `NotImplementedError`; record-of-record →
`NotImplementedError`; per-group inner int out of range → `IndexError`;
`to_strings()` on trailing-dim leaf → raise; `to_numpy` on a non-rectangular
targeted axis → raise.

**Rust:** unit tests for `nested_pack` (two-level gather correctness, empty
groups, `(2,·)` outer form) and `nested_gather` (mask/int-array, empty result,
recomputed `O0`).

## Section 10 — Migration & compatibility within Spec C

- Public `seqpro.rag.Ragged` still points at the awkward type; this work is
  internal to the `_core.py` path. No user-facing behavior ships (swap is Spec D).
- `from_offsets`'s list form and `from_lengths`'s nested form are additive on the
  new core path; the single-level forms are unchanged.
- `skills/seqpro/SKILL.md` / `docs/ragged.md`: **not** updated in Spec C (no
  observable public surface change yet; documented at the Spec D cutover). *If any
  Spec C API becomes observably usable by users before Spec D, revisit and update
  the skill in the same PR per the repo's skill-update rule.*

## Decision log (this spec)

- **2026-06-20** — **R = 2 cap.** Implement exactly two nested ragged levels;
  R ≥ 3 → `NotImplementedError`. Chosen over the roadmap's "arbitrary depth"
  framing by YAGNI (every surveyed consumer tops out at R = 2). Algorithms kept
  generalizable but the contract is R ≤ 2. *Divergence from the roadmap — logged
  in the roadmap decision log too.*
- **2026-06-20** — **Full numpy-style nested indexing**, including per-group inner
  selection (`rag[:, k]`, `rag[:, a:b]`, `rag[:, mask]`/`int_array`). Requires the
  nested-gather kernel. Positional logical-axis tuple mapping.
- **2026-06-20** — **Lazy nested offsets (approach A).** Inner offsets stay
  global; outer/per-group ops use `(2,·)` gather forms; pack only forced for
  irregular middle gathers. Rejected eager normalization (copies on every slice,
  violates the no-naive-copy hot-path rule).
- **2026-06-20** — **Per-axis `to_padded`.** `axis`/`length` choose which ragged
  axis/axes to densify, leaving the rest ragged (inner → trailing regular dim;
  outer → uniform count; both → fully dense).
- **2026-06-20** — **String-under-axis is standalone + a record field.** Non-empty
  `offsets` + `str_offsets`; each record field keeps its own `str_offsets` (the
  shared-offsets invariant covers only counted axes). `to_chars`/`to_strings`
  promote/demote the innermost real offset level ↔ `str_offsets`.
- **2026-06-20** — **Two new Rust kernels** (nested-pack, nested-gather); outer/
  inner-int paths reuse `select`. Record-of-record stays out of scope.

## Open questions deferred to Spec D

- Removing awkward, the public `Ragged` swap, and `tokenize`/`translate`
  adaptation to nested/string-under-axis shapes.
- Final public nested + string-under-axis surface and docs/skill updates.
- Whether R ≥ 3 is ever needed (revisit only if a consumer shape appears).
