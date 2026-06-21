# Roadmap: Rust-native `Ragged`

**Status:** In progress (epic)
**Started:** 2026-06-19
**Branch:** `feat/rust-ragged`

## ⚠️ This file is the single source of truth (SSoT)

This document is the **authoritative status and decision record** for the
Rust-native `Ragged` epic. Any work on Rust-native `Ragged` (Spec A–D, or any
follow-up) **MUST**:

1. **Read this roadmap first** — locked decisions, scope constraints, and the
   spec sequence below are binding context for all related work.
2. **Update this roadmap as part of the same PR** — when a spec lands, a decision
   changes, scope shifts, or a divergence from an earlier spec is introduced,
   amend the relevant section (spec entry, decision log, or locked decisions) in
   the PR that makes the change. A Rust-`Ragged` PR that does not update this file
   when it should is incomplete.

Per-spec design docs in `docs/superpowers/specs/` hold the detailed design; this
roadmap holds the status, the locked epic-wide decisions, and the decision log.
On conflict, the most recent decision-log entry here wins.

## Motivation

`seqpro.rag.Ragged` is currently a subclass of `awkward.Array` restricted to
**exactly one** ragged dimension. Awkward gives us the nested-offset machinery
(indexing, slicing, ufunc dispatch, packing) for free, but at a cost:

- **Awkward is extremely general** — it models branching, tree-like data with
  unions and arbitrary per-branch raggedness. SeqPro needs none of that
  generality, but pays for it in dependency weight, surprising behaviors, and
  the awkward-subclass plumbing in `_array.py` (behavior patching, layout
  walking, `__typestr__` hacks, indexed-record projection).
- **One ragged axis is not enough.** Real consumers (genoray, GenVarLoader,
  genvarformer) need up to two *nested* ragged levels — e.g. alt/ref alleles
  `(batch, ploidy, ~variants, ~allele_len)` and tokenized variant windows
  `(batch, ploidy, ~variants, ~window)`. Today these live as raw awkward layouts
  *outside* `Ragged`, defeating the abstraction.

This epic replaces the awkward backing with a **Rust-native, non-branching**
ragged array that supports arbitrary nesting depth along a single (linear) path,
plus native struct-of-arrays records — and removes `awkward` from the dependency
tree entirely.

## Scope: non-branching only

We deliberately support **only non-branching** raggedness. Every slice at a given
depth has the **same axis structure** — no siblings with differing depth, no
unions, no tree. Concretely, a `Ragged` is a linear chain of axes:

```
(*leading_regular, None × R, *trailing_regular)
```

- `leading_regular`: zero or more fixed-size batch/grid axes.
- `None × R`: a **contiguous run** of `R ≥ 1` ragged levels (one offsets array each).
- `trailing_regular`: zero or more fixed-size per-element feature axes
  (e.g. OHE `×4`, embedding `×d`).

A **regular axis never sits between two ragged axes** — confirmed empirically
across all three consumer projects (see Shape survey). This single constraint is
what lets us drop awkward's general machinery.

## Locked design decisions

| Decision | Choice |
|---|---|
| Awkward end-state | **Full removal, drop-in.** New type *is* `Ragged`; `awkward` leaves the dependency tree. Public API stays as close to today as possible. |
| Rust/Python boundary | **Python holds the NumPy buffers; Rust does the layout algebra** (nested indexing/slicing math, multi-level pack/pad, validation). ufuncs and zero-copy NumPy interop stay pure-Python. Mirrors the existing kshuffle/tokenize/translate kernel pattern. |
| Axis model | **Ragged run + regular ends.** `(*leading_int, None×R, *trailing_int)`. No regular axis between ragged axes. |
| String/char duality | **ASCII has two intentional interpretations, distinguished by dtype-as-descriptor; storage is a flat single-byte buffer either way.** *Opaque string:* `.dtype == np.dtype('S')` (itemsize 0), shape `(N,)` — the byte-length is **not** an axis (lives in `str_offsets`). *Ascii chars:* `.dtype == np.dtype('S1')` (itemsize 1), shape `(N, ~length)` — the length **is** a ragged axis (real `offsets`). Zero-copy `to_chars()`/`to_strings()` retag between them (promote/demote `str_offsets ↔ offsets`). This lets a sequence either stay opaque (length need not align with siblings on `zip`) or decompose to chars (must align on `~length`). numpy floor stays `>=1.26`. |

### Consequence of the string/char decision

The flat `.data` (S1 buffer) and byte-offsets for a sequence `Ragged` are
*physically unchanged* — `tokenize`/`translate` kernels are byte-identical. What
changes is shape/dtype bookkeeping: an **opaque-string** collection's `.shape`
drops its `None` (`(N, None)` → `(N,)`) and `.dtype` reports `np.dtype('S')`
(decoupled from the `S1` storage buffer — the descriptor-not-storage pattern),
while **chars** are an ordinary `S1` ragged with shape `(N, ~length)`. Records
that must length-align a sequence with numeric siblings use chars (same shape,
one shared offsets); opaque strings are for when length should not align. A
small, contained set of seqpro-owned call sites adapt; no kernel rewrites.
Downstream gets clean string/char constructors + zero-copy conversions.

## Shape survey (genoray / GenVarLoader / genvarformer)

Surveyed 2026-06-19 across `~/projects/{genoray,GenVarLoader,genvarformer}`.

| Pattern | Shape | Ragged levels | Where |
|---|---|---|---|
| Sparse genotypes / dosages / mutcat | `(samples, ploidy, ~variants)` | 1 | genoray `_svar.py` (record: genos+dosages+mutcat) |
| Multi-range read | `(ranges, samples, ploidy, ~variants)` | 1 | genoray `_svar.py` |
| Annotated haplotypes | `(batch, ploidy, ~length)` | 1 | GVL `_ragged.py` (3-field record) |
| Ragged intervals | `(batch, n_tracks, ~intervals)` | 1 | GVL `_ragged.py` (3-field record) |
| Token windows / embeddings | `(N, ~, L)` / `(n_queries, ~, d_emb)` | 1 + trailing regular | genvarformer |
| **Alt/ref alleles** | `(batch, ploidy, ~variants, ~allele_len)` | **1 + string leaf** | GVL `_haps.py` / `_flat_variants.py` |
| **Flat variant windows** | `(batch, ploidy, ~variants, ~window)` | **2** | GVL `_flat_variants.py` |
| **Codon annotations** | `(regions, ~genes, ~codons)` | **2** (record: codon_pos, strand, variant_ptr) | genvarformer `annotations.py` |

**Findings:** raggedness is 1 level almost everywhere, 2 nested levels in three
hot cases; **zero** instances of a regular axis between two ragged axes; records
(SoA) are pervasive and routinely mix string and numeric fields at a shared level
(in the Rust-native model, such sequence fields are expressed as `S1` **chars** so
they share one offsets object with their numeric siblings — see string/char duality).

## Spec sequence

Dependency-ordered. Each spec is a standalone sub-project with its own design
doc → implementation plan → build cycle.

1. **Spec A — Core single-level Rust-native `Ragged`** (this session).
   Replace the `ak.Array` base class. Data model, constructors, properties,
   indexing/slicing, `view`/`squeeze`/`reshape`/`to_numpy`, ufunc + NumPy
   interop, string-leaf semantics, single-level `to_packed`/`to_padded`. Rust:
   single-level index/slice/pack/validate kernels.
   *Exit:* existing single-level, non-record tests pass with awkward removed
   from this path.

2. **Spec B — Core string/char duality + records / struct-of-arrays.**
   *(Landed 2026-06-20 —
   [design doc](../superpowers/specs/2026-06-20-rust-ragged-records-design.md).)*
   **Prerequisite core refinement (modifies Spec A):** the string/char duality —
   opaque `'S'` strings (`(N,)`, `str_offsets`) vs `'S1'` chars (`(N, ~length)`,
   real `offsets`), `.dtype`-as-descriptor, zero-copy `to_chars()`/`to_strings()`,
   `from_lengths(S1, …)` defaults to opaque.
   **Records:** native `RecordLayout` composing per-field `RaggedLayout`s over
   **one shared offsets object with the same shape**, fields **numeric and/or
   `S1` chars**, native `from_fields` / `rag.zip` (replaces `ak.zip`), per-field
   dict `.dtype` (structured, descriptor-only) / `.data`, zero-copy field access +
   `__setitem__` mutation, record-aware `to_packed`, per-field `to_numpy`/`to_padded`
   dicts. **`.parts` dropped** (was listed originally; superseded). `view`/ufunc
   raise on records. **Opaque `'S'` strings are standalone only** (not record
   fields); `view`/ufunc raise on records. No new Rust kernels.
   *Exit:* genoray + single-level GVL/GVF record cases work (sequences as chars).

3. **Spec C — Nested raggedness (R = 2) + `'S'`-under-an-axis.**
   *(Landed 2026-06-21 —
   [design doc](../superpowers/specs/2026-06-20-rust-ragged-nested-design.md).)*
   Generalize to **R = 2** (capped; R ≥ 3 → `NotImplementedError` — see decision
   log): nested offset list `[O0, O1]`, **full numpy-style nested indexing**
   (incl. per-group inner selection `rag[:, k]`/`rag[:, mask]`), nested
   constructors, record-aware `to_packed`, **per-axis `to_padded`** (pad chosen
   ragged axes, leave the rest ragged), nested records. Also the
   opaque-string-**under**-a-ragged-axis leaf (non-empty `offsets` **and** its own
   `str_offsets`), **standalone and as a record field** (each field keeps its own
   `str_offsets`), with nested `to_chars`/`to_strings`. Lazy nested offsets
   (approach A: inner offsets global, `(2,·)` gather forms, pack only on irregular
   gathers). Rust: nested-pack + nested-gather kernels.
   *Exit:* the three doubly-ragged cases (alleles, flat windows, codon
   annotations) work.

4. **Spec D — Cutover & cleanup.**
   Remove `awkward` from dependencies entirely; adapt `tokenize`/`translate` to
   string-leaf shapes; benchmark vs the old awkward path; rewrite
   `docs/ragged.md`; update `skills/seqpro/SKILL.md`; downstream migration notes.
   **Gated by the throughput milestone** ([design
   doc](../superpowers/specs/2026-06-21-ragged-throughput-gate-design.md)): a
   transitional, local A-vs-B benchmark (`benchmarks/bench_ragged_backends.py`,
   `pixi run -e bench rag-gate`) must show rust-native `Ragged` ≥ awkward (per-op
   wall-clock, `rust <= awkward * (1 + tol)`, default `tol = 0.10`) across Core +
   records + nested R=2 ops **before** `awkward` is dropped. The gate runs during
   the coexistence window, is retired at cutover, and its rust-side timings fold
   into the CodSpeed bench for forward regression tracking.
   *Exit:* throughput gate green, `import awkward` gone, full suite green,
   docs/skill current.

## Decision log

- **2026-06-21** — Throughput gate **GREEN: 23/23 passed (tol=10.00%), exit 0**
  after optimizing the 5 single-level regressors. All categories now pass with
  rust at or below awkward; worst ratio is `to_packed` i64 (unpacked) at 0.986
  (a genuine, fair, narrow win). Three changes, all **Python-only** (no
  `src/*.rs` edit, no `maturin` rebuild — the rust/numba kernels were already
  competitive):

  1. **Opt-in validation (default off).** `Ragged.__init__` previously ran
     `validate_layout` on *every* construction (including every `__getitem__`
     result), checking R=1 monotonicity **twice** (numpy `_is_monotonic` +
     rust `_ragged_validate`) — awkward validates nothing on construct. This
     both violated the documented "validation is opt-in and front-loaded via a
     `validate=` flag" convention (CLAUDE.md) and was the `construct` perf bug.
     Now `from_offsets`/`from_lengths`/`__init__` take `validate: bool = False`;
     `validate=True` is the one obvious way to ask "is this input clean?" Fixed
     `construct` (i64 1.79×→0.094, S1 1.59×→0.084) and removed re-validation on
     indexing results.
  2. **Slice indexing fast-path.** For a `slice`, `__getitem__` now takes
     `starts[sl]`/`stops[sl]` views instead of materializing `np.arange(n)` +
     `np.where` + a rust gather. Fixed `index[slice]` (1.21×→0.226). Verified
     equivalent to the old path across normal/full/empty/step/negative slice
     forms.
  3. **Benchmark fairness fix for `to_packed`.** The prior `to_packed` cells fed
     **already-packed** data, where awkward's `to_packed` is zero-copy (shares
     the buffer) while rust's `to_packed(copy=True)` does a defensive
     `data.copy()` — comparing *unequal work*, contrary to the gate's
     same-logical-work principle. The single-level `to_packed` cells now pack
     **unpacked** (masked) input so both backends perform the real gather; rust
     wins (i64 0.986, S1 0.866). This corrected the earlier "4.25× blocker"
     framing: the numba `_pack` kernel was never the problem — it beats awkward
     on genuine pack work; the cell was measuring awkward's degenerate
     already-packed shortcut. Records/nested `to_packed` cells were left on
     packed input (rust wins there regardless, as awkward does real structural
     work).

  Net: the 5 entries below are **resolved**; none remain Spec D blockers. Full
  rag pytest suite green (3 pre-existing pyarrow `PyExtensionType` failures
  unrelated/unchanged); ruff clean.

- **2026-06-21** — Throughput gate ran; `rag-gate` pixi task wired.
  `pixi run -e bench rag-gate` now runs the full A-vs-B benchmark
  (`benchmarks/bench_ragged_backends.py`). Result: **18/23 passed
  (tol=10.00%), exit code 1**. Categories: records ✓ (4/4), nested ✓ (8/8),
  string ✓ (2/2), single ✗ (4/9). The 5 FAILing ops are all in the
  single-level category and are **Spec D blockers** — must be optimised before
  cutover:

  | op | shape | awk (µs) | rust (µs) | rust/awk | backing layer |
  |---|---|---|---|---|---|
  | construct | 8000×~11-60 i64 | 6.11 | 10.92 | **1.787** | rust ingest (`from_offsets`) |
  | index[slice] | 8000×~11-60 i64 | 15.76 | 19.07 | **1.211** | rust |
  | to_packed | 8000×~11-60 i64 | 27.80 | 118.11 | **4.248** | numba `_pack` (via `_pack_parts`) |
  | construct | 8000×~11-60 S1 | 6.79 | 10.80 | **1.590** | rust ingest (`from_offsets`) |
  | to_packed | 8000×~11-60 S1 | 28.80 | 44.66 | **1.551** | numba `_pack` (via `_pack_parts`) |

  Worst regressor: `to_packed` i64 at 4.25× slower. The single-level
  `to_packed` regressions are in the **numba** kernel `seqpro.rag._ops._pack`
  (via `_pack_parts`) — NOT in Rust. (The nested R=2 `to_packed`, backed by the
  rust `_ragged_nested_pack` extension, passed the gate.) The `construct` and
  `index[slice]` FAILs are in the rust ingest path. These are the priority
  targets for the Spec D performance sprint.

- **2026-06-21** — Spec D throughput gate designed
  ([design doc](../superpowers/specs/2026-06-21-ragged-throughput-gate-design.md)).
  A transitional, local A-vs-B benchmark gates the Spec D cutover: rust-native
  `Ragged` must be ≥ awkward (per-op wall-clock, `rust <= awkward * (1 + tol)`,
  `tol = 0.10` default) across Core + records + nested R=2 ops. Decided during
  brainstorming: standalone script (`benchmarks/bench_ragged_backends.py` +
  `rag-gate` pixi task) over a pytest test, since the gate is one-time/throwaway;
  per-op-with-tolerance pass bar over strict/aggregate; min-of-repeats
  wall-clock with warmup + autoscaled batches; records/R=2 rows compare
  awkward-native (`ak.*`) vs
  rust-native (public awkward wrapper is 1-level only). Not a permanent CI fixture
  — CodSpeed keeps tracking forward regressions of the rust path; the gate is
  retired at cutover and its rust-side timings migrate into the CodSpeed bench.
- **2026-06-19** — Epic kicked off. Locked the four design decisions above after
  surveying consumer shapes. Chose to decompose into 4 sequential specs rather
  than front-load all specs before proving the core model.
- **2026-06-19** — Spec A landed: Rust-native single-level `Ragged` in
  `rag/_core.py` (+ `_layout.py`, `_ingest.py`, `src/ragged.rs`), fully tested
  against the awkward oracle. Public `seqpro.rag.Ragged` still points at the
  awkward type; the swap + tokenize/translate adaptation are deferred to Spec D
  (records to Spec B, nesting to Spec C). Confirmed string-leaf shape change
  `(N, None) -> (N,)` for byte collections.
- **2026-06-20** — Roadmap declared the SSoT for this epic (see directive at top):
  all Rust-`Ragged` work must read it and update it in the same PR.
- **2026-06-20** — Spec B design approved (records / struct-of-arrays). Locked:
  separate `RecordLayout` composing per-field `RaggedLayout`s over one shared
  offsets object; `from_fields` canonical + `rag.zip` alias (fields offset-equal,
  canonicalized to one shared object); `__setitem__` field mutation; row `int`
  index → dict, `slice`/`mask` → record `Ragged`; `to_numpy`/`to_padded` → per-field
  SoA dicts that raise on string fields; `view`/ufunc raise on records; no new
  Rust kernels (reuse Spec A per field). **`.parts` dropped** — `_core.py` already
  omits it and `RagParts` is gone; `.data`/`.offsets`/`.dtype` cover its uses.
  `.dtype` returns a structured dtype as a descriptor only (SoA, not AoS). Spec B
  stays internal/oracle-tested; public swap remains Spec D.
- **2026-06-20** — String/char duality adopted (supersedes the original
  "String leaf" locked decision). ASCII has two intentional interpretations:
  opaque `'S'` strings (`(N,)`, length in `str_offsets`, not an axis) and `'S1'`
  chars (`(N, ~length)`, length is a counted ragged axis), distinguished by
  `.dtype` as a descriptor (storage is a flat single-byte buffer either way).
  Zero-copy `to_chars()`/`to_strings()` retag between them. `from_lengths(S1, …)`
  defaults to opaque. `np.dtype('S')` (itemsize 0) chosen over `StringDType`
  (numpy 2.0-only) / `np.str_` to keep the **numpy `>=1.26` floor**. This pulls
  the core string refinement into Spec B as a prerequisite (modifies landed
  Spec A) and **simplifies Spec B records**: sequence fields that length-align
  with numeric siblings are chars (same shape, one shared offsets), so the record
  invariant reverts to **shared offsets + same shape**; the earlier
  shared-offsets-not-shape correction is withdrawn. **Opaque `'S'` strings are
  standalone only in Spec B**; the `'S'`-under-a-ragged-axis leaf (alleles)
  moves to **Spec C**.
- **2026-06-20** — Spec B landed: string/char duality + native records in
  `_core.py` — `np.dtype('S')` opaque-string descriptor with zero-copy
  `to_chars()`/`to_strings()`; `None`-in-shape disambiguates chars vs opaque;
  `RecordLayout` + `from_fields`/`rag.zip`; record field access/mutation; row
  indexing (`slice`/`mask` → record, `int` → dict); record
  `to_packed`/`to_numpy`/`to_padded`; awkward bridge (`layout_from_ak`/`to_ak`);
  differential-tested vs the awkward oracle. Two in-flight plan corrections:
  record `is_base` relaxed to the single-level one-indirection rule (dropping a
  redundant per-field copy in record `to_packed`); `__array__` raises `TypeError`
  on records (-O-safe) rather than relying on a bare assert. Public
  `seqpro.rag.Ragged` remains awkward-backed; the swap + tokenize/translate
  adaptation remain Spec D. `skills/seqpro/SKILL.md` not updated (Spec B is
  internal-only; skill update is Spec D). Known latent item: opaque-string detection routes on dtype.kind=='S', which over-accepts multi-byte S4/S100 (no caller constructs it today; tighten to S1-only in Spec D).
- **2026-06-20** — Spec C design approved (nested raggedness + string-under-axis).
  Four forks decided during brainstorming:
  1. **R = 2 cap** — implement exactly two nested ragged levels; R ≥ 3 →
     `NotImplementedError`. **Divergence from this roadmap's "arbitrary depth"
     framing**, chosen by YAGNI (every surveyed consumer tops out at R = 2).
     Algorithms kept generalizable but the contract is R ≤ 2.
  2. **Full numpy-style nested indexing** — positional logical-axis selectors
     incl. per-group inner selection (`rag[:, k]`, `rag[:, a:b]`,
     `rag[:, mask]`/`int_array`); adds a nested-gather Rust kernel.
  3. **Per-axis `to_padded`** — `axis`/`length` choose which ragged axis/axes to
     densify (inner → trailing regular dim; outer → uniform count; both → dense),
     leaving the rest ragged.
  4. **String-under-axis is standalone + a record field** — non-empty `offsets` +
     own `str_offsets`; the shared-offsets record invariant covers only counted
     axes, so each field keeps its private `str_offsets`. `to_chars`/`to_strings`
     promote/demote the innermost real offset level ↔ `str_offsets`.
  Layout: lazy nested offsets (inner global, `(2,·)` gather forms, pack only on
  irregular middle gathers); two new Rust kernels (nested-pack, nested-gather);
  record-of-record stays out of scope. Spec C stays internal/oracle-tested; public
  swap + tokenize/translate adaptation remain Spec D.
- **2026-06-21** — Spec C landed: nested raggedness (R = 2) + string-under-axis in
  `_core.py` / `_layout.py` / `_ops.py` / `src/ragged.rs`. What shipped:
  - **R = 2 cap**: `shape.count(None) >= 3` → `NotImplementedError`; record-of-record
    and R ≥ 3 fields rejected at construction time.
  - **Full nested indexing**: outer `slice`/`mask`/`int` (lazy gather, no data copy);
    `rag[i, j]` tuple chaining (chains single-key `__getitem__` calls); `rag[:, k]`
    (uniform int — k-th middle of each group, R=1 result); `rag[:, a:b]` (per-group
    slice, R=2 result); `rag[:, mask]` (mask over global middle axis, R=2 result);
    `rag[:, idx]` (uniform int-array, multi-column gather, R=2 result).
  - **String-under-axis leaf** — standalone (`from_offsets(..., str_offsets=...)`) and
    as a record field (each field retains its own `str_offsets`; shared-offsets
    record invariant covers only counted ragged axes). Zero-copy `to_chars()` /
    `to_strings()` promote/demote the innermost real offset level ↔ `str_offsets`.
    Bug fixed: string-under-axis integer indexing correctly uses `str_offsets` to
    locate byte boundaries when a real `offsets` level is also present.
  - **Nested records** — `from_fields` accepts R=2 fields; fields share the full
    `[O0, O1]` offsets list (same object, zero-copy SoA). Per-field `str_offsets`
    live in each field's `RaggedLayout`, not in the shared record list.
  - **Two new Rust kernels**: `nested_gather` (mask over global middle axis, returns
    per-group counts + selected indices) and `nested_pack` (pack scattered R=2
    layout to contiguous canonical `[O0_1D, O1_1D]`).
  - **`to_packed` / `to_padded` / `to_numpy`** for R=2: `to_packed` calls
    `_nested_pack_parts` (Rust `nested_pack`) per field, sharing the packed offsets
    list across fields. **`to_padded(axis=None, both)` composes two single-level
    pads** (no third "nested pad" kernel): inner R=1 view padded to dense
    `(M_total, K)`, then outer R=1 view (with trailing `K` dim) padded to
    `(*leading, M, K)`. `to_padded(axis=-1)` pads inner only → R=1 result.
    Single-level `to_padded` was generalized to support a trailing regular dim
    (needed for the outer pad step). `to_numpy` for R=2 delegates to
    `to_padded(axis=None)` after confirming both axes are uniform.
  - **`_ingest` bridge** updated: R=2 layouts and string-under-axis layouts handled
    in `layout_from_ak` / `to_ak`.
  - **Differential Hypothesis suite** (`tests/test_ragged_nested_diff.py`): R=2
    construct/index/pack (8 property tests covering all indexing forms, records,
    string-under-axis) vs the awkward oracle.
  - **Consumer-case exit tests** (`tests/test_ragged_nested_consumers.py`): the three
    doubly-ragged shapes from the survey — alleles string-under-axis, flat variant
    windows (dense tensor), codon annotations record.
  - **Deliberately deferred to Spec D / follow-up (NOT shipped)**:
    (a) `to_padded(axis=-2)` outer-only-with-ragged-inner structural pad — currently
    raises `NotImplementedError("to_padded(axis=-2) ... not supported in Spec C")`;
    its result would require feature (b).
    (b) Multi-leading-dim grid integer indexing (`rag[i, j]` peeling a leading int
    dim on `(d0, d1, ..., None)` arrays) — core flattens leading dims today; a
    proper grid peel would need a new outer-slice path.
    (c) Public swap + `tokenize`/`translate` adaptation — Spec D (unchanged).
  - `skills/seqpro/SKILL.md` not updated (Spec C is internal-only, consistent with
    Spec B; skill update is Spec D). Known latent item: S4/S100 dtype
    over-acceptance carries forward to Spec D.
