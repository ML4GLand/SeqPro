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

3. **Spec C — Arbitrary-depth nested raggedness + `'S'`-under-an-axis.**
   Generalize to `R ≥ 2`: nested offset list, multi-level indexing/slicing,
   nested constructors, multi-level `to_packed`/`to_padded`, nested records. Also
   the opaque-string-**under**-a-ragged-axis leaf (a field carrying both shared
   `offsets` and its own `str_offsets`), which the alleles case needs.
   Rust: nested-pack / nested-index kernels.
   *Exit:* the three doubly-ragged cases (alleles, flat windows, codon
   annotations) work.

4. **Spec D — Cutover & cleanup.**
   Remove `awkward` from dependencies entirely; adapt `tokenize`/`translate` to
   string-leaf shapes; benchmark vs the old awkward path; rewrite
   `docs/ragged.md`; update `skills/seqpro/SKILL.md`; downstream migration notes.
   *Exit:* `import awkward` gone, full suite green, docs/skill current.

## Decision log

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
  internal-only; skill update is Spec D).
