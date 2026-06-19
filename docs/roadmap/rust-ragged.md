# Roadmap: Rust-native `Ragged`

**Status:** In progress (epic)
**Started:** 2026-06-19
**Branch:** `feat/rust-ragged`

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
| String leaf | **A bytes/`S1` element is an opaque variable-width leaf; its byte-length is never an axis.** A collection of `N` sequences has shape `(N,)`, dtype bytes. String byte-offsets are stored but not counted in `.shape`/`.offsets`. This keeps records clean: a string field and a numeric field at the same level share identical axes. |

### Consequence of the string-leaf decision

The flat `.data` (S1 buffer) and byte-offsets for a sequence `Ragged` are
*physically unchanged* — `tokenize`/`translate` kernels are byte-identical. What
changes is shape bookkeeping: a byte collection's `.shape` drops its `None`
(`(N, None)` → `(N,)`), and constructors learn "bytes dtype ⇒ trailing offsets
are the string leaf, not an axis." A small, contained set of seqpro-owned call
sites adapt; no kernel rewrites. Downstream gets a clean string-leaf constructor.

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
(SoA) are pervasive and routinely mix string and numeric fields at a shared level.

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

2. **Spec B — Records / struct-of-arrays.**
   Native record `Ragged`: shared offset list across fields, mixed
   string+numeric leaves, native `zip`/record constructor (replaces `ak.zip`),
   per-field dict `.dtype`/`.data`/`.parts`, zero-copy field access,
   record-aware `to_packed`.
   *Exit:* genoray + single-level GVL/GVF record cases work.

3. **Spec C — Arbitrary-depth nested raggedness.**
   Generalize to `R ≥ 2`: nested offset list, multi-level indexing/slicing,
   nested constructors, multi-level `to_packed`/`to_padded`, nested records.
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
