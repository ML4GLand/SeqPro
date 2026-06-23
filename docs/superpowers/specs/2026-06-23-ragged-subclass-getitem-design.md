# Ragged `__getitem__` numpy-consistency + subclass-preserving transforms

**Date:** 2026-06-23
**Status:** Design approved; ready for implementation planning
**Scope:** Two independent seqpro PRs, landed in order. PR1 (bug fix) first, PR2 (subclass seam) second. A downstream GenVarLoader cleanup follows once both land.

## Problem

`seqpro.rag._core.Ragged` has two related shortcomings that force downstream
consumers (GenVarLoader's `RaggedVariants`, and similarly genoray) to wrap
`Ragged` in composition objects and re-implement structural operations:

1. **`__getitem__` is not numpy-consistent for multi-leading-axis records.**
   numpy guarantees a non-tuple key is treated as a 1-tuple: `A[x]` ≡ `A[(x,)]`.
   `Ragged` violates this for **record** layouts with `rag_dim > 1`
   (e.g. shape `(batch, ploidy, ~variants)`):

   | index | returns | correct? |
   |---|---|---|
   | `rec[1:3]` | `(2, None)` | ❌ leading fixed axes flattened away |
   | `rec[(slice(1,3),)]` | `(2, 2, None)` | ✅ numpy-like (peels axis 0) |
   | `rec[0]` | `dict` of raw arrays | ❌ should peel axis 0 → `(2, None)` Ragged |
   | `rec[(0,)]` | `(2, None)` Ragged | ✅ |

   Root cause: the non-tuple record path `_getitem_record` → `_getitem_record_rows`
   (`_core.py:807,983`) flattens **all** leading fixed axes and gathers over the
   flattened shared ragged axis, while the tuple path `_getitem_tuple_multidim`
   (`_core.py:633`) correctly peels only the first axis. Non-record multidim
   arrays already peel-first correctly via `_getitem_multidim` — only the
   record-rows path is wrong.

   GenVarLoader currently works around this in `RaggedVariants.__getitem__`:
   ```python
   if rag._is_record and rag.rag_dim > 1 and not isinstance(idx, tuple):
       result = rag[(idx,)]   # force the correct (numpy-like) tuple path
   ```

2. **Structural transforms always return base `Ragged`, never a subclass.**
   Every transform hardcodes `return Ragged(...)` / `Ragged.from_offsets(...)` /
   `Ragged.from_fields(...)` at ~25 sites (`__getitem__`, `reshape`, `squeeze`,
   `to_packed`, `concatenate`, ufuncs). There is no `type(self)` / `cls` /
   constructor hook, so a subclass's type is dropped by every operation. This is
   why `RaggedVariants` is a *composition wrapper* (`self._rag: Ragged`) that
   overrides `reshape`/`to_packed`/`squeeze`/`__getitem__` solely to re-wrap the
   result as `RaggedVariants.from_record(...)`.

## PR1 — `__getitem__` numpy-consistency for record layouts

**Goal:** make a non-tuple key behave exactly like the equivalent 1-tuple for
record layouts, restoring the numpy contract `rec[x] == rec[(x,)]`.

**Change:** in `_getitem_record` (`_core.py:807`), before dispatching to
`_getitem_record_rows`, normalize a non-string, non-tuple key to a 1-tuple when
the record has more than one leading fixed axis:

```python
def _getitem_record(self, where):
    rec = self._layout
    if isinstance(where, str):
        return Ragged(rec.fields[where])          # field extraction — unchanged (illustrative; keep the real KeyError wrap)
    # numpy: A[x] ≡ A[(x,)]. Call the multidim peel path directly under the SAME
    # guard it already requires (single ragged axis + >1 leading fixed axis), so
    # we never re-dispatch through __getitem__.
    if (
        not isinstance(where, tuple)
        and self.rag_dim > 1
        and rec.shape.count(None) == 1
    ):
        return self._getitem_tuple_multidim((where,))
    return self._getitem_record_rows(where)
```

This routes `rec[0]`, `rec[1:3]`, `rec[mask]`, `rec[idx_array]` through the
existing, already-correct `_getitem_tuple_multidim` path for single-ragged-axis,
`rag_dim > 1` records. `rag_dim == 1` records keep their current behavior (a
single leading axis; non-tuple and 1-tuple already agree there).

**Recursion hazard (must avoid):** do NOT implement this as `return self[(where,)]`.
For a record that is *not* single-None (nested ragged with `rag_dim > 1`), the
1-tuple would miss the `_getitem_tuple_multidim` guard at `_core.py:457`, fall
through to the per-key loop (`_core.py:499`), and re-enter `_getitem_record` with
the original non-tuple key → infinite recursion. Calling `_getitem_tuple_multidim`
directly under the matching guard sidesteps this; non-single-None records fall
through to `_getitem_record_rows` unchanged (out of scope, not regressed).

**Out of scope for PR1:** changing `rag_dim == 1` record semantics (e.g. whether
`rec[0]` on a `(batch, ~v)` record returns a dict). Not part of the numpy-
consistency bug; leave behavior unchanged.

**Tests:**
- New parametrized test: for a `(d0, d1, None)` record (and a `(d0,d1,d2,None)`
  record from the genoray-style `from_offsets`), assert `rec[k]` deep-equals
  `rec[(k,)]` for `k ∈ {int, slice, bool-mask, int-array}`, checking both
  `.shape` and field values.
- Regression: confirm non-record multidim and `rag_dim==1` record indexing are
  unchanged.

**Beneficiaries:** GenVarLoader (lets the workaround branch be deleted in the
follow-up) and genoray (same `(n_ranges, n_samples, ploidy, None)` record shape).

## PR2 — subclass-preserving transforms (Design B)

**Goal:** structural transforms return `type(self)`, so a `Ragged` subclass
survives `__getitem__`/`reshape`/`squeeze`/`to_packed`/`concatenate` without the
subclass overriding any of them.

**Key enabling fact:** a subclass like `RaggedVariants` carries **no instance
state beyond the layout** — its only purpose is a domain vocabulary (typed/derived
properties, `rc_`, construction invariants) layered on a record `Ragged`. So the
subclass can declare `__slots__ = ()` and be reconstructed purely from a layout.

**Change:** add one private constructor and route structural construction through
it:

```python
def _with_layout(self, layout) -> "Ragged":
    obj = object.__new__(type(self))   # subclass-preserving; bypasses __init__
    obj._layout = layout
    return obj
```

Classify the ~25 construction sites into two buckets:

- **Structural (use `self._with_layout(...)`):** positional `__getitem__` results,
  `reshape`, `squeeze`, `to_packed`, `concatenate`, and the `None`/newaxis paths —
  operations that return "the same logical container, transformed". These preserve
  the subclass.
- **Extraction / new-container (keep `Ragged(...)`):** string-key field access
  (`Ragged(field)` in `_getitem_record`), per-field results, and any site that
  returns a *different* logical thing than `self`. These stay base `Ragged`.

`from_offsets` / `from_fields` / `from_lengths` remain base-typed staticmethods
(callers that want a subclass use the subclass's own constructor).

**Consumer migration (GenVarLoader, separate follow-up PR after both land):**
- Make `RaggedVariants` subclass `Ragged` with `__slots__ = ()`; drop the `_rag`
  composition field (it *is* the record `Ragged`).
- Keep domain methods/properties: `alt`/`ref`/`start`/`dosage`, derived `ilen`/`end`,
  `rc_`, `_alt_chars`, construction invariants in `__init__`.
- **Delete** the structural overrides — `reshape`, `to_packed`, `squeeze`, and the
  entire `__getitem__` override: with PR1, base positional indexing is numpy-correct
  and preserves ploidy; with PR2, it returns `RaggedVariants`; string-key access
  already returns a bare field `Ragged` from base `_getitem_record`. Net: the
  override boilerplate goes away.

**Tests:**
- A minimal `Ragged` subclass in the seqpro test suite (`class _Sub(Ragged): __slots__=()`)
  asserting `type(sub[...])`, `type(sub.reshape(...))`, `type(sub.to_packed())`,
  `type(sub.squeeze())`, `type(concatenate([sub, sub]))` are all `_Sub`, while
  string-key field extraction returns base `Ragged`.
- Construction invariants are correctly skipped on transformed results
  (`object.__new__` bypass), and validated only via the public `__init__`.

**Constraint / trade-off:** Design B works only while a subclass holds no state
beyond `_layout`. If a future subclass needs extra instance state, it must either
add a `_with_layout` override that copies that state, or we revisit with the
`_constructor`-hook variant (Design A). Documented as a subclassing contract.

## Risks

- PR1 narrowly gated (`_is_record and rag_dim > 1 and not tuple and not str`), low
  blast radius; the target path already exists and is exercised by GVL today.
- PR2 requires correct site classification; mis-tagging an *extraction* site as
  structural would wrongly return a subclass for a single field. The subclass test
  matrix guards this. Landing PR1 first keeps the indexing-correctness change
  separate from the type-preservation change, so a regression is easy to localize.

## Sequencing

1. PR1 (bug fix) → merge → patch release.
2. PR2 (subclass seam) → merge → release.
3. GenVarLoader follow-up: subclass `RaggedVariants` on `Ragged`, delete workaround
   + structural overrides, pin the seqpro version that includes both.
