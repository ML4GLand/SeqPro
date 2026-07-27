# Design: integer indexing on a string-under-axis `Ragged` (issue #71)

## Problem

For a **string-under-axis** `Ragged` — non-empty `offsets` plus `str_offsets` set,
per Spec C Section 2 — integer indexing collapses a whole group into one
concatenated blob, silently discarding the per-string boundaries that
`str_offsets` is already carrying.

```python
data  = np.frombuffer(b"AGGTC", dtype="S1")
outer = np.array([0, 2, 3])      # group  -> string index
inner = np.array([0, 1, 3, 5])   # string -> byte index

s = Ragged.from_offsets(data, (2, None), outer, str_offsets=inner)
n = Ragged.from_offsets(np.array([10, 20, 30], np.int32), (2, None), outer)
```

```
s.lengths -> [2 1]
n[0] -> array([10, 20], dtype=int32)   # 2 elements, matches lengths[0]
s[0] -> b'AGG'                         # ONE bytes; is that 'A'+'GG' or 'AG'+'G'?
```

`s.lengths[0] == 2` but `len(s[0]) == 3`. The boundary at `str_offsets[1] == 1` is
dropped.

### It fails silently, data-dependently, and in two places

Consumers zip a numeric field against a string field on the shared axis. While
every string is one byte the lengths coincide and the misalignment is invisible;
the first multi-byte entry (an indel) breaks it.

The issue reports one site. There are **two**:

| # | Site | Layout |
|---|---|---|
| 1 | `Ragged.__getitem__` integer branch, `python/seqpro/rag/_core.py:722-731` | plain string-under-axis |
| 2 | `_getitem_record_rows` integer branch, `python/seqpro/rag/_core.py:1216-1226` | opaque-string **field** of a record |

Site 2 is the one GenVarLoader actually hits. `RaggedVariants` is a record
`Ragged` with shape `(batch, ploidy, ~variants)` whose `alt`/`ref` are
opaque-string fields sharing one offsets object with numeric `start`/`ilen`
(`genvarloader/_dataset/_rag_variants.py:193`). Peeling a row today gives:

```
rec[0] -> {'alt':   array([b'A', b'G', b'G'], dtype='|S1'),   # 3 chars
           'start': array([1, 2], dtype=int32)}                # 2 variants
```

The string field is not even `bytes` — it is the raw concatenated `S1` char
buffer, sitting next to a per-variant numeric array in the same dict. This is
strictly worse than the reported symptom.

`_getitem_record_rows_r2` (`_core.py:1243`) delegates to `Ragged(fl)[where]`, so
it inherits site 1's behavior and needs no separate change.

### Why fix rather than document

`to_numpy()` and `to_packed()` both *refuse* on this layout rather than return
something lossy (`_core.py:1777-1780`, `:1801-1805`, `:1282-1285`). Integer
indexing is the one accessor that quietly returns a wrong-shaped answer. No
information is lost — `str_offsets` holds the boundaries — so this is purely a
question of what the accessor hands back.

## Decision

**Integer indexing that peels the last real ragged level off a string-under-axis
`Ragged` returns the standalone opaque-string layout: `offsets == []`,
`str_offsets` set, `shape == (k,)`, `is_string == True`.**

This is not a new return type. Spec C's own layout table
(`docs/superpowers/specs/2026-06-20-rust-ragged-nested-design.md:129-135`) defines
the standalone opaque string of Spec A/B as *"the zero-real-level special case"*
of string-under-axis:

| | standalone string (Spec B) | string-under-axis (Spec C) | chars |
|---|---|---|---|
| `offsets` | `[]` | `[O0]` | `[O0, O1]` |
| `str_offsets` | set | set | `None` |
| `.shape` | `(N,)` | `(*leading, None)` | `(*leading, None, None)` |

Peeling the one real level off `(N, ~var)` therefore lands exactly on `(k,)`.
"Return the layout with one fewer real level" is what integer indexing already
means everywhere else in the class; this makes the string case obey it too.

The returned layout already supports everything a consumer needs (verified on
0.21.2):

```
flat.shape (3,)   is_string True   len 3
flat[0] b'cat'    flat[-1] b'there'
list(flat)  -> [b'cat', b'hi', b'there']
zip(...)    -> [(1, b'cat'), (2, b'hi'), (3, b'there')]
flat[5]     -> IndexError          flat.lengths -> [3 2 5]
```

So `zip(rv.start[0][h], rv.alt[0][h])` — the GenVarLoader pattern from
mcvickerlab/GenVarLoader#330 — becomes correct, and `len(s[i]) == s.lengths[i]`
holds.

### Alternatives rejected

- **Object array of `bytes`** (issue option 1). Same ergonomics as the chosen
  design but allocates `k` Python `bytes` objects on every row access, against
  the repo's "no naive NumPy / no Python loops in hot paths" rule — and this is a
  per-batch-item accessor. It also introduces a return type the layout algebra
  does not otherwise use. Strictly dominated.
- **Char `Ragged`, i.e. `to_chars()[i]`** (issue option 2). Shape `(k, None)` of
  `S1`; forces every consumer to re-assemble strings. Loses the "these are
  strings" information the layout is carrying.
- **Raise `NotImplementedError`** (issue option 3). Follows the
  `to_numpy()`/`to_packed()` precedent, but indexing is far more central than
  those, and the information needed to answer correctly is present. Would leave
  GenVarLoader with no ergonomic path at all.

## Design

### Site 1 — `_core.py:722-731`

Mirror the existing `_slice_contig_string` (`_core.py:511-541`), which already
performs exactly this narrowing for the slice case:

```python
if isinstance(where, (int, np.integer)):
    lo, hi = int(starts[where]), int(stops[where])
    if self._rl.str_offsets is not None and self._layout.offsets:
        # string-under-axis: peel the real level -> standalone opaque string (k,)
        so = self._rl.str_offsets
        b0 = int(so[lo])
        return Ragged(RaggedLayout(
            data=self._rl.data[b0 : int(so[hi])],
            offsets=[],
            shape=(hi - lo,),
            str_offsets=so[lo : hi + 1] - b0,
        ))
    ...
```

- Data is a view; only the small `(k+1,)` offsets slice is copied, to rebase to
  zero as `is_contiguous` requires for opaque strings (`_core.py:319-321`).
- Correct whether `offsets[0]` is 1-D canonical or a lazy `(2, M)` gather layout:
  `starts`/`stops` come from `_starts_stops()`, and a group is a contiguous run in
  string-index space under either encoding.
- `lo == hi` (empty group) yields a well-formed length-0 result.

### Site 2 — `_getitem_record_rows`, `_core.py:1216-1226`

Apply the same construction per opaque-string field; numeric and char fields are
untouched. The returned dict then has one entry per field, all of length
`hi - lo`.

Each field keeps its **own** `str_offsets` (Spec C Section 5), so fields are
narrowed independently against the shared `lo`/`hi`.

### Explicitly unchanged

- **The standalone/flat case** (`offsets == []`). `flat[0]` still returns `bytes`
  — that is the terminal peel and what Spec A/B consumers depend on. The guard
  `self._layout.offsets` already distinguishes the two.
- **Slicing, masking, fancy indexing.** Already correct; they preserve
  `str_offsets`.
- **`_getitem_record_rows_r2`**, which delegates to site 1.
- **Rust (`src/`, `crates/`).** Python-layer accessor only.

## Compatibility

Breaking for anyone relying on the concatenation. Warrants a minor bump and a
CHANGELOG entry. The old value remains reachable as
`b"".join(s[i])` for a caller who genuinely wanted the concatenated group.

`tests/test_ragged_core.py:733` (`test_string_under_axis_integer_index`) pins the
current behavior and must be updated. Note it uses exactly one string per group,
so it could never have distinguished the two interpretations — that is why the
bug survived.

## Testing

1. The issue's exact repro: `len(s[0]) == s.lengths[0]`, `s[0][0] == b'A'`,
   `s[0][1] == b'GG'`, `list(s[1]) == [b'TC']`.
2. Numeric/string parity: `len(s[i]) == len(n[i])` for every `i`, for a
   string-under-axis and a numeric `Ragged` sharing one offsets object.
3. Multi-dim `(batch, ploidy, ~variants)`: `s[0][h]` yields per-variant strings
   (the shape that reaches site 1 via `_getitem_multidim`).
4. Record row: `rec[0]` — every field the same length, `zip` aligned, with at
   least one multi-byte (indel) entry so 1-byte coincidence cannot mask a
   regression.
5. Zero-copy: the result's data buffer shares memory with the parent's.
6. Empty group (`lo == hi`) → length-0 result, `len(...) == 0`.
7. Agreement with `to_chars()`: `s[i][j] == s.to_chars()[i][j].tobytes()`.
8. Negative index and out-of-bounds `IndexError` preserved.
9. Standalone string regression: `flat[0]` still returns `bytes`.

## Out of scope

- Changing `to_numpy()` / `to_packed()` behavior on this layout.
- The GenVarLoader-side fix (mcvickerlab/GenVarLoader#330) — tracked separately;
  this change is what unblocks it.

## Docs

`skills/seqpro/SKILL.md` must be updated in the same PR — CLAUDE.md requires it
for any breaking change or public-behavior change.
