# Ragged getitem fast paths: contiguous slice, `to_fixed`, lean `from_offsets`

**Date:** 2026-06-24
**Status:** approved (design)
**Scope:** `python/seqpro/rag/_core.py` + tests. No Rust this round.

## Motivation

A microbench (GenVarLoader-side, comparing gvl's internal `_Flat` transport against
`seqpro.rag.Ragged` on the getitem hot-path ops) showed `Ragged` paying a large,
avoidable cost on three operations. Best-of-5, dev env, shape `(B=128, P=2, ~2000)`:

| op | `_Flat` (µs) | `Ragged` (µs) | gap |
|---|---|---|---|
| `from_offsets` | 0.35 | 1.08 | 3.1× |
| leading-axis slice | 3.0 | 8.2 | 2.7× |
| `view(dtype)` | 0.37 | 0.77 | 2.1× |
| slice **+ to_packed** | — | 25–43 | — |
| uniform densify | 0.34 (`to_fixed`) | 4.8 (`to_numpy`) | 14× |

> **Correction.** An earlier draft of this table reported densify as 378× vs
> `to_padded` (128 µs). That was a wrong-baseline artifact: `to_numpy()` *already*
> does a zero-copy uniform reshape (verified: it shares the data buffer) at 4.8 µs.
> The honest gap is 14×, and it is entirely `to_numpy`'s uniformity scan
> (`np.diff` + `np.all(lengths == lengths[0])`) plus the `is_base` check — not the
> reshape. See §2.

The wins are **algorithmic, not language-bound** (per-op costs are 0.3–8 µs; a pyo3
FFI hop is ~0.1–0.5 µs of overhead, which would eat much of the gain). Two root
causes:

1. **Slice always gathers to `(2, N)` starts/stops over the full buffer.** Both the
   R=1 path (`_getitem` → `_row_gather` → `np.stack`) and the `rag_dim>1` path
   (`_getitem_multidim`) produce a non-contiguous `(2, N)` offsets layout and keep
   `data` pointed at the *entire* parent buffer. Any downstream code that needs
   contiguous `(N+1,)` offsets must then call `to_packed()` — a full copy (the
   25–43 µs row).
2. **`to_numpy()`'s uniform reshape always pays a uniformity scan.** The reshape
   itself is zero-copy, but `to_numpy()` unconditionally scans `np.diff(offsets)` to
   verify every row is the same length before reshaping — ~4.5 µs the caller can't
   skip even when it constructed the buffer and *knows* it is uniform.

This blocks GenVarLoader from retiring its `_Flat`/`_FlatAnnotatedHaps`/`FlatIntervals`
shadow layer, which exists *only* to avoid these costs on the getitem hot path.

## Decisions (settled)

- **Substrate:** Python fast-paths in `_core.py` now; defer any Rust port to a
  tracking issue (hybrid). Rationale: the wins are algorithmic and the ops are
  µs-scale, so FFI overhead would erode the benefit.
- **Uniform densify:** **no new method.** `to_numpy()` already does the zero-copy
  uniform reshape; add a `validate: bool = True` keyword to it that, when `False`,
  skips the uniformity scan and trusts the caller (the "trust-me" path for callers
  that control buffer construction, like gvl). See §2.
- **`from_offsets` size check:** gated behind `validate` (default `validate=False`
  does no size validation) — accepted default-safety regression for the full
  construction win. See §3.
- **Fast-path layout coverage:** all layouts this round — R=1, R=2, opaque-string
  (flat and under-axis), record R=1, record R=2 — including string-under-axis
  records (which `to_packed` currently rejects under Spec C; the slice fast path
  can narrow them correctly and will, with the asymmetry noted in a comment).

## §1 — Contiguous-slice fast path (`__getitem__`)

### Unifying model

An outer step-1 slice is the same operation repeated at each offsets *level*: given
a contiguous 1-D offsets array `O` and a slice `[a:b]` in that level's units, emit

```
O_new   = O[a : b+1] - O[a]          # contiguous (M+1,), zero-based
narrow the next level down to [O[a] : O[b]]
```

Chained across levels:

- **R=1:** `O0 → data`
- **R=2:** `O0 → O1 → data`
- **string:** `O0 → str_offsets → data` (or `str_offsets → data` for a flat
  string collection with no axis)
- **record:** rebase the shared `O0` (and `O1` for R=2) once, narrow **each field's**
  `data` independently (string fields use the string chain per field)

The result is the **already-packed** slice computed from views + small offset
subtractions: no pad/pack kernel, no `(2, N)` drift, and `data` is a narrowed view.
We verified `is_base`/`is_contiguous` are both true on the result (`data[a:b]` is an
owned view; rebased offsets are zero-based and end at the narrowed data length).

### Gate (single, checkable, parity-safe)

```python
if isinstance(where, slice):
    start, stop, step = where.indices(<outer_len>)
    if step == 1 and self.is_contiguous:
        return <fast path>
# else: existing path, byte-for-byte unchanged
```

`is_contiguous` (`_core.py:300-315`) is exactly the right predicate — it already
checks: all offsets levels are 1-D, `data` is C-contiguous, every record field's
`data` is C-contiguous, and (for strings) `str_offsets` is 1-D and zero-based. So
the gate fires only when every level is in canonical contiguous form.

Everything else falls through to today's code: non-slice keys (int / bool mask /
int-array), `step != 1`, negative-step, and any already-`(2, N)` (post-gather)
array. Those paths are **not touched**.

### Per-layout narrowing

Each row below is an independent implementation+test unit. `n_inner` is the product
of the fixed dims between the outer axis and the ragged axis (1 when `rag_dim == 1`);
the outer slice `[start:stop]` is converted to segment units `[start*n_inner :
stop*n_inner]` before rebasing.

| Layout | Levels rebased | Buffer narrowed |
|---|---|---|
| R=1 numeric/char | `O0` | `data` |
| R=2 numeric/char | `O0`, `O1`-window | `data` |
| string (flat) | `str_offsets` | `data` |
| string (under-axis) | `O0`, `str_offsets`-window | `data` |
| record R=1 | shared `O0` | each field's `data` |
| record R=2 | shared `O0`, `O1`-window | each field's `data` |

Subclass identity is preserved by the existing `__getitem__` wrapper
(`_with_layout`, used for non-string keys).

### Observable behavior change

A plain contiguous step-1 slice now returns `(N+1,)` contiguous offsets over a
**narrowed** `data` view, instead of `(2, N)` starts/stops over the full buffer.
**Values are identical**; only internal representation changes (strictly toward
"already packed"). seqpro tests that assert the old `(2, N)` / full-buffer shape
post-slice are updated. GenVarLoader is unaffected — it uses `_Flat` on this path
and only constructs `Ragged` at the return boundary.

### Why the surface area is safe

The gate guarantees we *only ever* change the representation of plain contiguous
step-1 slices; every other index runs the unchanged code. Each layout variant has a
property test asserting `fast_path(x, sl)` equals `force_old_path(x, sl)`
element-wise (compare `to_packed` data + per-row lengths) over randomized shapes and
dtypes. A bug in any variant is caught before merge, never shipped.

## §2 — `to_numpy(..., *, validate: bool = True)`

No new method. `to_numpy()` already reshapes a uniform Ragged zero-copy
(`_core.py:1592-1600`); the only avoidable cost is its mandatory uniformity scan.
Add a keyword-only `validate` flag that gates that scan.

```python
def to_numpy(self, allow_missing: bool = False, *, validate: bool = True):
    ...
    # single-level (R=1) path:
    if validate:
        lengths = self.lengths
        if lengths.size and not np.all(lengths == lengths.flat[0]):
            raise ValueError("cannot convert a jagged Ragged to a dense array")
        row_len = int(lengths.flat[0]) if lengths.size else 0
    else:
        # trust the caller: infer row_len from offsets without scanning for uniformity.
        # numpy's reshape still total-checks for free, so a grossly wrong buffer raises;
        # a non-uniform buffer whose total happens to match is the caller's contract to avoid.
        n_rows = <product of leading fixed dims>
        row_len = (self.offsets[-1] // n_rows) if n_rows else 0
    packed = self if self.is_base else self.to_packed()
    leading = packed.shape[: packed.rag_dim]
    return packed._rl.data.reshape(*(leading or (-1,)), row_len, *packed._rl.data.shape[1:])
```

- **`validate=True` (default):** unchanged behavior — scans and raises `ValueError`
  on a jagged array. Safe.
- **`validate=False`:** skips the `np.diff`/`np.all` uniformity scan; infers
  `row_len` from offsets and reshapes. numpy's reshape still rejects a total-size
  mismatch for free, so the only thing the caller takes responsibility for is
  per-row uniformity. This is the ~0.34 µs hot path.
- The flag threads through the **R=2** path (gate the `grp_lens`/`mid_lens`
  uniformity `ValueError`s) and the **record** path (passed through to per-field
  recursion), so `validate=False` is consistent across layouts.
- `allow_missing` is unchanged and orthogonal.

**Consumer (gvl, separate change):** replace `_Flat.to_fixed(L)` with
`ragged.to_numpy(validate=False)` — gvl controls buffer construction in
fixed-output mode, so it can assert uniformity by construction and skip the scan.

## §3 — Lean `from_offsets`

Close the 3.1× construction gap:

- Skip the `np.ascontiguousarray(o, dtype=OFFSET_TYPE)` copy-check when an offsets
  array is **already** C-contiguous `int64` (use it as-is).
- **Gate the eager data-size check behind `validate`.** Today the check at
  `_core.py:130-144` runs unconditionally; move it inside the `validate` branch so
  the default (`validate=False`) path does no size validation.

### Behavior change (documented)

With `validate=False` (the default), `from_offsets` no longer raises on a
data/offsets size mismatch — a malformed array is constructed and the error
surfaces later (e.g. an out-of-bounds index or a wrong slice). Callers that want
the eager guard pass `validate=True`. This trades a small default-safety regression
for the full construction win and matches the "trusted internal constructor"
role the hot path needs.

- `validate=True` keeps **all** checks (size check + `_ragged_validate` + monotonic
  offsets, as today).
- The same elision applies to `str_offsets` (`ascontiguousarray` only when not
  already canonical) and to the `validate`-gated size check for the string and R≥1
  paths.

Acceptance is the bench (below): target `from_offsets` within ~1.1× of `_Flat`.

## Testing & acceptance

1. **Parity property tests (per layout):** `fast_path` result == forced old-gather
   result, element-wise (`to_packed` data + lengths), across random shapes/dtypes,
   including `rag_dim>1`, R=2, all string and record kinds, empty slices, and
   `step != 1` / non-contiguous → fallback.
2. **`to_numpy(validate=...)`:** `validate=True` (default) result equals
   `validate=False` result on uniform input and shares the data buffer (zero-copy);
   `validate=True` still raises `ValueError` on a jagged array while `validate=False`
   does not scan; consistent across R=1, R=2, and record layouts.
3. **Full existing seqpro suite green** (update any test asserting `(2, N)` /
   full-buffer post-slice; `test_ragged_core.py`, `test_ragged_core_records.py`,
   `test_rag_to_packed.py`).
4. **`from_offsets` validate gate:** `validate=True` still raises `ValueError` on a
   data/offsets size mismatch; `validate=False` (default) constructs without raising.
   Update any existing test that relied on the eager default raise to pass
   `validate=True`.
5. **Re-run the microbench**, gates: slice within ~1.2× of `_Flat`;
   `to_numpy(validate=False)` within ~1.2×; `from_offsets` within ~1.1×.
6. **Cross-repo parity:** run GenVarLoader's suite against an editable install of
   this branch (the byte-identical parity harness) — must stay byte-identical.
7. **Docs:** update the `seqpro` agent skill for `to_numpy(validate=...)` + the
   slice-contiguity note; file the deferred-Rust tracking issue.

## Out of scope (this round)

Rust port of any of this; changes to `to_padded` semantics or to `to_numpy`'s
default (`validate=True`) behavior beyond the new opt-in flag; fast paths for
non-step-1 or already-`(2, N)` slices (they keep using the gather path).
