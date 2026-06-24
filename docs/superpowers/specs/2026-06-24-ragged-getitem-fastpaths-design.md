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
| uniform densify | 0.34 (`to_fixed`) | 128 (`to_padded`) | 378× |

The wins are **algorithmic, not language-bound** (per-op costs are 0.3–8 µs; a pyo3
FFI hop is ~0.1–0.5 µs of overhead, which would eat much of the gain). Two root
causes:

1. **Slice always gathers to `(2, N)` starts/stops over the full buffer.** Both the
   R=1 path (`_getitem` → `_row_gather` → `np.stack`) and the `rag_dim>1` path
   (`_getitem_multidim`) produce a non-contiguous `(2, N)` offsets layout and keep
   `data` pointed at the *entire* parent buffer. Any downstream code that needs
   contiguous `(N+1,)` offsets must then call `to_packed()` — a full copy (the
   25–43 µs row).
2. **No uniform-length fast path for densify.** When every row has the same length
   (fixed-output mode), the only tool is `to_padded`, which runs the pad kernel
   (128 µs) where a pure reshape (0.34 µs) suffices.

This blocks GenVarLoader from retiring its `_Flat`/`_FlatAnnotatedHaps`/`FlatIntervals`
shadow layer, which exists *only* to avoid these costs on the getitem hot path.

## Decisions (settled)

- **Substrate:** Python fast-paths in `_core.py` now; defer any Rust port to a
  tracking issue (hybrid). Rationale: the wins are algorithmic and the ops are
  µs-scale, so FFI overhead would erode the benefit.
- **Uniform densify API:** new explicit `Ragged.to_fixed(length)` method (mirrors
  gvl `_Flat.to_fixed`), not implicit magic inside `to_numpy`.
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

## §2 — `Ragged.to_fixed(length: int) -> NDArray`

Same contract as `to_padded`: non-record, ragged-axis-last.

```python
def to_fixed(self, length: int) -> NDArray:
    rag = self if self.is_contiguous else self.to_packed()
    lengths = np.diff(rag.offsets)            # cheap; O(n_segments)
    if rag.offsets.size and not (lengths == length).all():
        raise ValueError(...)                 # non-uniform
    return rag._rl.data.reshape(*leading, length)   # zero-copy when contiguous+uniform
```

- Raises `ValueError` on any row whose length `!= length`.
- Zero-copy reshape when already contiguous + uniform (the hot-path case); one
  `to_packed` copy first if the input drifted to `(2, N)`.
- Record / non-ragged-axis-last inputs raise `NotImplementedError`, matching
  `to_padded`.

## §3 — Lean `from_offsets`

Close most of the 3.1× construction gap without weakening the default safety
contract:

- Skip the `np.ascontiguousarray(o, dtype=OFFSET_TYPE)` copy-check when an offsets
  array is **already** C-contiguous `int64` (use it as-is).
- Keep the eager data-size check on by default — it is a single integer compare and
  catches real user error.

Acceptance is the bench (below). If profiling shows the size-check itself dominates
the residual gap, gate it behind `validate` with a documented default change — but
default-safe is the goal, and the `ascontiguousarray` elision is expected to be the
bulk of the win.

## Testing & acceptance

1. **Parity property tests (per layout):** `fast_path` result == forced old-gather
   result, element-wise (`to_packed` data + lengths), across random shapes/dtypes,
   including `rag_dim>1`, R=2, all string and record kinds, empty slices, and
   `step != 1` / non-contiguous → fallback.
2. **`to_fixed`:** equals `to_padded` for uniform input; raises on non-uniform;
   zero-copy (shares base) when contiguous+uniform.
3. **Full existing seqpro suite green** (update any test asserting `(2, N)` /
   full-buffer post-slice; `test_ragged_core.py`, `test_ragged_core_records.py`,
   `test_rag_to_packed.py`).
4. **Re-run the microbench**, gates: slice and `to_fixed` within ~1.2× of `_Flat`;
   `from_offsets` gap < 1.5×.
5. **Cross-repo parity:** run GenVarLoader's suite against an editable install of
   this branch (the byte-identical parity harness) — must stay byte-identical.
6. **Docs:** update the `seqpro` agent skill for `to_fixed` + the slice-contiguity
   note; file the deferred-Rust tracking issue.

## Out of scope (this round)

Rust port of any of this; changes to `to_numpy` / `to_padded` semantics; fast paths
for non-step-1 or already-`(2, N)` slices (they keep using the gather path).
