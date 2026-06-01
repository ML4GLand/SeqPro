# Flat-buffer `to_padded` for `Ragged`

## Background

`seqpro.rag.reverse_complement` (released in 0.12.1) replaced the awkward-array
idiom for per-batch reverse-complement with a single-pass, parallel Numba kernel
over the flat `(data, offsets)` buffer. Downstream (GenVarLoader) it cut the RC
step from ~13 ms to ~0.4 ms and 92 MB to ~0 allocation at buffered-loader scale.

The remaining awkward hotspot in the same per-batch path is **densifying a
`Ragged` into a rectilinear, right-padded array**. GenVarLoader implements this
today as `to_padded(rag, pad_value)`:

```python
length = int(rag.lengths.max())
if is_rag_dtype(rag, np.bytes_):
    arr = Ragged(ak_str.rpad(rag, length, pad_value)).to_numpy()
else:
    rag = ak.pad_none(rag, length, axis=-1, clip=True)
    arr = ak.to_numpy(ak.fill_none(rag, pad_value)).astype(rag.dtype, copy=False)
```

This goes through awkward's `rpad` / `pad_none` + `fill_none` + `to_numpy`,
which the profile flagged as the top `awkward array_module.empty` allocator. It
is the natural next flat-buffer target, and it belongs upstream in seqpro
(seqpro owns the `Ragged` type), alongside `reverse_complement`.

`Ragged.to_numpy(allow_missing=...)` already exists but is awkward-backed and is
about *dense conversion of an already-rectangular* ragged array, not padding to
a target length. `seqpro.pad_seqs` pads *dense* sequence arrays, not `Ragged`.
So a flat-buffer `to_padded` on `Ragged` is genuinely new surface.

## Goal

Add `seqpro.rag.to_padded(rag, pad_value, *, length=None)`: a flat-buffer,
parallel densify-and-right-pad for `Ragged` arrays that is byte-identical to the
awkward idiom above, generic over the stored dtype (S1, integer, float).

Non-goals (YAGNI): left/both-side padding (gvl only right-pads); fixed trailing
dims / one-hot `(batch, None, K)` layouts (no current consumer; ragged-axis-last
only, matching `reverse_complement`); record-layout `Ragged` (convert fields
individually, like `to_numpy`).

## API

```python
def to_padded(
    rag: Ragged[RDTYPE],
    pad_value,
    *,
    length: int | None = None,
) -> NDArray[RDTYPE]:
    """Densify a Ragged into a right-padded rectilinear array via a flat-buffer kernel.

    Flat-buffer alternative to the awkward idiom
    ``Ragged(ak_str.rpad(rag, L, v)).to_numpy()`` / ``ak.pad_none + fill_none + to_numpy``:
    each row is copied once into a pre-filled output buffer in a single parallel pass.

    Parameters
    ----------
    rag
        Ragged array with exactly one ragged dimension and no fixed trailing
        dimensions (the ragged axis is last). Any fixed-itemsize dtype (S1,
        integer, float).
    pad_value
        Fill value for positions past each row's length. Must be castable to
        ``rag.data.dtype`` (e.g. ``b"N"`` for S1, ``-1`` for int32, ``0.0`` for
        float32); numpy raises on an incompatible value.
    length
        Target length of the (formerly ragged) last axis. ``None`` (default)
        uses ``rag.lengths.max()`` — the batch maximum. An explicit ``length``
        right-pads shorter rows and **truncates** longer rows to exactly
        ``length``.

    Returns
    -------
    NDArray
        Dense array of dtype ``rag.data.dtype`` and shape
        ``(*rag.shape[:rag_dim], out_len)`` where ``out_len`` is ``length`` or
        the batch max. A fresh allocation (densify cannot be in place).
    """
```

`pad_value` is positional (matches gvl's `to_padded(rag, pad_value)` so the
gvl call sites port with a one-line import swap). `length` is keyword-only.

## Implementation (approach A: one generic byte-copy kernel)

Densification is memory-bandwidth-bound, so a single dtype-agnostic kernel over
the `uint8` view is both simplest and as fast as dtype-specialized variants.

1. **Guards** (mirror `reverse_complement`):
   - record-layout (`_parts` is a dict) → `NotImplementedError`, convert fields individually;
   - any fixed trailing dim after `rag.rag_dim` → `ValueError` (ragged-axis-last only);
   - if `not rag.is_contiguous`: `rag = Ragged(ak.to_packed(rag))`.
2. **Resolve dims**: `offsets = ascontiguousarray(rag.offsets, int64)`;
   `n_rows = offsets.shape[0] - 1`; `out_len = length if length is not None else
   (int(rag.lengths.max()) if n_rows else 0)`; `itemsize = rag.data.dtype.itemsize`.
3. **Pre-fill output**: `out = np.full((n_rows, out_len), pad_value,
   dtype=rag.data.dtype)` — numpy performs the dtype cast (incl. `b"N"` → S1).
   `np.full` is C-contiguous, so the flat byte view is row-major.
4. **Copy kernel** (`@nb.njit(parallel=True, nogil=True, cache=True)`), over the
   `uint8` views of `out` and `rag.data`:
   ```python
   for i in nb.prange(n_rows):
       row_len = offsets[i + 1] - offsets[i]
       ncopy = min(row_len, out_len)            # truncation falls out for free
       src = offsets[i] * itemsize
       dst = i * out_len * itemsize
       for b in range(ncopy * itemsize):
           out_u1[dst + b] = data_u1[src + b]
   ```
   Padding needs no work — pre-filled positions past `ncopy` keep `pad_value`.
5. **Reshape leading dims**: if `rag` had leading non-ragged dims (e.g.
   `(batch, ploidy, None)`), reshape `out` from `(n_rows, out_len)` to
   `(*leading, out_len)` (the C-order flatten of the leading dims equals the
   ragged-row order, same contract as `reverse_complement`'s mask).

Then export `to_padded` from `seqpro/rag/__init__.py` and add it to `__all__`.

### Edge cases

| Case | Behavior |
|---|---|
| `n_rows == 0` | `out_len = 0` (avoid `max()` of empty); returns shape `(0, 0)` |
| all rows empty, `length=None` | `out_len = 0`; returns `(n_rows, 0)` |
| `length == 0` | every row truncated to empty; returns `(*leading, 0)` |
| row longer than `length` | truncated to `length` (first `length` elements kept) |
| `pad_value` not castable | numpy raises in `np.full` (no silent corruption) |
| non-contiguous offsets (post-slice) | packed via `ak.to_packed` before the kernel |

## Testing

- **Byte-identical to the awkward idiom** across randomized batches, for each
  consumer dtype/pad pairing gvl uses: S1 + `b"N"`, int32 + `-1`, int32 +
  `np.iinfo(int32).max`, float32 + `0.0`. Include zero-length rows, a single
  row, and an all-empty batch. Reference = the gvl awkward expression.
- **Fixed `length`**: pad case (`length > max`), exact case (`length == max`),
  truncate case (`length < max`), and `length == 0`.
- **Leading dims**: a `(batch, ploidy, None)` input densifies to
  `(batch, ploidy, out_len)` with rows in the right cells.
- **Guards**: record-layout raises; a `(batch, None, K)` trailing-dim input
  raises; non-contiguous (sliced) input still produces the correct result.
- **Microbench** (not a CI assertion): `to_padded` vs the awkward idiom at
  buffered-loader scale (~1024 rows × several-kb), reporting ms/call and peak
  allocation, as was done for `reverse_complement`.

## Downstream (GenVarLoader, separate PR)

Once released, gvl's `to_padded` in `_ragged.py` becomes a thin pass-through to
`seqpro.rag.to_padded` (same `(rag, pad_value)` positional contract), and gvl's
fixed-length `Ragged.to_numpy()` densify path can optionally route through
`to_padded(..., length=...)`. Out of scope for this seqpro spec; tracked in the
gvl REGRESSIONS doc's "0.6.1 throughput parity" section.
