**Date:** 2026-05-31
**Status:** Approved

## Goal

Add a Numba-parallelized `to_packed()` to `seqpro.rag` that replaces `awkward.to_packed()` for `Ragged` arrays. Separate work found a parallel gather-copy kernel is substantially faster than `ak.to_packed`, even when the source buffer is a `np.memmap`. This work:

1. Adds `Ragged.to_packed()` (method) and `seqpro.rag.to_packed()` (free function).
2. Swaps the internal `ak.to_packed` call sites to the new implementation.
3. Removes the dead `as_contiguous` packing path in `unbox`.
4. Microbenchmarks throughput to (a) verify it beats `ak.to_packed`, and (b) rank which input parameters most affect throughput.

## Why a kernel wins

`ak.to_packed` on a single-ragged-dim array gathers each row's `data[start:stop]` into a fresh contiguous buffer and rewrites offsets to start at zero. That is exactly a parallel gather-copy: one contiguous read + one contiguous write per row. A Numba `prange` kernel does this directly, scales across cores, and keeps each row a single contiguous span — which is what makes it bandwidth-friendly and fast even against a memmapped source.

## API

A method on `Ragged` plus a free function in `rag/_ops.py` (mirroring `reverse_complement`); the method delegates to the free function.

```python
# python/seqpro/rag/_array.py — method on Ragged
def to_packed(self, copy: bool = True) -> Ragged[RDTYPE_co]: ...

# python/seqpro/rag/_ops.py — free function
def to_packed(rag: Ragged, *, copy: bool = True) -> Ragged: ...
```

`__all__` in `_ops.py` gains `"to_packed"`.

### Copy semantics

- `copy=True` *(default)* — always returns a freshly allocated, contiguous, zero-based `Ragged` with **1-D** offsets. You own the buffer (safe to then mutate in place, e.g. feed `reverse_complement(copy=False)`). Already-packed input is still copied.
- `copy=False` — zero-copy passthrough **iff** the input is already packed (1-D offsets, `offsets[0] == 0`, C-contiguous data of exactly `offsets[-1] * elem` bytes). Otherwise raises `ValueError`. This is an assertion ("give me the buffer without copying, or fail"), never a silent allocation.

Unlike `view`/`to_numpy`, the method is defined for record-layout `Ragged` (packing fields is well-defined).

## Implementation — the kernel (`rag/_ops.py`)

Approach A: a single generic byte-view kernel. The data buffer is viewed as raveled `uint8`; offsets are byte-scaled by `elem = itemsize * prod(trailing_fixed_dims)`. One kernel covers every dtype, trailing fixed dims, and bytes.

```python
@nb.njit(parallel=True, nogil=True, cache=True)
def _pack(src_bytes, starts, stops, out_bytes, out_starts):
    for i in nb.prange(starts.shape[0]):
        n = stops[i] - starts[i]
        out_bytes[out_starts[i]:out_starts[i] + n] = src_bytes[starts[i]:stops[i]]
```

Driver (plain NumPy, outside the kernel):

1. `unbox(rag)` (zero-copy) → `(data, offsets, shape)`. Detect 1-D offsets (`ListOffsetArray`) vs 2-D `[starts, stops]` (`ListArray`).
2. Derive per-row `starts`/`stops`:
   - 1-D offsets: `starts = offsets[:-1]`, `stops = offsets[1:]`.
   - 2-D offsets: `starts = offsets[0]`, `stops = offsets[1]`.
3. `elem = itemsize * prod(trailing_fixed_dims)`. View `data` as raveled `uint8`; byte-scale `starts`/`stops` by `elem`.
4. `lengths = stops - starts`; `out_offsets = concat([0], cumsum(lengths))` (element units); byte-scaled `out_starts` for the kernel.
5. Already-packed test (see copy semantics). If packed: `copy=False` → return input as-is; `copy=True` → copy.
6. `out_bytes = np.empty(out_offsets[-1] * elem, uint8)`; run `_pack`.
7. Rebuild via `Ragged.from_offsets(out_view, shape, out_offsets_1d)`. **Output always has 1-D offsets.**

### Record layouts

Loop the kernel over each field's data buffer, reusing one shared (byte-scaled) offsets computation. Reassemble as a record `Ragged` with shared 1-D offsets.

## Swap internal call sites

Replace `Ragged(ak.to_packed(rag))` with `rag.to_packed()` — or `.to_packed(copy=False)` where ownership is already guaranteed — at the call sites in:

- `python/seqpro/_encoders.py` (4 sites)
- `python/seqpro/alphabets/_alphabets.py` (1 site; note the following `seqs.offsets` 1-D assumption still holds)
- `python/seqpro/rag/_ops.py` (`reverse_complement` non-contiguous fallback)

Existing tests guard correctness.

## Remove dead `as_contiguous` path in `unbox`

`unbox(arr, as_contiguous=False)` packs via `ak.to_packed` when `as_contiguous=True`:

```python
if as_contiguous:
    arr = ak.to_packed(arr)
```

No caller in the package or tests passes `as_contiguous=True` — every call uses the default. Remove the `as_contiguous` parameter and its branch so `to_packed` is the single packing implementation in the module. Update the docstring (drop the "guaranteed zero-copy if `as_contiguous` is False" caveat — it is now always zero-copy).

## Benchmark (`benchmarks/bench_to_packed.py`, `bench` env)

Standalone argparse script → tidy CSV + seaborn figures. Three implementations compared:

- **`seqpro`** — the new kernel.
- **`ak.to_packed`** — primary baseline.
- **pure-NumPy gather** — secondary baseline (`np.repeat`/`cumsum` index trick; serial).

Metric: **throughput (GB/s)** = bytes packed / wall-time, plus speedup ratio over `ak.to_packed`. Numba JIT is warmed before timing; report median of N repeats.

Swept axes (values illustrative, tunable via argparse):

| Axis | Values |
|---|---|
| `n_rows` | 1e3, 1e4, 1e5, 1e6 |
| mean row length | 16, 256, 4096 |
| length distribution | uniform vs long-tailed (load imbalance across threads) |
| source | in-RAM vs `np.memmap` |
| dtype itemsize | uint8 (S1) vs float64 |
| threads | `NUMBA_NUM_THREADS` ∈ {1, 2, 4, 8} |

Outputs:

- Tidy CSV: one row per (config × impl) with throughput, time, speedup.
- Faceted seaborn plots: throughput vs `n_rows` faceted by length / source / dtype; a thread-scaling line plot.
- Printed summary ranking axes by effect size on the speedup ratio (goal 2).

## Tests (`tests/`)

TDD against `rag/_ops.py` / the method. Correctness compared to `ak.to_packed`:

- 2-D starts/stops (`ListArray`, e.g. after a reversing slice) — core unpacked case.
- Trailing fixed dims (OHE `(total, 4)`).
- Record layout (multi-field SoA).
- Already-packed passthrough: `copy=False` returns the same buffer (identity check); `copy=True` returns a distinct buffer.
- `copy=False` raises `ValueError` on unpacked input.
- dtypes: bytes (`S1`) and `float64`.
- Empty rows / zero-length entries.

## Skill update

`skills/seqpro/SKILL.md` gains a `Ragged.to_packed()` entry (new public feature — required by `CLAUDE.md`).

## Out of scope

- Multi-ragged-dimension arrays (`Ragged` is single-ragged-dim by construction).
- Union-type layouts (unsupported by `Ragged`).
- Changing the on-disk / awkward layout conventions.
