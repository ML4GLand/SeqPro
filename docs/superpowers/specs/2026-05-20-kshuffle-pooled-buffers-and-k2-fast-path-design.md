# K-Shuffle: Pooled Buffers + k=2 Fast Path

**Status:** Spec
**Date:** 2026-05-20
**Owner:** d-laub
**Scope:** `src/kshuffle.rs`. Builds on the optimization shipped in `2026-05-20-kshuffle-optimization-design.md`.

## Goal

Two related optimizations in a single PR:

1. **Sparse-reset thread-local lookup buffer** — eliminate the per-row 64 KB LUT zero-fill that dominates k=8. Target: k=8 ≥ 1.8× faster (≈18.4 ms → ≈9 ms or better on the 10K × 200bp benchmark), with no regression on smaller k.
2. **k=2 fast path** — specialize the dinucleotide-shuffle case to skip LUT and codes entirely (`vertex_id = b2c[seq[i]]`). Kept in the PR only if it shows ≥ 15% improvement over the pooled general path at k=2; otherwise dropped.

This addresses items 1 and 3 (in the priority order from the investigation) of the deferred "Approach C" from the prior spec.

## Non-goals

- API changes.
- Algorithm changes.
- Pooling `HashLut` (rare path; keep simple).
- SIMD k-mer encoding, merge of Wilson passes, k=3+ specializations — deferred.

## Investigation summary (background)

The 10K × 200bp criterion benchmark on the current `feat/kshuffle-rust-opt`:

| k | DirectLut, full LUT zero-fill | HashLut (force-routed) |
|---|-------------------------------|------------------------|
| 2 | 6.79 ms | 6.65 ms |
| 4 | 6.96 ms | 6.55 ms |
| 6 | 7.13 ms | 8.61 ms |
| 8 | **18.44 ms** | **8.96 ms** |

Routing k=8 through the HashMap halves its runtime — the cost wasn't the algorithm, it was zeroing a fresh 16K-entry `Vec<u32>` once per row × 10K rows. Sparse reset of a reused buffer avoids the zero-fill entirely.

## Architecture

One new type `ShuffleBuffers` owns all per-thread reusable state and is threaded through `k_shuffle1` by `&mut`. Each rayon worker gets one via `map_init`.

```rust
pub(crate) struct ShuffleBuffers {
    lut: Box<[u32]>,       // length MAX_LUT, initialized to u32::MAX
    lut_written: Vec<u32>, // codes written this row; used for sparse reset
    codes: Vec<u32>,       // n_windows codes; reused
    vertices: Vec<Vertex>, // reused
    indices: Vec<u32>,     // reused
}
```

`ShuffleBuffers::new()` allocates `Box<[u32; MAX_LUT]>` once. Sparse reset restores only the written positions to `u32::MAX`. The `codes`, `vertices`, `indices` Vecs are cleared and re-grown per row — small enough that their cost is dominated by use, not by reallocation, but pooling them removes the per-row alloc/free overhead anyway.

The existing `KmerIndex` trait and `DirectLut` struct are removed; their logic becomes methods on `ShuffleBuffers`. `HashLut` survives as a small private helper for the fallback path (protein / large k), without pooling — it keeps its current per-row allocation.

### k=2 specialization

In `k_shuffle1`, after argument checks, dispatch:

```rust
if k == 2 {
    return k_shuffle1_k2(seq, &mut rng, out, alphabet_bytes, buffers);
}
// existing general path follows...
```

`k_shuffle1_k2` does one pass over the sequence with `vertex_id = b2c[seq[i]]` inline, tracking which vertex ids have been seen via a stack-allocated `[bool; 256]`. It populates `buffers.vertices` and `buffers.indices` directly without touching `buffers.lut` or `buffers.codes`. It then calls `wilson_random_spanning_tree` and `random_walk` unchanged.

### Outer `k_shuffle`

Replace the existing `par_bridge().map(...)` with `par_bridge().map_init(ShuffleBuffers::new, ...)`. Per-row seed derivation stays as-is.

## Components

**`ShuffleBuffers`** — the new struct above. Public to the crate, private to `kshuffle.rs`.
- `new() -> Self` — allocates the boxed LUT, leaves Vecs empty.
- `reset_lut(&mut self)` — restores `u32::MAX` at every position in `lut_written`, then clears `lut_written`.
- `build_direct_index(&mut self, seq, k_minus_1, alphabet_size, b2c) -> u32` — replaces `DirectLut::build`. Resets LUT, walks `seq` with rolling encoder, fills `self.codes` and `self.lut`, returns `n_vertices`. The caller reads codes via `&self.codes`.
- `build_hash_index(&mut self, seq, k_minus_1) -> u32` — fallback path. Clears `self.codes`, builds a per-call `HashMap<&[u8], u32, Xxh3Builder>`, fills `self.codes` with vertex ids in window order, returns `n_vertices`. No long-lived hash state.

**`k_shuffle1` signature change**
```rust
fn k_shuffle1(
    seq: ArrayView1<u8>,
    k: usize,
    seed: Option<u64>,
    out: ArrayViewMut1<u8>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
    buffers: &mut ShuffleBuffers,
) -> Result<()>;
```

**`k_shuffle1_inner` updated** to use `buffers` for `codes`, `vertices`, `indices`. The current `window_vid` local Vec disappears (it was a copy of `codes` after `lut.lookup`; we now use `buffers.codes` directly as the vertex-id sequence).

**`k_shuffle1_k2` (new)** — ~50 LoC. Inline vertex resolution, no LUT, no codes.

**`wilson_random_spanning_tree` and `random_walk`** unchanged (already take `&mut [Vertex]` / `&mut [u32]`).

## Data flow

**General path:**
```
seq + buffers
  ↓ reset_lut(); codes.clear(); vertices.clear(); indices.clear()
  ↓ build_direct_index → fills lut, lut_written, codes; returns n_vertices
  ↓ vertices.resize_with(n_vertices, Vertex::default)
  ↓ fill n_indices, i_sequence from codes (one pass)
  ↓ prefix-sum idx_offset
  ↓ indices.resize(n_lets - 1, 0); fill from codes pairs
  ↓ Wilson → random_walk → out
```

**k=2 path:**
```
seq + buffers
  ↓ vertices.clear(); indices.clear()
  ↓ one pass: vertex_id = b2c[seq[i]]; track seen in [bool; 256]
  ↓ vertices.resize_with(n_vertices, Vertex::default)
  ↓ fill n_indices, i_sequence inline
  ↓ prefix-sum, fill indices
  ↓ Wilson → random_walk → out
```

## Invariants

1. **Output distribution identical** to the current `feat/kshuffle-rust-opt` impl. The LUT change is a storage refactor; the k=2 path constructs the same graph by a different route. The existing equivalence-with-reference test (k ∈ {2..8}) must still pass — including for k=2, which proves the specialization matches.
2. **k-mer frequency preservation** unchanged.
3. **Per-row seed determinism** unchanged — buffers carry no state across rows (every row begins with reset/clear).
4. **No data races** — each rayon worker owns its own `ShuffleBuffers` via `map_init`. Cross-thread reuse only of immutable inputs.

## Error handling

No new error paths. `ShuffleBuffers::new` is infallible (OOM panics, as today). The k=2 path's `[bool; 256]` is stack-allocated. Existing `KShuffleError` variants and `assert!(alphabet_size >= 2)` unchanged.

## Testing

1. **All existing tests must pass** — `equivalence_with_reference_impl_for_k_2_through_8`, `k_shuffle_preserves_kmer_frequencies` (proptest, 256 cases), `determinism_across_runs_and_thread_counts`, the `direct_lut_*` and `hash_lut_*` unit tests (some may need adaptation to the new `ShuffleBuffers`-based API), full Python suite.

2. **New: buffer-reuse correctness test.** Call `k_shuffle1` repeatedly with the *same* `ShuffleBuffers` on different sequences/seeds and compare each call's output to the same call with a fresh `ShuffleBuffers`. Catches missing-reset bugs. Cover k ∈ {2, 4, 6, 8}, ≥ 5 sequences per k.

3. **Determinism across thread counts** — re-run existing test under `RAYON_NUM_THREADS=1` and `4`.

## Verification gates (mandatory before merging)

### Correctness

- All tests above pass.
- `cargo clippy --all-targets -- -D warnings` clean.

### Performance — LUT pooling

Measured against `feat/kshuffle-rust-opt` HEAD on the 10K × 200bp criterion benchmark:

- **k=8: ≥ 1.8× faster** (target ≈ 9 ms or better; current 18.44 ms).
- **k=2, 4, 6: no regression** (within ±5%).

If k=8 misses the gate, investigate before merging — likely culprit is a bug in `reset_lut` or `build_direct_index` (e.g., resetting too much or zeroing the LUT in full despite the written list).

### Performance — k=2 specialization

After LUT pooling is in place, compare k=2 with the specialization vs k=2 forced through the general (pooled) path:

- **k=2 must be ≥ 15% faster with the specialization.**
- If < 15%: **revert the k=2 specialization** before merging. The LUT pooling stays. Code complexity that doesn't earn its keep gets dropped, not paid for in hope.

## Out of scope (deferred)

- Pooling for `HashLut`.
- SIMD k-mer encoding.
- Merging Wilson's two traversal passes.
- k=3 / k=4 specializations.
- Larger benchmarks (e.g., 1M × 100bp, 100 × 10Kbp) — relevant later but not gating this PR.

These remain on the Approach C wishlist; decide after this PR ships and we have new measurements.
