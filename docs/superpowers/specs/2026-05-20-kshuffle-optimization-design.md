# K-Shuffle Rust Optimization

**Status:** Spec
**Date:** 2026-05-20
**Owner:** d-laub
**Scope:** `src/kshuffle.rs` (Rust extension) and its Python wrapper invariants in `python/seqpro/_modifiers.py`.

## Goal

Speed up `k_shuffle` on the dominant SeqPro workload — large batches of short DNA/RNA sequences (≫1 row, k ≤ 8, α = 4) — without changing the algorithm or its statistical guarantees. Target: ≥3× wall-clock improvement on a 10K × 200bp benchmark; expected 5–10× from removing per-row allocations and replacing the k-mer HashMap with a direct lookup table.

A secondary bug fix is included: user-supplied seeds currently produce identical shuffles for every row in a batch; this is corrected by deriving per-row seeds from a parent RNG.

## Non-goals

- API changes to the Python surface.
- Algorithm changes (output distribution must remain that of Altschul-Erickson + Wilson).
- Buffer pooling, SIMD, k=2 specialization, or merging Wilson's two passes — sketched below as "future (Approach C)".
- Optimizing protein / large-k workloads (they still work via the fallback path but aren't tuned).

## Architecture

`k_shuffle` and `k_shuffle1` keep their public signatures. Per-row work is restructured behind a small strategy trait selected once at the top of `k_shuffle1`:

```rust
trait KmerIndex {
    fn lookup(&self, kmer_code: u32, kmer_bytes: &[u8]) -> u32;  // -> vertex_id
    fn n_vertices(&self) -> u32;
}
```

Two implementations:

- **`DirectLut`** (fast path). Used when `α^(k-1) <= MAX_LUT` (16_384). Stores `Vec<u32>` of size `α^(k-1)`, mapping encoded (k−1)-mer → vertex id (`u32::MAX` = unseen). Built in one pass with a rolling integer encoder.
- **`HashLut`** (fallback). `HashMap<&'a [u8], u32, Xxh3Builder>` keyed on a borrowed slice into the input sequence — no per-lookup allocation. Used for protein / large k where the LUT would be too large.

Selection rule (deterministic per call):
```
use DirectLut iff alphabet_size.checked_pow((k-1) as u32) is Some(n) && n <= MAX_LUT
```

### Outer parallelization

Replace `par_bridge` over `rows()` with native parallel iteration:

```rust
let n_rows = seqs.len_of(Axis(0));
let row_seeds: Vec<u64> = {
    let mut parent = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None    => SmallRng::from_entropy(),
    };
    (0..n_rows).map(|_| parent.gen()).collect()
};

(seqs.axis_iter(Axis(0)), out.axis_iter_mut(Axis(0)), &row_seeds)
    .into_par_iter()
    .try_for_each(|(in_row, out_row, &seed)| {
        k_shuffle1(in_row, k, Some(seed), out_row, alphabet_size)
    })?;
```

This gives true work-stealing parallelism and a documented, deterministic seed contract: same `(seed, n_rows, k)` → byte-equal output across runs and across rayon thread counts.

## Components

### `kmer_encode` module (new, ~30 LoC)

- `build_byte_to_code(alphabet_bytes: &[u8]) -> [u8; 256]` — 256-entry table mapping any input byte to its alphabet code. Built once outside the parallel region and shared by `&`. For DNA this maps `A→0, C→1, G→2, T→3`; other bytes map to a sentinel (`u8::MAX`) which would indicate input the caller should have sanitized — current code does not check this either, and we keep that contract.
- `encode_first(seq: &[u8], k_minus_1: usize, b2c: &[u8; 256]) -> u32` — initial (k−1)-mer integer.
- `roll(prev: u32, drop_byte: u8, add_byte: u8, base_pow_km2: u32, alpha: u32, b2c: &[u8; 256]) -> u32` — rolling update: `(prev - b2c[drop]*base_pow_km2) * alpha + b2c[add]`. Branch-free.

### `Vertex`, simplified

```rust
struct Vertex {
    idx_offset: u32,
    n_indices:  u32,   // 0 sentinel (no Option)
    i_indices:  u32,
    next:       u32,
    i_sequence: u32,
    intree:     bool,
}
```

`u32` fields throughout. Sequence length is capped at `u32::MAX` (the current `usize` cap is also far above any realistic biological sequence). Halves memory footprint vs `usize` on 64-bit and improves cache behavior. `derive_builder` is removed; plain mutable construction.

### `k_shuffle1`, restructured

Four phases, total of 3 passes over the sequence (down from 5):

1. **Build index** (1 pass): rolling encoder + DirectLut (or HashLut). Assigns vertex ids on first sight.
2. **Allocate `vertices: Vec<Vertex>`** of size `n_vertices`; fill `n_indices` and `i_sequence` in 1 pass (the current code does these in two separate iterations).
3. **Prefix-sum `idx_offset`; allocate `indices: Vec<u32>`; populate adjacency** (1 pass over `zip(seq[..-1].windows, seq[1..].windows)`, reusing the rolling encoder so neither side re-hashes).
4. **Wilson → random_walk**: logic unchanged, field types switched from `usize` to `u32`.

The hot path performs zero allocations per k-mer lookup (vs `Vec<u8>` allocation on the current code's every lookup, ~3·L allocations per sequence).

## Data flow

```
ArrayView1<u8>
   │
   ▼  rolling encoder (uses shared byte_to_code[256])
encoded (k-1)-mer as u32
   │
   ▼  DirectLut::lookup   (O(1) array index)   ── or HashLut::lookup (borrowed-slice key)
vertex_id : u32
   │
   ▼
Vec<Vertex>, Vec<u32> indices
   │
   ▼
Wilson random arborescence → random_walk → ArrayViewMut1<u8> out
```

## Invariants preserved

1. **Output distribution unchanged.** Same algorithm, same RNG family (`SmallRng`), same per-row sampling pattern.
2. **k-mer frequency preservation** — the algorithm's whole point — is unaffected; only the lookup data structure changes.
3. First and last `k-1` bytes of output equal those of input.
4. Existing `KShuffleError` semantics for `k < 1`, `k >= L`, `n_lets` too small.

## Seed contract (new, documented)

- `seed = None` → parent RNG `SmallRng::from_entropy()`; non-reproducible (matches current behavior).
- `seed = Some(s)` → parent RNG `SmallRng::seed_from_u64(s)`; row `i` uses the parent's `i`-th `u64` draw. Same `(s, n_rows, k)` produces byte-identical output across runs and across rayon thread counts. Changing batch size changes per-row seeds (acceptable: matches typical numpy `default_rng` expectations).

This is a behavior change relative to current code, which seeded every row from the same `s`. Documented in the function docstring and the changelog.

## Error handling

- Existing `KShuffleError` variants kept and checked at the top of `k_shuffle1` before any optimization paths run.
- Add `assert!(alphabet_size >= 2)` at entry (size-1 alphabets are not meaningful for k-shuffle).
- Replace `unsafe { Array::uninit(...).assume_init() }` with `Array::from_elem(shape, 0u8)`. Sound either way for `u8`, but removes a code-review trip-hazard.

## Testing

1. **Equivalence test (Rust).** Keep the current HashMap implementation as a test-only `kshuffle_ref` module. For each k ∈ {2..8}, fixed seed, random sequences of length ∈ {16, 64, 256, 1024}: assert byte-equal output between `kshuffle_ref` and the new `DirectLut` path. Proves the optimization is a pure refactor.
2. **K-mer frequency invariant (Rust, proptest).** Upgrade the existing `same_freq` test to use `proptest`: for random DNA sequences (length 8..1024) and k ∈ {2..8}, assert input and output k-mer multisets are equal.
3. **Determinism test.** `(seed, batch)` → byte-equal output across runs, across `RAYON_NUM_THREADS ∈ {1, 4}`.
4. **Benchmark (`benches/kshuffle.rs`, criterion).** 10K × 200bp, k ∈ {2, 4, 6, 8}. Report median and before/after ratio in the PR.

## Verification gates

- All existing tests pass.
- Equivalence test passes for k ∈ {2..8}.
- Benchmark shows ≥3× improvement on the target workload, else investigate before declaring success.
- `cargo clippy --all-targets -- -D warnings` clean.

## Future direction (Approach C, deferred)

If the target benchmark still shows headroom after this work:

- **Thread-local buffer pools** for `vertices`, `indices`, and the LUT, reused across rows on each rayon worker. Eliminates the remaining per-row `Vec` allocations.
- **SIMD k-mer encoding** for the rolling encoder (likely diminishing returns once allocations are gone).
- **Merge Wilson's two traversal passes** (currently does an erase-loop then a follow-loop separately).
- **k=2 specialization** (dinucleotide shuffle has a known simple Eulerian-circuit form).
- **`DirectLut` as a thread-local `Box<[u32]>`** sized to the project-wide max, reset by index-list rather than zeroed in full each call.

These are intentionally out of scope here. Each would warrant its own benchmark-driven decision.
