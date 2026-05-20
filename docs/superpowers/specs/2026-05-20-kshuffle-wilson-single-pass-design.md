# K-Shuffle: Single-Pass Wilson

**Status:** Spec
**Date:** 2026-05-20
**Owner:** d-laub
**Scope:** `src/kshuffle.rs::wilson_random_spanning_tree`.

## Goal

Merge Wilson's current two-pass loop-erased random walk into a single pass with explicit cycle-detection. Eliminates the second traversal per starting vertex.

## Non-goals

- Algorithm changes.
- Touching `random_walk` or the index-build phases.

## Approach

For each `i` not yet in the tree, walk randomly along outgoing edges. Maintain `path: Vec<u32>` — the stack of vertex ids on the current walk — and `on_path: bool` per vertex.

- Visit a vertex `u`:
  - If `intree[u]`: stop. Commit `path` to the tree.
  - Else if `on_path[u]`: pop the stack until its top is `u` (clearing `on_path` on popped entries). This is loop erasure.
  - Else: push `u` onto `path`, set `on_path[u] = true`.
  - Roll `vertices[u].next = rng.gen_range(0..vertices[u].n_indices)`. Step to next.

When the walk hits the tree, commit `path`: for each `v` in `path`, set `intree[v] = true` and `on_path[v] = false`. Clear `path`.

**Why this preserves the reference algorithm's output:** the rolls of `next` happen at every visit, in the same order, for the same vertex ids, with the same RNG state. The set of vertices left with `intree = true` after each `i`-iteration is exactly the loop-erased path — identical to what the current code's pass 2 marks.

## Changes

- `src/kshuffle.rs`:
  - Add `on_path: bool` to `Vertex`. Likely fits in existing padding (no size increase).
  - Add `path: Vec<u32>` to `ShuffleBuffers`. Cleared between calls (already follows the existing pool-clear pattern).
  - Replace `wilson_random_spanning_tree` body with the single-pass algorithm above. Signature changes only by adding `path: &mut Vec<u32>` (or carrying via `&mut ShuffleBuffers`).
  - In `k_shuffle1_inner` and (if any other caller) update the call site to pass the buffer.

No changes to the public API or `random_walk`.

## Invariants

1. **Output distribution identical** to the current impl. The equivalence test (byte-equal vs the reference snapshot for k ∈ {2..8}) is the safety net.
2. **k-mer frequency preservation** unaffected.
3. **No allocations on hot path** — `path` is pooled.
4. **No data races** — `path` lives on `ShuffleBuffers`, which is per-rayon-worker.

## Testing

- All existing tests must pass (equivalence vs reference for k ∈ {2..8}, proptest k-mer frequency invariant, determinism across thread counts, buffer-reuse correctness, Python suite).
- No new tests required — the equivalence test already exercises every Wilson code path with random sequences and seeds, and would catch any divergence from the original.
- `cargo clippy --all-targets -- -D warnings` clean.

## Verification gate

Run the 10K × 200bp criterion benchmark before and after. Decision rule:

- **Keep the change if ≥ 3% faster on at least one of k ∈ {2, 4, 6, 8}**, with no regression > 2% on any other k.
- **Otherwise revert.** Code complexity that doesn't earn measurable improvement gets dropped.

The equivalence test guarantees correctness either way; the gate is purely about whether the simpler-by-traversal-count code is also faster.

## Out of scope

SIMD k-mer encoding and other Approach C optimizations remain deferred. After this work ships (or is reverted), the remaining Approach C item is SIMD-only.
