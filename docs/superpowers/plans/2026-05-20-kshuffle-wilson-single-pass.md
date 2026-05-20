# Single-Pass Wilson Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Wilson's two-pass loop-erased random walk with a single-pass cycle-detection variant; keep only if it shows ≥ 3% improvement on at least one k.

**Architecture:** Add a `path: Vec<u32>` to `ShuffleBuffers` and an `on_path: bool` to `Vertex`. The new `wilson_random_spanning_tree` walks once, pushing each visited vertex onto `path` and marking `on_path = true`. When a revisit is detected (target's `on_path` is true), pop the stack back to the duplicate (clearing `on_path` on popped entries) — this is loop erasure. When the walk hits the tree, commit `path` to the tree by setting `intree = true` on each entry and clearing `on_path`. Output distribution is identical to the current impl (same RNG calls in same order); the equivalence test catches any divergence.

**Tech Stack:** Rust (rand, rayon, ndarray). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-kshuffle-wilson-single-pass-design.md`

---

## File Structure

**Will modify:**
- `src/kshuffle.rs` — add `on_path` to `Vertex`, add `path` to `ShuffleBuffers`, rewrite `wilson_random_spanning_tree`, update the one call site in `k_shuffle1_inner`.

No new files, no dependencies.

**Test-environment note:** Use `pixi run -e dev cargo test --lib <pattern>` for tests. Use `DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 2 --measurement-time 5` for benches. Bare `cargo test` and bare `cargo bench` fail with libpython rpath errors.

---

## Task 1: Add storage — `on_path` on `Vertex`, `path` on `ShuffleBuffers`

**Files:**
- Modify: `src/kshuffle.rs`

- [ ] **Step 1: Add `on_path: bool` to `Vertex` struct**

Find the `Vertex` struct (currently around lines 184-192 of `src/kshuffle.rs`):

```rust
#[derive(Clone, Default)]
struct Vertex {
    idx_offset: u32,
    n_indices: u32,   // 0 = sentinel
    i_indices: u32,
    next: u32,
    i_sequence: u32,
    intree: bool,
}
```

Replace with:

```rust
#[derive(Clone, Default)]
struct Vertex {
    idx_offset: u32,
    n_indices: u32,   // 0 = sentinel
    i_indices: u32,
    next: u32,
    i_sequence: u32,
    intree: bool,
    on_path: bool,    // true while vertex sits on the current Wilson walk's stack
}
```

(Adds 1 byte; likely fits in `intree`'s existing 3-byte padding tail, so the struct size should not grow.)

- [ ] **Step 2: Add `path: Vec<u32>` to `ShuffleBuffers`**

Find the `ShuffleBuffers` struct (currently around lines 25-39 of `src/kshuffle.rs`). It looks like:

```rust
pub(crate) struct ShuffleBuffers {
    lut: Box<[u32]>,
    lut_written: Vec<u32>,
    codes: Vec<u32>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}
```

Add a `path` field at the end:

```rust
pub(crate) struct ShuffleBuffers {
    lut: Box<[u32]>,
    lut_written: Vec<u32>,
    codes: Vec<u32>,
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    path: Vec<u32>,
}
```

In `impl ShuffleBuffers::new()`, add the corresponding initializer:

```rust
pub(crate) fn new() -> Self {
    Self {
        lut: vec![u32::MAX; MAX_LUT as usize].into_boxed_slice(),
        lut_written: Vec::new(),
        codes: Vec::new(),
        vertices: Vec::new(),
        indices: Vec::new(),
        path: Vec::new(),
    }
}
```

- [ ] **Step 3: Verify it still builds**

```bash
pixi run -e dev cargo check --lib
```
Expected: clean. (The new field is unused for now — a `dead_code` warning is acceptable until Task 2.)

- [ ] **Step 4: Run all existing tests to confirm no regression**

```bash
pixi run -e dev cargo test --lib
```
Expected: all tests pass. The added field is initialized to `false` via `Default` and is currently unread.

- [ ] **Step 5: Commit**

```bash
git add src/kshuffle.rs
git commit -m "feat: add on_path flag to Vertex and path stack to ShuffleBuffers"
```

---

## Task 2: Rewrite `wilson_random_spanning_tree` as single-pass

**Files:**
- Modify: `src/kshuffle.rs`

- [ ] **Step 1: Replace `wilson_random_spanning_tree` function body**

Find the existing `fn wilson_random_spanning_tree<R: Rng>(...)` (currently around lines 304-353 of `src/kshuffle.rs`). It uses two passes per starting vertex. Replace the entire function with:

```rust
fn wilson_random_spanning_tree<R: Rng>(
    vertices: &mut [Vertex],
    indices: &[u32],
    root_idx: usize,
    rng: &mut R,
    path: &mut Vec<u32>,
) {
    vertices[root_idx].intree = true;

    for i in 0..vertices.len() {
        if vertices[i].intree {
            continue;
        }
        debug_assert!(path.is_empty());

        let mut u_idx = i;
        loop {
            // Walk hits the existing tree → commit `path` and stop.
            if vertices[u_idx].intree {
                break;
            }
            // Walk revisits a vertex already on the current path → loop erasure.
            if vertices[u_idx].on_path {
                // Pop until the stack top is u_idx; clear on_path on popped entries.
                while let Some(&top) = path.last() {
                    if top as usize == u_idx {
                        break;
                    }
                    vertices[top as usize].on_path = false;
                    path.pop();
                }
                // u_idx is now back at the top of the path; on_path[u_idx] stays true.
            } else {
                vertices[u_idx].on_path = true;
                path.push(u_idx as u32);
            }

            // Re-roll u's next choice on every visit (matches original behavior).
            let u = &mut vertices[u_idx];
            u.next = rng.gen_range(0..u.n_indices);
            u_idx = indices[(u.idx_offset + u.next) as usize] as usize;
        }

        // Commit the loop-erased path to the tree.
        for &v in path.iter() {
            let vert = &mut vertices[v as usize];
            vert.intree = true;
            vert.on_path = false;
        }
        path.clear();
    }
}
```

Note the new `path: &mut Vec<u32>` parameter at the end of the signature.

- [ ] **Step 2: Update the call site in `k_shuffle1_inner`**

Find the call (currently around line 298 of `src/kshuffle.rs`):

```rust
wilson_random_spanning_tree(&mut buffers.vertices, &buffers.indices, root_idx, rng);
```

This needs a split borrow because we now pass `&mut buffers.vertices` and `&mut buffers.path` from the same struct. Replace with:

```rust
let ShuffleBuffers {
    ref mut vertices,
    ref indices,
    ref mut path,
    ..
} = *buffers;
wilson_random_spanning_tree(vertices, indices, root_idx, rng, path);
random_walk(vertices, &mut buffers.indices, root_idx, rng, seq, k, out);
```

That destructure trips up the subsequent `random_walk` call because `random_walk` needs `&mut indices` but the destructure borrows it as `&indices`. The cleanest workaround: don't destructure; instead, use explicit raw fields with `&mut` carefully. Replace the original two-line call site:

```rust
wilson_random_spanning_tree(&mut buffers.vertices, &buffers.indices, root_idx, rng);
random_walk(&mut buffers.vertices, &mut buffers.indices, root_idx, rng, seq, k, out);
```

with:

```rust
// Wilson needs &mut vertices, &indices, &mut path — three disjoint borrows
// from buffers. Rust's borrow checker allows simultaneous borrows of
// distinct fields, but we have to spell them out at the call site rather
// than rely on a single &mut buffers.
wilson_random_spanning_tree(
    &mut buffers.vertices,
    &buffers.indices,
    root_idx,
    rng,
    &mut buffers.path,
);
random_walk(
    &mut buffers.vertices,
    &mut buffers.indices,
    root_idx,
    rng,
    seq,
    k,
    out,
);
```

This works because Rust permits multiple borrows of distinct named fields of a struct (split borrows).

- [ ] **Step 3: Build**

```bash
pixi run -e dev cargo check --lib
```
Expected: clean. If the borrow checker complains about overlapping borrows in `k_shuffle1_inner`, the most likely fix is to ensure each call uses fully-qualified `buffers.field` syntax (not aliased through a local) so that the split-borrow analysis sees distinct fields.

- [ ] **Step 4: Run all Rust tests, including the equivalence test**

```bash
pixi run -e dev cargo test --lib
```
Expected: all tests pass. The crucial ones are:
- `kshuffle::test::equivalence_with_reference_impl_for_k_2_through_8` — byte-equal vs the original reference impl; catches algorithmic drift.
- `kshuffle::test::k_shuffle_preserves_kmer_frequencies` — proptest, 256 cases.
- `kshuffle::test::buffer_reuse_matches_fresh_buffer_output` — ensures `path` is cleared between rows.
- `kshuffle::test::determinism_across_runs_and_thread_counts` — RNG order preserved.

**If the equivalence test fails:** the most likely cause is a subtle ordering bug. Specifically:
- Verify `path` is empty at the start of each `i`-iteration (the `debug_assert!` will catch this).
- Verify that on a revisit, we pop entries above `u_idx` but leave `u_idx` itself on the stack with `on_path` still true.
- Verify that `rng.gen_range` is called at every visit (including the revisit), so the RNG sequence matches.

- [ ] **Step 5: Python tests**

```bash
pixi run -e dev maturin develop
pixi run -e dev pytest tests/test_modifiers.py::test_k_shuffle tests/test_shape_matrix.py -v
```
Expected: pass.

- [ ] **Step 6: Clippy**

```bash
pixi run -e dev cargo clippy --all-targets -- -D warnings
```
Expected: clean. Likely candidates if not: `needless_range_loop` (use `iter().enumerate()`), `manual_swap`, `ptr_arg`.

- [ ] **Step 7: Commit**

```bash
git add src/kshuffle.rs
git commit -m "perf: replace two-pass Wilson with single-pass loop-erased random walk"
```

---

## Task 3: Verify perf gate

**Files:** maybe revert `src/kshuffle.rs`. Bench-only otherwise.

- [ ] **Step 1: Capture the baseline (before this PR's Wilson change)**

The baseline numbers from before this work, on the same branch:

| k | Baseline (post-pooling, pre-Wilson-merge) |
|---|------|
| 2 | 7.21 ms |
| 4 | 7.21 ms |
| 6 | 7.15 ms |
| 8 | 7.16 ms |

(These come from the final benchmark in the prior plan's Task 7. If you want to regenerate them, run on HEAD~3 — the commit before "add on_path flag to Vertex".)

- [ ] **Step 2: Run the benchmark with the single-pass Wilson**

```bash
DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 2 --measurement-time 5 2>&1 | tee /tmp/wilson_single_pass_bench.txt
```

Capture the medians for k ∈ {2, 4, 6, 8}.

- [ ] **Step 3: Apply the spec gate**

Decision rule (from the spec):

- **Keep** the change if ≥ 3% faster on at least one of k ∈ {2, 4, 6, 8}, with no regression > 2% on any other k.
- **Revert** otherwise.

Compute `new_median / baseline` for each k. A value < 0.97 means ≥ 3% faster. A value > 1.02 means > 2% regression.

- [ ] **Step 4: Either commit a note or revert**

**If kept:**
```bash
git commit --allow-empty -m "bench: confirm single-pass Wilson meets perf gate (k=X: A.AAx, k=Y: B.BBx, ...)"
```

**If reverted:** revert both Task 1 and Task 2 commits. The cleanest way:

```bash
# Identify the two commits to revert (the Wilson rewrite and the storage addition).
git log --oneline -5
# Revert in reverse order (newer first).
git revert --no-edit <wilson-rewrite-sha>
git revert --no-edit <storage-addition-sha>
```

Then run tests once more to confirm the revert is clean:

```bash
pixi run -e dev cargo test --lib
pixi run -e dev cargo clippy --all-targets -- -D warnings
```

Both should pass and be clean.

- [ ] **Step 5: Final verification**

Whether kept or reverted, end the task by confirming:

```bash
pixi run -e dev cargo test --lib
pixi run -e dev maturin develop
pixi run -e dev pytest tests/test_modifiers.py::test_k_shuffle -v
pixi run -e dev cargo clippy --all-targets -- -D warnings
```

All should pass / be clean.
