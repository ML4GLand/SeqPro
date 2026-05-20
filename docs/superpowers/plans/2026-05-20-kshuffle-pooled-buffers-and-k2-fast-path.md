# K-Shuffle Pooled Buffers + k=2 Fast Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the per-row 64 KB LUT zero-fill that dominates k=8 (~2× win expected) by introducing a thread-local sparse-reset `ShuffleBuffers`, and add a k=2 specialization that is kept only if it shows ≥ 15% improvement over the pooled general path.

**Architecture:** A single `ShuffleBuffers` struct owns reusable storage (LUT + written-list, codes, vertices, indices). Each rayon worker gets one via `map_init`. The old `KmerIndex` trait and `DirectLut` / `HashLut` structs are removed; their logic becomes methods on `ShuffleBuffers`. Hash fallback is inlined into one method (no pooled hash state). A `k=2` branch in `k_shuffle1` dispatches to a specialized worker that skips LUT/codes entirely.

**Tech Stack:** Rust (rayon `par_bridge` + `map_init`, ndarray, rand). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-kshuffle-pooled-buffers-and-k2-fast-path-design.md`

---

## File Structure

**Will modify:**
- `src/kshuffle.rs` — add `ShuffleBuffers`, migrate `DirectLut`/`HashLut` to methods, change `k_shuffle1` signature, update `k_shuffle` to `map_init`, add k=2 specialization, adapt existing tests.

No new or deleted files. No new dependencies.

**Test-environment note (applies to every task):**
- Rust unit tests: `pixi run -e dev cargo test --lib <pattern>`
- Rust benches: `DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3`
- Python tests: `pixi run -e dev maturin develop && pixi run -e dev pytest tests/...`
- Bare `cargo test` will fail with a libpython rpath error — always go through `pixi run -e dev`.

---

## Task 1: Introduce `ShuffleBuffers` with sparse-reset LUT (no integration yet)

**Files:**
- Modify: `src/kshuffle.rs`

Add the struct and unit-test it in isolation. Wiring into `k_shuffle1` happens in Task 2.

- [ ] **Step 1: Add `ShuffleBuffers` struct and `new` / `reset_lut` to `src/kshuffle.rs`**

Insert immediately after the existing `MAX_LUT` constant and `lut_size` helper, before the `KmerIndex` trait:

```rust
/// Per-thread reusable storage for one row of k-shuffle work.
///
/// Each rayon worker owns one of these (allocated via `map_init` in
/// `k_shuffle`) and reuses it across rows. The LUT uses a sparse-reset
/// strategy: only positions written this row are restored to `u32::MAX`.
pub(crate) struct ShuffleBuffers {
    /// Length `MAX_LUT`. Each entry is either `u32::MAX` (unseen) or the
    /// vertex id of the (k-1)-mer with this encoded value. The whole
    /// allocation is initialized to `u32::MAX` exactly once, in `new`.
    lut: Box<[u32]>,
    /// Encoded (k-1)-mer values that were written into `lut` this row.
    /// Used by `reset_lut` to undo only those writes.
    lut_written: Vec<u32>,
    /// Vertex id of each (k-1)-mer window, in window order.
    codes: Vec<u32>,
    /// Vertex storage for one row.
    vertices: Vec<Vertex>,
    /// Adjacency-list storage for one row.
    indices: Vec<u32>,
}

impl ShuffleBuffers {
    pub(crate) fn new() -> Self {
        Self {
            lut: vec![u32::MAX; MAX_LUT as usize].into_boxed_slice(),
            lut_written: Vec::new(),
            codes: Vec::new(),
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Restore `u32::MAX` at every position previously written.
    /// O(written count), not O(MAX_LUT).
    fn reset_lut(&mut self) {
        for &c in &self.lut_written {
            self.lut[c as usize] = u32::MAX;
        }
        self.lut_written.clear();
    }
}
```

- [ ] **Step 2: Add unit test inside the existing `mod test` block at the bottom of `src/kshuffle.rs`**

```rust
#[test]
fn shuffle_buffers_sparse_reset_only_touches_written_positions() {
    let mut buf = ShuffleBuffers::new();
    // Initially all u32::MAX.
    assert!(buf.lut.iter().all(|&x| x == u32::MAX));
    // Write a few positions.
    buf.lut[3] = 7;
    buf.lut[100] = 9;
    buf.lut_written.extend_from_slice(&[3, 100]);
    // Sparse reset.
    buf.reset_lut();
    // Restored at those positions.
    assert_eq!(buf.lut[3], u32::MAX);
    assert_eq!(buf.lut[100], u32::MAX);
    // lut_written cleared.
    assert!(buf.lut_written.is_empty());
    // Second reset is a no-op (regression check on a missing clear).
    buf.lut[42] = 5;
    buf.lut_written.push(42);
    buf.reset_lut();
    assert_eq!(buf.lut[42], u32::MAX);
    assert!(buf.lut_written.is_empty());
}
```

- [ ] **Step 3: Build and run the new test**

```bash
pixi run -e dev cargo test --lib kshuffle::test::shuffle_buffers_sparse_reset
```
Expected: pass. (Other existing tests still pass too; this task only adds.)

- [ ] **Step 4: Commit**

```bash
git add src/kshuffle.rs
git commit -m "feat: add ShuffleBuffers with sparse-reset LUT"
```

---

## Task 2: Migrate `DirectLut`/`HashLut` into `ShuffleBuffers` methods; wire through `k_shuffle1` + `k_shuffle`

**Files:**
- Modify: `src/kshuffle.rs`

This is the biggest task. It removes the `KmerIndex` trait and the `DirectLut`/`HashLut` structs (replaced by methods), changes `k_shuffle1`'s signature, updates `k_shuffle` to use `map_init`, and adapts the existing `direct_lut_*` / `hash_lut_*` unit tests.

- [ ] **Step 1: Replace the `KmerIndex` trait + `DirectLut` + `HashLut` block with two methods on `ShuffleBuffers`**

Delete from `src/kshuffle.rs`:
- The `pub(crate) trait KmerIndex { ... }` block.
- The `pub(crate) struct DirectLut { ... }` and its `impl DirectLut { ... }` and `impl KmerIndex for DirectLut { ... }`.
- The `pub(crate) struct HashLut<'a> { ... }` and its `impl<'a> HashLut<'a> { ... }` and `impl<'a> KmerIndex for HashLut<'a> { ... }`.

Add the following two methods to the existing `impl ShuffleBuffers` block (the one introduced in Task 1):

```rust
/// Build the direct lookup index. Fills `self.lut`, `self.lut_written`,
/// and `self.codes`. Returns `n_vertices`.
///
/// Walks `seq` once with a rolling encoder. The first time a code is
/// encountered, it's recorded in `self.lut[code] = next_vertex_id` and
/// the code is appended to `self.lut_written` for later sparse reset.
/// `self.codes[i]` is the vertex id of the (k-1)-mer at position i.
///
/// `lut_capacity` is `alphabet_size^(k-1)`; `self.lut` is sized to
/// `MAX_LUT` ≥ `lut_capacity`, so we only touch the first `lut_capacity`
/// entries.
fn build_direct_index(
    &mut self,
    seq: &[u8],
    k_minus_1: usize,
    alphabet_size: u32,
    b2c: &[u8; 256],
    lut_capacity: u32,
    max_uniq_lets: u32,
) -> u32 {
    debug_assert!(lut_capacity as usize <= self.lut.len());
    self.reset_lut();
    self.codes.clear();

    let n_windows = seq.len() - k_minus_1 + 1;
    self.codes.reserve(n_windows);
    let mut n_vertices: u32 = 0;

    let base_pow_km2 = if k_minus_1 >= 1 {
        alphabet_size.pow((k_minus_1 - 1) as u32)
    } else {
        1
    };

    let mut code = kmer_encode::encode_first(seq, k_minus_1, alphabet_size, b2c);
    if self.lut[code as usize] == u32::MAX && n_vertices < max_uniq_lets {
        self.lut[code as usize] = n_vertices;
        self.lut_written.push(code);
        n_vertices += 1;
    }
    self.codes.push(self.lut[code as usize]);

    for i in 0..(n_windows - 1) {
        code = kmer_encode::roll(
            code,
            seq[i],
            seq[i + k_minus_1],
            base_pow_km2,
            alphabet_size,
            b2c,
        );
        if self.lut[code as usize] == u32::MAX && n_vertices < max_uniq_lets {
            self.lut[code as usize] = n_vertices;
            self.lut_written.push(code);
            n_vertices += 1;
        }
        self.codes.push(self.lut[code as usize]);
    }

    n_vertices
}

/// Build the hash-based index. Fills `self.codes`. Returns `n_vertices`.
///
/// Used for the fallback path (protein / large k where `α^(k-1) > MAX_LUT`).
/// Allocates a fresh `HashMap` per call — no pooled hash state.
fn build_hash_index(
    &mut self,
    seq: &[u8],
    k_minus_1: usize,
    max_uniq_lets: u32,
) -> u32 {
    self.codes.clear();
    let n_windows = seq.len() - k_minus_1 + 1;
    self.codes.reserve(n_windows);

    let mut map: HashMap<&[u8], u32, Xxh3Builder> =
        HashMap::with_capacity_and_hasher(max_uniq_lets as usize, Xxh3Builder::new());
    let mut n_vertices: u32 = 0;
    for i in 0..n_windows {
        let kmer = &seq[i..i + k_minus_1];
        let id = if let Some(&id) = map.get(kmer) {
            id
        } else if n_vertices < max_uniq_lets {
            let id = n_vertices;
            map.insert(kmer, id);
            n_vertices += 1;
            id
        } else {
            // Bound hit: keep returning the last-known id (matches the
            // original code's behavior of silently capping at max_uniq_lets).
            // In practice this path is unreachable because max_uniq_lets is
            // already an upper bound on distinct (k-1)-mers in `seq`.
            *map.values().next().unwrap()
        };
        self.codes.push(id);
    }

    n_vertices
}
```

- [ ] **Step 2: Update `k_shuffle1` signature and body to thread `&mut ShuffleBuffers`**

Replace the existing `fn k_shuffle1(...)` body with:

```rust
fn k_shuffle1(
    seq: ArrayView1<u8>,
    k: usize,
    seed: Option<u64>,
    mut out: ArrayViewMut1<u8>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
    buffers: &mut ShuffleBuffers,
) -> Result<()> {
    let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = seq.len();

    if k >= l { seq.assign_to(out); return Ok(()); }
    if k < 1 { bail!(KShuffleError::KLessThanOne); }
    assert!(alphabet_size >= 2, "alphabet_size must be >= 2");

    if k == 1 {
        seq.assign_to(&mut out);
        out.as_slice_mut().unwrap().shuffle(&mut rng);
        return Ok(());
    }

    // (k=2 specialization slot — added in Task 5.)

    // Ensure contiguous slice access. Non-contiguous rows are copied.
    if seq.is_standard_layout() {
        k_shuffle1_inner(seq, k, &mut rng, out, alphabet_size, alphabet_bytes, buffers)
    } else {
        let owned: ndarray::Array1<u8> = seq.to_owned();
        k_shuffle1_inner(owned.view(), k, &mut rng, out, alphabet_size, alphabet_bytes, buffers)
    }
}
```

(Preserve the existing non-contiguous fallback that was added during the earlier optimization PR.)

- [ ] **Step 3: Update `k_shuffle1_inner` to use `buffers`**

Replace the existing `fn k_shuffle1_inner(...)` with:

```rust
fn k_shuffle1_inner(
    seq: ArrayView1<u8>,
    k: usize,
    rng: &mut SmallRng,
    out: ArrayViewMut1<u8>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
    buffers: &mut ShuffleBuffers,
) -> Result<()> {
    let l = seq.len();
    let seq_slice = seq.as_slice().expect("k_shuffle1_inner requires contiguous row");
    let k_minus_1 = k - 1;
    let n_lets = l - k + 2;
    let max_uniq_lets = n_lets.min(alphabet_size.pow(k_minus_1 as u32)) as u32;
    let alpha_u32 = alphabet_size as u32;
    let b2c = kmer_encode::build_byte_to_code(alphabet_bytes);

    // Phase 1: build the index, fills buffers.codes (vertex id per window).
    let n_vertices: u32 = match lut_size(alphabet_size, k_minus_1 as u32) {
        Some(cap) => buffers.build_direct_index(
            seq_slice, k_minus_1, alpha_u32, &b2c, cap, max_uniq_lets,
        ),
        None => buffers.build_hash_index(seq_slice, k_minus_1, max_uniq_lets),
    };

    // Phase 2: vertices.
    buffers.vertices.clear();
    buffers.vertices.resize_with(n_vertices as usize, Vertex::default);
    for (i, &v) in buffers.codes.iter().enumerate() {
        let vertex = &mut buffers.vertices[v as usize];
        if i < (n_lets - 1) {
            vertex.n_indices += 1;
        }
        vertex.i_sequence = i as u32; // last-write-wins, matches original
    }

    // Phase 3a: prefix-sum idx_offset.
    let mut current_idx: u32 = 0;
    for v in buffers.vertices.iter_mut() {
        v.idx_offset = current_idx;
        current_idx += v.n_indices;
    }

    // Phase 3b: adjacency.
    buffers.indices.clear();
    buffers.indices.resize(n_lets - 1, 0u32);
    for i in 0..(n_lets - 1) {
        let u_id = buffers.codes[i] as usize;
        let v_id = buffers.codes[i + 1];
        let u = &mut buffers.vertices[u_id];
        if u.n_indices > 0 {
            buffers.indices[(u.idx_offset + u.i_indices) as usize] = v_id;
            u.i_indices += 1;
        }
    }

    // Phase 4: Wilson + random_walk.
    let root_idx = buffers.codes[buffers.codes.len() - 1] as usize;
    wilson_random_spanning_tree(&mut buffers.vertices, &buffers.indices, root_idx, rng);
    random_walk(&mut buffers.vertices, &mut buffers.indices, root_idx, rng, seq, k, out);

    Ok(())
}
```

- [ ] **Step 4: Update `k_shuffle` (outer) to use `map_init`**

Replace the existing `pub fn k_shuffle<D: Dimension>(...)` body with:

```rust
pub fn k_shuffle<D: Dimension>(
    seqs: ArrayView<u8, D>,
    k: usize,
    seed: Option<u64>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
) -> Array<u8, D> {
    let mut out = Array::from_elem(seqs.raw_dim(), 0u8);

    let n_rows: usize = out.rows().into_iter().count();
    let mut parent = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let row_seeds: Vec<u64> = (0..n_rows).map(|_| parent.gen()).collect();

    let results: Vec<Result<()>> = out
        .rows_mut()
        .into_iter()
        .zip(seqs.rows())
        .zip(row_seeds)
        .par_bridge()
        .map_init(
            ShuffleBuffers::new,
            |buffers, ((out_row, row), row_seed)| {
                k_shuffle1(row, k, Some(row_seed), out_row, alphabet_size, alphabet_bytes, buffers)
            },
        )
        .collect();

    for result in results {
        result.expect("k_shuffle error");
    }
    out
}
```

- [ ] **Step 5: Replace the existing `direct_lut_*` and `hash_lut_*` unit tests**

Find the three tests in `mod test`:
- `direct_lut_assigns_ids_in_first_seen_order`
- `hash_lut_assigns_ids_in_first_seen_order`
- `direct_lut_and_hash_lut_agree_on_ids`

Replace all three with these three replacements that test the new methods on `ShuffleBuffers`:

```rust
#[test]
fn build_direct_index_assigns_ids_in_first_seen_order() {
    let b2c = crate::kmer_encode::build_byte_to_code(b"ACGT");
    let mut buf = ShuffleBuffers::new();
    // Sequence AACGAA, k-1=2 → windows: AA, AC, CG, GA, AA
    // First-seen unique: AA, AC, CG, GA → 4 vertices, ids 0..3
    let n = buf.build_direct_index(b"AACGAA", 2, 4, &b2c, 16, 100);
    assert_eq!(n, 4);
    // codes are vertex ids: AA=0, AC=1, CG=2, GA=3, AA=0
    assert_eq!(buf.codes, vec![0, 1, 2, 3, 0]);
    // lut_written records the encoded k-mer values that were assigned ids.
    // AA=0, AC=1, CG=6, GA=8 → encoded values 0,1,6,8 (any order).
    let mut written = buf.lut_written.clone();
    written.sort_unstable();
    assert_eq!(written, vec![0, 1, 6, 8]);
}

#[test]
fn build_hash_index_assigns_ids_in_first_seen_order() {
    let mut buf = ShuffleBuffers::new();
    let n = buf.build_hash_index(b"AACGAA", 2, 100);
    assert_eq!(n, 4);
    assert_eq!(buf.codes, vec![0, 1, 2, 3, 0]);
}

#[test]
fn build_direct_and_hash_index_agree_on_codes() {
    let b2c = crate::kmer_encode::build_byte_to_code(b"ACGT");
    let mut bd = ShuffleBuffers::new();
    let mut bh = ShuffleBuffers::new();
    let seq: &[u8] = b"ACGTACGTACGT";
    let nd = bd.build_direct_index(seq, 3, 4, &b2c, 64, 100);
    let nh = bh.build_hash_index(seq, 3, 100);
    assert_eq!(nd, nh);
    assert_eq!(bd.codes, bh.codes);
}
```

- [ ] **Step 6: Update the `equivalence_with_reference_impl_for_k_2_through_8` test to pass a buffer**

Find the test and update the `k_shuffle1` call site to pass a freshly-allocated buffer:

```rust
let mut buffers = ShuffleBuffers::new();
super::k_shuffle1(
    seq_arr.view(), k, Some(row_seed),
    out_new.view_mut(), alphabet_size, alphabet_bytes, &mut buffers,
).unwrap();
```

(The reference call to `kshuffle_ref::k_shuffle1_ref` is unchanged.)

- [ ] **Step 7: Update the proptest `k_shuffle_preserves_kmer_frequencies` similarly**

Find the proptest and add a buffer:

```rust
let mut buffers = super::ShuffleBuffers::new();
super::k_shuffle1(
    seq_arr.view(), k, Some(seed),
    out.view_mut(), 4, b"ACGT", &mut buffers,
).unwrap();
```

- [ ] **Step 8: Build and run all Rust tests**

```bash
pixi run -e dev cargo test --lib
```
Expected: all tests pass (including `same_freq` proptest variant, all `*_index*` tests, `equivalence_*`, `determinism_*`, `shuffle_buffers_sparse_reset_*`).

- [ ] **Step 9: Rebuild the Python extension and run Python tests**

```bash
pixi run -e dev maturin develop
pixi run -e dev pytest tests/test_modifiers.py::test_k_shuffle tests/test_shape_matrix.py -v
```
Expected: all pass.

- [ ] **Step 10: Clippy**

```bash
pixi run -e dev cargo clippy --all-targets -- -D warnings
```
Expected: clean. If clippy complains about anything in the new methods, fix inline (likely candidates: `needless_range_loop` — rewrite as `.iter().enumerate()`; `redundant_field_names` — switch to shorthand).

- [ ] **Step 11: Commit**

```bash
git add src/kshuffle.rs
git commit -m "refactor: pool LUT and reuse buffers via ShuffleBuffers"
```

---

## Task 3: Buffer-reuse correctness test

**Files:**
- Modify: `src/kshuffle.rs` (add test in `mod test`)

This test catches missing-reset bugs in `ShuffleBuffers` — the kind of error pooling makes possible and that the existing equivalence test (which uses fresh buffers each call) would miss.

- [ ] **Step 1: Add the test to `mod test`**

```rust
#[test]
fn buffer_reuse_matches_fresh_buffer_output() {
    use ndarray::Array1;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let alphabet_size = 4;
    let alphabet_bytes = b"ACGT";
    let mut seedgen = SmallRng::seed_from_u64(0xCAFEBABE);

    // Build a fixed list of (seq, k, seed) triples spanning the supported range.
    let mut cases: Vec<(Vec<u8>, usize, u64)> = Vec::new();
    for &len in &[16usize, 64, 256] {
        for &k in &[2usize, 3, 4, 6, 8] {
            if k >= len { continue; }
            for _ in 0..5 {
                let seq: Vec<u8> = (0..len)
                    .map(|_| alphabet_bytes[seedgen.gen_range(0..4)])
                    .collect();
                cases.push((seq, k, seedgen.gen()));
            }
        }
    }

    // Run each case twice: once with a fresh buffer, once with a reused buffer.
    let mut reused = ShuffleBuffers::new();
    for (seq, k, seed) in &cases {
        let seq_arr = Array1::from(seq.clone());
        let len = seq.len();

        let mut out_fresh = Array1::<u8>::zeros(len);
        let mut fresh = ShuffleBuffers::new();
        super::k_shuffle1(
            seq_arr.view(), *k, Some(*seed),
            out_fresh.view_mut(), alphabet_size, alphabet_bytes, &mut fresh,
        ).unwrap();

        let mut out_reused = Array1::<u8>::zeros(len);
        super::k_shuffle1(
            seq_arr.view(), *k, Some(*seed),
            out_reused.view_mut(), alphabet_size, alphabet_bytes, &mut reused,
        ).unwrap();

        assert_eq!(
            out_fresh, out_reused,
            "mismatch at len={} k={} seed={}", len, k, seed
        );
    }
}
```

- [ ] **Step 2: Run the test**

```bash
pixi run -e dev cargo test --lib kshuffle::test::buffer_reuse_matches_fresh_buffer_output
```
Expected: pass.

**If this fails:** the most likely cause is a missing `clear()` / `reset_lut()` somewhere in `k_shuffle1_inner` or one of the `build_*_index` methods. Trace the first failing case and check what state leaks across rows.

- [ ] **Step 3: Commit**

```bash
git add src/kshuffle.rs
git commit -m "test: assert pooled and fresh ShuffleBuffers produce identical output"
```

---

## Task 4: Measure LUT-pooling perf gate

**Files:** none modified. Bench-only.

- [ ] **Step 1: Run the criterion benchmark on the current branch (with pooling)**

```bash
DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3 2>&1 | tee /tmp/kshuffle_bench_pooled.txt
```

Capture the median times for k ∈ {2, 4, 6, 8}.

- [ ] **Step 2: Compare to the pre-pooling baseline**

Baseline (from `feat/kshuffle-rust-opt` head before this PR):

| k | Baseline |
|---|----------|
| 2 | 6.79 ms |
| 4 | 6.96 ms |
| 6 | 7.13 ms |
| 8 | 18.44 ms |

Compute ratios `(baseline / new median)`. Check against the spec's gates:

- **k=8: ≥ 1.8× faster.** Target ≈ 9 ms or better.
- **k=2, 4, 6: within ±5% of baseline.**

- [ ] **Step 3: Decide**

- If gates pass: continue to Task 5.
- If k=8 misses (< 1.8×): the most likely cause is a bug in `reset_lut` or `build_direct_index` (e.g., resetting the wrong thing, or accidentally allocating a fresh Vec instead of clearing in place). **Investigate and fix before continuing.** Do not proceed to k=2 specialization with the LUT gate failing.
- If k=2/4/6 regress > 5%: also investigate — pooling should not slow these down. Possible cause: cleared but not preallocated `Vec` triggering reallocation per row; switch to `vec.clear(); vec.reserve(n)` patterns where missing.

- [ ] **Step 4: Record the numbers in a brief commit message**

If the gate is met, no code commit is required from this task. If you adjusted the code to meet the gate, commit those adjustments with a descriptive message.

---

## Task 5: Add k=2 specialization (`k_shuffle1_k2`)

**Files:**
- Modify: `src/kshuffle.rs`

- [ ] **Step 1: Add `k_shuffle1_k2` worker function**

Insert after `k_shuffle1_inner` in `src/kshuffle.rs`:

```rust
/// Specialized k-shuffle path for k=2 (dinucleotide shuffle).
///
/// For k=2, the (k-1)-mer is a single byte, so the vertex id of position
/// `i` is just `b2c[seq[i]]`. This bypasses the LUT/codes machinery
/// entirely. The Wilson + random_walk phases are unchanged.
fn k_shuffle1_k2(
    seq: ArrayView1<u8>,
    rng: &mut SmallRng,
    out: ArrayViewMut1<u8>,
    alphabet_bytes: &[u8],
    buffers: &mut ShuffleBuffers,
) -> Result<()> {
    let seq_slice = seq.as_slice().expect("k_shuffle1_k2 requires contiguous row");
    let l = seq_slice.len();
    let n_lets = l; // for k=2, n_lets = l - k + 2 = l
    let b2c = kmer_encode::build_byte_to_code(alphabet_bytes);

    // Phase 1: resolve vertex ids inline. Vertex space ≤ 256; we assign
    // dense ids 0..n_vertices in first-seen order via a 256-entry table.
    let mut byte_to_vid = [u32::MAX; 256];
    let mut n_vertices: u32 = 0;
    buffers.codes.clear();
    buffers.codes.reserve(l);
    for &b in seq_slice {
        let c = b2c[b as usize] as usize;
        let mut id = byte_to_vid[c];
        if id == u32::MAX {
            id = n_vertices;
            byte_to_vid[c] = id;
            n_vertices += 1;
        }
        buffers.codes.push(id);
    }

    // Phase 2: vertices.
    buffers.vertices.clear();
    buffers.vertices.resize_with(n_vertices as usize, Vertex::default);
    for (i, &v) in buffers.codes.iter().enumerate() {
        let vertex = &mut buffers.vertices[v as usize];
        if i < (n_lets - 1) {
            vertex.n_indices += 1;
        }
        vertex.i_sequence = i as u32;
    }

    // Phase 3a: prefix-sum idx_offset.
    let mut current_idx: u32 = 0;
    for v in buffers.vertices.iter_mut() {
        v.idx_offset = current_idx;
        current_idx += v.n_indices;
    }

    // Phase 3b: adjacency.
    buffers.indices.clear();
    buffers.indices.resize(n_lets - 1, 0u32);
    for i in 0..(n_lets - 1) {
        let u_id = buffers.codes[i] as usize;
        let v_id = buffers.codes[i + 1];
        let u = &mut buffers.vertices[u_id];
        if u.n_indices > 0 {
            buffers.indices[(u.idx_offset + u.i_indices) as usize] = v_id;
            u.i_indices += 1;
        }
    }

    // Phase 4: Wilson + random_walk.
    let root_idx = buffers.codes[buffers.codes.len() - 1] as usize;
    wilson_random_spanning_tree(&mut buffers.vertices, &buffers.indices, root_idx, rng);
    random_walk(&mut buffers.vertices, &mut buffers.indices, root_idx, rng, seq, 2, out);

    Ok(())
}
```

- [ ] **Step 2: Wire k=2 dispatch into `k_shuffle1`**

In `k_shuffle1`, replace the comment `// (k=2 specialization slot — added in Task 5.)` with the actual dispatch. After the `k == 1` block, before the `if seq.is_standard_layout() { ... }` block, insert:

```rust
if k == 2 {
    if seq.is_standard_layout() {
        return k_shuffle1_k2(seq, &mut rng, out, alphabet_bytes, buffers);
    } else {
        let owned: ndarray::Array1<u8> = seq.to_owned();
        return k_shuffle1_k2(owned.view(), &mut rng, out, alphabet_bytes, buffers);
    }
}
```

- [ ] **Step 3: Verify all existing tests still pass**

```bash
pixi run -e dev cargo test --lib
```
Expected: all tests pass. The `equivalence_with_reference_impl_for_k_2_through_8` test is especially load-bearing here — it asserts byte-equal output between the new k=2 specialization and the original reference impl. **If it fails for k=2 cases, debug before continuing.** Likely cause: a divergence in vertex-id assignment order between the inline code and the original `htable.entry().or_insert_with` order.

- [ ] **Step 4: Verify Python tests**

```bash
pixi run -e dev maturin develop
pixi run -e dev pytest tests/test_modifiers.py::test_k_shuffle -v
```
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/kshuffle.rs
git commit -m "feat: specialize k=2 path to skip LUT and codes lookup"
```

---

## Task 6: Measure k=2 perf gate and decide whether to keep the specialization

**Files:** maybe revert `src/kshuffle.rs`. Bench-only otherwise.

- [ ] **Step 1: Run the criterion benchmark with the k=2 specialization in place**

```bash
DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3 2>&1 | tee /tmp/kshuffle_bench_k2_specialized.txt
```

Capture the k=2 median.

- [ ] **Step 2: Disable the k=2 dispatch and re-bench**

Temporarily comment out the `if k == 2 { ... return ... }` block in `k_shuffle1` (do NOT delete it). Re-run the benchmark:

```bash
DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3 2>&1 | tee /tmp/kshuffle_bench_k2_general.txt
```

Capture the k=2 median (now going through the general path, with pooled buffers).

- [ ] **Step 3: Compute the ratio**

`ratio = general_median / specialized_median`

- **If ratio ≥ 1.15** (specialization is ≥ 15% faster): un-comment the dispatch, commit a note in the message confirming the gate was met.
- **If ratio < 1.15**: revert the k=2 specialization (delete `k_shuffle1_k2` and the dispatch) — the LUT pooling stays. Commit the revert with a message explaining the measured ratio.

- [ ] **Step 4: Restore the chosen state**

If kept:
```bash
# (un-comment the k==2 dispatch first)
git add src/kshuffle.rs
git commit -m "bench: confirm k=2 specialization meets >=15% gate (measured X.XXx)"
```

If reverted:
```bash
# (delete k_shuffle1_k2 and the if k == 2 { ... } block in k_shuffle1)
git add src/kshuffle.rs
git commit -m "revert: drop k=2 specialization; measured speedup X.XXx < 15% gate"
```

- [ ] **Step 5: Re-run the full test suite to confirm correctness after the decision**

```bash
pixi run -e dev cargo test --lib
pixi run -e dev pytest tests/test_modifiers.py::test_k_shuffle -v
```
Expected: pass.

---

## Task 7: Final verification

- [ ] **Step 1: Full Rust test suite**

```bash
pixi run -e dev cargo test --lib
```
Expected: all tests pass — `kmer_encode::tests::*`, `kshuffle::test::*` (including `shuffle_buffers_sparse_reset_*`, `build_direct_index_*`, `build_hash_index_*`, `buffer_reuse_*`, `equivalence_*`, `determinism_*`, `k_shuffle_preserves_kmer_frequencies` proptest).

- [ ] **Step 2: Determinism across thread counts**

```bash
RAYON_NUM_THREADS=1 pixi run -e dev cargo test --lib kshuffle::test::determinism
RAYON_NUM_THREADS=4 pixi run -e dev cargo test --lib kshuffle::test::determinism
```
Expected: both pass.

- [ ] **Step 3: Full Python test suite**

```bash
pixi run -e dev maturin develop
pixi run -e dev pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 4: Clippy**

```bash
pixi run -e dev cargo clippy --all-targets -- -D warnings
```
Expected: clean.

- [ ] **Step 5: Final benchmark headline**

```bash
DYLD_LIBRARY_PATH=/Users/david/.pixi/envs/python/lib cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3 2>&1 | tee /tmp/kshuffle_bench_final.txt
```

Record the final numbers (k ∈ {2,4,6,8}) for the PR description. Confirm all spec gates from Section "Verification gates" of the spec are met.

- [ ] **Step 6: Confirm all spec gates from `docs/superpowers/specs/2026-05-20-kshuffle-pooled-buffers-and-k2-fast-path-design.md`**

- [x] All tests pass (Step 1, 3).
- [x] `cargo clippy --all-targets -- -D warnings` clean (Step 4).
- [x] k=8: ≥ 1.8× faster than pre-pooling baseline (from Task 4).
- [x] k=2, 4, 6: within ±5% of pre-pooling baseline OR improved (from Task 4).
- [x] k=2 specialization decision recorded with measured ratio (from Task 6).

If any gate fails, investigate before declaring success.
