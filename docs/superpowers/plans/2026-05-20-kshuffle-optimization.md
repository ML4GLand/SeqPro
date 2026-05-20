# K-Shuffle Rust Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-row `HashMap<Vec<u8>, _>` k-mer index in `src/kshuffle.rs` with an integer-encoded direct lookup table (DNA/RNA fast path) plus a borrowed-slice HashMap fallback, fix the per-row seed-reuse bug, and validate via a Rust equivalence test and proptest-based k-mer frequency invariant.

**Architecture:** Strategy trait `KmerIndex` chosen once per call. Fast path `DirectLut` (used when `α^(k-1) ≤ 16_384`) is a `Vec<u32>` indexed by a rolling integer encoding of each (k−1)-mer. Fallback `HashLut` uses `HashMap<&[u8], u32, Xxh3Builder>` with borrowed keys. `Vertex` shrinks to `u32` fields, no builder. Outer `k_shuffle` uses native rayon parallel iteration with per-row seeds derived from a parent RNG.

**Tech Stack:** Rust (pyo3, ndarray, rayon, rand, xxhash-rust), proptest (new dev-dep), criterion (new dev-dep). Python wrapper unchanged except for docstring.

**Spec:** `docs/superpowers/specs/2026-05-20-kshuffle-optimization-design.md`

---

## File Structure

**Will modify:**
- `src/kshuffle.rs` — refactor `Vertex`, `k_shuffle`, `k_shuffle1`; add `KmerIndex` trait + impls; rewrite tests.
- `Cargo.toml` — add `proptest` and `criterion` as dev-dependencies.
- `python/seqpro/_modifiers.py` — update `k_shuffle` docstring to document new seed-per-row contract.

**Will create:**
- `src/kmer_encode.rs` — rolling integer encoder utilities.
- `src/kshuffle_ref.rs` — preserved copy of the current implementation, gated behind `#[cfg(test)]`, used by the equivalence test only.
- `benches/kshuffle.rs` — criterion benchmark.

**Why this split:** `kmer_encode` is general-purpose (could be reused later for tokenize/ohe optimizations); keeping it separate from `kshuffle.rs` makes it independently testable. `kshuffle_ref.rs` is the trusted oracle for equivalence testing — keeping it byte-for-byte intact maximizes confidence that we haven't perturbed the algorithm.

---

## Task 1: Snapshot current implementation as `kshuffle_ref` for equivalence testing

**Files:**
- Create: `src/kshuffle_ref.rs`
- Modify: `src/lib.rs` (add `#[cfg(test)] mod kshuffle_ref;`)

- [ ] **Step 1: Copy current `src/kshuffle.rs` verbatim to `src/kshuffle_ref.rs`**

```bash
cp src/kshuffle.rs src/kshuffle_ref.rs
```

- [ ] **Step 2: Strip the `#[cfg(test)] mod test { ... }` block from `src/kshuffle_ref.rs`**

The reference module is itself test-only; it does not need nested tests. Open `src/kshuffle_ref.rs` and delete the `#[cfg(test)] mod test { ... }` block at the bottom (lines starting with `#[cfg(test)]` through the closing `}` of `mod test`).

- [ ] **Step 3: Rename the public function in `src/kshuffle_ref.rs` to avoid symbol collision**

Edit `src/kshuffle_ref.rs`: rename `pub fn k_shuffle` → `pub fn k_shuffle_ref` and `fn k_shuffle1` → `fn k_shuffle1_ref`. Update the internal call from `k_shuffle1` to `k_shuffle1_ref` (one site, inside `k_shuffle_ref`).

- [ ] **Step 4: Register the module behind `#[cfg(test)]` in `src/lib.rs`**

Edit `src/lib.rs`, add immediately after `pub mod kshuffle;`:

```rust
#[cfg(test)]
mod kshuffle_ref;
```

- [ ] **Step 5: Verify compilation**

```bash
cargo check --tests
```
Expected: success, no warnings about `kshuffle_ref` being unused (it's `#[cfg(test)]` so dead-code is suppressed under non-test builds; under test build it's referenced by the equivalence test we add later — for this commit, a `dead_code` warning is acceptable).

- [ ] **Step 6: Commit**

```bash
git add src/kshuffle_ref.rs src/lib.rs
git commit -m "test: snapshot current kshuffle impl as reference for equivalence testing"
```

---

## Task 2: Add `kmer_encode` module with rolling integer encoder

**Files:**
- Create: `src/kmer_encode.rs`
- Modify: `src/lib.rs` (add `pub mod kmer_encode;`)

- [ ] **Step 1: Write the module skeleton with failing tests**

Create `src/kmer_encode.rs`:

```rust
//! Rolling integer encoder for k-mers over an arbitrary byte alphabet.
//!
//! The encoder maps each (k-1)-byte window of a sequence to a `u32` integer
//! in `0..alphabet_size.pow(k-1)`. A 256-entry byte-to-code table allows
//! arbitrary alphabets (DNA = ACGT, RNA = ACGU, protein, etc.).

/// Build a 256-entry table mapping ASCII byte → alphabet code.
/// Bytes not present in `alphabet_bytes` map to `u8::MAX` (sentinel; callers
/// are expected to have sanitized input — current code does not check either).
pub fn build_byte_to_code(alphabet_bytes: &[u8]) -> [u8; 256] {
    let mut table = [u8::MAX; 256];
    for (code, &b) in alphabet_bytes.iter().enumerate() {
        debug_assert!(code < 256);
        table[b as usize] = code as u8;
    }
    table
}

/// Encode the first (k-1)-mer of `seq` as a base-`alphabet_size` integer.
/// Most-significant digit is `seq[0]`. Panics if `seq.len() < k_minus_1`.
pub fn encode_first(seq: &[u8], k_minus_1: usize, alphabet_size: u32, b2c: &[u8; 256]) -> u32 {
    let mut code: u32 = 0;
    for &b in &seq[..k_minus_1] {
        code = code * alphabet_size + b2c[b as usize] as u32;
    }
    code
}

/// Rolling update: given previous (k-1)-mer code at position `p`, return
/// the code at position `p+1`. `base_pow_km2 = alphabet_size^(k-2)` (precomputed).
/// `drop_byte = seq[p]`, `add_byte = seq[p + k - 1]`.
#[inline]
pub fn roll(
    prev: u32,
    drop_byte: u8,
    add_byte: u8,
    base_pow_km2: u32,
    alphabet_size: u32,
    b2c: &[u8; 256],
) -> u32 {
    let drop_code = b2c[drop_byte as usize] as u32;
    let add_code = b2c[add_byte as usize] as u32;
    (prev - drop_code * base_pow_km2) * alphabet_size + add_code
}

#[cfg(test)]
mod tests {
    use super::*;

    const DNA: &[u8] = b"ACGT";

    #[test]
    fn byte_to_code_dna() {
        let b2c = build_byte_to_code(DNA);
        assert_eq!(b2c[b'A' as usize], 0);
        assert_eq!(b2c[b'C' as usize], 1);
        assert_eq!(b2c[b'G' as usize], 2);
        assert_eq!(b2c[b'T' as usize], 3);
        assert_eq!(b2c[b'N' as usize], u8::MAX);
    }

    #[test]
    fn encode_first_acgt_k4() {
        // (k-1) = 3, alphabet=4: ACG -> 0*16 + 1*4 + 2 = 6
        let b2c = build_byte_to_code(DNA);
        assert_eq!(encode_first(b"ACGT", 3, 4, &b2c), 0 * 16 + 1 * 4 + 2);
    }

    #[test]
    fn rolling_matches_recomputed() {
        // For sequence ACGTACGT and k-1 = 3:
        // windows: ACG, CGT, GTA, TAC, ACG, CGT
        let b2c = build_byte_to_code(DNA);
        let seq = b"ACGTACGT";
        let k_minus_1 = 3;
        let alpha = 4u32;
        let base_pow_km2 = alpha.pow((k_minus_1 - 1) as u32); // 4^2 = 16

        let mut prev = encode_first(seq, k_minus_1, alpha, &b2c);
        for i in 0..(seq.len() - k_minus_1) {
            let recomputed = encode_first(&seq[i..], k_minus_1, alpha, &b2c);
            assert_eq!(prev, recomputed, "mismatch at window {}", i);
            if i + 1 + k_minus_1 <= seq.len() {
                prev = roll(prev, seq[i], seq[i + k_minus_1], base_pow_km2, alpha, &b2c);
            }
        }
    }
}
```

- [ ] **Step 2: Register the module in `src/lib.rs`**

Add `pub mod kmer_encode;` near the top of `src/lib.rs` (after `pub mod kshuffle;`).

- [ ] **Step 3: Run the unit tests, verify they pass**

```bash
cargo test --lib kmer_encode
```
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/kmer_encode.rs src/lib.rs
git commit -m "feat: add kmer_encode module with rolling integer encoder"
```

---

## Task 3: Add `KmerIndex` trait with `DirectLut` and `HashLut` impls (unit-tested in isolation)

**Files:**
- Modify: `src/kshuffle.rs`

This task adds the trait and impls but does not yet wire them into `k_shuffle1`. We unit-test them in isolation first to keep correctness easy to localize.

- [ ] **Step 1: Add the trait + impls at the top of `src/kshuffle.rs`**

Insert after the existing `use` statements and before `KShuffleError`:

```rust
use crate::kmer_encode;

const MAX_LUT: u32 = 16_384;

/// Returns `Some(α^(k-1))` if `DirectLut` fits, else `None`.
fn lut_size(alphabet_size: usize, k_minus_1: u32) -> Option<u32> {
    let n = (alphabet_size as u32).checked_pow(k_minus_1)?;
    if n <= MAX_LUT { Some(n) } else { None }
}

/// Maps (k-1)-mers in a sequence to dense vertex ids 0..n_vertices.
pub(crate) trait KmerIndex {
    /// Lookup by integer code (fast path) OR byte slice (slow path).
    /// Implementations use whichever they need; the other is ignored.
    fn lookup(&self, code: u32, bytes: &[u8]) -> u32;
    fn n_vertices(&self) -> u32;
}

/// Direct-indexed lookup table. Used when α^(k-1) ≤ MAX_LUT.
pub(crate) struct DirectLut {
    /// table[encoded_kmer] = vertex_id, or u32::MAX if unseen.
    table: Vec<u32>,
    n_vertices: u32,
}

impl DirectLut {
    /// Build by walking the sequence once with a rolling encoder.
    /// Stops assigning new vertex ids once the bound `max_uniq_lets` is hit
    /// (matches the current code's behavior).
    pub(crate) fn build(
        seq: &[u8],
        k_minus_1: usize,
        alphabet_size: u32,
        b2c: &[u8; 256],
        lut_capacity: u32,
        max_uniq_lets: u32,
    ) -> (Self, Vec<u32>) {
        let mut table = vec![u32::MAX; lut_capacity as usize];
        let n_windows = seq.len() - k_minus_1 + 1;
        let mut codes: Vec<u32> = Vec::with_capacity(n_windows);
        let mut n_vertices: u32 = 0;

        let base_pow_km2 = if k_minus_1 >= 1 {
            alphabet_size.pow((k_minus_1 - 1) as u32)
        } else {
            1
        };

        let mut code = kmer_encode::encode_first(seq, k_minus_1, alphabet_size, b2c);
        codes.push(code);
        if table[code as usize] == u32::MAX && n_vertices < max_uniq_lets {
            table[code as usize] = n_vertices;
            n_vertices += 1;
        }

        for i in 0..(n_windows - 1) {
            code = kmer_encode::roll(
                code,
                seq[i],
                seq[i + k_minus_1],
                base_pow_km2,
                alphabet_size,
                b2c,
            );
            codes.push(code);
            if table[code as usize] == u32::MAX && n_vertices < max_uniq_lets {
                table[code as usize] = n_vertices;
                n_vertices += 1;
            }
        }

        (Self { table, n_vertices }, codes)
    }
}

impl KmerIndex for DirectLut {
    #[inline]
    fn lookup(&self, code: u32, _bytes: &[u8]) -> u32 {
        self.table[code as usize]
    }
    fn n_vertices(&self) -> u32 {
        self.n_vertices
    }
}

/// Borrowed-slice HashMap fallback for protein / large k.
pub(crate) struct HashLut<'a> {
    map: HashMap<&'a [u8], u32, Xxh3Builder>,
    n_vertices: u32,
}

impl<'a> HashLut<'a> {
    pub(crate) fn build(
        seq: &'a [u8],
        k_minus_1: usize,
        max_uniq_lets: u32,
    ) -> Self {
        let mut map: HashMap<&'a [u8], u32, Xxh3Builder> =
            HashMap::with_capacity_and_hasher(max_uniq_lets as usize, Xxh3Builder::new());
        let mut n_vertices: u32 = 0;
        for kmer in seq.windows(k_minus_1) {
            if n_vertices >= max_uniq_lets { break; }
            map.entry(kmer).or_insert_with(|| {
                let id = n_vertices;
                n_vertices += 1;
                id
            });
        }
        Self { map, n_vertices }
    }
}

impl<'a> KmerIndex for HashLut<'a> {
    #[inline]
    fn lookup(&self, _code: u32, bytes: &[u8]) -> u32 {
        *self.map.get(bytes).expect("k-mer must be present (built from same seq)")
    }
    fn n_vertices(&self) -> u32 {
        self.n_vertices
    }
}
```

- [ ] **Step 2: Add unit tests for both impls at the bottom of `src/kshuffle.rs` (inside the existing `mod test`)**

```rust
#[test]
fn direct_lut_assigns_ids_in_first_seen_order() {
    let b2c = crate::kmer_encode::build_byte_to_code(b"ACGT");
    // Sequence AACGAA, k-1=2 → windows: AA, AC, CG, GA, AA
    // First-seen unique: AA, AC, CG, GA → 4 vertices, ids 0..3
    let (lut, codes) = DirectLut::build(b"AACGAA", 2, 4, &b2c, 16, 100);
    assert_eq!(lut.n_vertices(), 4);
    // codes for windows: AA=0, AC=1, CG=6, GA=8, AA=0
    assert_eq!(codes, vec![0, 1, 6, 8, 0]);
    assert_eq!(lut.lookup(0, b"AA"), 0);
    assert_eq!(lut.lookup(1, b"AC"), 1);
    assert_eq!(lut.lookup(6, b"CG"), 2);
    assert_eq!(lut.lookup(8, b"GA"), 3);
}

#[test]
fn hash_lut_assigns_ids_in_first_seen_order() {
    let seq: &[u8] = b"AACGAA";
    let h = HashLut::build(seq, 2, 100);
    assert_eq!(h.n_vertices(), 4);
    assert_eq!(h.lookup(0, b"AA"), 0);
    assert_eq!(h.lookup(0, b"AC"), 1);
    assert_eq!(h.lookup(0, b"CG"), 2);
    assert_eq!(h.lookup(0, b"GA"), 3);
}

#[test]
fn direct_lut_and_hash_lut_agree_on_ids() {
    let b2c = crate::kmer_encode::build_byte_to_code(b"ACGT");
    let seq: &[u8] = b"ACGTACGTACGT";
    let k_minus_1 = 3;
    let (lut, codes) = DirectLut::build(seq, k_minus_1, 4, &b2c, 64, 100);
    let h = HashLut::build(seq, k_minus_1, 100);
    assert_eq!(lut.n_vertices(), h.n_vertices());
    for (i, &code) in codes.iter().enumerate() {
        let bytes = &seq[i..i + k_minus_1];
        assert_eq!(lut.lookup(code, bytes), h.lookup(code, bytes));
    }
}
```

- [ ] **Step 3: Run the new tests, verify they pass**

```bash
cargo test --lib kshuffle::test::direct_lut kshuffle::test::hash_lut
```
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/kshuffle.rs
git commit -m "feat: add KmerIndex trait with DirectLut and HashLut impls"
```

---

## Task 4: Refactor `Vertex` to plain `u32`-field struct (no builder, no Option)

**Files:**
- Modify: `src/kshuffle.rs`
- Modify: `Cargo.toml` (remove `derive_builder` dependency)

- [ ] **Step 1: Replace the `Vertex` definition in `src/kshuffle.rs`**

Find:
```rust
#[derive(Builder)]
#[builder(pattern = "immutable")]
struct Vertex {
    idx_offset: usize,
    n_indices: usize,
    i_indices: usize,
    intree: bool,
    next: usize,
    i_sequence: usize,
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
}
```

- [ ] **Step 2: Remove `derive_builder` use statement**

In `src/kshuffle.rs`, delete the line `use derive_builder::Builder;`.

- [ ] **Step 3: Verify it still compiles (existing `k_shuffle1` will break — that's expected)**

```bash
cargo check --lib 2>&1 | head -40
```
Expected: errors come only from `k_shuffle1`'s use of `VertexBuilder`/`Option<usize>`/`usize` fields. These will be fixed in Task 5. Note any unrelated errors and stop if found.

(This task does not commit alone — it pairs with Task 5. Skip directly to Task 5; Task 5's commit covers both.)

---

## Task 5: Rewrite `k_shuffle1` to use the strategy trait + `u32` Vertex

**Files:**
- Modify: `src/kshuffle.rs`

This is the meatiest task. It restructures `k_shuffle1` into 4 phases as specified.

- [ ] **Step 1: Replace `k_shuffle1` with a thin entry + inner worker**

Delete the entire existing `fn k_shuffle1(...)` body and replace it with these two functions:

```rust
fn k_shuffle1(
    seq: ArrayView1<u8>,
    k: usize,
    seed: Option<u64>,
    mut out: ArrayViewMut1<u8>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
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

    k_shuffle1_inner(seq, k, &mut rng, out, alphabet_size, alphabet_bytes)
}

fn k_shuffle1_inner(
    seq: ArrayView1<u8>,
    k: usize,
    rng: &mut SmallRng,
    out: ArrayViewMut1<u8>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
) -> Result<()> {
    let l = seq.len();
    let seq_slice = seq.as_slice().expect("k_shuffle1 requires contiguous row");
    let k_minus_1 = k - 1;
    let n_lets = l - k + 2;
    let max_uniq_lets = n_lets.min(alphabet_size.pow(k_minus_1 as u32)) as u32;
    let alpha_u32 = alphabet_size as u32;
    let b2c = kmer_encode::build_byte_to_code(alphabet_bytes);

    // Phase 1: build k-mer index, materialize vertex id per window position.
    // `window_vid[i]` = vertex id of the (k-1)-mer starting at position i.
    let n_windows = l - k_minus_1 + 1;
    let mut window_vid: Vec<u32> = Vec::with_capacity(n_windows);

    let n_vertices: u32 = match lut_size(alphabet_size, k_minus_1 as u32) {
        Some(cap) => {
            let (lut, codes) =
                DirectLut::build(seq_slice, k_minus_1, alpha_u32, &b2c, cap, max_uniq_lets);
            for &c in &codes {
                window_vid.push(lut.lookup(c, &[]));
            }
            lut.n_vertices()
        }
        None => {
            let h = HashLut::build(seq_slice, k_minus_1, max_uniq_lets);
            for i in 0..n_windows {
                window_vid.push(h.lookup(0, &seq_slice[i..i + k_minus_1]));
            }
            h.n_vertices()
        }
    };

    // Phase 2: allocate vertices; fill n_indices and i_sequence in one pass.
    // Matches the original code's rule: for i < n_lets - 1, increment n_indices
    // and update i_sequence; for i == n_lets - 1 (the final window), update
    // i_sequence only (this is the "root" k-mer at the sequence's tail).
    let mut vertices: Vec<Vertex> = vec![Vertex::default(); n_vertices as usize];
    for (i, &v) in window_vid.iter().enumerate() {
        let vertex = &mut vertices[v as usize];
        if i < (n_lets - 1) {
            vertex.n_indices += 1;
        }
        vertex.i_sequence = i as u32; // last-write-wins, matches original
    }

    // Phase 3a: prefix-sum idx_offset.
    let mut current_idx: u32 = 0;
    for v in vertices.iter_mut() {
        v.idx_offset = current_idx;
        current_idx += v.n_indices;
    }

    // Phase 3b: populate adjacency. For each consecutive window pair (u, v)
    // in the sequence (excluding the final window as `u`), append v's id to
    // u's outgoing edge list.
    let mut indices: Vec<u32> = vec![0u32; n_lets - 1];
    for i in 0..(n_lets - 1) {
        let u_id = window_vid[i] as usize;
        let v_id = window_vid[i + 1];
        let u = &mut vertices[u_id];
        if u.n_indices > 0 {
            indices[(u.idx_offset + u.i_indices) as usize] = v_id;
            u.i_indices += 1;
        }
    }

    // Phase 4: Wilson + random_walk. Root = the final window's vertex id.
    let root_idx = window_vid[n_windows - 1] as usize;
    wilson_random_spanning_tree(&mut vertices, &indices, root_idx, rng);
    random_walk(&mut vertices, &mut indices, root_idx, rng, seq, k, out);

    Ok(())
}
```

**Equivalence-with-original note for reviewers:** The original code's `htable.entry().or_insert_with` assigned vertex ids in first-seen order. `DirectLut::build` and `HashLut::build` both preserve that order (each scans positions 0..n_windows and assigns ids on first sight). The `i_sequence` rule above is "last write wins on every occurrence," matching the original (which executed `vertices[..] = v.i_sequence(hentry.i_sequence)` on every iteration, with an additional `n_indices` bump on non-final positions).

- [ ] **Step 2: Update `wilson_random_spanning_tree` and `random_walk` for `u32` Vertex fields**

In `wilson_random_spanning_tree`:
- Change `indices: &Vec<usize>` → `indices: &Vec<u32>`.
- Inside, `rng.gen_range(0..u.n_indices)` is fine (`u32` range).
- `u_idx = indices[u.idx_offset + u.next]` — both are `u32`; index needs `as usize`. Update:
  ```rust
  u_idx = indices[(u.idx_offset + u.next) as usize] as usize;
  ```
- `u.idx_offset + u.next` may overflow u32 if both are near max — sequence lengths in this regime never approach that.

In `random_walk`:
- Change `indices: &mut Vec<usize>` → `indices: &mut Vec<u32>`.
- Update arithmetic: `let idx = u.n_indices - 1;` (u32 ok). `indices[(u.idx_offset + idx) as usize]` everywhere.
- `j = indices[...]` is now `u32`; the subsequent `j = v.i_sequence + k - 2` reassignment makes `j` a different type. Rename the index-domain `j` to `tmp_idx` (`u32`) and keep `j` as `usize` for `seq[j]` indexing.
- `indices[u.idx_offset..u.idx_offset + idx].shuffle(rng)` needs explicit `as usize`:
  ```rust
  let lo = u.idx_offset as usize;
  let hi = (u.idx_offset + idx) as usize;
  indices[lo..hi].shuffle(rng);
  ```
- For the root branch: `indices[u.idx_offset..u.idx_offset + u.n_indices].shuffle(rng)` — same treatment.
- `u.i_indices = 0;` — fine (u32).
- The walk loop: `u.i_indices >= u.n_indices` (u32), `u_idx` should be `usize` for indexing into `vertices: &mut Vec<Vertex>`. Where reading `indices[u.idx_offset + u.i_indices]`, do `as usize` on the address; the value is `u32` which is the next vertex id — cast `as usize`.

Apply changes mechanically; if compiler complains about mixing types, prefer keeping `usize` only for `Vec`/slice indexing operands and `u32` for stored fields.

- [ ] **Step 3: Update `k_shuffle` (outer fn) to pass `alphabet_bytes`**

The outer `k_shuffle` needs the alphabet bytes too. Update its signature:

```rust
pub fn k_shuffle<D: Dimension>(
    seqs: ArrayView<u8, D>,
    k: usize,
    seed: Option<u64>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
) -> Array<u8, D> {
    let mut out = Array::from_elem(seqs.raw_dim(), 0u8);
    let results = out
        .rows_mut()
        .into_iter()
        .zip(seqs.rows())
        .par_bridge()
        .map(|(out_row, row)| {
            k_shuffle1(row, k, seed, out_row, alphabet_size, alphabet_bytes)
        })
        .collect::<Vec<_>>();
    for result in results {
        result.expect("k_shuffle error");
    }
    out
}
```

(Per-row seed derivation comes in Task 6; this step only threads `alphabet_bytes` through.)

- [ ] **Step 4: Update `src/lib.rs` `_k_shuffle` pyfunction to accept and pass `alphabet_bytes`**

Update the pyfunction signature:

```rust
#[pyfunction]
fn _k_shuffle<'py>(
    py: Python<'py>,
    seqs: PyReadonlyArray<'py, u8, IxDyn>,
    k: usize,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
    seed: Option<u64>,
) -> &'py PyArray<u8, IxDyn> {
    let seqs = seqs.as_array();
    let out = kshuffle::k_shuffle(seqs, k, seed, alphabet_size, alphabet_bytes);
    out.into_pyarray(py)
}
```

- [ ] **Step 5: Update Python wrapper in `python/seqpro/_modifiers.py`**

Find the line:
```python
shuffled = _k_shuffle(seqs.view("u1"), k, len(alphabet), seed).view("S1")
```
Replace with:
```python
shuffled = _k_shuffle(seqs.view("u1"), k, len(alphabet), alphabet.array.view("u1").tobytes(), seed).view("S1")
```

If `alphabet.array` isn't the right accessor, inspect `python/seqpro/alphabets/_alphabets.py` for the attribute that gives the ordered bytes (likely `alphabet.array` or `alphabet.alphabet`). Use what's actually there.

- [ ] **Step 6: Update existing `same_freq` test to pass `alphabet_bytes`**

In `src/kshuffle.rs`, find:
```rust
let res = k_shuffle1(seq.view(), k, Some(1), shuffled.view_mut(), alphabet_size);
```
Replace with:
```rust
let res = k_shuffle1(seq.view(), k, Some(1), shuffled.view_mut(), alphabet_size, b"ACGT");
```

- [ ] **Step 7: Build Rust + Python, run all Rust tests**

```bash
maturin develop
cargo test --lib
```
Expected: all tests pass including `same_freq`, `direct_lut_*`, `hash_lut_*`, and `kmer_encode::tests::*`.

- [ ] **Step 8: Run existing Python tests for `k_shuffle`**

```bash
pytest tests/test_modifiers.py::test_k_shuffle tests/test_shape_matrix.py::test_k_shuffle_bytes_shape_matrix -v
```
Expected: all pass.

- [ ] **Step 9: Remove `derive_builder` from `Cargo.toml`**

Edit `Cargo.toml`, delete the line `derive_builder = "0.13.1"`.

Re-run `cargo check --lib` to confirm nothing else used it.

- [ ] **Step 10: Commit**

```bash
git add src/kshuffle.rs src/lib.rs python/seqpro/_modifiers.py Cargo.toml
git commit -m "refactor: rewrite k_shuffle1 with KmerIndex strategy + u32 Vertex"
```

---

## Task 6: Add per-row seed derivation + native rayon parallel iteration

**Files:**
- Modify: `src/kshuffle.rs`

- [ ] **Step 1: Replace `k_shuffle` outer function**

Find the current `pub fn k_shuffle` and replace with:

```rust
pub fn k_shuffle<D: Dimension>(
    seqs: ArrayView<u8, D>,
    k: usize,
    seed: Option<u64>,
    alphabet_size: usize,
    alphabet_bytes: &[u8],
) -> Array<u8, D> {
    let mut out = Array::from_elem(seqs.raw_dim(), 0u8);

    // Derive a per-row seed from a parent RNG so that:
    //  - same user seed + same batch produces identical output across runs
    //    and across rayon thread counts;
    //  - rows within a batch get independent shuffles.
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
        .zip(row_seeds.into_iter())
        .par_bridge()
        .map(|((out_row, row), row_seed)| {
            k_shuffle1(row, k, Some(row_seed), out_row, alphabet_size, alphabet_bytes)
        })
        .collect();

    for result in results {
        result.expect("k_shuffle error");
    }
    out
}
```

**Note:** `par_bridge` remains because ndarray's `rows()` doesn't expose a clean `IndexedParallelIterator`. Determinism is preserved because each row's seed is precomputed deterministically from the parent RNG before parallel execution.

- [ ] **Step 2: Add a determinism test in `mod test`**

```rust
#[test]
fn determinism_across_runs_and_thread_counts() {
    use ndarray::Array2;
    let alphabet_size = 4;
    let k = 4;
    // 8 rows of length 32, repeated DNA-like content
    let mut seqs = Array2::<u8>::zeros((8, 32));
    for (i, mut row) in seqs.rows_mut().into_iter().enumerate() {
        for (j, b) in row.iter_mut().enumerate() {
            *b = b"ACGT"[(i + j) % 4];
        }
    }

    let run = || {
        crate::kshuffle::k_shuffle(seqs.view(), k, Some(42), alphabet_size, b"ACGT")
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "k_shuffle must be deterministic with a fixed seed");
}
```

- [ ] **Step 3: Run the new test**

```bash
cargo test --lib kshuffle::test::determinism_across_runs_and_thread_counts
```
Expected: pass.

- [ ] **Step 4: Verify with thread-count change**

```bash
RAYON_NUM_THREADS=1 cargo test --lib kshuffle::test::determinism
RAYON_NUM_THREADS=4 cargo test --lib kshuffle::test::determinism
```
Expected: both pass. (The same `(seed, batch)` produces the same output regardless of thread count.)

- [ ] **Step 5: Commit**

```bash
git add src/kshuffle.rs
git commit -m "fix: derive per-row seeds so batches get independent shuffles"
```

---

## Task 7: Add equivalence test (new impl vs `kshuffle_ref` reference)

**Files:**
- Modify: `src/kshuffle.rs` (add test in `mod test`)

This test proves the optimization is a pure data-structure refactor: same algorithm, same RNG, same output.

**Note on expected equivalence:** The new impl and the reference share `SmallRng` and the same per-call sampling pattern, so a single-row call with the same `seed` MUST produce byte-equal output. We test the per-row function (`k_shuffle1`) directly to bypass the new batch-level seed derivation.

- [ ] **Step 1: Write the equivalence test**

Add to `mod test`:

```rust
#[test]
fn equivalence_with_reference_impl_for_k_2_through_8() {
    use ndarray::Array1;
    use rand::{Rng, SeedableRng};
    use rand::rngs::SmallRng;

    let alphabet_size = 4;
    let alphabet_bytes = b"ACGT";
    let mut seedgen = SmallRng::seed_from_u64(0xDEAD_BEEF);

    for &len in &[16usize, 64, 256, 1024] {
        for &k in &[2usize, 3, 4, 5, 6, 7, 8] {
            if k >= len { continue; }
            // Random DNA sequence.
            let seq: Vec<u8> = (0..len).map(|_| alphabet_bytes[seedgen.gen_range(0..4)]).collect();
            let seq_arr = Array1::from(seq.clone());
            let row_seed: u64 = seedgen.gen();

            let mut out_new = Array1::<u8>::zeros(len);
            let mut out_ref = Array1::<u8>::zeros(len);

            // New impl
            super::k_shuffle1(
                seq_arr.view(), k, Some(row_seed),
                out_new.view_mut(), alphabet_size, alphabet_bytes,
            ).unwrap();

            // Reference impl
            crate::kshuffle_ref::k_shuffle1_ref(
                seq_arr.view(), k, Some(row_seed),
                out_ref.view_mut(), alphabet_size,
            ).unwrap();

            assert_eq!(
                out_new, out_ref,
                "mismatch at len={} k={} seed={}", len, k, row_seed
            );
        }
    }
}
```

**Note:** `k_shuffle1_ref` is private in `kshuffle_ref`. Make it `pub(crate)` in `src/kshuffle_ref.rs` (change `fn k_shuffle1_ref` → `pub(crate) fn k_shuffle1_ref`).

- [ ] **Step 2: Run the equivalence test**

```bash
cargo test --lib kshuffle::test::equivalence_with_reference_impl_for_k_2_through_8
```
Expected: pass.

**If this test fails,** the optimization changed behavior. Investigate before proceeding — the most likely culprits are (a) vertex-id assignment order differs between old and new (the old code assigned ids by first-seen via `htable.entry().or_insert_with`; new code must do the same), or (b) the `i_sequence` "last write wins" rule is off (see Task 5 Step 1).

- [ ] **Step 3: Commit**

```bash
git add src/kshuffle.rs src/kshuffle_ref.rs
git commit -m "test: add equivalence test between new and reference k_shuffle impls"
```

---

## Task 8: Add proptest-based k-mer frequency invariant test

**Files:**
- Modify: `Cargo.toml` (add `proptest` as dev-dep)
- Modify: `src/kshuffle.rs` (replace the simple `same_freq` test)

- [ ] **Step 1: Add `proptest` as a dev-dependency in `Cargo.toml`**

Add under a new section (or extend existing if present):

```toml
[dev-dependencies]
proptest = "1.4"
```

- [ ] **Step 2: Replace `same_freq` test with proptest-based version**

Find the existing `#[test] fn same_freq()` in `src/kshuffle.rs` and replace the body of `mod test` with (preserving the `kmer_frequencies` helper):

```rust
#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    fn kmer_frequencies(seq: &[u8], k: usize) -> HashMap<&[u8], u32> {
        let mut freqs = HashMap::new();
        for kmer in seq.windows(k) {
            *freqs.entry(kmer).or_insert(0) += 1;
        }
        freqs
    }

    proptest! {
        #[test]
        fn k_shuffle_preserves_kmer_frequencies(
            len in 8usize..256,
            k in 2usize..8,
            seed in any::<u64>(),
            bytes in proptest::collection::vec(0u8..4, 256),
        ) {
            prop_assume!(k < len);
            let seq: Vec<u8> = bytes.into_iter().take(len).map(|c| b"ACGT"[c as usize]).collect();
            let seq_arr = ndarray::Array1::from(seq.clone());
            let mut out = ndarray::Array1::<u8>::zeros(len);
            super::k_shuffle1(seq_arr.view(), k, Some(seed), out.view_mut(), 4, b"ACGT").unwrap();

            let in_freqs = kmer_frequencies(seq_arr.as_slice().unwrap(), k);
            let out_freqs = kmer_frequencies(out.as_slice().unwrap(), k);
            prop_assert_eq!(in_freqs, out_freqs);
        }
    }

    // ... (keep the existing direct_lut_*, hash_lut_*, determinism_*, equivalence_* tests here)
}
```

**Important:** Keep all the other tests (`direct_lut_*`, `hash_lut_*`, `determinism_*`, `equivalence_*`) added in earlier tasks. Only the original `same_freq` is replaced; the rest stay.

- [ ] **Step 3: Run the proptest**

```bash
cargo test --lib kshuffle::test::k_shuffle_preserves_kmer_frequencies
```
Expected: pass (default 256 proptest cases).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml src/kshuffle.rs
git commit -m "test: add proptest-based k-mer frequency invariant"
```

---

## Task 9: Add criterion benchmark

**Files:**
- Create: `benches/kshuffle.rs`
- Modify: `Cargo.toml` (add `criterion` dev-dep + `[[bench]]` table)

- [ ] **Step 1: Update `Cargo.toml`**

Add to `[dev-dependencies]`:
```toml
criterion = { version = "0.5", features = ["html_reports"] }
```

Add at the bottom:
```toml
[[bench]]
name = "kshuffle"
harness = false
```

- [ ] **Step 2: Create `benches/kshuffle.rs`**

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::Array2;
use seqpro::kshuffle::k_shuffle;

fn make_batch(n_rows: usize, len: usize) -> Array2<u8> {
    let mut arr = Array2::<u8>::zeros((n_rows, len));
    for (i, mut row) in arr.rows_mut().into_iter().enumerate() {
        for (j, b) in row.iter_mut().enumerate() {
            *b = b"ACGT"[((i.wrapping_mul(7919) + j.wrapping_mul(31)) ^ (i + j)) % 4];
        }
    }
    arr
}

fn bench_kshuffle(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_shuffle/10k_x_200bp");
    let batch = make_batch(10_000, 200);
    for &k in &[2usize, 4, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let out = k_shuffle(black_box(batch.view()), k, Some(42), 4, b"ACGT");
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kshuffle);
criterion_main!(benches);
```

**Note:** This requires `kshuffle::k_shuffle` to be publicly reachable from the `seqpro` crate root (it's `pub mod kshuffle;` in `lib.rs` and `pub fn k_shuffle`, so this works).

- [ ] **Step 3: Run the benchmark to confirm it builds and produces numbers**

```bash
cargo bench --bench kshuffle -- --warm-up-time 1 --measurement-time 3
```
Expected: completes; reports a time per iteration for each k. Capture the numbers — these are the "after" baseline. (The "before" numbers come from running the same bench against `main` prior to this branch's changes — capture separately for the PR description.)

- [ ] **Step 4: Verify the ≥3× target on the target workload**

Compare against `main` HEAD numbers. If improvement is below 3×, **do not** mark the work complete — investigate (likely culprits: rayon overhead dominating on tiny rows, LUT zeroing dominating, or HashLut accidentally being selected). Add findings to the PR description.

- [ ] **Step 5: Commit**

```bash
git add benches/kshuffle.rs Cargo.toml
git commit -m "bench: add criterion benchmark for k_shuffle"
```

---

## Task 10: Update Python docstring for new seed contract

**Files:**
- Modify: `python/seqpro/_modifiers.py`

- [ ] **Step 1: Update the `k_shuffle` docstring**

Find the `seed` parameter description in `python/seqpro/_modifiers.py:39` (`def k_shuffle`):

```
    seed
        Seed or generator for shuffling.
```

Replace with:

```
    seed
        Seed or generator for shuffling. When given a fixed integer seed, the
        same ``(seed, batch_size, k)`` produces byte-identical output across
        runs and across thread counts; each row in a batch receives an
        independent shuffle derived from a parent RNG seeded by this value.
        Changing batch size changes the per-row seeds.
```

- [ ] **Step 2: Run Python tests again to confirm no regression**

```bash
pytest tests/test_modifiers.py tests/test_shape_matrix.py -v
```
Expected: all pass.

- [ ] **Step 3: Lint**

```bash
ruff check python/seqpro/_modifiers.py
ruff format --check python/seqpro/_modifiers.py
cargo clippy --all-targets -- -D warnings
```
Expected: all clean.

- [ ] **Step 4: Commit**

```bash
git add python/seqpro/_modifiers.py
git commit -m "docs: document per-row seed semantics in k_shuffle"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full Rust test suite**

```bash
cargo test --lib
```
Expected: all pass.

- [ ] **Step 2: Full Python test suite (the parts that touch k_shuffle and any cross-feature tests)**

```bash
pytest tests/ -v
```
Expected: all pass.

- [ ] **Step 3: Benchmark headline**

```bash
cargo bench --bench kshuffle 2>&1 | tee /tmp/kshuffle_bench_after.txt
```
Record headline numbers (k=2,4,6,8) and the before/after ratios for the PR description.

- [ ] **Step 4: Confirm verification gates from the spec**

- [x] All existing tests pass.
- [x] Equivalence test passes for k ∈ {2..8}.
- [x] Benchmark shows ≥3× improvement on 10K × 200bp workload.
- [x] `cargo clippy --all-targets -- -D warnings` clean.

If any gate fails, investigate before declaring success. Do not claim the work is done based on plan progress alone — the gates are the bar.
