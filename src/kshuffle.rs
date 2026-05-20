use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3Builder;

use anyhow::{bail, Result};
use ndarray::prelude::*;
use rand::rngs::SmallRng;

use crate::kmer_encode;

const MAX_LUT: u32 = 16_384;

/// Returns `Some(α^(k-1))` if `DirectLut` fits, else `None`.
fn lut_size(alphabet_size: usize, k_minus_1: u32) -> Option<u32> {
    let n = (alphabet_size as u32).checked_pow(k_minus_1)?;
    if n <= MAX_LUT { Some(n) } else { None }
}

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
    /// Wilson walk path stack.
    path: Vec<u32>,
}

impl ShuffleBuffers {
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

    /// Restore `u32::MAX` at every position previously written.
    /// O(written count), not O(MAX_LUT).
    fn reset_lut(&mut self) {
        for &c in &self.lut_written {
            self.lut[c as usize] = u32::MAX;
        }
        self.lut_written.clear();
    }

    /// Build the direct lookup index. Fills `self.lut`, `self.lut_written`,
    /// and `self.codes`. Returns `n_vertices`.
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
    /// Fallback for protein / large k where `α^(k-1) > MAX_LUT`.
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
                *map.values().next().unwrap()
            };
            self.codes.push(id);
        }

        n_vertices
    }
}


#[derive(thiserror::Error, Debug)]
pub enum KShuffleError {
    #[error("k must be greater than 0")]
    KLessThanOne,
    #[error("k must be less than the length of the sequence")]
    KLargerThanLength,
    #[error("k must be less than the length of the sequence minus 1")]
    NEntriesTooSmall,
}

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

    if seq.is_standard_layout() {
        k_shuffle1_inner(seq, k, &mut rng, out, alphabet_size, alphabet_bytes, buffers)
    } else {
        let owned: ndarray::Array1<u8> = seq.to_owned();
        k_shuffle1_inner(owned.view(), k, &mut rng, out, alphabet_size, alphabet_bytes, buffers)
    }
}

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
    random_walk(&mut buffers.vertices, &mut buffers.indices, root_idx, rng, seq, k, out);

    Ok(())
}

fn wilson_random_spanning_tree<R: Rng>(
    vertices: &mut [Vertex],
    indices: &[u32],
    root_idx: usize,
    rng: &mut R,
) {
    let root_vertex = &mut vertices[root_idx];
    root_vertex.intree = true;

    for i in 0..vertices.len() {
        {
            let mut u_idx = i;
            loop {
                {
                    let u = &vertices[u_idx];
                    if u.intree {
                        break;
                    }
                }
                {
                    let u = &mut vertices[u_idx];
                    u.next = rng.gen_range(0..u.n_indices);
                }
                {
                    let u = &vertices[u_idx];
                    u_idx = indices[(u.idx_offset + u.next) as usize] as usize;
                }
            }
        }
        {
            let mut u_idx = i;
            loop {
                {
                    let u = &vertices[u_idx];
                    if u.intree {
                        break;
                    }
                }
                {
                    let u = &mut vertices[u_idx];
                    u.intree = true;
                }
                {
                    let u = &vertices[u_idx];
                    u_idx = indices[(u.idx_offset + u.next) as usize] as usize;
                }
            }
        }
    }
}

fn random_walk<R: Rng>(
    vertices: &mut [Vertex],
    indices: &mut [u32],
    root_idx: usize,
    rng: &mut R,
    seq: ArrayView1<u8>,
    k: usize,
    mut out: ArrayViewMut1<u8>,
) {
    for (i, u) in vertices.iter_mut().enumerate() {
        if i != root_idx {
            let idx = u.n_indices - 1;
            let off = u.idx_offset as usize;
            let next = u.next as usize;
            let idx_us = idx as usize;
            indices.swap(off + idx_us, off + next);
            indices[off..off + idx_us].shuffle(rng);
        } else {
            let off = u.idx_offset as usize;
            let n = u.n_indices as usize;
            indices[off..off + n].shuffle(rng);
        }
        u.i_indices = 0;
    }

    // walk the graph
    let out = out.as_slice_mut().unwrap();
    out[..k - 1].clone_from_slice(seq.slice(s![..k - 1]).as_slice().unwrap());
    let mut i = k - 1;
    let mut u_idx = 0usize;
    loop {
        let v_idx_u32 = {
            let u = &vertices[u_idx];
            if u.i_indices >= u.n_indices {
                break;
            }
            indices[(u.idx_offset + u.i_indices) as usize]
        };
        let v_idx = v_idx_u32 as usize;
        {
            let j: usize;
            if u_idx != v_idx {
                let v = &vertices[v_idx];
                j = v.i_sequence as usize + k - 2;
                out[i] = seq[j];
                i += 1;
                vertices[u_idx].i_indices += 1;
            } else {
                let v = &mut vertices[v_idx];
                j = v.i_sequence as usize + k - 2;
                out[i] = seq[j];
                i += 1;
                v.i_indices += 1;
            }
        }
        u_idx = v_idx;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    fn kmer_frequencies(seq: &[u8], k: usize) -> HashMap<&[u8], u32> {
        let mut freqs = HashMap::new();

        for kmer in seq.windows(k) {
            let count = freqs.entry(kmer).or_insert(0);
            *count += 1;
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
            let mut buffers = super::ShuffleBuffers::new();
            super::k_shuffle1(seq_arr.view(), k, Some(seed), out.view_mut(), 4, b"ACGT", &mut buffers).unwrap();

            let in_freqs = kmer_frequencies(seq_arr.as_slice().unwrap(), k);
            let out_freqs = kmer_frequencies(out.as_slice().unwrap(), k);
            prop_assert_eq!(in_freqs, out_freqs);
        }
    }

    #[test]
    fn build_direct_index_assigns_ids_in_first_seen_order() {
        let b2c = crate::kmer_encode::build_byte_to_code(b"ACGT");
        let mut buf = ShuffleBuffers::new();
        // Sequence AACGAA, k-1=2 → windows: AA, AC, CG, GA, AA
        let n = buf.build_direct_index(b"AACGAA", 2, 4, &b2c, 16, 100);
        assert_eq!(n, 4);
        assert_eq!(buf.codes, vec![0, 1, 2, 3, 0]);
        // lut_written records encoded k-mer values that were assigned ids.
        // AA=0, AC=1, CG=6, GA=8.
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

    #[test]
    fn determinism_across_runs_and_thread_counts() {
        use ndarray::Array2;
        let alphabet_size = 4;
        let k = 4;
        // 8 rows of length 32, deterministic DNA-like content
        let mut seqs = Array2::<u8>::zeros((8, 32));
        for (i, mut row) in seqs.rows_mut().into_iter().enumerate() {
            for (j, b) in row.iter_mut().enumerate() {
                *b = b"ACGT"[(i + j) % 4];
            }
        }

        let run = || {
            super::k_shuffle(seqs.view(), k, Some(42), alphabet_size, b"ACGT")
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "k_shuffle must be deterministic with a fixed seed");

        // Rows within the batch should NOT all be identical to each other
        // (regression check on the seed-reuse bug).
        let row0 = a.row(0).to_owned();
        let row1 = a.row(1).to_owned();
        assert_ne!(
            row0, row1,
            "rows in a batch should get independent shuffles, not identical ones"
        );
    }

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
                let mut buffers = super::ShuffleBuffers::new();
                super::k_shuffle1(
                    seq_arr.view(), k, Some(row_seed),
                    out_new.view_mut(), alphabet_size, alphabet_bytes, &mut buffers,
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
}
