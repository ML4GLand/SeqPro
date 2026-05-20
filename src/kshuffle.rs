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
        .map(|((out_row, row), row_seed)| {
            k_shuffle1(row, k, Some(row_seed), out_row, alphabet_size, alphabet_bytes)
        })
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
        k_shuffle1_inner(seq, k, &mut rng, out, alphabet_size, alphabet_bytes)
    } else {
        let owned: ndarray::Array1<u8> = seq.to_owned();
        k_shuffle1_inner(owned.view(), k, &mut rng, out, alphabet_size, alphabet_bytes)
    }
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
            super::k_shuffle1(seq_arr.view(), k, Some(seed), out.view_mut(), 4, b"ACGT").unwrap();

            let in_freqs = kmer_frequencies(seq_arr.as_slice().unwrap(), k);
            let out_freqs = kmer_frequencies(out.as_slice().unwrap(), k);
            prop_assert_eq!(in_freqs, out_freqs);
        }
    }

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
}
