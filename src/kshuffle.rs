use derive_builder::Builder;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3Builder;

use anyhow::{bail, Result};
use ndarray::prelude::*;
use rand::rngs::SmallRng;

#[derive(thiserror::Error, Debug)]
pub enum KShuffleError {
    #[error("k must be greater than 0")]
    KLessThanOne,
    #[error("k must be less than the length of the sequence")]
    KLargerThanLength,
    #[error("k must be less than the length of the sequence minus 1")]
    NEntriesTooSmall,
}

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

struct HEntry {
    /// Number of unique k-mers that come before this k-mer
    i_vertices: usize,
    /// First index where k-mer appears
    i_sequence: usize,
}

pub fn k_shuffle<D: Dimension>(
    seqs: ArrayView<u8, D>,
    k: usize,
    seed: Option<u64>,
    alphabet_size: usize,
) -> Array<u8, D> {
    let mut out = unsafe { Array::uninit(seqs.raw_dim()).assume_init() };

    let results = out
        .rows_mut()
        .into_iter()
        .zip(seqs.rows())
        .par_bridge()
        .map(|(out_row, row)| k_shuffle1(row, k, seed, out_row, alphabet_size))
        .collect::<Vec<_>>();

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
) -> Result<()> {
    let seed = seed.unwrap_or_else(|| rand::thread_rng().gen());
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = seq.len();

    if k >= l {
        seq.assign_to(out);
        return Ok(());
    }

    if k < 1 {
        bail!(KShuffleError::KLessThanOne);
    }

    if k == 1 {
        seq.assign_to(&mut out);
        out.as_slice_mut().unwrap().shuffle(&mut rng);
        return Ok(());
    }

    let n_lets = l - k + 2;
    let max_uniq_lets = n_lets.min(alphabet_size.pow((k - 1) as u32));
    let mut htable = HashMap::with_capacity_and_hasher(max_uniq_lets, Xxh3Builder::new());

    // find distinct verticess
    let mut n_vertices = 0;
    for (pos, kmer) in seq.windows(k - 1).into_iter().enumerate() {
        if n_vertices < max_uniq_lets {
            htable.entry(kmer.to_vec()).or_insert_with(|| {
                let hentry = HEntry {
                    i_vertices: n_vertices,
                    i_sequence: pos,
                };
                n_vertices += 1;
                hentry
            });
        }
    }

    let root = seq.slice(s![-(k as isize - 1)..]).to_vec();
    let mut indices = vec![0 as usize; n_lets - 1];
    let mut vertices = (0..n_vertices)
        .map(|_| VertexBuilder::default().intree(false).n_indices(0).next(0))
        .collect::<Vec<_>>();

    // set i_sequence and n_indices for each vertex
    for (i, kmer) in seq.windows(k - 1).into_iter().enumerate() {
        let hentry = htable.get(&kmer.to_vec()).unwrap();
        let v = &mut vertices[hentry.i_vertices];

        if i < n_lets - 1 {
            let n_indices = v.n_indices.map_or(1, |n| n + 1);
            vertices[hentry.i_vertices] = v.i_sequence(hentry.i_sequence).n_indices(n_indices);
        } else {
            vertices[hentry.i_vertices] = v.i_sequence(hentry.i_sequence);
        }
    }

    // distribute indices
    let mut current_idx = 0usize;
    for v in &mut vertices {
        *v = v.idx_offset(current_idx).into();
        current_idx += v.n_indices.unwrap();
    }

    let mut vertices = vertices
        .into_iter()
        .map(|v| v.i_indices(0).build().unwrap())
        .collect::<Vec<_>>();

    // populate indices for each vertex
    for (kmer1, kmer2) in seq
        .slice(s![..-1])
        .windows(k - 1)
        .into_iter()
        .zip(seq.slice(s![1..]).windows(k - 1))
    {
        let eu = htable.get(&kmer1.to_vec()).unwrap();
        let ev = htable.get(&kmer2.to_vec()).unwrap();

        let u = &mut vertices[eu.i_vertices];
        if u.n_indices > 0 {
            let i_indices = u.i_indices;
            indices[u.idx_offset + i_indices] = ev.i_vertices;
            u.i_indices += 1;
        }
    }

    // Wilson algorithm for random arborescence
    let root_idx = htable.get(&root).unwrap().i_vertices;
    wilson_random_spanning_tree(&mut vertices, &indices, root_idx, &mut rng);
    random_walk(&mut vertices, &mut indices, root_idx, &mut rng, seq, k, out);

    Ok(())
}

fn wilson_random_spanning_tree<R: Rng>(
    vertices: &mut Vec<Vertex>,
    indices: &Vec<usize>,
    root_idx: usize,
    rng: &mut R,
) {
    let root_vertex = &mut vertices[root_idx];
    root_vertex.intree = true;

    for i in 0..vertices.len() {
        // let mut u = &mut vertices[i];
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
                    u_idx = indices[u.idx_offset + u.next];
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
                    u_idx = indices[u.idx_offset + u.next];
                }
            }
        }
    }
}

fn random_walk<R: Rng>(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<usize>,
    root_idx: usize,
    rng: &mut R,
    seq: ArrayView1<u8>,
    k: usize,
    mut out: ArrayViewMut1<u8>,
) {
    let mut j;
    for (i, u) in vertices.iter_mut().enumerate() {
        if i != root_idx {
            let idx = u.n_indices - 1;
            j = indices[u.idx_offset + idx];
            indices[u.idx_offset + idx] = indices[u.idx_offset + u.next];
            let next = u.next;
            indices[u.idx_offset + next] = j;
            indices[u.idx_offset..u.idx_offset + idx].shuffle(rng);
        } else {
            indices[u.idx_offset..u.idx_offset + u.n_indices].shuffle(rng);
        }
        u.i_indices = 0;
    }

    // walk the graph
    let out = out.as_slice_mut().unwrap();
    out[..k - 1].clone_from_slice(seq.slice(s![..k - 1]).as_slice().unwrap());
    let mut i = k - 1;
    let mut u_idx = 0;
    loop {
        let v_idx = {
            let u = &vertices[u_idx];
            if u.i_indices >= u.n_indices {
                break;
            }
            indices[u.idx_offset + u.i_indices]
        };
        {
            if u_idx != v_idx {
                let v = &vertices[v_idx];
                j = v.i_sequence + k - 2;
                out[i] = seq[j];
                i += 1;
                vertices[u_idx].i_indices += 1;
            } else {
                let v = &mut vertices[v_idx];
                j = v.i_sequence + k - 2;
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

    fn kmer_frequencies(seq: &[u8], k: usize) -> HashMap<&[u8], u32> {
        let mut freqs = HashMap::new();

        for kmer in seq.windows(k) {
            let count = freqs.entry(kmer).or_insert(0);
            *count += 1;
        }

        freqs
    }

    #[test]
    fn same_freq() {
        let k = 2;
        let alphabet_size = 4;
        let seq = ArrayView1::from(b"AATAT");

        let freqs = kmer_frequencies(seq.as_slice().unwrap(), k);
        let mut shuffled = unsafe { Array::uninit(seq.len()).assume_init() };
        let res = k_shuffle1(seq.view(), k, Some(1), shuffled.view_mut(), alphabet_size);
        assert!(res.is_ok());

        let shuffled_freqs = kmer_frequencies(shuffled.as_slice().unwrap(), k);

        println!("{:?}", shuffled);
        assert_eq!(freqs, shuffled_freqs);
    }
}
