use rand::{seq::SliceRandom, Rng, SeedableRng};
use std::collections::HashMap;
use xxhash_rust::xxh3::Xxh3Builder;

use indextree::Arena;
use ndarray::prelude::*;
use rand::rngs::SmallRng;
use anyhow::{bail, Result};

#[derive(thiserror::Error, Debug)]
pub enum KShuffleError {
    #[error("k must be greater than 0")]
    KLessThanOne,
    #[error("k must be less than the length of the sequence")]
    KLargerThanLength,
    #[error("k must be less than the length of the sequence minus 1")]
    NEntriesTooSmall,
}

#[derive(Clone)]
struct Vertex {
    indices: Vec<usize>,
    n_indices: usize,
    i_indices: usize,
    intree: bool,
    next: usize,
    i_sequence: usize,
}

struct HEntry {
    next: Option<Box<HEntry>>,
    i_vertices: usize,
    i_sequence: usize,
}

fn kshuffle(
    arr: ArrayView1<u8>,
    k: usize,
    seed: Option<u64>,
) -> Result<Array1<u8>> {
    let seed = seed.unwrap_or(0);
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = arr.len();

    if k >= l {
        return Ok(arr.to_owned());
    }

    if k < 1 {
        bail!(KShuffleError::KLessThanOne);
    }

    if k == 1 {
        let mut out = arr.to_owned().to_vec();
        out.shuffle(&mut rng);
        let out = Array1::from_vec(out);
        return Ok(out);
    }

    let n_lets = l - k + 2;
    let mut htable = HashMap::with_capacity_and_hasher(n_lets, Xxh3Builder::new());

    for kmer in arr.windows(k) {
        let hentry = HEntry {
            next: None,
            i_vertices: 0,
            i_sequence: 0,
        };
        htable.insert(kmer.to_vec(), hentry);
    }

    let root = arr.slice(s![-(k as isize)..]).to_vec();
    let n_vertices = htable.len();
    let tree = Arena::<Vertex>::with_capacity(n_vertices);
    let mut vertices = vec![
        Vertex {
            indices: vec![],
            n_indices: 0,
            i_indices: 0,
            intree: false,
            next: 0,
            i_sequence: 0,
        };
        n_vertices
    ];

    for (i, kmer) in arr.windows(k).into_iter().enumerate() {
        let hentry = htable.get(&kmer.to_vec()).unwrap();
        let v = &mut vertices[hentry.i_vertices];

        v.i_sequence = hentry.i_sequence;
        if i < n_lets - 1 {
            v.n_indices += 1;
        }
    }

    let indices = vec![0 as usize; n_vertices];
    let mut j = 0;

    for v in &mut vertices {
        v.indices = indices[j..j + v.n_indices].to_vec();
        j += v.n_indices;
    }

    for (kmer1, kmer2) in arr
        .slice(s![..-1])
        .windows(k)
        .into_iter()
        .zip(arr.slice(s![1..]).windows(k))
    {
        let eu = htable.get(&kmer1.to_vec()).unwrap();
        let ev = htable.get(&kmer2.to_vec()).unwrap();
        let u = &mut vertices[eu.i_vertices];

        u.indices[u.i_indices] = ev.i_vertices;
        u.i_indices += 1;
    }

    let root_idx = htable.get(&root).unwrap().i_vertices;
    let root_vertex = &mut vertices[root_idx];
    root_vertex.intree = true;

    // TODO: doesn't compile due to multiple mutable borrows
    for v in &mut vertices {
        let mut u = v;
        while !u.intree {
            u.next = rng.gen_range(0..u.n_indices);
            u = &mut vertices[u.indices[u.next]];
        }
        let mut u = v;
        while !u.intree {
            u.intree = true;
            u = &mut vertices[u.indices[u.next]];
        }
    }
    
    // TODO: doesn't compile due to multiple mutable borrows
    for (i, v) in vertices.iter_mut().enumerate() {
        if i != root_idx {
            j = v.indices[v.n_indices - 1];
            v.indices[v.n_indices - 1] = v.indices[v.next];
            v.indices[v.next] = j;
            v.indices[0..v.n_indices - 1].shuffle(&mut rng);
        } else {
            v.indices.shuffle(&mut rng);
        }
        v.i_indices = 0;
    }

    let mut out: Vec<u8> = vec![0; l];
    out[..k].clone_from_slice(arr.slice(s![..k]).as_slice().unwrap());
    let u = &mut vertices[0];
    let mut i = k - 1;
    while u.i_indices < u.n_indices {
        let v = &mut vertices[u.indices[u.i_indices]];
        j = v.i_sequence + k - 2;
        out[i] = arr[j];
        i += 1;
        u.i_indices += 1;
        let u = v;
    }

    Ok(Array1::from(out))
}

#[cfg(test)]
mod test {
    use super::*;

    fn kmer_frequencies(seq: Vec<u8>, k: usize) -> HashMap<&[u8], u32> {
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
        let seq = Array1::from(vec![b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T']);

        let freqs = kmer_frequencies(seq.to_vec(), k);
        let shuffled = kshuffle(seq.view(), k, Some(0)).unwrap();
        let shuffled_freqs = kmer_frequencies(shuffled.to_vec(), k);

        assert_eq!(freqs, shuffled_freqs);
    }
}
