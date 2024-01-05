use std::collections::HashMap;
use rand::{seq::SliceRandom, SeedableRng, Rng};
use thiserror::Error;
use xxhash_rust::xxh3::{Xxh3, Xxh3Builder};

use rand::rngs::SmallRng;
use ndarray::prelude::*;

type Arr_u8 = ArrayView1<u8>;

#[derive(Error, Debug)]
pub enum KShuffleError {
    KLessThanOne,
    KLargerThanLength,
    NEntriesTooSmall,
}

struct Vertex {
    indices: [usize],
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

const HASHER: Xxh3 = Xxh3Builder::new().build();

fn kshuffle(arr: Arr_u8, k: usize, seed: u64) -> Result<Array1<u8>, Error> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = arr.len();

    if k >= l {
        return Ok(arr.to_owned())
    }
    
    if k < 1 {
        return KShuffleError::KLessThanOne()
    }

    if k == 1 {
        let out = arr.to_owned().as_slice_mut()?;
        out.shuffle(&mut rng);
        let out = Array1::from(out);
        return Ok(out)
    }

    let n_lets = l - k + 2;
    let mut htable = HashMap::with_capacity(n_lets);
    
    for kmer in arr.windows(k) {
        let hentry = HEntry {
            next: None,
            i_vertices: 0,
            i_sequence: 0,
        };
        htable.insert(kmer.as_slice()?, hentry);
    }
    
    let root = arr.slice(s![-k..]).as_slice()?;
    let n_vertices = htable.len();
    let vertices = vec![Vertex {
        indices: vec![],
        n_indices: 0,
        i_indices: 0,
        intree: false,
        next: 0,
        i_sequence: 0,
    }; n_vertices];

    for (i, kmer) in arr.windows(k).into_iter().enumerate() {
        let hentry = htable.get(kmer.as_slice()?)?;
        let v = &mut vertices[hentry.i_vertices];

        v.i_sequence = hentry.i_sequence;
        if i < n_lets - 1 {
            v.n_indices += 1;
        }
    }

    let indices = vec![0 as usize; n_vertices];
    let mut j = 0;

    for v in vertices {
        v.indices = indices[j..j + v.n_indices];
        j += v.n_indices;
    }

    for (kmer1, kmer2) in arr.slice(s![..-1]).windows(k).into_iter().zip(arr.slice(s![1..]).windows(k)) {
        let eu = htable.get(kmer1.as_slice()?)?;
        let ev = htable.get(kmer2.as_slice()?)?;
        let u = vertices[eu.i_vertices];

        u.indices[u.i_indices] = ev.i_vertices;
        u.i_indices += 1;
    }

    let root_vertex = vertices[htable.get(root)?.i_vertices];
    root_vertex.intree = true;

    for v in vertices {
        let u = v;
        while !u.intree {
            u.next = rng.gen_range(0..u.n_indices);
            u = vertices[u.indices[u.next]];
        }
        let u = v;
        while !u.intree {
            u.intree = true;
            u = vertices[u.indices[u.next]];
        }
    }

    for v in &mut vertices {
        if v != root_vertex {
            j = v.indices[v.n_indices - 1];
            v.indices[v.n_indices - 1] = v.indices[v.next];
            v.indices[v.next] = j;
            v.indices[0..v.n_indices - 1].shuffle(&mut rng);
        } else {
            v.indices.shuffle(&mut rng);
        }
        v.i_indices = 0;
    }

    let mut out: Vec<u8> = Vec::with_capacity(l);
    out[0..k] = arr.slice(s![..k]).as_slice()?;
    let mut u = vertices[0];
    let mut i = k - 1;
    while u.i_indices < u.n_indices {
        let v = vertices[u.indices[u.i_indices]];
        j = v.i_sequence + k - 2;
        out[i] = arr[j];
        i += 1;
        u.i_indices += 1;
        u = v;
    }

    Ok(Array1::from(out))
}
