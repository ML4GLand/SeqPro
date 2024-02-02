use derive_builder::Builder;
use rand::{seq::SliceRandom, Rng, SeedableRng};
use rayon::prelude::*;
use std::cell::RefCell;
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
#[builder(pattern = "owned")]
struct Vertex<'a> {
    indices: &'a mut [usize],
    n_indices: usize,
    i_indices: usize,
    intree: bool,
    next: usize,
    i_sequence: usize,
}

struct HEntry {
    i_vertices: usize,
    i_sequence: usize,
}

pub fn k_shuffle<D: Dimension>(
    seqs: ArrayView<u8, D>,
    k: usize,
    seed: Option<u64>,
) -> Array<u8, D> {
    let mut out = unsafe { Array::uninit(seqs.raw_dim()).assume_init() };

    let results = out
        .rows_mut()
        .into_iter()
        .zip(seqs.rows())
        .par_bridge()
        .map(|(out_row, row)| k_shuffle1(row, k, seed, out_row))
        .collect::<Vec<_>>();

    for result in results {
        result.expect("k_shuffle error");
    }

    out
}

fn k_shuffle1(
    arr: ArrayView1<u8>,
    k: usize,
    seed: Option<u64>,
    mut out: ArrayViewMut1<u8>,
) -> Result<()> {
    let seed = seed.unwrap_or_else(|| {
        rand::thread_rng().gen()
    });
    let mut rng = SmallRng::seed_from_u64(seed);
    let l = arr.len();

    if k >= l {
        arr.assign_to(out);
        return Ok(());
    }

    if k < 1 {
        bail!(KShuffleError::KLessThanOne);
    }

    if k == 1 {
        arr.assign_to(&mut out);
        out.as_slice_mut().unwrap().shuffle(&mut rng);
        return Ok(());
    }

    let n_lets = l - k + 2;
    let mut htable = HashMap::with_capacity_and_hasher(n_lets, Xxh3Builder::new());

    // find distinct verticess
    let mut n_vertices = 0;
    for (pos, kmer) in arr.windows(k - 1).into_iter().enumerate() {
        htable.entry(kmer.to_vec()).or_insert_with(|| {
            let hentry = HEntry {
                i_vertices: n_vertices,
                i_sequence: pos,
            };
            n_vertices += 1;
            hentry
        });
    }

    let n_vertices = htable.len();
    let root = arr.slice(s![-(k as isize - 1)..]).to_vec();
    let mut indices = vec![0 as usize; n_lets - 1];
    let mut vertices = (0..n_vertices)
        .map(|_| RefCell::new(VertexBuilder::default().intree(false).n_indices(0).next(0)))
        .collect::<Vec<_>>();

    // set i_sequence and n_indices for each vertex
    for (i, kmer) in arr.windows(k - 1).into_iter().enumerate() {
        let hentry = htable.get(&kmer.to_vec()).unwrap();
        let v = vertices[hentry.i_vertices].take();

        if i < n_lets - 1 {
            let n_indices = v.n_indices.map_or(1, |n| n + 1);
            vertices[hentry.i_vertices] =
                v.i_sequence(hentry.i_sequence).n_indices(n_indices).into();
        } else {
            vertices[hentry.i_vertices] = v.i_sequence(hentry.i_sequence).into();
        }
    }

    // distribute indices
    let mut for_vertex: &mut [usize];
    let mut indices_slice = indices.as_mut_slice();
    for v in &mut vertices {
        let temp = v.take();
        (for_vertex, indices_slice) = indices_slice.split_at_mut(temp.n_indices.unwrap());
        *v = temp.indices(for_vertex).into();
    }

    let vertices = vertices
        .into_iter()
        .map(|v| RefCell::new(v.take().i_indices(0).build().unwrap()))
        .collect::<Vec<_>>();

    // populate indices for each vertex
    for (kmer1, kmer2) in arr
        .slice(s![..-1])
        .windows(k - 1)
        .into_iter()
        .zip(arr.slice(s![1..]).windows(k - 1))
    {
        let eu = htable.get(&kmer1.to_vec()).unwrap();
        let ev = htable.get(&kmer2.to_vec()).unwrap();

        let mut u = vertices[eu.i_vertices].borrow_mut();
        if u.n_indices > 0 {
            let i_indices = u.i_indices;
            u.indices[i_indices] = ev.i_vertices;
            u.i_indices += 1;
        }
    }

    // Wilson algorithm for random arborescence
    let root_idx = htable.get(&root).unwrap().i_vertices;
    {
        let mut root_vertex = vertices[root_idx].borrow_mut();
        root_vertex.intree = true;
    }

    for i in 0..vertices.len() {
        // let mut u = &mut vertices[i];
        {
            let mut u_idx = i;
            loop {
                {
                    let u = vertices[u_idx].borrow();
                    if u.intree {
                        break;
                    }
                }
                {
                    let mut u = vertices[u_idx].borrow_mut();
                    u.next = rng.gen_range(0..u.n_indices);
                }
                {
                    let u = vertices[u_idx].borrow();
                    u_idx = u.indices[u.next];
                }
            }
        }
        {
            let mut u_idx = i;
            loop {
                {
                    let u = vertices[u_idx].borrow();
                    if u.intree {
                        break;
                    }
                }
                {
                    let mut u = vertices[u_idx].borrow_mut();
                    u.intree = true;
                }
                {
                    let u = vertices[u_idx].borrow();
                    u_idx = u.indices[u.next];
                }
            }
        }
    }

    // shuffle indices to prepare for walk
    let mut j;
    for (i, mut u) in vertices.iter().map(|v| v.borrow_mut()).enumerate() {
        if i != root_idx {
            let idx = u.n_indices - 1;
            j = u.indices[idx];
            u.indices[idx] = u.indices[u.next];
            let next = u.next;
            u.indices[next] = j;
            u.indices[0..idx].shuffle(&mut rng);
        } else {
            u.indices.shuffle(&mut rng);
        }
        u.i_indices = 0;
    }

    // walk the graph
    let out = out.as_slice_mut().unwrap();
    out[..k - 1].clone_from_slice(arr.slice(s![..k - 1]).as_slice().unwrap());
    let mut i = k - 1;
    let mut u_idx = 0;
    loop {
        let v_idx = {
            let u = vertices[u_idx].borrow();
            if u.i_indices >= u.n_indices {
                break;
            }
            u.indices[u.i_indices]
        };
        {
            if u_idx != v_idx {
                let v = vertices[v_idx].borrow();
                j = v.i_sequence + k - 2;
                out[i] = arr[j];
                i += 1;
                vertices[u_idx].borrow_mut().i_indices += 1;
            } else {
                let mut v = vertices[v_idx].borrow_mut();
                j = v.i_sequence + k - 2;
                out[i] = arr[j];
                i += 1;
                v.i_indices += 1;
            }
        }
        u_idx = v_idx;
    }

    Ok(())
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
        let seq = ArrayView1::from(b"AATAT");

        let freqs = kmer_frequencies(seq.as_slice().unwrap(), k);
        let mut shuffled = unsafe { Array::uninit(seq.len()).assume_init() };
        let res = k_shuffle1(seq.view(), k, Some(1), shuffled.view_mut());
        assert!(res.is_ok());

        let shuffled_freqs = kmer_frequencies(shuffled.as_slice().unwrap(), k);

        println!("{:?}", shuffled);
        assert_eq!(freqs, shuffled_freqs);
    }
}
