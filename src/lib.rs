pub mod kshuffle;
pub mod kmer_encode;
pub mod tokenize;
pub mod translate;
#[cfg(test)]
#[allow(dead_code, clippy::all)]
mod kshuffle_ref;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;

#[pymodule]
fn seqpro(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_k_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(tokenize::_tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_drop, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_stop_ends, m)?)?;
    Ok(())
}

#[pyfunction]
/// Shuffle sequences while preserving k-mer frequencies.
///
/// Parameters
/// ----------
/// seqs : NDArray[uint8]
///    Sequences to shuffle. Array must be contiguous with the last dimension
///    being the sequence length.
/// k : int
///    Length of k-mers to preserve frequencies of.
/// alphabet_size : int
///    Number of unique characters in the alphabet.
/// alphabet_bytes : bytes
///    Ordered alphabet bytes (e.g. b"ACGT").
/// seed : int, optional
///   Seed for the random number generator.
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
