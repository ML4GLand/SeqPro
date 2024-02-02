pub mod kshuffle;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray};
use pyo3::prelude::*;

#[pymodule]
fn seqpro(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_k_shuffle, m)?)?;
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
/// seed : int, optional
///   Seed for the random number generator.
fn _k_shuffle<'py>(
    py: Python<'py>,
    seqs: PyReadonlyArray<'py, u8, IxDyn>,
    k: usize,
    seed: Option<u64>,
) -> &'py PyArray<u8, IxDyn> {
    let seqs = seqs.as_array();
    let out = kshuffle::k_shuffle(seqs, k, seed);
    out.into_pyarray(py)
}
