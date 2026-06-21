pub mod kmer_encode;
pub mod kshuffle;
#[cfg(test)]
#[allow(dead_code, clippy::all)]
mod kshuffle_ref;
pub mod ragged;
pub mod translate;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type RaggedSelectResult<'py> = PyResult<(&'py PyArray<i64, Ix1>, &'py PyArray<i64, Ix1>)>;

#[pymodule]
fn seqpro(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_k_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_drop, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_stop_ends, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_ohe, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_ohe_drop, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_validate, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_select, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_nested_gather, m)?)?;
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

#[pyfunction]
fn _ragged_validate(offsets: PyReadonlyArray1<i64>, n_data: i64, n_segments: i64) -> PyResult<()> {
    ragged::validate(offsets.as_array(), n_data, n_segments).map_err(PyValueError::new_err)
}

#[pyfunction]
fn _ragged_nested_gather<'py>(
    py: Python<'py>,
    o0_starts: PyReadonlyArray1<'py, i64>,
    o0_stops: PyReadonlyArray1<'py, i64>,
    mask: PyReadonlyArray1<'py, bool>,
) -> RaggedSelectResult<'py> {
    let (counts, idx) =
        ragged::nested_gather(o0_starts.as_array(), o0_stops.as_array(), mask.as_array())
            .map_err(PyValueError::new_err)?;
    Ok((counts.into_pyarray(py), idx.into_pyarray(py)))
}

#[pyfunction]
fn _ragged_select<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<'py, i64>,
    stops: PyReadonlyArray1<'py, i64>,
    idx: PyReadonlyArray1<'py, i64>,
) -> RaggedSelectResult<'py> {
    let (s, e) = ragged::select(starts.as_array(), stops.as_array(), idx.as_array())
        .map_err(PyValueError::new_err)?;
    Ok((s.into_pyarray(py), e.into_pyarray(py)))
}
