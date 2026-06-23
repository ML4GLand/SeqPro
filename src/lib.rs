pub mod kmer_encode;
pub mod kshuffle;
#[cfg(test)]
#[allow(dead_code, clippy::all)]
mod kshuffle_ref;
pub mod ragged;
pub mod translate;

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray, PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

type RaggedSelectResult<'py> =
    PyResult<(Bound<'py, PyArray<i64, Ix1>>, Bound<'py, PyArray<i64, Ix1>>)>;
type NestedPackResult<'py> = PyResult<(
    Bound<'py, PyArray<i64, Ix1>>,
    Bound<'py, PyArray<i64, Ix1>>,
    Bound<'py, PyArray<u8, Ix1>>,
)>;

#[pymodule]
fn seqpro(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_k_shuffle, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_drop, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_stop_ends, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_ohe, m)?)?;
    m.add_function(wrap_pyfunction!(translate::_translate_ohe_drop, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_validate, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_select, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_nested_gather, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_nested_pack, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_pack, m)?)?;
    m.add_function(wrap_pyfunction!(_ragged_concat, m)?)?;
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
) -> Bound<'py, PyArray<u8, IxDyn>> {
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
fn _ragged_nested_pack<'py>(
    py: Python<'py>,
    o0_starts: PyReadonlyArray1<'py, i64>,
    o0_stops: PyReadonlyArray1<'py, i64>,
    o1_starts: PyReadonlyArray1<'py, i64>,
    o1_stops: PyReadonlyArray1<'py, i64>,
    src: PyReadonlyArray1<'py, u8>,
    elem: i64,
) -> NestedPackResult<'py> {
    let (o0, o1, out_bytes) = ragged::nested_pack(
        o0_starts.as_array(),
        o0_stops.as_array(),
        o1_starts.as_array(),
        o1_stops.as_array(),
        src.as_array(),
        elem,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        o0.into_pyarray(py),
        o1.into_pyarray(py),
        out_bytes.into_pyarray(py),
    ))
}

#[pyfunction]
fn _ragged_pack<'py>(
    starts: PyReadonlyArray1<'py, i64>,
    stops: PyReadonlyArray1<'py, i64>,
    src: PyReadonlyArray1<'py, u8>,
    elem: i64,
    mut out: PyReadwriteArray1<'py, u8>,
) -> PyResult<()> {
    ragged::pack_into(
        starts.as_array(),
        stops.as_array(),
        src.as_array(),
        elem,
        out.as_array_mut()
            .as_slice_mut()
            .ok_or_else(|| PyValueError::new_err("out must be contiguous"))?,
    )
    .map_err(PyValueError::new_err)
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

type RaggedConcatResult<'py> =
    PyResult<(Bound<'py, PyArray<u8, Ix1>>, Bound<'py, PyArray<i64, Ix1>>)>;

/// Concatenate N ragged arrays along their ragged axis.
///
/// Parameters
/// ----------
/// data_list : list of NDArray[uint8]
///     Byte-view of each packed data buffer (contiguous uint8, 1-D).
/// offsets_list : list of NDArray[int64]
///     1-D (G+1,) offset arrays for each input (element units, not bytes).
/// elem : int
///     Byte size of one logical element (e.g. 4 for int32/float32).
///
/// Returns
/// -------
/// (out_data, out_offsets) : (NDArray[uint8], NDArray[int64])
///     Concatenated byte buffer and (G+1,) cumulative offsets in element units.
#[pyfunction]
fn _ragged_concat<'py>(
    py: Python<'py>,
    data_list: &Bound<'py, PyList>,
    offsets_list: &Bound<'py, PyList>,
    elem: usize,
) -> RaggedConcatResult<'py> {
    // Extract contiguous u8 slices from each data array.
    let mut data_bufs: Vec<PyReadonlyArray1<'py, u8>> = Vec::with_capacity(data_list.len());
    for item in data_list.iter() {
        let arr: PyReadonlyArray1<u8> = item.extract().map_err(|e| {
            PyValueError::new_err(format!("data_list element is not a 1-D uint8 array: {}", e))
        })?;
        data_bufs.push(arr);
    }

    // Extract contiguous i64 slices from each offsets array.
    let mut off_bufs: Vec<PyReadonlyArray1<'py, i64>> = Vec::with_capacity(offsets_list.len());
    for item in offsets_list.iter() {
        let arr: PyReadonlyArray1<i64> = item.extract().map_err(|e| {
            PyValueError::new_err(format!(
                "offsets_list element is not a 1-D int64 array: {}",
                e
            ))
        })?;
        off_bufs.push(arr);
    }

    let data_slices: Vec<&[u8]> = data_bufs
        .iter()
        .map(|a| {
            a.as_slice()
                .map_err(|_| PyValueError::new_err("data array must be contiguous"))
        })
        .collect::<PyResult<_>>()?;

    let off_slices: Vec<&[i64]> = off_bufs
        .iter()
        .map(|a| {
            a.as_slice()
                .map_err(|_| PyValueError::new_err("offsets array must be contiguous"))
        })
        .collect::<PyResult<_>>()?;

    let (out_data, out_offsets) =
        ragged::ragged_concat(data_slices, off_slices, elem).map_err(PyValueError::new_err)?;

    Ok((
        Array1::from_vec(out_data).into_pyarray(py),
        Array1::from_vec(out_offsets).into_pyarray(py),
    ))
}
