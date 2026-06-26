use ndarray::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use digest::Digest;
use rapidhash::v1::{rapidhash_v1, rapidhash_v1_seeded, RapidSecrets};

/// Hash each delimited byte run with a RustCrypto `Digest` (md5, sha256, ...).
///
/// `offsets` has `N + 1` entries delimiting `N` runs in `data`. Returns a flat
/// `N * output_size` buffer (row-major, one digest per run).
fn hash_elems<D: Digest>(data: &[u8], offsets: &[i64]) -> Vec<u8> {
    let n = offsets.len().saturating_sub(1);
    let out_size = <D as Digest>::output_size();
    let mut out = vec![0u8; n * out_size];
    out.par_chunks_mut(out_size)
        .zip(offsets.par_windows(2))
        .for_each(|(chunk, w)| {
            let start = w[0] as usize;
            let stop = w[1] as usize;
            let digest = D::digest(&data[start..stop]);
            chunk.copy_from_slice(digest.as_slice());
        });
    out
}

/// One-shot rapidhash of a byte run using the portable, C++-compatible v1
/// algorithm (stable across crate versions and machines).
fn rapidhash_one(bytes: &[u8], seed: Option<u64>) -> u64 {
    match seed {
        Some(s) => rapidhash_v1_seeded(bytes, &RapidSecrets::seed(s)),
        None => rapidhash_v1(bytes),
    }
}

/// Hash each string in a packed string buffer.
///
/// Parameters
/// ----------
/// data : NDArray[uint8]
///     Flat, contiguous byte buffer holding all strings concatenated.
/// str_offsets : NDArray[int64]
///     `(N + 1,)` zero-based byte boundaries delimiting `N` strings in `data`.
/// algo : str
///     One of "md5", "sha256", "rapidhash".
/// seed : int, optional
///     Seed for rapidhash only.
#[pyfunction]
pub fn _ragged_hash<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    str_offsets: PyReadonlyArray1<'py, i64>,
    algo: &str,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>> {
    let data = data.as_slice()?;
    let offsets = str_offsets.as_slice()?;
    let n = offsets.len().saturating_sub(1);
    match algo {
        "md5" => {
            let out = py.detach(|| hash_elems::<md5::Md5>(data, offsets));
            let arr = Array2::from_shape_vec((n, 16), out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(arr.into_pyarray(py).into_any())
        }
        "sha256" => {
            let out = py.detach(|| hash_elems::<sha2::Sha256>(data, offsets));
            let arr = Array2::from_shape_vec((n, 32), out)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(arr.into_pyarray(py).into_any())
        }
        "rapidhash" => {
            let out: Vec<u64> = py.detach(|| {
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let start = offsets[i] as usize;
                        let stop = offsets[i + 1] as usize;
                        rapidhash_one(&data[start..stop], seed)
                    })
                    .collect()
            });
            Ok(Array1::from_vec(out).into_pyarray(py).into_any())
        }
        other => Err(PyValueError::new_err(format!(
            "unknown algo {other:?}; expected one of 'md5', 'sha256', 'rapidhash'"
        ))),
    }
}
