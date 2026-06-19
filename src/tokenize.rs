//! LUT gather for `seqpro.tokenize`: `out[i] = lut[seq[i]]` over a flat
//! `u8` buffer, choosing serial vs. rayon-parallel by an element-count
//! threshold. The 256-entry `i32` LUT is built in Python (NumPy).

use rayon::prelude::*;

/// Element count at or above which the rayon gather overtakes the serial one.
/// Re-measured for rayon in `benches/` (Task 5); the old Numba constant (40k)
/// was tuned to Numba's ~96µs thread-launch floor and does not transfer.
pub const TOKENIZE_PARALLEL_THRESHOLD: usize = 40_000;

/// Serial gather. `out` and `seq` must have equal length; `lut` has 256 entries.
pub fn gather_serial(seq: &[u8], lut: &[i32], out: &mut [i32]) {
    for (o, &s) in out.iter_mut().zip(seq.iter()) {
        *o = lut[s as usize];
    }
}

/// Parallel gather over contiguous slices.
pub fn gather_parallel(seq: &[u8], lut: &[i32], out: &mut [i32]) {
    out.par_iter_mut()
        .zip(seq.par_iter())
        .for_each(|(o, &s)| *o = lut[s as usize]);
}

use numpy::{IntoPyArray, PyArray, PyReadonlyArray, PyReadonlyArray1, PyReadwriteArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Tokenize a `u8` array via a 256-entry `i32` LUT: `out[i] = lut[seq[i]]`.
///
/// `parallel`: `None` → threshold heuristic, `Some(true/false)` → forced.
/// A non-C-contiguous `out` can only be written serially; with `parallel=Some(true)`
/// that is an error (the Python layer raises first; this is a backstop).
#[pyfunction]
#[pyo3(signature = (seqs, lut, out=None, parallel=None))]
pub fn _tokenize<'py>(
    py: Python<'py>,
    seqs: PyReadonlyArray<'py, u8, numpy::IxDyn>,
    lut: PyReadonlyArray1<'py, i32>,
    out: Option<&'py PyArray<i32, numpy::IxDyn>>,
    parallel: Option<bool>,
) -> PyResult<&'py PyArray<i32, numpy::IxDyn>> {
    let seq_arr = seqs.as_array();
    let lut = lut.as_slice()?;
    let n = seq_arr.len();

    // Flatten input to a contiguous Vec<u8> (input is read-only; cheap and
    // lets both serial and parallel paths run on a slice).
    let seq_owned: Vec<u8> = match seq_arr.as_slice() {
        Some(s) => s.to_vec(),
        None => seq_arr.iter().copied().collect(),
    };

    let want_parallel = parallel.unwrap_or(n >= TOKENIZE_PARALLEL_THRESHOLD);

    match out {
        Some(o) => {
            let mut rw: PyReadwriteArray<i32, numpy::IxDyn> = o.readwrite();
            let contiguous = rw.as_array().is_standard_layout();
            if !contiguous && parallel == Some(true) {
                return Err(PyValueError::new_err(
                    "parallel=True requires a C-contiguous out array, got a \
                     non-contiguous out. Use parallel=None/False or a contiguous out.",
                ));
            }
            if contiguous {
                let slice = rw.as_slice_mut()?;
                if want_parallel {
                    gather_parallel(&seq_owned, lut, slice);
                } else {
                    gather_serial(&seq_owned, lut, slice);
                }
            } else {
                // Strided out: serial write through the ndarray view.
                let mut view = rw.as_array_mut();
                for (o, &s) in view.iter_mut().zip(seq_owned.iter()) {
                    *o = lut[s as usize];
                }
            }
            Ok(o)
        }
        None => {
            let mut buf = vec![0i32; n];
            if want_parallel {
                gather_parallel(&seq_owned, lut, &mut buf);
            } else {
                gather_serial(&seq_owned, lut, &mut buf);
            }
            let shape: Vec<usize> = seq_arr.shape().to_vec();
            let arr = ndarray::Array::from_shape_vec(numpy::IxDyn(&shape), buf)
                .expect("shape/len mismatch");
            Ok(arr.into_pyarray(py))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lut() -> Vec<i32> {
        let mut l = vec![-1i32; 256];
        l[b'A' as usize] = 0;
        l[b'C' as usize] = 1;
        l[b'G' as usize] = 2;
        l[b'T' as usize] = 3;
        l
    }

    #[test]
    fn serial_matches_expected() {
        let seq = b"ACGTN";
        let mut out = vec![0i32; seq.len()];
        gather_serial(seq, &lut(), &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, -1]);
    }

    #[test]
    fn parallel_matches_serial() {
        let seq: Vec<u8> = (0..10_000u32).map(|i| b"ACGT"[(i % 4) as usize]).collect();
        let l = lut();
        let mut a = vec![0i32; seq.len()];
        let mut b = vec![0i32; seq.len()];
        gather_serial(&seq, &l, &mut a);
        gather_parallel(&seq, &l, &mut b);
        assert_eq!(a, b);
    }
}
