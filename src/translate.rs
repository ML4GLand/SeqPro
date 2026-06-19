//! Codon-stride translation for `AminoAlphabet.translate`. Reads a flat `u8`
//! nucleotide buffer in `codon_size` strides and emits one AA byte per codon.
//! The standard ACGT/k=3 path uses a 64-entry LUT keyed by the same bit-hash
//! as Python `_pack_codon_index`; non-standard alphabets use a linear scan.

pub const TRANSLATE_PARALLEL_THRESHOLD: usize = 40_000;

/// Pack a canonical ACGT codon into a 6-bit LUT index `[0,63]`.
/// Mirrors Python `_pack_codon_index`. Caller guarantees bytes are upper-cased ACGT.
#[inline]
pub fn pack_codon_index(b0: u8, b1: u8, b2: u8) -> usize {
    let n0 = ((b0 >> 1) & 3) as usize;
    let n1 = ((b1 >> 1) & 3) as usize;
    let n2 = ((b2 >> 1) & 3) as usize;
    (n0 << 4) | (n1 << 2) | n2
}

#[inline]
fn is_acgt(b: u8) -> bool {
    b == b'A' || b == b'C' || b == b'G' || b == b'T'
}

/// Standard ACGT/k=3 LUT translate. Non-canonical codons → `marker`.
pub fn translate_lut_into(buf: &[u8], codon_size: usize, lut: &[u8], marker: u8, out: &mut [u8]) {
    debug_assert_eq!(codon_size, 3);
    for (o, codon) in out.iter_mut().zip(buf.chunks_exact(codon_size)) {
        let b0 = codon[0] & 0xDF;
        let b1 = codon[1] & 0xDF;
        let b2 = codon[2] & 0xDF;
        *o = if is_acgt(b0) && is_acgt(b1) && is_acgt(b2) {
            lut[pack_codon_index(b0, b1, b2)]
        } else {
            marker
        };
    }
}

/// Generic case-insensitive linear-scan translate for non-standard alphabets.
pub fn translate_scan_into(
    buf: &[u8],
    codon_size: usize,
    keys: &[u8],
    values: &[u8],
    marker: u8,
    out: &mut [u8],
) {
    let n_codons_table = values.len();
    for (o, codon) in out.iter_mut().zip(buf.chunks_exact(codon_size)) {
        let mut hit = marker;
        'keys: for i in 0..n_codons_table {
            let key = &keys[i * codon_size..(i + 1) * codon_size];
            for j in 0..codon_size {
                if (codon[j] & 0xDF) != (key[j] & 0xDF) {
                    continue 'keys;
                }
            }
            hit = values[i];
            break;
        }
        *o = hit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal standard genetic code subset for tests: ATG->M, TAA->*, AAA->K.
    fn lut() -> Vec<u8> {
        let mut l = vec![b'X'; 64];
        l[pack_codon_index(b'A', b'T', b'G')] = b'M';
        l[pack_codon_index(b'T', b'A', b'A')] = b'*';
        l[pack_codon_index(b'A', b'A', b'A')] = b'K';
        l
    }

    #[test]
    fn lut_translates_and_case_folds() {
        let buf = b"ATGtaaAAANNN"; // includes lowercase + non-canonical NNN
        let mut out = vec![0u8; buf.len() / 3];
        translate_lut_into(buf, 3, &lut(), b'X', &mut out);
        assert_eq!(out, b"M*KX");
    }

    #[test]
    fn scan_matches_lut_on_standard() {
        let buf = b"ATGTAAAAA";
        let keys = b"ATGTAAAAA"; // 3 codons
        let values = b"M*K";
        let mut out = vec![0u8; 3];
        translate_scan_into(buf, 3, keys, values, b'X', &mut out);
        assert_eq!(out, b"M*K");
    }
}

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Translate a flat nucleotide buffer (length multiple of `codon_size`) to a
/// flat AA buffer of length `buf.len()/codon_size`.
#[pyfunction]
#[pyo3(signature = (buf, codon_size, lut=None, keys=None, values=None, marker=88))]
pub fn _translate_bytes<'py>(
    py: Python<'py>,
    buf: PyReadonlyArray1<'py, u8>,
    codon_size: usize,
    lut: Option<PyReadonlyArray1<'py, u8>>,
    keys: Option<PyReadonlyArray1<'py, u8>>,
    values: Option<PyReadonlyArray1<'py, u8>>,
    marker: u8,
) -> PyResult<&'py PyArray1<u8>> {
    let buf = buf.as_slice()?;
    if codon_size == 0 || buf.len() % codon_size != 0 {
        return Err(PyValueError::new_err(
            "buffer length must be a positive multiple of codon_size",
        ));
    }
    let n_codons = buf.len() / codon_size;
    let mut out = vec![0u8; n_codons];
    match lut {
        Some(lut) => translate_lut_into(buf, codon_size, lut.as_slice()?, marker, &mut out),
        None => {
            let keys = keys.ok_or_else(|| PyValueError::new_err("keys required without lut"))?;
            let values =
                values.ok_or_else(|| PyValueError::new_err("values required without lut"))?;
            translate_scan_into(
                buf,
                codon_size,
                keys.as_slice()?,
                values.as_slice()?,
                marker,
                &mut out,
            );
        }
    }
    Ok(out.into_pyarray(py))
}
