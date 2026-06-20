//! Codon-stride translation for `AminoAlphabet.translate`. Reads a flat `u8`
//! nucleotide buffer in `codon_size` strides and emits one AA byte per codon.
//! The standard ACGT/k=3 path uses a 64-entry LUT keyed by the same bit-hash
//! as Python `_pack_codon_index`; non-standard alphabets use a linear scan.

/// Crossover threshold (in codons) above which `_translate_bytes` uses the
/// rayon parallel path (`translate_lut_into_par` / `translate_scan_into_par`).
/// Measured on a 10-core Apple M-series Mac: serial ≈ 67 Mcodon/s flat;
/// parallel overtakes at ~10k codons (tie) and reaches 395 Mcodon/s at 333k
/// (5.9x). 20_000 is chosen as a conservative crossover (1.8x win there),
/// safely above the ~10k break-even. WIRED.
pub const TRANSLATE_PARALLEL_THRESHOLD: usize = 20_000;

/// Pack a canonical ACGT codon into a 6-bit LUT index `[0,63]`.
/// Mirrors Python `_pack_codon_index`. Caller guarantees bytes are upper-cased ACGT.
#[inline]
pub fn pack_codon_index(b0: u8, b1: u8, b2: u8) -> usize {
    let n0 = ((b0 >> 1) & 3) as usize;
    let n1 = ((b1 >> 1) & 3) as usize;
    let n2 = ((b2 >> 1) & 3) as usize;
    (n0 << 4) | (n1 << 2) | n2
}

const fn build_is_acgt() -> [bool; 256] {
    let mut t = [false; 256];
    t[b'A' as usize] = true;
    t[b'a' as usize] = true;
    t[b'C' as usize] = true;
    t[b'c' as usize] = true;
    t[b'G' as usize] = true;
    t[b'g' as usize] = true;
    t[b'T' as usize] = true;
    t[b't' as usize] = true;
    t
}
static IS_ACGT: [bool; 256] = build_is_acgt();

/// Translate a single codon using the 64-entry LUT. Non-canonical → `marker`.
/// Uses a 256-entry validity table (non-short-circuit `&`) to avoid branch mispredictions.
/// The `& 0xDF` case fold is now implicit: IS_ACGT accepts both 'A'/'a' etc., and
/// the index computation via `(b >> 1) & 3` produces the same result for upper/lower pairs.
#[inline]
fn translate_codon_lut(codon: &[u8], lut: &[u8], marker: u8) -> u8 {
    let b0 = codon[0];
    let b1 = codon[1];
    let b2 = codon[2];
    // non-short-circuit `&` avoids extra branches; 3 table lookups replace ~12 comparisons.
    if IS_ACGT[b0 as usize] & IS_ACGT[b1 as usize] & IS_ACGT[b2 as usize] {
        let idx = (((b0 >> 1) & 3) as usize) << 4
            | (((b1 >> 1) & 3) as usize) << 2
            | ((b2 >> 1) & 3) as usize;
        lut[idx]
    } else {
        marker
    }
}

/// Translate a single codon by linear scan against `keys`/`values`. No match → `marker`.
#[inline]
fn translate_codon_scan(
    codon: &[u8],
    codon_size: usize,
    keys: &[u8],
    values: &[u8],
    marker: u8,
) -> u8 {
    let n_codons_table = values.len();
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
    hit
}

/// Standard ACGT/k=3 LUT translate (serial). Non-canonical codons → `marker`.
pub fn translate_lut_into(buf: &[u8], codon_size: usize, lut: &[u8], marker: u8, out: &mut [u8]) {
    debug_assert_eq!(codon_size, 3);
    for (o, codon) in out.iter_mut().zip(buf.chunks_exact(codon_size)) {
        *o = translate_codon_lut(codon, lut, marker);
    }
}

/// Generic case-insensitive linear-scan translate for non-standard alphabets (serial).
pub fn translate_scan_into(
    buf: &[u8],
    codon_size: usize,
    keys: &[u8],
    values: &[u8],
    marker: u8,
    out: &mut [u8],
) {
    for (o, codon) in out.iter_mut().zip(buf.chunks_exact(codon_size)) {
        *o = translate_codon_scan(codon, codon_size, keys, values, marker);
    }
}

/// Parallel LUT translate using rayon. Bit-identical to `translate_lut_into`.
/// Uses coarse-grained chunking (8192 codons per task) to amortize rayon scheduling
/// overhead — per-codon parallelism has prohibitive dispatch cost at ~3 bytes/item.
pub fn translate_lut_into_par(
    buf: &[u8],
    codon_size: usize,
    lut: &[u8],
    marker: u8,
    out: &mut [u8],
) {
    use rayon::prelude::*;
    const CHUNK_CODONS: usize = 8192;
    out.par_chunks_mut(CHUNK_CODONS)
        .zip(buf.par_chunks(CHUNK_CODONS * codon_size))
        .for_each(|(out_chunk, buf_chunk)| {
            for (o, codon) in out_chunk.iter_mut().zip(buf_chunk.chunks_exact(codon_size)) {
                *o = translate_codon_lut(codon, lut, marker);
            }
        });
}

/// Parallel scan translate using rayon. Bit-identical to `translate_scan_into`.
pub fn translate_scan_into_par(
    buf: &[u8],
    codon_size: usize,
    keys: &[u8],
    values: &[u8],
    marker: u8,
    out: &mut [u8],
) {
    use rayon::prelude::*;
    out.par_iter_mut()
        .zip(buf.par_chunks_exact(codon_size))
        .for_each(|(o, codon)| *o = translate_codon_scan(codon, codon_size, keys, values, marker));
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

    #[test]
    fn parallel_lut_matches_serial() {
        // Build a ~50k-codon random-ish ACGT buffer
        let codons = 50_000usize;
        let buf: Vec<u8> = (0..codons * 3)
            .map(|i| [b'A', b'C', b'G', b'T'][i % 4])
            .collect();
        let mut out_serial = vec![0u8; codons];
        let mut out_par = vec![0u8; codons];
        translate_lut_into(&buf, 3, &lut(), b'X', &mut out_serial);
        translate_lut_into_par(&buf, 3, &lut(), b'X', &mut out_par);
        assert_eq!(out_serial, out_par);
    }

    #[test]
    fn parallel_scan_matches_serial() {
        let keys = b"ATGTAAAAA";
        let values = b"M*K";
        let base: Vec<u8> = b"ATGTAAAAA".repeat(5000);
        let codons = base.len() / 3;
        let mut out_serial = vec![0u8; codons];
        let mut out_par = vec![0u8; codons];
        translate_scan_into(&base, 3, keys, values, b'X', &mut out_serial);
        translate_scan_into_par(&base, 3, keys, values, b'X', &mut out_par);
        assert_eq!(out_serial, out_par);
    }
}

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
    let parallel = n_codons >= TRANSLATE_PARALLEL_THRESHOLD;
    match lut {
        Some(lut) => {
            let lut = lut.as_slice()?;
            if parallel {
                translate_lut_into_par(buf, codon_size, lut, marker, &mut out);
            } else {
                translate_lut_into(buf, codon_size, lut, marker, &mut out);
            }
        }
        None => {
            let keys = keys.ok_or_else(|| PyValueError::new_err("keys required without lut"))?;
            let values =
                values.ok_or_else(|| PyValueError::new_err("values required without lut"))?;
            let keys = keys.as_slice()?;
            let values = values.as_slice()?;
            if parallel {
                translate_scan_into_par(buf, codon_size, keys, values, marker, &mut out);
            } else {
                translate_scan_into(buf, codon_size, keys, values, marker, &mut out);
            }
        }
    }
    Ok(out.into_pyarray(py))
}

/// First-stop truncation: per sequence, end = index just past the first
/// `stop_char` in `[starts[i], full_ends[i])`, else `full_ends[i]`.
/// Mirrors Python `_nb_find_stop_ends`.
#[pyfunction]
pub fn _translate_stop_ends<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    starts: PyReadonlyArray1<'py, i64>,
    full_ends: PyReadonlyArray1<'py, i64>,
    stop_char: u8,
) -> PyResult<&'py PyArray1<i64>> {
    let data = data.as_slice()?;
    let starts = starts.as_slice()?;
    let full_ends = full_ends.as_slice()?;
    let n = starts.len();
    let mut ends = vec![0i64; n];
    for i in 0..n {
        let mut end = full_ends[i];
        for j in starts[i]..full_ends[i] {
            if data[j as usize] == stop_char {
                end = j + 1;
                break;
            }
        }
        ends[i] = end;
    }
    Ok(ndarray::Array1::from(ends).into_pyarray(py))
}

/// Compact a flat translated AA buffer, dropping codons containing a byte whose
/// upper-cased form is not in `valid_upper`. `offsets` are codon-indexed.
/// Mirrors Python `_nb_drop_unknown_codons`.
#[pyfunction]
pub fn _translate_drop<'py>(
    py: Python<'py>,
    translated: PyReadonlyArray1<'py, u8>,
    codons: PyReadonlyArray2<'py, u8>,
    offsets: PyReadonlyArray1<'py, i64>,
    valid_upper: PyReadonlyArray1<'py, u8>,
) -> PyResult<(&'py PyArray1<u8>, &'py PyArray1<i64>)> {
    let translated = translated.as_slice()?;
    let codons = codons.as_array(); // (n_codons, codon_size)
    let offsets = offsets.as_slice()?;
    let valid = valid_upper.as_slice()?;
    let k = codons.shape()[1];
    let n = offsets.len() - 1;

    let mut out: Vec<u8> = Vec::with_capacity(translated.len());
    let mut new_offsets: Vec<i64> = Vec::with_capacity(n + 1);
    new_offsets.push(0);
    for s in 0..n {
        let start = offsets[s] as usize;
        let end = offsets[s + 1] as usize;
        for c in start..end {
            let mut keep = true;
            for j in 0..k {
                let b = codons[[c, j]] & 0xDF;
                if !valid.iter().any(|&v| v == b) {
                    keep = false;
                    break;
                }
            }
            if keep {
                out.push(translated[c]);
            }
        }
        new_offsets.push(out.len() as i64);
    }
    Ok((
        out.into_pyarray(py),
        ndarray::Array1::from(new_offsets).into_pyarray(py),
    ))
}

/// Decode each one-hot row to a nucleotide byte; all-zero (or no-1) row -> 0 sentinel
/// (a non-canonical byte that downstream validity/translation treats as invalid).
fn decode_ohe_rows(data: ndarray::ArrayView2<u8>, nuc: &[u8]) -> Vec<u8> {
    let total = data.shape()[0];
    let n_nuc = data.shape()[1];
    let mut nuc_buf = vec![0u8; total];
    for r in 0..total {
        let mut found = 0u8;
        for c in 0..n_nuc {
            if data[[r, c]] == 1 {
                found = nuc[c];
                break;
            }
        }
        nuc_buf[r] = found;
    }
    nuc_buf
}

/// Fused OHE->codon->AA->OHE translation for OHE-ragged input, avoiding any
/// round-trip through the Numba-backed ohe/decode_ohe. `data` is (total, n_nuc);
/// returns (total/codon_size, n_aa).
#[pyfunction]
#[pyo3(signature = (data, nuc_bytes, codon_size, lut=None, keys=None, values=None, aa_bytes=None, marker=88))]
pub fn _translate_ohe<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, u8>,
    nuc_bytes: PyReadonlyArray1<'py, u8>,
    codon_size: usize,
    lut: Option<PyReadonlyArray1<'py, u8>>,
    keys: Option<PyReadonlyArray1<'py, u8>>,
    values: Option<PyReadonlyArray1<'py, u8>>,
    aa_bytes: Option<PyReadonlyArray1<'py, u8>>,
    marker: u8,
) -> PyResult<&'py PyArray2<u8>> {
    let data = data.as_array(); // (total, n_nuc)
    let nuc = nuc_bytes.as_slice()?;
    let aa_bytes = aa_bytes.ok_or_else(|| PyValueError::new_err("aa_bytes required"))?;
    let aa_bytes = aa_bytes.as_slice()?;
    let total = data.shape()[0];
    let n_aa = aa_bytes.len();
    if codon_size == 0 || total % codon_size != 0 {
        return Err(PyValueError::new_err(
            "total rows must be a positive multiple of codon_size",
        ));
    }

    // 1) Decode each OHE row to a nucleotide byte (all-zero row -> 0 sentinel).
    let nuc_buf = decode_ohe_rows(data, nuc);

    // 2) Translate all codons to AA bytes.
    // NOTE: serial path intentional — surrounding decode/compaction loops are serial;
    // parallel translate could be wired here in a future pass if OHE becomes a bottleneck.
    let n_codons = total / codon_size;
    let mut aa = vec![0u8; n_codons];
    match lut {
        Some(lut) => translate_lut_into(&nuc_buf, codon_size, lut.as_slice()?, marker, &mut aa),
        None => {
            let keys = keys.ok_or_else(|| PyValueError::new_err("keys required without lut"))?;
            let values =
                values.ok_or_else(|| PyValueError::new_err("values required without lut"))?;
            translate_scan_into(
                &nuc_buf,
                codon_size,
                keys.as_slice()?,
                values.as_slice()?,
                marker,
                &mut aa,
            );
        }
    }

    // 3) Re-encode AA bytes one-hot against aa_bytes (no match -> all-zero row).
    let mut out = ndarray::Array2::<u8>::zeros((n_codons, n_aa));
    for i in 0..n_codons {
        for (j, &ab) in aa_bytes.iter().enumerate() {
            if ab == aa[i] {
                out[[i, j]] = 1;
                break;
            }
        }
    }
    Ok(out.into_pyarray(py))
}

/// Fused OHE->codon->AA->OHE with unknown="drop" compaction. Drops any codon
/// containing a nucleotide row whose decoded byte (`& 0xDF`) is not in
/// `valid_upper` (all-zero/unknown rows decode to a sentinel that is never
/// valid). Returns the compacted (n_kept, n_aa) OHE array and codon-indexed new
/// offsets. `offsets` are nucleotide-row offsets (multiples of codon_size).
#[pyfunction]
#[pyo3(signature = (data, nuc_bytes, codon_size, valid_upper, offsets, lut=None, keys=None, values=None, aa_bytes=None, marker=88))]
pub fn _translate_ohe_drop<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, u8>,
    nuc_bytes: PyReadonlyArray1<'py, u8>,
    codon_size: usize,
    valid_upper: PyReadonlyArray1<'py, u8>,
    offsets: PyReadonlyArray1<'py, i64>,
    lut: Option<PyReadonlyArray1<'py, u8>>,
    keys: Option<PyReadonlyArray1<'py, u8>>,
    values: Option<PyReadonlyArray1<'py, u8>>,
    aa_bytes: Option<PyReadonlyArray1<'py, u8>>,
    marker: u8,
) -> PyResult<(&'py PyArray2<u8>, &'py PyArray1<i64>)> {
    let data = data.as_array();
    let nuc = nuc_bytes.as_slice()?;
    let valid = valid_upper.as_slice()?;
    let offsets = offsets.as_slice()?;
    let aa_bytes = aa_bytes.ok_or_else(|| PyValueError::new_err("aa_bytes required"))?;
    let aa_bytes = aa_bytes.as_slice()?;
    let total = data.shape()[0];
    let n_aa = aa_bytes.len();
    if codon_size == 0 || total % codon_size != 0 {
        return Err(PyValueError::new_err(
            "total rows must be a positive multiple of codon_size",
        ));
    }

    // 1) Decode rows to nucleotide bytes.
    let nuc_buf = decode_ohe_rows(data, nuc);

    // 2) Translate all codons to AA bytes.
    // NOTE: serial path intentional — surrounding decode/compaction loops are serial;
    // parallel translate could be wired here in a future pass if OHE becomes a bottleneck.
    let n_codons = total / codon_size;
    let mut aa = vec![0u8; n_codons];
    match lut {
        Some(lut) => translate_lut_into(&nuc_buf, codon_size, lut.as_slice()?, marker, &mut aa),
        None => {
            let keys = keys.ok_or_else(|| PyValueError::new_err("keys required without lut"))?;
            let values =
                values.ok_or_else(|| PyValueError::new_err("values required without lut"))?;
            translate_scan_into(
                &nuc_buf,
                codon_size,
                keys.as_slice()?,
                values.as_slice()?,
                marker,
                &mut aa,
            );
        }
    }

    // 3) Per sequence, keep codons whose every nucleotide row is valid; collect
    //    kept AA bytes; build codon-indexed new offsets.
    let n_seq = offsets.len() - 1;
    let mut kept_aa: Vec<u8> = Vec::new();
    let mut new_offsets: Vec<i64> = Vec::with_capacity(n_seq + 1);
    new_offsets.push(0);
    for s in 0..n_seq {
        let codon_start = (offsets[s] as usize) / codon_size;
        let codon_end = (offsets[s + 1] as usize) / codon_size;
        for c in codon_start..codon_end {
            let mut keep = true;
            for j in 0..codon_size {
                let b = nuc_buf[c * codon_size + j] & 0xDF;
                if !valid.iter().any(|&v| v == b) {
                    keep = false;
                    break;
                }
            }
            if keep {
                kept_aa.push(aa[c]);
            }
        }
        new_offsets.push(kept_aa.len() as i64);
    }

    // 4) One-hot encode kept AA bytes against aa_bytes (no match -> all-zero row).
    let n_kept = kept_aa.len();
    let mut out = ndarray::Array2::<u8>::zeros((n_kept, n_aa));
    for (i, &ab_byte) in kept_aa.iter().enumerate() {
        for (j, &ab) in aa_bytes.iter().enumerate() {
            if ab == ab_byte {
                out[[i, j]] = 1;
                break;
            }
        }
    }
    Ok((
        out.into_pyarray(py),
        ndarray::Array1::from(new_offsets).into_pyarray(py),
    ))
}
