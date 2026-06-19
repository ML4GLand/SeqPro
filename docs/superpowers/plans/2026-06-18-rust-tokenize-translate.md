# Rust `tokenize` / `translate` Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `seqpro.tokenize` and `AminoAlphabet.translate` from Numba to the existing Rust/PyO3 extension at full feature parity, with a benchmark harness to decide keep-or-revert.

**Architecture:** Thin Rust kernels (mirroring the existing `kshuffle` pattern): Python keeps all orchestration — `cast_seqs`, `check_axes`, ragged `to_packed()`/`from_offsets`, and all LUT/codon-LUT construction (NumPy, not Numba) — and calls new Rust functions `_tokenize` / `_translate*` for the inner compute. Parallelism uses rayon (already a dependency); per-kernel element-count thresholds live in Rust. Numba kernels stay in `_numba.py` during the experiment for a cheap revert.

**Tech Stack:** Rust + PyO3 0.20 (`abi3-py39`) + `numpy` 0.20 + `ndarray` 0.15 + rayon; maturin build; Python 3.10+; NumPy; pytest + Hypothesis + pytest-benchmark; pixi env management.

## Global Constraints

- Public Python signatures of `tokenize` and `translate` MUST NOT change (verbatim parity, including overloads).
- `out=` for `tokenize` must have dtype `int32`, else `TypeError` (message: `out must have dtype int32, got {dtype}.`).
- `tokenize` with `parallel=True` and a non-C-contiguous `out` MUST raise `ValueError`.
- Canonical in-memory format: `|S1` bytes for strings, `uint8` for OHE; token IDs are `int32`.
- Case-insensitivity in `translate` is unconditional (upper-case via `& 0xDF`).
- `unknown="drop"` always returns a `Ragged`, even for dense input.
- Build the Rust extension with `maturin develop` after any `src/` change; run tests with `pixi run test`.
- Conventional commits (`feat:`, `fix:`, `test:`, `bench:`, `docs:`, `chore:`).
- Leave all Numba kernels in `_numba.py` in place until the keep decision (do not delete in this plan).
- The pack hash (`_pack_codon_index`) and the 64-entry codon LUT stay built in Python and are passed into Rust — single source of truth.

---

## File Structure

- `src/tokenize.rs` — **create.** Flat-buffer LUT gather (serial + rayon), strided-out handling, threshold constant. Exposes `_tokenize`.
- `src/translate.rs` — **create.** Codon-stride translate (standard LUT + generic scan), drop-compaction, stop-truncation, OHE↔AA fused path, threshold constant. Exposes `_translate_bytes`, `_translate_drop`, `_translate_stop_ends`, `_translate_ohe`.
- `src/lib.rs` — **modify.** Declare `mod tokenize; mod translate;` and register the new `#[pyfunction]`s in the `#[pymodule]`.
- `python/seqpro/_encoders.py` — **modify.** `tokenize` calls `_tokenize`; remove the `lut_gather`/`np.take` branch.
- `python/seqpro/alphabets/_alphabets.py` — **modify.** `AminoAlphabet.translate` calls the Rust entry points instead of the `gufunc_*`/`_nb_*` kernels.
- `tests/test_tokenize_rust.py` — **create.** Differential + edge-case tests for `tokenize`.
- `tests/test_translate_rust.py` — **create.** Differential + edge-case tests for `translate`.
- `benches/bench_tokenize_translate.py` — **create.** pytest-benchmark sweep.
- `pyproject.toml` — **modify.** Add `pytest-benchmark` to the dev dependency group.

---

## Task 1: Scaffold `src/tokenize.rs` gather kernel (Rust unit-tested)

**Files:**
- Create: `src/tokenize.rs`
- Modify: `src/lib.rs:1-15`

**Interfaces:**
- Produces (Rust, internal): `pub const TOKENIZE_PARALLEL_THRESHOLD: usize`; `fn gather_serial(seq: &[u8], lut: &[i32], out: &mut [i32])`; `fn gather_parallel(seq: &[u8], lut: &[i32], out: &mut [i32])`.

- [ ] **Step 1: Write the failing Rust unit tests**

Create `src/tokenize.rs`:

```rust
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
```

Add to `src/lib.rs` after the existing `pub mod` lines (line 2):

```rust
pub mod tokenize;
```

- [ ] **Step 2: Run the tests to verify they pass after implementation is in place**

Run: `cargo test --lib tokenize`
Expected: `serial_matches_expected` and `parallel_matches_serial` PASS. (Implementation and tests are written together here because the kernel is trivial and the test is the spec.)

- [ ] **Step 3: Commit**

```bash
git add src/tokenize.rs src/lib.rs
git commit -m "feat(rust): add tokenize LUT gather kernel"
```

---

## Task 2: Expose `_tokenize` to Python and wire the dense path

**Files:**
- Modify: `src/tokenize.rs`
- Modify: `src/lib.rs:11-15`
- Modify: `python/seqpro/_encoders.py:249-328`
- Create: `tests/test_tokenize_rust.py`

**Interfaces:**
- Consumes: `gather_serial`, `gather_parallel`, `TOKENIZE_PARALLEL_THRESHOLD` (Task 1).
- Produces (Python-visible): `seqpro.seqpro._tokenize(seqs: ndarray[u8], lut: ndarray[i32, 1d], out: ndarray[i32] | None, parallel: bool | None) -> ndarray[i32]`. Output shape equals `seqs.shape`. When `out` is given it is filled and returned. `parallel=None` → threshold heuristic; `True`/`False` → forced. A non-C-contiguous `out` with `parallel=True` raises `ValueError` from Rust as a backstop (Python raises first).

- [ ] **Step 1: Write the failing Python differential test**

Create `tests/test_tokenize_rust.py`:

```python
import numpy as np
import pytest

import seqpro as sp

TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
UNK = 7


def _ref_tokenize(seqs, token_map, unk):
    """Pure-NumPy reference (LUT + np.take), independent of the impl under test."""
    keys = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    vals = np.array(list(token_map.values()), dtype=np.int32)
    lut = np.full(256, np.int32(unk), dtype=np.int32)
    lut[keys] = vals
    arr = sp.cast_seqs(seqs)
    return np.take(lut, arr.view(np.uint8))


@pytest.mark.parametrize("parallel", [None, False, True])
@pytest.mark.parametrize("n", [0, 1, 5, 100, 50_000])
def test_tokenize_matches_reference_1d(n, parallel):
    rng = np.random.default_rng(0)
    seqs = rng.choice(np.frombuffer(b"ACGTN", "S1"), size=n)
    got = sp.tokenize(seqs, TOKEN_MAP, UNK, parallel=parallel)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))


def test_tokenize_multidim_preserves_shape():
    rng = np.random.default_rng(1)
    seqs = rng.choice(np.frombuffer(b"ACGT", "S1"), size=(4, 8, 3))
    got = sp.tokenize(seqs, TOKEN_MAP, UNK)
    assert got.shape == (4, 8, 3)
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_tokenize_rust.py -q`
Expected: FAIL (Rust `_tokenize` not yet exposed / not yet called by `sp.tokenize`).

- [ ] **Step 3: Add the `_tokenize` pyfunction**

Append to `src/tokenize.rs`:

```rust
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
```

Register in `src/lib.rs` `#[pymodule]` (after the `_k_shuffle` line):

```rust
m.add_function(wrap_pyfunction!(tokenize::_tokenize, m)?)?;
```

- [ ] **Step 4: Rewrite the dense branch of `tokenize` in `_encoders.py`**

Replace the dense (non-Ragged) tail of `tokenize` (currently `python/seqpro/_encoders.py:309-328`) with:

```python
    _seqs = cast_seqs(seqs)
    u8 = _seqs.view(np.uint8)
    # A strided out= cannot be written through by the parallel kernel; reject
    # the impossible combination early with a clear message (Rust also guards).
    out_blocks_parallel = out is not None and not out.flags.c_contiguous
    if parallel is True and out_blocks_parallel:
        raise ValueError(
            "parallel=True requires a C-contiguous out array, got a "
            "non-contiguous out. Use parallel=None/False or a contiguous out."
        )
    return _tokenize(u8, lut, out, parallel)
```

Add the import near the top of `_encoders.py` (after the existing imports, around line 16):

```python
from .seqpro import _tokenize  # type: ignore[missing-import]  # compiled Rust extension
```

Leave the `lut`/`keys`/`vals` construction above (lines 289-292) unchanged. Remove the now-dead module constant usage only inside the dense branch; keep `_TOKENIZE_PARALLEL_THRESHOLD` defined (still used by the Ragged branch until Task 3).

- [ ] **Step 5: Build and run the tests**

Run: `maturin develop && pixi run -e dev pytest tests/test_tokenize_rust.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/tokenize.rs src/lib.rs python/seqpro/_encoders.py tests/test_tokenize_rust.py
git commit -m "feat(tokenize): route dense path through Rust _tokenize"
```

---

## Task 3: Route the Ragged `tokenize` path through Rust + edge tests

**Files:**
- Modify: `python/seqpro/_encoders.py:294-307`
- Modify: `tests/test_tokenize_rust.py`

**Interfaces:**
- Consumes: `_tokenize` (Task 2); `Ragged.to_packed()`, `Ragged.from_offsets(data, shape, offsets)`, `.data`, `.offsets`, `.lengths` (existing).

- [ ] **Step 1: Write the failing Ragged + edge tests**

Append to `tests/test_tokenize_rust.py`:

```python
from seqpro.rag import Ragged


def test_tokenize_ragged_matches_reference():
    data = np.frombuffer(b"ACGTACG", "S1")
    offsets = np.array([0, 3, 7], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.tokenize(rag, TOKEN_MAP, UNK)
    np.testing.assert_array_equal(got.data, _ref_tokenize(data, TOKEN_MAP, UNK))
    np.testing.assert_array_equal(got.offsets, offsets)


def test_tokenize_out_dtype_typeerror():
    seqs = np.frombuffer(b"ACGT", "S1")
    bad = np.empty(4, dtype=np.int64)
    with pytest.raises(TypeError, match="int32"):
        sp.tokenize(seqs, TOKEN_MAP, UNK, out=bad)


def test_tokenize_parallel_true_strided_out_valueerror():
    seqs = np.frombuffer(b"ACGTACGT", "S1")
    out = np.empty(16, dtype=np.int32)[::2]  # non-contiguous
    assert not out.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        sp.tokenize(seqs, TOKEN_MAP, UNK, out=out, parallel=True)


def test_tokenize_out_strided_serial_ok():
    seqs = np.frombuffer(b"ACGTACGT", "S1")
    out = np.empty(16, dtype=np.int32)[::2]
    got = sp.tokenize(seqs, TOKEN_MAP, UNK, out=out, parallel=False)
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))
```

- [ ] **Step 2: Run to verify the Ragged test fails**

Run: `pixi run -e dev pytest tests/test_tokenize_rust.py -q`
Expected: `test_tokenize_ragged_matches_reference` FAILs (Ragged branch still uses Numba `lut_gather`/`np.take`); the `out`/`parallel` tests should already PASS from Task 2.

- [ ] **Step 3: Rewrite the Ragged branch**

Replace the Ragged branch body (currently `python/seqpro/_encoders.py:294-307`) with:

```python
    if isinstance(seqs, Ragged):
        seqs = seqs.to_packed()
        n = len(seqs.lengths.ravel())
        trailing = seqs.data.shape[1:]
        u8 = seqs.data.view(np.uint8)
        flat = _tokenize(u8, lut, None, parallel)
        return Ragged.from_offsets(flat, (n, None, *trailing), seqs.offsets)
```

Now `_TOKENIZE_PARALLEL_THRESHOLD` is unused in Python; delete its definition (`_encoders.py:18-21`) and remove the `lut_gather` import from the `._numba` import block at the top of `_encoders.py` (leave the other Numba imports used by `decode_tokens`).

- [ ] **Step 4: Build and run the full tokenize suite + existing tests**

Run: `maturin develop && pixi run -e dev pytest tests/test_tokenize_rust.py tests/test_tokenize.py -q`
(Substitute the actual existing tokenize test path if named differently — discover with `ls tests | grep -i token`.)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_encoders.py tests/test_tokenize_rust.py
git commit -m "feat(tokenize): route ragged path through Rust, drop Numba gather"
```

---

## Task 4: Scaffold `src/translate.rs` — standard LUT + generic scan kernels (Rust unit-tested)

**Files:**
- Create: `src/translate.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Produces (Rust, internal): `pub const TRANSLATE_PARALLEL_THRESHOLD: usize`; `fn pack_codon_index(b0:u8,b1:u8,b2:u8)->usize`; `fn translate_lut_into(buf:&[u8], codon_size:usize, lut:&[u8], marker:u8, out:&mut [u8])`; `fn translate_scan_into(buf:&[u8], codon_size:usize, keys:&[u8], values:&[u8], marker:u8, out:&mut [u8])`.

Notes: `buf.len()` is a multiple of `codon_size`; `out.len() == buf.len()/codon_size`. `keys` is the flattened `(n_codons, codon_size)` key table; `values[i]` is the AA byte for `keys[i*codon_size .. (i+1)*codon_size]`. Matching is case-insensitive (`& 0xDF`). `pack_codon_index` mirrors Python `_pack_codon_index`: `((b>>1)&3)` per byte, packed `(n0<<4)|(n1<<2)|n2`.

- [ ] **Step 1: Write `src/translate.rs` with unit tests**

```rust
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
```

Add `pub mod translate;` to `src/lib.rs`.

- [ ] **Step 2: Run the unit tests**

Run: `cargo test --lib translate`
Expected: `lut_translates_and_case_folds`, `scan_matches_lut_on_standard` PASS.

- [ ] **Step 3: Commit**

```bash
git add src/translate.rs src/lib.rs
git commit -m "feat(rust): add codon-stride translate kernels"
```

---

## Task 5: Expose `_translate_bytes` and wire the dense `pad` path

**Files:**
- Modify: `src/translate.rs`
- Modify: `src/lib.rs`
- Modify: `python/seqpro/alphabets/_alphabets.py:552-570`
- Create: `tests/test_translate_rust.py`

**Interfaces:**
- Consumes: `translate_lut_into`, `translate_scan_into`, `TRANSLATE_PARALLEL_THRESHOLD`.
- Produces (Python-visible): `seqpro.seqpro._translate_bytes(buf: ndarray[u8, 1d], codon_size: int, lut: ndarray[u8,1d] | None, keys: ndarray[u8,1d] | None, values: ndarray[u8,1d] | None, marker: int) -> ndarray[u8, 1d]`. Returns one AA byte per codon. Exactly one of (`lut`) or (`keys`+`values`) is non-None. For this task only the contiguous 1-D nucleotide axis is handled (trailing axes handled by the caller flattening; see Step 4).

- [ ] **Step 1: Write the failing dense differential test**

Create `tests/test_translate_rust.py`:

```python
import numpy as np
import pytest

import seqpro as sp
from seqpro import AA  # standard AminoAlphabet


def _bio_like_translate(seq_str: str) -> str:
    """Reference using AA.codon_to_aa directly (independent of the impl)."""
    out = []
    for i in range(0, len(seq_str), 3):
        out.append(AA.codon_to_aa.get(seq_str[i : i + 3], "X"))
    return "".join(out)


@pytest.mark.parametrize(
    "seq",
    ["ATGAAATAA", "atgaaataa", "ATGNNNTAA", ""],
)
def test_translate_dense_pad_matches_reference(seq):
    arr = np.frombuffer(seq.encode("ascii"), "S1")
    got = sp.AA.translate(arr)
    exp = _bio_like_translate(seq.upper())
    assert got.tobytes().decode() == exp
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/test_translate_rust.py -q`
Expected: FAIL (`_translate_bytes` not exposed; `translate` still uses gufuncs — note: it may *pass* if the existing Numba path is still wired. That is expected; the test asserts behavior, not impl. Proceed to wire Rust and keep the test green.)

- [ ] **Step 3: Add `_translate_bytes` pyfunction**

Append to `src/translate.rs`:

```rust
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
```

Register in `src/lib.rs`: `m.add_function(wrap_pyfunction!(translate::_translate_bytes, m)?)?;`

- [ ] **Step 4: Wire the dense `pad` branch of `translate`**

In `python/seqpro/alphabets/_alphabets.py`, add the import near the top (after line 19):

```python
from ..seqpro import _translate_bytes  # type: ignore[missing-import]  # compiled Rust extension
```

Add a private helper on `AminoAlphabet` (place after `_check_ohe_rows`):

```python
    def _rust_translate_flat(
        self, nuc_u8: NDArray[np.uint8], marker_byte: np.uint8
    ) -> NDArray[np.uint8]:
        """Translate a contiguous 1-D nucleotide u8 buffer to a flat AA u8 buffer.

        Dispatches to the Rust LUT path for the standard ACGT/k=3 alphabet and
        the Rust linear-scan path otherwise. One AA byte per codon.
        """
        codon_size = self.codon_array.shape[-1]
        flat = np.ascontiguousarray(nuc_u8).reshape(-1)
        if self.codon_lut is not None:
            return _translate_bytes(
                flat, codon_size, self.codon_lut, None, None, int(marker_byte)
            )
        keys = np.ascontiguousarray(self.codon_array.view(np.uint8)).reshape(-1)
        values = np.ascontiguousarray(self.aa_array.view(np.uint8)).reshape(-1)
        return _translate_bytes(
            flat, codon_size, None, keys, values, int(marker_byte)
        )
```

Replace the dense `pad` branch (currently `_alphabets.py:552-570`, the block starting `# pad: shape-preserving dense output`) with:

```python
            # pad: shape-preserving dense output. Move length axis last so codons
            # are contiguous per row, translate the flat buffer, reshape back.
            norm = np.moveaxis(seqs, length_axis, -1)
            lead_shape = norm.shape[:-1]
            seq_len = norm.shape[-1]
            n_codons = seq_len // codon_size
            flat = np.ascontiguousarray(norm.reshape(-1)).view(np.uint8)
            aa = self._rust_translate_flat(flat, marker_byte)  # (total_codons,)
            aa = aa.view("S1").reshape(*lead_shape, n_codons)
            return np.moveaxis(aa, -1, length_axis)
```

(Per-row contiguity holds because each row's nucleotides are contiguous after `moveaxis(...-1)` + `reshape(-1)`, and codons never straddle rows since `seq_len % codon_size == 0`.)

- [ ] **Step 5: Build and run**

Run: `maturin develop && pixi run -e dev pytest tests/test_translate_rust.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/translate.rs src/lib.rs python/seqpro/alphabets/_alphabets.py tests/test_translate_rust.py
git commit -m "feat(translate): route dense pad path through Rust"
```

---

## Task 6: Route the Ragged bytes `pad` path through Rust

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py:603-635`
- Modify: `tests/test_translate_rust.py`

**Interfaces:**
- Consumes: `_rust_translate_flat` (Task 5); `Ragged.to_packed`, `.data`, `.offsets`, `.lengths`, `Ragged.from_offsets`.

- [ ] **Step 1: Write the failing Ragged test**

Append to `tests/test_translate_rust.py`:

```python
from seqpro.rag import Ragged


def test_translate_ragged_bytes_pad():
    # Two sequences, lengths 9 and 6 (both divisible by 3).
    data = np.frombuffer(b"ATGAAATAA" b"AAATAA", "S1")
    offsets = np.array([0, 9, 15], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.AA.translate(rag)
    assert got.data.tobytes().decode() == _bio_like_translate("ATGAAATAAAAATAA")
    np.testing.assert_array_equal(got.offsets, offsets // 3)
```

- [ ] **Step 2: Run to verify behavior**

Run: `pixi run -e dev pytest tests/test_translate_rust.py::test_translate_ragged_bytes_pad -q`
Expected: PASS if the existing Numba ragged path is still wired (asserts behavior). The point of this task is to swap the impl while keeping it green.

- [ ] **Step 3: Replace the flat-buffer translate in the Ragged branch**

In the Ragged branch of `translate` (`_alphabets.py:603-635`), replace the `if total > 0:` block that builds `codons`/`translated_flat` via `sliding_window_view` + `gufunc_translate*` with the Rust call. The bytes case (`not is_ohe`, no trailing) becomes:

```python
        total = nuc_bytes_flat.shape[0]
        trailing = nuc_bytes_flat.shape[1:]
        if total > 0 and not trailing:
            translated_flat = self._rust_translate_flat(
                nuc_bytes_flat.view(np.uint8), marker_byte
            ).view("S1")
        elif total > 0 and trailing:
            # Multi-track bytes: translate each trailing column via the flat
            # codon stride. Move codon axis contiguous per column.
            n_codons = total // codon_size
            cols = int(np.prod(trailing))
            buf = np.ascontiguousarray(
                np.moveaxis(nuc_bytes_flat.reshape(total, cols), 0, -1)
            ).reshape(-1)  # (cols*total,)
            aa = self._rust_translate_flat(buf.view(np.uint8), marker_byte)
            aa = aa.view("S1").reshape(cols, n_codons)
            translated_flat = np.ascontiguousarray(
                np.moveaxis(aa, 0, -1)
            ).reshape(n_codons, *trailing)
        else:
            translated_flat = np.empty((0, *trailing), dtype="S1")
        codons_u1 = (
            np.ascontiguousarray(nuc_bytes_flat.view(np.uint8)).reshape(
                total // codon_size if total else 0, codon_size
            )
            if not trailing
            else np.empty((0, codon_size), dtype=np.uint8)
        )
```

Keep the existing `new_offsets = offsets // codon_size` line and everything after it (drop / truncate_stop / OHE re-encode) unchanged for now — later tasks replace those. The `codons_u1` recomputation above preserves the variable the drop branch expects.

> Note for the implementer: the OHE branch and the `is_drop` / `truncate_stop` branches still call the Numba `_nb_*` kernels at this point — that is intentional and removed in Tasks 7–9. Confirm `gufunc_translate*` is no longer referenced in the bytes path.

- [ ] **Step 4: Build and run translate + existing translate tests**

Run: `maturin develop && pixi run -e dev pytest tests/test_translate_rust.py tests/test_translate.py -q`
(Discover the existing test path: `ls tests | grep -i translate`.)
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate_rust.py
git commit -m "feat(translate): route ragged bytes pad path through Rust"
```

---

## Task 7: `unknown="drop"` compaction in Rust + wire dense and ragged drop

**Files:**
- Modify: `src/translate.rs`
- Modify: `src/lib.rs`
- Modify: `python/seqpro/alphabets/_alphabets.py` (dense drop block ~506-550; ragged drop block ~637-655)
- Modify: `tests/test_translate_rust.py`

**Interfaces:**
- Produces (Python-visible): `_translate_drop(translated: ndarray[u8,1d], codons: ndarray[u8,2d (n_codons,codon_size)], offsets: ndarray[i64,1d], valid_upper: ndarray[u8,1d]) -> (ndarray[u8,1d], ndarray[i64,1d])`. Drops any codon containing a byte whose `& 0xDF` is not in `valid_upper`; returns compacted AA bytes and new monotonic offsets. Mirrors `_nb_drop_unknown_codons`.

- [ ] **Step 1: Write the failing drop test**

Append to `tests/test_translate_rust.py`:

```python
def test_translate_dense_drop_removes_noncanonical():
    # ATG (M), NNN (drop), TAA (*). Drop returns a Ragged even for dense input.
    arr = np.frombuffer(b"ATGNNNTAA", "S1")
    got = sp.AA.translate(arr, unknown="drop")
    assert got.data.tobytes().decode() == "M*"
    np.testing.assert_array_equal(got.offsets, np.array([0, 2], dtype=np.int64))
```

- [ ] **Step 2: Run to verify it stays green (behavior) — then swap impl**

Run: `pixi run -e dev pytest tests/test_translate_rust.py::test_translate_dense_drop_removes_noncanonical -q`
Expected: PASS via the existing Numba `_nb_drop_unknown_codons`. Swap to Rust while keeping it green.

- [ ] **Step 3: Add `_translate_drop` pyfunction**

Append to `src/translate.rs`:

```rust
use numpy::PyReadonlyArray2;

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
```

Register: `m.add_function(wrap_pyfunction!(translate::_translate_drop, m)?)?;`

- [ ] **Step 4: Swap both drop call sites to Rust**

In the dense `is_drop` branch (`_alphabets.py` ~506-550) and the ragged `is_drop` branch (~637-655), replace the `_nb_drop_unknown_codons(...)` call with:

```python
                out_u1, new_offsets = _translate_drop(
                    translated_flat,  # u8, 1-D
                    np.ascontiguousarray(codons_flat),  # (n_codons, codon_size) u8
                    offsets.astype(np.int64),
                    self._valid_upper_bytes,
                )
```

Adapt variable names to each branch (`offsets` is `offsets`/`new_offsets` respectively; `codons_flat` vs `codons_u1`). Ensure `translated_flat` is the flat u8 AA buffer (use `_rust_translate_flat` to produce it instead of the gufunc). Add the import: `from ..seqpro import _translate_drop` to the top-of-file extension import.

- [ ] **Step 5: Build and run**

Run: `maturin develop && pixi run -e dev pytest tests/test_translate_rust.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/translate.rs src/lib.rs python/seqpro/alphabets/_alphabets.py tests/test_translate_rust.py
git commit -m "feat(translate): route unknown=drop compaction through Rust"
```

---

## Task 8: `truncate_stop` end-finding in Rust + wire

**Files:**
- Modify: `src/translate.rs`
- Modify: `src/lib.rs`
- Modify: `python/seqpro/alphabets/_alphabets.py:657-668`
- Modify: `tests/test_translate_rust.py`

**Interfaces:**
- Produces (Python-visible): `_translate_stop_ends(data: ndarray[u8,1d], starts: ndarray[i64,1d], full_ends: ndarray[i64,1d], stop_char: int) -> ndarray[i64,1d]`. For each sequence, returns the index just past the first `stop_char` (inclusive), else `full_ends[i]`. Mirrors `_nb_find_stop_ends`.

- [ ] **Step 1: Write the failing truncate test**

Append to `tests/test_translate_rust.py`:

```python
def test_translate_ragged_truncate_stop():
    data = np.frombuffer(b"ATGTAAAAA" b"AAAAAA", "S1")  # seq0: M * K ; seq1: K K
    offsets = np.array([0, 9, 15], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.AA.translate(rag, truncate_stop=True)
    # seq0 truncates after '*' -> "M*"; seq1 has no stop -> "KK"
    seq0 = got[0].data.tobytes().decode()
    seq1 = got[1].data.tobytes().decode()
    assert seq0 == "M*"
    assert seq1 == "KK"
```

- [ ] **Step 2: Run (green via Numba), then swap impl**

Run: `pixi run -e dev pytest tests/test_translate_rust.py::test_translate_ragged_truncate_stop -q`
Expected: PASS via existing `_nb_find_stop_ends`.

- [ ] **Step 3: Add `_translate_stop_ends`**

Append to `src/translate.rs`:

```rust
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
```

Register: `m.add_function(wrap_pyfunction!(translate::_translate_stop_ends, m)?)?;`

- [ ] **Step 4: Swap the call site**

Replace the `_nb_find_stop_ends(...)` call (`_alphabets.py:660-662`) with:

```python
            ends = _translate_stop_ends(
                translated_flat.view(np.uint8), starts, full_ends, np.uint8(ord("*"))
            )
```

Add `_translate_stop_ends` to the extension import.

- [ ] **Step 5: Build and run**

Run: `maturin develop && pixi run -e dev pytest tests/test_translate_rust.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/translate.rs src/lib.rs python/seqpro/alphabets/_alphabets.py tests/test_translate_rust.py
git commit -m "feat(translate): route truncate_stop through Rust"
```

---

## Task 9: Self-contained Rust OHE↔AA path + wire OHE-ragged translate

**Files:**
- Modify: `src/translate.rs`
- Modify: `src/lib.rs`
- Modify: `python/seqpro/alphabets/_alphabets.py:589-598, 672-677`
- Modify: `tests/test_translate_rust.py`

**Interfaces:**
- Produces (Python-visible): `_translate_ohe(data: ndarray[u8,2d (total,n_nuc)], nuc_bytes: ndarray[u8,1d (n_nuc)], codon_size: int, lut: ndarray[u8,1d]|None, keys: ndarray[u8,1d]|None, values: ndarray[u8,1d]|None, aa_bytes: ndarray[u8,1d (n_aa)], marker: int, unknown_aa: int) -> ndarray[u8,2d (total//codon_size, n_aa)]`. One pass OHE→codon→AA→OHE. Each nucleotide row is decoded by argmax-of-one-hot to a byte from `nuc_bytes` (all-zero row → a non-canonical sentinel byte, e.g. 0, which forces `marker`); each output AA byte is re-encoded one-hot against `aa_bytes` (AA not in `aa_bytes`, e.g. `marker`/`unknown_aa`, → all-zero row).

- [ ] **Step 1: Write the failing OHE test**

Append to `tests/test_translate_rust.py`:

```python
def test_translate_ohe_ragged_roundtrips():
    # Build OHE ragged DNA for "ATGTAA" (len 6, 2 codons -> "M*").
    seq = np.frombuffer(b"ATGTAA", "S1")
    ohe = sp.DNA.ohe(seq)  # (6, 4)
    offsets = np.array([0, 6], dtype=np.int64)
    rag = Ragged.from_offsets(ohe, (1, None, 4), offsets)
    got = sp.AA.translate(rag, nuc_alphabet=sp.DNA)
    # Decode AA OHE back to bytes for assertion.
    aa_bytes = sp.AA.decode_ohe(got.data, ohe_axis=-1)
    assert aa_bytes.tobytes().decode() == _bio_like_translate("ATGTAA")
```

- [ ] **Step 2: Run (green via Numba decode_ohe path), then swap impl**

Run: `pixi run -e dev pytest tests/test_translate_rust.py::test_translate_ohe_ragged_roundtrips -q`
Expected: PASS via the existing `decode_ohe`→translate→`ohe` path.

- [ ] **Step 3: Add `_translate_ohe`**

Append to `src/translate.rs`:

```rust
/// Fused OHE→codon→AA→OHE translation for OHE-ragged input, avoiding any
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
    let aa_bytes = aa_bytes
        .ok_or_else(|| PyValueError::new_err("aa_bytes required"))?;
    let aa_bytes = aa_bytes.as_slice()?;
    let total = data.shape()[0];
    let n_nuc = data.shape()[1];
    let n_aa = aa_bytes.len();
    if codon_size == 0 || total % codon_size != 0 {
        return Err(PyValueError::new_err(
            "total rows must be a positive multiple of codon_size",
        ));
    }

    // 1) Decode each OHE row to a nucleotide byte (all-zero row -> 0 sentinel).
    let mut nuc_buf = vec![0u8; total];
    for r in 0..total {
        let mut found = 0u8; // sentinel: non-canonical, forces marker downstream
        for c in 0..n_nuc {
            if data[[r, c]] == 1 {
                found = nuc[c];
                break;
            }
        }
        nuc_buf[r] = found;
    }

    // 2) Translate to AA bytes.
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
```

Register: `m.add_function(wrap_pyfunction!(translate::_translate_ohe, m)?)?;`

- [ ] **Step 4: Wire the OHE branch**

In the Ragged branch, when `is_ohe` is True, replace the `decode_ohe` → translate → `ohe` sequence with a single Rust call. Concretely, replace the OHE decode block (`_alphabets.py:589-598`) and the OHE re-encode block (`672-677`) so that, for the OHE case, the code computes:

```python
        if is_ohe:
            if nuc_alphabet is None:
                raise ValueError("nuc_alphabet is required for OHE Ragged input.")
            if validate:
                self._check_ohe_rows(seqs.data, len(nuc_alphabet.array))
            n_aa = len(self.aa_array)
            if self.codon_lut is not None:
                ohe_flat = _translate_ohe(
                    seqs.data, nuc_alphabet.array.view(np.uint8), codon_size,
                    self.codon_lut, None, None,
                    self.aa_array.view(np.uint8), int(marker_byte),
                )
            else:
                keys = np.ascontiguousarray(self.codon_array.view(np.uint8)).reshape(-1)
                values = np.ascontiguousarray(self.aa_array.view(np.uint8)).reshape(-1)
                ohe_flat = _translate_ohe(
                    seqs.data, nuc_alphabet.array.view(np.uint8), codon_size,
                    None, keys, values,
                    self.aa_array.view(np.uint8), int(marker_byte),
                )
            new_offsets = offsets // codon_size
            # OHE input does not support unknown="drop" or trailing tracks (existing rule).
            if truncate_stop:
                # Stop char in OHE space: find first all-zero?? No — stop is '*',
                # which is a real AA row. Decode the stop column index once.
                stop_col = int(np.where(self.aa_array.view(np.uint8) == ord("*"))[0][0])
                stop_rows = (ohe_flat[:, stop_col] == 1).view(np.uint8)
                starts = new_offsets[:-1].astype(np.int64)
                full_ends = new_offsets[1:].astype(np.int64)
                ends = _translate_stop_ends(stop_rows, starts, full_ends, np.uint8(1))
                out_offsets = np.stack([starts, ends])
            else:
                out_offsets = new_offsets
            return Ragged.from_offsets(ohe_flat, (n, None, n_aa), out_offsets)
```

This OHE branch now returns early; ensure it is placed before the bytes-path translate logic and that the bytes path is taken only when `not is_ohe`. Remove the now-unused OHE decode/re-encode lines from the old flow.

> Implementer note: `truncate_stop` for OHE previously fell through the shared `_nb_find_stop_ends` on decoded bytes. The OHE-native version above finds the stop row by its one-hot column. Verify against the existing OHE+truncate test (if any); if none exists, the differential suite in Task 11 covers it.

- [ ] **Step 5: Build and run**

Run: `maturin develop && pixi run -e dev pytest tests/test_translate_rust.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/translate.rs src/lib.rs python/seqpro/alphabets/_alphabets.py tests/test_translate_rust.py
git commit -m "feat(translate): self-contained Rust OHE<->AA path"
```

---

## Task 10: Differential property tests (Hypothesis) + full existing suite

**Files:**
- Modify: `tests/test_tokenize_rust.py`
- Modify: `tests/test_translate_rust.py`

**Interfaces:**
- Consumes: the public `sp.tokenize` / `sp.AA.translate` (now Rust-backed) and pure-NumPy references defined earlier.

- [ ] **Step 1: Add a Hypothesis differential test for tokenize**

Append to `tests/test_tokenize_rust.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp


@settings(max_examples=200, deadline=None)
@given(
    seqs=hnp.arrays(
        dtype="S1",
        shape=hnp.array_shapes(min_dims=1, max_dims=3, min_side=0, max_side=20),
        elements=st.sampled_from([b"A", b"C", b"G", b"T", b"N", b"x"]),
    ),
    parallel=st.sampled_from([None, True, False]),
)
def test_tokenize_differential(seqs, parallel):
    got = sp.tokenize(seqs, TOKEN_MAP, UNK, parallel=parallel)
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))
```

- [ ] **Step 2: Add a Hypothesis differential test for translate**

Append to `tests/test_translate_rust.py`:

```python
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=200, deadline=None)
@given(
    n_codons=st.integers(min_value=0, max_value=30),
    data=st.data(),
    unknown=st.sampled_from(["X", "drop"]),
)
def test_translate_dense_differential(n_codons, data, unknown):
    chars = data.draw(
        st.lists(
            st.sampled_from(list("ACGTacgtN")),
            min_size=n_codons * 3,
            max_size=n_codons * 3,
        )
    )
    seq = "".join(chars)
    arr = np.frombuffer(seq.encode("ascii"), "S1")
    got = sp.AA.translate(arr, unknown=unknown)
    if unknown == "drop":
        # Reference: translate then drop non-canonical codons.
        ref = []
        for i in range(0, len(seq), 3):
            cod = seq[i : i + 3].upper()
            if set(cod) <= set("ACGT"):
                ref.append(sp.AA.codon_to_aa.get(cod, "X"))
        assert got.data.tobytes().decode() == "".join(ref)
    else:
        assert got.tobytes().decode() == _bio_like_translate(seq.upper())
```

- [ ] **Step 3: Run the differential tests**

Run: `pixi run -e dev pytest tests/test_tokenize_rust.py tests/test_translate_rust.py -q`
Expected: PASS.

- [ ] **Step 4: Run the entire existing suite (parity gate)**

Run: `pixi run test`
Expected: PASS — no regressions in any existing test (`tests/`), including `cargo test --lib` for Rust units (`cargo test`).

- [ ] **Step 5: Commit**

```bash
git add tests/test_tokenize_rust.py tests/test_translate_rust.py
git commit -m "test: differential property tests for Rust tokenize/translate"
```

---

## Task 11: Benchmark harness + threshold re-measurement

**Files:**
- Create: `benches/bench_tokenize_translate.py`
- Modify: `pyproject.toml:31-37`
- Modify: `src/tokenize.rs` (threshold constant), `src/translate.rs` (threshold constant)

**Interfaces:**
- Consumes: public `sp.tokenize` / `sp.AA.translate`.

- [ ] **Step 1: Add `pytest-benchmark` to the dev group**

In `pyproject.toml`, under `[dependency-groups] dev = [...]`, add:

```toml
    "pytest-benchmark>=4.0",
```

Run: `pixi install` (or `pixi run -e dev python -c "import pytest_benchmark"`).
Expected: import succeeds.

- [ ] **Step 2: Write the benchmark sweep**

Create `benches/bench_tokenize_translate.py`:

```python
"""End-to-end Python-level benchmarks for the Rust tokenize/translate port.

Run: pixi run -e dev pytest benches/bench_tokenize_translate.py --benchmark-only

Measures the public API (including PyO3 marshalling), which Rust-only criterion
misses and which decides the small-array regime. To compare against the Numba
baseline, check out the pre-port commit and run the same file.
"""

import numpy as np
import pytest

import seqpro as sp

TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
SIZES = [100, 10_000, 1_000_000]


@pytest.mark.parametrize("n", SIZES)
def test_bench_tokenize_dense(benchmark, n):
    rng = np.random.default_rng(0)
    seqs = rng.choice(np.frombuffer(b"ACGT", "S1"), size=n)
    benchmark(lambda: sp.tokenize(seqs, TOKEN_MAP, 7))


@pytest.mark.parametrize("n_codons", [33, 3_333, 333_333])
def test_bench_translate_dense(benchmark, n_codons):
    rng = np.random.default_rng(1)
    seqs = rng.choice(np.frombuffer(b"ACGT", "S1"), size=n_codons * 3)
    benchmark(lambda: sp.AA.translate(seqs))
```

- [ ] **Step 3: Run the sweep on the port**

Run: `pixi run -e dev pytest benches/bench_tokenize_translate.py --benchmark-only -q`
Expected: a benchmark table prints; record min times per size.

- [ ] **Step 4: Re-measure thresholds and bake constants**

From the sweep, find the element count where the parallel (rayon) path overtakes serial for each kernel. Temporarily force `parallel=True`/`False` (tokenize) and add a forced-serial bench variant (translate) to locate the crossover. Update `TOKENIZE_PARALLEL_THRESHOLD` in `src/tokenize.rs` and `TRANSLATE_PARALLEL_THRESHOLD` in `src/translate.rs` to the measured values, updating the doc comments with the measured number and machine core count (mirroring the original comment style). Rebuild: `maturin develop`.

- [ ] **Step 5: Commit**

```bash
git add benches/bench_tokenize_translate.py pyproject.toml src/tokenize.rs src/translate.rs
git commit -m "bench: end-to-end tokenize/translate sweep; re-measure rayon thresholds"
```

---

## Task 12: Collect results, update the design doc, decide

**Files:**
- Modify: `docs/superpowers/specs/2026-06-18-rust-tokenize-translate-design.md`
- (Conditionally) Modify: `skills/seqpro/SKILL.md`

**Interfaces:** none (analysis + docs).

- [ ] **Step 1: Capture the baseline numbers**

Check out the pre-port commit (the commit before Task 1) in a scratch worktree, run `pixi run -e dev pytest benches/bench_tokenize_translate.py --benchmark-only -q`, and record both **cold** (first call, Numba JIT included) and **warm** times.

```bash
git worktree add ../seqpro-baseline HEAD~$(git rev-list --count f177d09..HEAD)
# run benches there, record numbers, then:
git worktree remove ../seqpro-baseline
```

(If counting commits is error-prone, instead `git stash` the benches file onto the pre-port commit manually; the goal is baseline numbers on the same machine.)

- [ ] **Step 2: Tabulate port vs. baseline and apply the decision rule**

Add a "Results" section to the design doc with a table: rows = (function × size × layout), columns = Numba-cold, Numba-warm, Rust. Apply the rule: **keep if Rust wins clearly in ≥1 regime and never meaningfully regresses others.** Write the explicit keep/revert conclusion.

- [ ] **Step 3: Confirm or update the skill**

Verify public signatures are unchanged: `git diff f177d09^ -- python/seqpro/_encoders.py python/seqpro/alphabets/_alphabets.py | grep -E '^\+.*def (tokenize|translate)'` shows no signature changes. If any documented behavior shifted (e.g. the `parallel=` note wording), update `skills/seqpro/SKILL.md`; otherwise note "no skill change required" in the design doc.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-18-rust-tokenize-translate-design.md skills/seqpro/SKILL.md
git commit -m "docs: tokenize/translate Rust port benchmark results and decision"
```

- [ ] **Step 5 (only if decision = keep): remove dead Numba kernels**

If kept, delete the now-unused kernels from `python/seqpro/_numba.py` (`lut_gather`, `gufunc_translate`, `gufunc_translate_lut`, `_nb_drop_unknown_codons`, `_nb_find_stop_ends`) and their imports, run `pixi run test` to confirm green, and commit:

```bash
git commit -am "chore: remove Numba kernels superseded by Rust tokenize/translate"
```

If decision = revert, instead revert the Python wiring commits (keep the design doc + results) so the experiment's findings are preserved.

---

## Self-Review

**Spec coverage:**
- Motivation/3 goals → Tasks 11–12 (speed + small-array via sweep; "validate kill Numba" via removing Numba from these paths in Tasks 3/6/7/8/9).
- Decision rule → Task 12 Step 2.
- Full parity (dense, ragged, OHE-ragged, drop, truncate_stop, out=, parallel hatch) → Tasks 2 (out/parallel), 3 (ragged), 5/6 (dense+ragged pad), 7 (drop), 8 (truncate_stop), 9 (OHE).
- Self-contained OHE↔AA → Task 9.
- Thin-kernels-then-measure (option C) → Tasks 1–9 thin kernels; Task 11 measure; thickening deferred to post-decision follow-up (noted, not forced).
- Re-measured thresholds → Task 11 Step 4.
- Benchmark harness (sizes, layouts, variants, warm/cold) → Task 11 + Task 12 Step 1.
- Differential + existing-suite testing → Task 10.
- Leave Numba kernels until keep decision → Global Constraints + Task 12 Step 5.
- Skill check → Task 12 Step 3.

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to" — all steps carry concrete code or exact commands. Discovery commands (`ls tests | grep`) are explicit fallbacks for test-file naming, not placeholders.

**Type consistency:** `_tokenize(seqs, lut, out, parallel)`, `_translate_bytes(buf, codon_size, lut, keys, values, marker)`, `_translate_drop(translated, codons, offsets, valid_upper) -> (out, new_offsets)`, `_translate_stop_ends(data, starts, full_ends, stop_char) -> ends`, `_translate_ohe(data, nuc_bytes, codon_size, lut, keys, values, aa_bytes, marker) -> (n_codons, n_aa)`. `_rust_translate_flat(nuc_u8, marker_byte)` used consistently across Tasks 5–7. Rust kernel names (`gather_serial`, `gather_parallel`, `translate_lut_into`, `translate_scan_into`, `pack_codon_index`) consistent across tasks.

**Known follow-up (out of this plan's scope):** boundary-thickening for small-array regime (option C's second phase) is gated on Task 11 results; if the sweep shows Python overhead dominating small inputs, a follow-up plan moves `cast_seqs`/ragged orchestration into Rust.
