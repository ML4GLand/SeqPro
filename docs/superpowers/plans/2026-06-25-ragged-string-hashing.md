# Ragged String Hashing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `md5`, `sha256`, and `rapidhash` per-string hashing of `Ragged` string containers, backed by a single rayon-parallel Rust kernel.

**Architecture:** A single Rust `#[pyfunction] _ragged_hash(data, str_offsets, algo, seed)` hashes each string (a byte run delimited by `str_offsets`) in parallel and returns a NumPy array (2-D `uint8` digests for md5/sha256, 1-D `uint64` for rapidhash). A Python free function `seqpro.rag.hash` (canonical implementation) normalizes any string representation — opaque-string leaf or chars/S1 leaf, at any ragged depth — to that `(buffer, delimiters)` contract, then shapes the output (regular `NDArray` when there is no grouping above the string level, `Ragged` reusing the input's outer offsets when there is). `Ragged.hash` is a thin delegator.

**Tech Stack:** Rust (PyO3 0.28, rayon, RustCrypto `md-5`/`sha2`/`digest`, `rapidhash`), Python (NumPy), pytest, pixi, maturin.

## Global Constraints

- All compute in a `#[pyfunction]` must run inside `py.detach(|| ...)`, capturing only `Ungil` slices (`&[u8]`, `&[i64]`); take slices while attached, `into_pyarray` after re-attaching. (project PyO3 convention)
- Canonical in-memory string format is `|S1` (single-byte) for chars and `'S'` opaque; OHE/`uint8` only for numeric. Offsets are `int64` (`OFFSET_TYPE` in `python/seqpro/rag/_utils.py`).
- New public feature → `skills/seqpro/SKILL.md` MUST be updated in this work (enforced by `CLAUDE.md`).
- Conventional commits: `feat:`, `test:`, `docs:`, etc.
- Strict Rust lint/format gates: `cargo fmt` and `cargo clippy` must be clean (no new violations); fix immediately, do not defer.
- Rust changes require `maturin develop` before Python tests can import the new symbol.
- `Ragged.from_offsets` rejects `shape.count(None) >= 3`, so **inputs are capped at R=2**; the deepest grouped output is a single-level (R=1) `Ragged`. (Do not attempt R=3 fixtures — they cannot be constructed.)
- Idiom established here (add to `CLAUDE.md`): public Ragged operations live as free functions in `_ops.py`; the `Ragged` method is a thin delegator.

**Environment:** run everything inside the pixi dev env, e.g. `pixi run -e dev <cmd>` or from `pixi shell -e dev`. Tests: `pixi run -e dev pytest`.

---

## File Structure

- `Cargo.toml` — add `digest`, `md-5`, `rapidhash`, `sha2` dependencies (rayon already present).
- `src/hashing.rs` (**new**) — the `_ragged_hash` kernel + generic `hash_elems` + `rapidhash_one` helper.
- `src/lib.rs` — `pub mod hashing;` and register `_ragged_hash`.
- `python/seqpro/rag/_ops.py` — `hash()` free function (canonical implementation); add to `__all__`.
- `python/seqpro/rag/_core.py` — `Ragged.hash()` thin delegator.
- `python/seqpro/rag/__init__.py` — export `hash`.
- `tests/test_ragged_hash.py` (**new**) — full test suite.
- `skills/seqpro/SKILL.md` — document the feature.
- `CLAUDE.md` — add the free-function/delegator idiom bullet.

---

## Task 1: Rust kernel `_ragged_hash` (md5, sha256, rapidhash)

**Files:**
- Modify: `Cargo.toml` (dependencies section)
- Create: `src/hashing.rs`
- Modify: `src/lib.rs` (module decl + registration)

**Interfaces:**
- Consumes: nothing (leaf task at the FFI boundary).
- Produces: `seqpro.seqpro._ragged_hash(data: NDArray[uint8], str_offsets: NDArray[int64], algo: str, seed: int | None) -> NDArray`. Returns a 2-D `uint8` array `(N, 16)` for `"md5"`, `(N, 32)` for `"sha256"`, and a 1-D `uint64` array `(N,)` for `"rapidhash"`, where `N = len(str_offsets) - 1`. Raises `ValueError` for an unknown `algo`.

- [ ] **Step 1: Add the Rust dependencies**

Edit `Cargo.toml`. In the `[dependencies]` section (alongside the existing `rayon = "1.11.0"`), add:

```toml
digest = "0.10"
md-5 = "0.10"
rapidhash = "4"
sha2 = "0.10"
```

- [ ] **Step 2: Create the kernel `src/hashing.rs`**

Create `src/hashing.rs` with exactly this content:

```rust
use ndarray::prelude::*;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use digest::Digest;
use rapidhash::v1::{rapidhash_v1, rapidhash_v1_seeded};

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
        Some(s) => rapidhash_v1_seeded(bytes, s),
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
```

Note on imports: `digest::Digest` is the trait both `md5::Md5` and `sha2::Sha256` implement (md-5 0.10 and sha2 0.10 share digest 0.10). `md5::Md5` is the type exported by the `md-5` crate. If `rapidhash::v1::{rapidhash_v1, rapidhash_v1_seeded}` does not resolve against the locked version, run `pixi run -e dev cargo doc -p rapidhash --open` (or inspect docs.rs for the locked version) and use the equivalent portable v1 one-shot functions — `rapidhash_v1(&[u8]) -> u64` and `rapidhash_v1_seeded(&[u8], u64) -> u64`.

- [ ] **Step 3: Register the module and function in `src/lib.rs`**

In `src/lib.rs`, add the module declaration next to the others (after `pub mod ragged;`):

```rust
pub mod hashing;
```

Inside the `fn seqpro(m: &Bound<'_, PyModule>) -> PyResult<()>` body, add this line next to the other `add_function` calls:

```rust
    m.add_function(wrap_pyfunction!(hashing::_ragged_hash, m)?)?;
```

- [ ] **Step 4: Build the extension**

Run: `pixi run -e dev maturin develop`
Expected: builds and installs without errors; `cargo fmt`/`clippy` clean. If clippy warns, fix immediately.

Verify clean lint/format:

Run: `pixi run -e dev cargo fmt --check && pixi run -e dev cargo clippy --all-targets -- -D warnings`
Expected: no output / exit 0.

- [ ] **Step 5: Write the failing kernel test**

Create `tests/test_ragged_hash.py` with this content (kernel-level tests; the Python wrapper comes in Task 2):

```python
import hashlib

import numpy as np
import pytest


def _pack(strings):
    """Return (uint8 data buffer, int64 offsets) for a list of byte strings."""
    data = np.frombuffer(b"".join(strings), dtype=np.uint8)
    offsets = np.concatenate(
        [[0], np.cumsum([len(s) for s in strings])]
    ).astype(np.int64)
    return np.ascontiguousarray(data), np.ascontiguousarray(offsets)


@pytest.mark.parametrize(
    "algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)]
)
def test_kernel_crypto_matches_hashlib(algo, hl):
    from seqpro.seqpro import _ragged_hash

    strings = [b"ACGT", b"", b"hello world", b"x", b"the quick brown fox"]
    data, offsets = _pack(strings)
    out = _ragged_hash(data, offsets, algo, None)
    assert out.dtype == np.uint8
    assert out.shape == (len(strings), hl().digest_size)
    expected = np.stack(
        [np.frombuffer(hl(s).digest(), dtype=np.uint8) for s in strings]
    )
    np.testing.assert_array_equal(out, expected)


def test_kernel_rapidhash_properties():
    from seqpro.seqpro import _ragged_hash

    strings = [b"abc", b"abc", b"abd", b""]
    data, offsets = _pack(strings)
    out = _ragged_hash(data, offsets, "rapidhash", None)
    assert out.dtype == np.uint64
    assert out.shape == (len(strings),)
    assert out[0] == out[1]  # identical input -> identical hash
    assert out[0] != out[2]  # different input -> different hash

    seeded = _ragged_hash(data, offsets, "rapidhash", 12345)
    assert seeded[0] != out[0]  # seed changes output
    again = _ragged_hash(data, offsets, "rapidhash", 12345)
    np.testing.assert_array_equal(seeded, again)  # deterministic


def test_kernel_unknown_algo_raises():
    from seqpro.seqpro import _ragged_hash

    data, offsets = _pack([b"AC"])
    with pytest.raises(ValueError, match="unknown algo"):
        _ragged_hash(data, offsets, "sha1", None)
```

- [ ] **Step 6: Run the kernel tests**

Run: `pixi run -e dev pytest tests/test_ragged_hash.py -v`
Expected: all PASS. If `test_kernel_crypto_matches_hashlib` fails on shape/values, recheck `hash_elems` slicing. If rapidhash import failed, recheck the symbol path per Step 2's note (then re-run `maturin develop`).

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/hashing.rs src/lib.rs tests/test_ragged_hash.py
git commit -m "feat(rust): add _ragged_hash kernel (md5/sha256/rapidhash, rayon)"
```

---

## Task 2: Python `hash()` free function + `Ragged.hash` method

**Files:**
- Modify: `python/seqpro/rag/_ops.py` (add `hash`, extend `__all__`, add `Literal` import)
- Modify: `python/seqpro/rag/_core.py` (add `Ragged.hash` method; ensure `Literal` imported)
- Modify: `python/seqpro/rag/__init__.py` (import + `__all__`)
- Test: `tests/test_ragged_hash.py` (append wrapper-level tests)

**Interfaces:**
- Consumes: `seqpro.seqpro._ragged_hash` (from Task 1); `Ragged._rl`, `Ragged._layout`, `Ragged._is_record`, `Ragged.to_packed()`, `Ragged.from_offsets(...)` (existing); `OFFSET_TYPE`.
- Produces:
  - `seqpro.rag.hash(rag, algo, *, seed=None) -> NDArray | Ragged`
  - `Ragged.hash(algo, *, seed=None) -> NDArray | Ragged`
  - Output is a regular `NDArray` when the input has no ragged level above the string level (flat opaque `(N,)` or chars R=1 `(N, None)`): shape `(*leading_fixed, 16/32)` `uint8` for crypto, `(*leading_fixed,)` `uint64` for rapidhash (where `prod(leading_fixed) == N`, or just `(N, …)` when there are none). Output is a single-level `Ragged` reusing the input's outer offsets otherwise: crypto leaf `uint8` with trailing fixed dim `(G, None, 16/32)`; rapidhash leaf scalar `uint64` `(G, None)`.

- [ ] **Step 1: Write the failing wrapper tests**

Append to `tests/test_ragged_hash.py`:

```python
from seqpro.rag._core import Ragged
import seqpro.rag as rag_mod


# --- fixtures: one per string representation ---------------------------------

def _opaque_flat(strings):
    """Opaque-string leaf, flat: is_string True, shape (N,)."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    return Ragged.from_lengths(data, lengths)


def _chars_r1(strings):
    """Chars / S1 leaf, R=1: shape (N, None), one string per row."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    offsets = np.concatenate(
        [[0], np.cumsum([len(s) for s in strings])]
    ).astype(np.int64)
    return Ragged.from_offsets(data, (len(strings), None), offsets)


def _opaque_under_axis(groups):
    """Opaque strings grouped: shape (G, None), str_offsets + outer o0."""
    flat = [s for g in groups for s in g]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    str_off = np.concatenate(
        [[0], np.cumsum([len(s) for s in flat])]
    ).astype(np.int64)
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    return Ragged.from_offsets(data, (len(groups), None), o0, str_offsets=str_off)


def _chars_r2(groups):
    """Chars / S1 leaf, R=2: shape (G, None, None)."""
    flat = [s for g in groups for s in g]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    o1 = np.concatenate(
        [[0], np.cumsum([len(s) for s in flat])]
    ).astype(np.int64)
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    return Ragged.from_offsets(data, (len(groups), None, None), [o0, o1])


def _digest_bytes(hl, s):
    return np.frombuffer(hl(s).digest(), dtype=np.uint8)


# --- regular (ungrouped) output: opaque flat + chars R=1 --------------------

@pytest.mark.parametrize(
    "algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)]
)
@pytest.mark.parametrize("ctor", [_opaque_flat, _chars_r1])
def test_crypto_regular_matches_hashlib(algo, hl, ctor):
    strings = [b"ACGT", b"", b"hello world", b"x"]
    r = ctor(strings)
    out = r.hash(algo)
    assert not isinstance(out, Ragged)
    assert out.dtype == np.uint8
    assert out.shape == (len(strings), hl().digest_size)
    expected = np.stack([_digest_bytes(hl, s) for s in strings])
    np.testing.assert_array_equal(out, expected)


# --- grouped output: opaque-under-axis + chars R=2 --------------------------

@pytest.mark.parametrize(
    "algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)]
)
@pytest.mark.parametrize("ctor", [_opaque_under_axis, _chars_r2])
def test_crypto_grouped_returns_ragged(algo, hl, ctor):
    groups = [[b"AA", b"B"], [b"CCC", b"DDDD"]]
    flat = [s for g in groups for s in g]
    r = ctor(groups)
    out = r.hash(algo)
    assert isinstance(out, Ragged)
    # output offsets identical to the input outer offsets
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    np.testing.assert_array_equal(out.offsets, o0)
    # per-string digests correct, in packed order
    packed = out.to_packed().data.reshape(len(flat), hl().digest_size)
    expected = np.stack([_digest_bytes(hl, s) for s in flat])
    np.testing.assert_array_equal(packed, expected)


def test_rapidhash_grouped_returns_ragged_uint64():
    groups = [[b"AA", b"B"], [b"CCC"]]
    r = _chars_r2(groups)
    out = r.hash("rapidhash")
    assert isinstance(out, Ragged)
    assert out.to_packed().data.dtype == np.uint64
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    np.testing.assert_array_equal(out.offsets, o0)


# --- equivalences and edges -------------------------------------------------

def test_free_function_matches_method():
    r = _chars_r1([b"AC", b"GT"])
    np.testing.assert_array_equal(rag_mod.hash(r, "md5"), r.hash("md5"))


def test_unpacked_input_is_packed_internally():
    r = _chars_r1([b"AAA", b"B", b"CC", b"DDDD"])
    sub = r[np.array([3, 0, 2])]  # gather -> non-contiguous offsets
    out = sub.hash("md5")
    expected = np.stack(
        [_digest_bytes(hashlib.md5, s) for s in [b"DDDD", b"AAA", b"CC"]]
    )
    np.testing.assert_array_equal(out, expected)


def test_empty_container():
    r = _chars_r1([])
    out = r.hash("sha256")
    assert out.shape == (0, 32)


# --- error handling ---------------------------------------------------------

def test_numeric_ragged_raises():
    r = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    with pytest.raises(ValueError, match="string/char"):
        r.hash("md5")


def test_record_ragged_raises():
    a = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), np.array([3, 3]))
    b = Ragged.from_lengths(np.frombuffer(b"birdho", "S1"), np.array([3, 3]))
    rec = Ragged.from_fields({"x": a, "y": b})
    with pytest.raises(NotImplementedError):
        rec.hash("md5")


def test_unknown_algo_raises_in_python():
    r = _chars_r1([b"AC"])
    with pytest.raises(ValueError, match="unknown algo"):
        r.hash("sha1")


def test_seed_on_crypto_raises():
    r = _chars_r1([b"AC"])
    with pytest.raises(ValueError, match="seed is only valid"):
        r.hash("md5", seed=1)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_ragged_hash.py -k "regular or grouped or free_function or unpacked or empty or numeric or record or unknown_algo_raises_in_python or seed_on_crypto" -v`
Expected: FAIL/ERROR — `Ragged` has no attribute `hash` (and `seqpro.rag` has no `hash`).

- [ ] **Step 3: Implement the `hash()` free function in `_ops.py`**

In `python/seqpro/rag/_ops.py`, change the typing import line `from typing import Any` to:

```python
from typing import Any, Literal
```

Change `__all__` to include `hash`:

```python
__all__ = ["concatenate", "hash", "reverse_complement", "to_packed", "to_padded"]
```

Append this function to `_ops.py`:

```python
def hash(
    rag: "Ragged[Any]",
    algo: Literal["md5", "sha256", "rapidhash"],
    *,
    seed: int | None = None,
) -> "NDArray[Any] | Ragged[Any]":
    """Hash each string in a Ragged container (md5, sha256, or rapidhash).

    Each string is hashed independently by a rayon-parallel Rust kernel. Works
    for both string representations — an opaque-string leaf (dtype ``'S'``) and
    a chars/S1 leaf (dtype ``|S1``) — at any ragged depth.

    Parameters
    ----------
    rag
        A string Ragged (opaque or S1 chars). Numeric leaves and record-layout
        arrays are rejected.
    algo
        ``"md5"`` (16-byte digest), ``"sha256"`` (32-byte digest), or
        ``"rapidhash"`` (8-byte ``uint64``).
    seed
        Only valid for ``"rapidhash"``; seeds the portable hash. Supplying it
        for md5/sha256 raises ``ValueError``.

    Returns
    -------
    NDArray or Ragged
        One digest per string, in packed order. When the input has no ragged
        level above the string level, a regular array: ``(*leading, 16/32)``
        ``uint8`` (md5/sha256) or ``(*leading,)`` ``uint64`` (rapidhash). When
        the input groups strings under an outer axis, a single-level ``Ragged``
        reusing those outer offsets, with a fixed-size digest leaf.
    """
    if not isinstance(rag, Ragged):
        rag = Ragged(rag)
    if rag._is_record:
        raise NotImplementedError(
            "hash is not defined on record-layout Ragged arrays; "
            "hash fields individually."
        )

    valid = ("md5", "sha256", "rapidhash")
    if algo not in valid:
        raise ValueError(f"unknown algo {algo!r}; expected one of {valid}")
    if seed is not None and algo != "rapidhash":
        raise ValueError(
            f"seed is only valid for algo='rapidhash', not {algo!r}"
        )

    rl = rag._rl
    if rl.str_offsets is None:
        # chars / S1 leaf: must be single-byte and have a ragged axis.
        if rl.data.dtype.kind != "S" or rl.data.dtype.itemsize != 1:
            raise ValueError(
                "hashing requires string/char data (opaque 'S' or |S1), "
                f"got dtype {rl.data.dtype!r}"
            )
        if not rag._layout.offsets:
            raise ValueError(
                "hashing requires a string collection (a ragged axis or opaque "
                "str_offsets); got a flat regular array"
            )

    packed = rag.to_packed()
    prl = packed._rl
    if prl.str_offsets is not None:
        delimiters = prl.str_offsets
        outer_offsets = list(packed._layout.offsets)
    else:
        delimiters = packed._layout.offsets[-1]
        outer_offsets = list(packed._layout.offsets[:-1])

    data_u1 = np.ascontiguousarray(prl.data).reshape(-1).view(np.uint8)
    delimiters = np.ascontiguousarray(delimiters, dtype=OFFSET_TYPE)

    from seqpro.seqpro import _ragged_hash  # type: ignore[missing-import]  # rust

    digests = _ragged_hash(data_u1, delimiters, algo, seed)

    if not outer_offsets:
        # No grouping above the string level -> regular array, reshaped to any
        # leading fixed dims (mirrors Ragged.lengths).
        shape = packed._layout.shape
        str_axis = shape.index(None) if None in shape else len(shape)
        leading = [d for d in shape[:str_axis] if d is not None]
        if leading:
            return digests.reshape(*leading, *digests.shape[1:])
        return digests

    # Grouped -> single-level Ragged reusing the input's outer offsets.
    o0 = outer_offsets[0]
    g = int(o0.shape[0]) - 1
    if digests.ndim == 2:
        out_shape: tuple[int | None, ...] = (g, None, digests.shape[1])
    else:
        out_shape = (g, None)
    return Ragged.from_offsets(digests, out_shape, o0)
```

- [ ] **Step 4: Add the `Ragged.hash` delegator in `_core.py`**

In `python/seqpro/rag/_core.py`, ensure `Literal` is importable. Check the `from typing import` line near the top; if `Literal` is not listed, add it. Then add this method to the `Ragged` class, immediately after the `to_padded` method (which ends around line 1700; place it after that method's `return`):

```python
    def hash(
        self,
        algo: "Literal['md5', 'sha256', 'rapidhash']",
        *,
        seed: "int | None" = None,
    ) -> "NDArray[Any] | Ragged[Any]":
        """Hash each string element. Thin delegator to :func:`seqpro.rag.hash`."""
        from ._ops import hash as _hash

        return _hash(self, algo, seed=seed)
```

- [ ] **Step 5: Export `hash` from `__init__.py`**

In `python/seqpro/rag/__init__.py`, update the `_ops` import and `__all__`:

```python
from ._ops import concatenate, hash, reverse_complement, to_packed, to_padded
```

Add `"hash"` to the `__all__` list (keep it alphabetically near `concatenate`):

```python
__all__ = [
    "OFFSET_TYPE",
    "DTYPE_co",
    "RDTYPE_co",
    "Ragged",
    "concatenate",
    "hash",
    "is_rag_dtype",
    "lengths_to_offsets",
    "reverse_complement",
    "to_packed",
    "to_padded",
    "zip",
]
```

- [ ] **Step 6: Run the full test suite**

Run: `pixi run -e dev pytest tests/test_ragged_hash.py -v`
Expected: all PASS (kernel tests from Task 1 + all wrapper tests).

- [ ] **Step 7: Lint/format Python**

Run: `pixi run -e dev ruff check python/seqpro/rag/ tests/test_ragged_hash.py && pixi run -e dev ruff format --check python/seqpro/rag/ tests/test_ragged_hash.py`
Expected: clean. If `ruff` flags `hash` shadowing the builtin, that is intentional (mirrors `rag.zip`); add a targeted `# noqa: A001` / `# noqa: A003` comment on the relevant `def`/`__all__` lines if required to pass, matching how `zip` is handled in `__init__.py`.

- [ ] **Step 8: Commit**

```bash
git add python/seqpro/rag/_ops.py python/seqpro/rag/_core.py python/seqpro/rag/__init__.py tests/test_ragged_hash.py
git commit -m "feat(rag): add Ragged.hash / sp.rag.hash (md5/sha256/rapidhash)"
```

---

## Task 3: Documentation (SKILL.md + CLAUDE.md idiom)

**Files:**
- Modify: `skills/seqpro/SKILL.md`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: the public API from Task 2 (`sp.rag.hash` / `Ragged.hash`).
- Produces: documentation only.

- [ ] **Step 1: Document the feature in `skills/seqpro/SKILL.md`**

In the `## `Ragged`` section (after the record-layout / `from_fields` example, around line 116), add this subsection:

````markdown
### Hashing strings

Hash each string in a `Ragged` (opaque-string or S1-chars leaf, any depth) with
a parallel Rust kernel:

```python
digests = rag.hash("sha256")        # (N, 32) uint8, one digest per string
md5s    = rag.hash("md5")           # (N, 16) uint8
fast    = rag.hash("rapidhash")     # (N,) uint64
seeded  = rag.hash("rapidhash", seed=42)   # seed valid for rapidhash only
# equivalently: sp.rag.hash(rag, "sha256")
```

Output mirrors the structure *above* the string level: a regular NumPy array
when strings aren't grouped (flat `(N, …)` / leading fixed dims), or a `Ragged`
reusing the outer offsets when they are (`(G, None, 16/32)` for md5/sha256,
`(G, None)` uint64 for rapidhash). Numeric and record-layout Rageds are
rejected.
````

If the `## Quick reference` table (around line 38) lists per-row ops, also add a row:

```markdown
| Hash ragged strings | `rag.hash("sha256"\|"md5"\|"rapidhash")` |
```

- [ ] **Step 2: Add the idiom convention to `CLAUDE.md`**

In `CLAUDE.md`, under `## Key Conventions`, add this bullet (after the `transforms/` bullet):

```markdown
- **Ragged operations: free function is canonical, method delegates.** Public
  Ragged/sequence operations live as free functions in the relevant `_ops`-style
  module (the implementation home, e.g. `rag/_ops.py`); the corresponding
  `Ragged` method is a thin one-line delegator to it (e.g. `Ragged.hash` →
  `rag._ops.hash`). Add new operations this way. This matches
  `reverse_complement` and `concatenate`.
```

- [ ] **Step 3: Verify docs reference the real API**

Run: `pixi run -e dev python -c "import seqpro as sp; import numpy as np; from seqpro.rag._core import Ragged; r = Ragged.from_lengths(np.frombuffer(b'catdog','S1'), np.array([3,3])); print(sp.rag.hash(r,'sha256').shape, r.hash('rapidhash').dtype)"`
Expected: prints `(2, 32) uint64` (confirms both the free function and method, and both an opaque-flat regular output and rapidhash dtype).

- [ ] **Step 4: Commit**

```bash
git add skills/seqpro/SKILL.md CLAUDE.md
git commit -m "docs: document Ragged string hashing and free-function/delegator idiom"
```

---

## Self-Review

**1. Spec coverage:**

| Spec requirement | Task |
|---|---|
| Per-string hashing, rayon kernel | Task 1 |
| md5 / sha256 (generic Digest) + rapidhash | Task 1 |
| `py.detach` compute, slices captured | Task 1 (Step 2) |
| Deps: `md-5`, `sha2`, `digest`, `rapidhash` | Task 1 (Step 1) |
| Free function canonical + method delegator | Task 2 (Steps 3–4) |
| Export from `__init__` | Task 2 (Step 5) |
| Unified delimiter rule (opaque vs chars) | Task 2 (Step 3) |
| Output: regular vs Ragged by outer structure | Task 2 (Step 3) + tests |
| Raw-bytes output `(N,16)/(N,32)/(N,)` | Task 1 + Task 2 |
| `seed` rapidhash-only; ValueError otherwise | Task 2 (Step 3) + test |
| Errors: numeric, non-collection, record, bad algo | Task 2 (Step 3) + tests |
| Edge: empty container, empty strings, unpacked input | Task 2 tests + Task 1 test (empty string) |
| Tests: opaque + chars at R=1 and R=2 | Task 2 tests |
| SKILL.md updated | Task 3 (Step 1) |
| CLAUDE.md idiom | Task 3 (Step 2) |

Note: the spec mentioned R=3 nesting in testing; `Ragged.from_offsets` rejects `count(None) >= 3`, so R=3 inputs cannot be constructed. Coverage is R=1 and R=2 (the supported maximum), which is correct.

**2. Placeholder scan:** No "TBD"/"handle appropriately"/"similar to" — every code step shows full code; the one external-API uncertainty (rapidhash symbol path) has a concrete verification command and exact target signatures.

**3. Type consistency:** Kernel returns `(N,16)`/`(N,32)` `uint8` and `(N,)` `uint64`; Python branches on `digests.ndim` (2 → crypto, 1 → rapidhash) consistently with that. `delimiters`/`outer_offsets` names are consistent across the function. `_ragged_hash(data, str_offsets, algo, seed)` signature matches its only call site. `Ragged.hash` ↔ `_ops.hash` names match the delegation.
