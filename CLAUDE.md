# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SeqPro is a Python/Rust hybrid package for processing biological sequences (DNA, RNA, protein) that makes almost zero compromises on speed. Python for-loops and comprehensions are avoided in anything that could be a hot loop (e.g. anything that isn't initializing a singleton). The Python layer handles high-level logic and NumPy operations (with Numba JIT for bottlenecks); the Rust extension (`src/`) handles performance-critical ops that can't be handled by numba like graph algorithms e.g. k-mer shuffling. The extension is implemented via PyO3/maturin and is compiled to `python/seqpro/seqpro.abi3.so`. Secondary to speed is convenience: a clean, simple, DRY API.

## Environment & Build

This project uses **pixi** for environment management (not conda/pip directly).

```bash
# Activate dev environment
pixi shell -e dev

# Run tests
pixi run test
# or: pytest tests/

# Run a single test file
pytest tests/test_ohe.py

# Build Rust extension (required after src/ changes)
maturin develop

# Lint / format
ruff check python/ tests/
ruff format python/ tests/

# Bump version (conventional commits, semver)
pixi run bump-dry  # dry run
```

Pre-commit hooks enforce ruff linting/formatting, conventional commits (commitizen), and `pixi lock` on push.

## Architecture

### Input type system

All public functions accept `SeqType` (defined in `python/seqpro/_utils.py`): a string, bytes, nested list of strings, or NumPy array with dtype `str_`, `object_`, `bytes_`, or `uint8` (OHE). `cast_seqs()` normalizes inputs to `|S1` bytes arrays or leaves OHE (`uint8`) untouched.

Most functions require callers to specify `length_axis` (which axis is sequence length) and `ohe_axis` (which axis is the one-hot dimension). `check_axes()` validates these and raises early.

### Module layout (`python/seqpro/`)

| Module | Contents |
|---|---|
| `_utils.py` | `cast_seqs`, `check_axes`, `array_slice` — shared low-level utilities |
| `alphabets/` | `NucleotideAlphabet`, `AminoAlphabet`, `DNA`, `RNA`, `AA` constants |
| `_encoders.py` | `ohe`, `decode_ohe`, `tokenize`, `decode_tokens`, `pad_seqs` (Numba-accelerated) |
| `_modifiers.py` | `reverse_complement`, `k_shuffle`, `jitter`, `random_seqs`, `bin_coverage` |
| `_analyzers.py` | `gc_content`, `nucleotide_content`, `length` |
| `_cleaners.py` | Sequence sanitization utilities |
| `_numba.py` | Raw Numba JIT kernels (called by encoders/modifiers) |
| `rag/` | `Ragged` array class for variable-length sequence collections |
| `transforms/` | Composable transform objects (`KShuffle`, `ReverseComplement`, `Jitter`, `Sequential`, `TMM`) |
| `bed.py`, `gtf.py` | Genomic interval I/O via polars/pyranges |
| `xr/` | Experimental xarray integration |

### Rust extension (`src/`)

Several modules are compiled and registered in the `seqpro` PyO3 module (`lib.rs`): `kshuffle.rs` (k-mer shuffle, `_k_shuffle`, called by `_modifiers.k_shuffle`), `translate.rs` (codon translation, the `_translate_*` functions), and `ragged.rs` (re-exports `seqpro-core::ragged` for the `_ragged_*` functions). `kmer_encode.rs` provides a rolling k-mer integer encoder used internally. Kernels expect contiguous arrays (e.g. k-shuffle takes a `uint8` ndarray with the last dimension as sequence length) and release the GIL via `py.detach()` during compute.

### Ragged arrays (`rag/`)

`Ragged` wraps variable-length sequences as a flat data array + offsets array, following the Arrow/awkward-array layout pattern. The `_array.py` file contains the full class; `_gufuncs.py` provides generalized ufunc helpers.

## Key Conventions

- All sequence arrays use `|S1` (single-byte ASCII) as the canonical in-memory format for string sequences, and `uint8` for OHE.
- Axis arguments (`length_axis`, `ohe_axis`) are always integers referring to the NumPy axis index; negative indexing is supported.
- The `transforms/` module wraps functional ops into callable objects with `__call__` for use in data pipelines.
- Conventional commits are enforced — use `feat:`, `fix:`, `ci:`, `bump:`, `refactor:`, `docs:`, etc. prefixes.
- **Validation is opt-in and front-loaded.** Add fast-fail/input validation via a `validate=` flag (or equivalent single opt-in), not per-feature `error` modes. There must be one obvious way to ask "is this input clean?" — don't duplicate the check across parameters.
- **No naive NumPy in hot paths.** Never use raw Python loops or naive NumPy (e.g. per-segment `np.concatenate`, Python `for` over sequences) where a Numba kernel is faster and leaner — unless the NumPy version is *verifiably* comparable in time and memory. When Numba is a poor fit (graph algorithms like k-shuffle), use the Rust/PyO3 extension (`src/`).
- **PyO3 perf tips for `src/` boundary code** (per the [PyO3 performance guide](https://github.com/PyO3/pyo3/blob/main/guide/src/performance.md)). When writing or editing a `#[pyfunction]`:
  - **Detach the interpreter during compute.** Any kernel doing real work (>~1ms, especially rayon-parallel ones) must run inside `py.detach(|| ...)`. Pattern: take slices/views (`as_slice`/`as_array`) and do all Python-touching work *while attached*, run the compute in `detach` capturing only `Ungil` slices/views (never a `PyReadonlyArray`/`Bound`/`Py`), then `into_pyarray(py)` *after* re-attaching. Add a `py: Python` param if the function lacks one.
  - Use the existing `py` token / `Bound::py()` — never `Python::attach` when a token is already in scope.
  - Prefer `cast::<T>()` over `extract::<T>()` for *native* Python types (`PyList`/`PyTuple`/etc.). Does not apply to numpy `PyReadonlyArray` extraction.
  - Pass Rust tuples (not `Bound<PyTuple>`) when calling back into Python, to hit the faster `vectorcall` path.

## Skills

This repo ships an installable skill at `skills/seqpro/SKILL.md` (skills.sh layout) describing how to use seqpro. **Any PR that adds a new public feature, changes a public signature, or makes a breaking change MUST update `skills/seqpro/SKILL.md` in the same PR.** Doc-only edits and internal refactors do not require updates unless they change conventions documented in the skill. Reviewers should reject feature/breaking PRs that don't touch the skill.
