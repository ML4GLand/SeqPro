"""Microbenchmarks for ``seqpro.tokenize`` (pytest-codspeed).

Collected by the normal test suite (runs each body once, ~free) and timed under
``pytest --codspeed`` (see the ``bench`` pixi task / bench.yaml CI workflow).
"""

from __future__ import annotations

import numpy as np
import pytest
import seqpro as sp
from seqpro._numba import gufunc_tokenize
from seqpro.rag import Ragged

pytest.importorskip("pytest_codspeed")

DNA_TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
UNKNOWN_TOKEN = 4
_BASES = np.frombuffer(b"ACGT", dtype="S1")


def _rng():
    # Argless default_rng would be nondeterministic; pin the seed.
    return np.random.default_rng(0)


def _dense(batch: int, length: int) -> np.ndarray:
    rng = _rng()
    idx = rng.integers(0, 4, size=(batch, length))
    return _BASES[idx]  # (batch, length) S1


def _ragged(n: int, low: int, high: int) -> Ragged:
    rng = _rng()
    lengths = rng.integers(low, high + 1, size=n).astype(np.int64)
    total = int(lengths.sum())
    data = _BASES[rng.integers(0, 4, size=total)]
    return Ragged.from_lengths(data, lengths)


def test_bench_dense_batch(benchmark):
    """Realistic training batch (512, 1024) DNA."""
    seqs = _dense(512, 1024)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_short_alleles(benchmark):
    """Thousands of very short sequences (both alleles)."""
    seqs = _ragged(8000, 1, 4)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_flanked_alleles(benchmark):
    """Thousands of >10 bp sequences (alleles with flank nucleotides)."""
    seqs = _ragged(8000, 11, 60)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_cres(benchmark):
    """Hundreds of 100-200 bp sequences (CREs)."""
    seqs = _ragged(500, 100, 200)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


# Candidate DNA fast path: a LUT built once at import, reused across calls,
# avoiding the per-call np.full(256)+scatter. Compared head-to-head below.
_DNA_LUT = np.full(256, np.int32(UNKNOWN_TOKEN), dtype=np.int32)
_DNA_LUT[np.frombuffer(b"ACGT", dtype="S1").view(np.uint8)] = np.arange(
    4, dtype=np.int32
)


def _generic_tokenize(u8: np.ndarray) -> np.ndarray:
    keys = np.array([c.encode("ascii") for c in DNA_TOKEN_MAP]).view(np.uint8)
    vals = np.array(list(DNA_TOKEN_MAP.values()), dtype=np.int32)
    lut = np.full(256, np.int32(UNKNOWN_TOKEN), dtype=np.int32)
    lut[keys] = vals
    return np.take(lut, u8)


def test_bench_dna_generic_lut(benchmark):
    u8 = _dense(512, 1024).view(np.uint8)
    benchmark(lambda: _generic_tokenize(u8))


def test_bench_dna_precomputed_lut(benchmark):
    u8 = _dense(512, 1024).view(np.uint8)
    benchmark(lambda: np.take(_DNA_LUT, u8))


# --- Baseline: the original parallel gufunc kernel, timed on the same inputs so
# CodSpeed flags any regression of the LUT-gather impl vs the old kernel. ---
_SOURCE = np.array([c.encode("ascii") for c in DNA_TOKEN_MAP]).view(np.uint8)
_TARGET = np.array(list(DNA_TOKEN_MAP.values()), dtype=np.int32)


def _gufunc(u8: np.ndarray) -> np.ndarray:
    return gufunc_tokenize(u8, _SOURCE, _TARGET, np.int32(UNKNOWN_TOKEN))


def test_bench_baseline_dense_batch(benchmark):
    """Old gufunc kernel on the dense (512, 1024) batch (baseline)."""
    u8 = _dense(512, 1024).view(np.uint8)
    benchmark(lambda: _gufunc(u8))


def test_bench_baseline_ragged_short_alleles(benchmark):
    # Pack inside the timed callable so the baseline pays the same to_packed()
    # copy the real ragged tokenize path does (fair head-to-head).
    seqs = _ragged(8000, 1, 4)
    benchmark(lambda: _gufunc(seqs.to_packed().data.view(np.uint8)))


def test_bench_baseline_ragged_flanked_alleles(benchmark):
    seqs = _ragged(8000, 11, 60)
    benchmark(lambda: _gufunc(seqs.to_packed().data.view(np.uint8)))


def test_bench_baseline_ragged_cres(benchmark):
    seqs = _ragged(500, 100, 200)
    benchmark(lambda: _gufunc(seqs.to_packed().data.view(np.uint8)))
