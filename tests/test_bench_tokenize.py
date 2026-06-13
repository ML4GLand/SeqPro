"""Microbenchmarks for ``seqpro.tokenize`` (pytest-codspeed).

Collected by the normal test suite (runs each body once, ~free) and timed under
``pytest --codspeed`` (see the ``bench`` pixi task / bench.yaml CI workflow).
"""

from __future__ import annotations

import numpy as np
import pytest
import seqpro as sp
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
