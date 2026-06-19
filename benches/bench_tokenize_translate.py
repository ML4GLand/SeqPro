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
    benchmark(lambda: sp.AA.translate(seqs, length_axis=0))
