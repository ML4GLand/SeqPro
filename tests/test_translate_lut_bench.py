"""Microbench: O(64) linear scan vs O(1) LUT in ``gufunc_translate``.

Skipped under the default test selection; run explicitly with::

    python tests/test_translate_lut_bench.py

Expected speedup: 20-50× depending on CPU + cache effects. The LUT is
64 bytes (fits in L1), the lookup is two shifts + two ors + one
dereference per codon, while the original scan does up to 64 byte-array
comparisons per codon.
"""

from __future__ import annotations

import time

import numpy as np
import seqpro as sp
from seqpro._numba import gufunc_translate, gufunc_translate_lut


def _bench(n_codons: int = 1_000_000, n_iters: int = 5) -> tuple[float, float, int]:
    """Time both paths over ``n_iters`` iterations.

    Returns
    -------
    (t_linear, t_lut, n_diff)
    """
    rng = np.random.default_rng(42)
    byte_for_idx = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
    codons = byte_for_idx[rng.choice(4, size=(n_codons, 3))]  # (n, 3) uint8
    kmer_keys = sp.AA.codon_array.view(np.uint8)
    kmer_values = sp.AA.aa_array.view(np.uint8)

    # Warm up (Numba caches compiled signatures across calls)
    _ = gufunc_translate(codons[:100], kmer_keys, kmer_values)
    _ = gufunc_translate_lut(codons[:100], sp.AA.codon_lut)

    times_linear, times_lut = [], []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        old = gufunc_translate(codons, kmer_keys, kmer_values)
        times_linear.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        new = gufunc_translate_lut(codons, sp.AA.codon_lut)
        times_lut.append(time.perf_counter() - t0)

    n_diff = int((old != new).sum())
    return min(times_linear), min(times_lut), n_diff


if __name__ == "__main__":
    n = 1_000_000
    t_old, t_new, n_diff = _bench(n)
    print(f"Benchmark: {n:,} codons (min over 5 iters)")
    print(f"  linear scan: {t_old * 1000:8.1f} ms  ({n / t_old / 1e6:.1f} M codons/s)")
    print(f"  LUT lookup:  {t_new * 1000:8.1f} ms  ({n / t_new / 1e6:.1f} M codons/s)")
    print(f"  speedup:     {t_old / t_new:.1f}x")
    print(f"  outputs differ in {n_diff} of {n} positions (0 = correct)")
    assert n_diff == 0
