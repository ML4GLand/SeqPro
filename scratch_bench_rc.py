"""Microbench: flat-buffer ragged reverse-complement vs gvl's awkward idiom.

Run: uv run --with attrs python scratch_bench_rc.py

Reproduces the exact awkward path gvl uses today in
``_dataset/_query.py:reverse_complement_ragged`` ->  ``_ragged.py:reverse_complement``:

    Ragged(ak.to_packed(ak.where(to_rc, rc_awkward(rag), rag)))

and compares wall-time + peak allocation against the Numba flat-buffer kernel
``seqpro.rag._ops.reverse_complement`` on a realistic per-batch shape
(batch=32, ~16 kb rows, ~50% negative-strand mask).
"""

from __future__ import annotations

import time
import tracemalloc

import awkward as ak
import awkward.operations.str as ak_str
import numpy as np
from awkward.contents import NumpyArray

import seqpro as sp
from seqpro.rag import Ragged, lengths_to_offsets
from seqpro.rag._ops import _reverse_complement_ragged, reverse_complement

COMP_LUT = sp.DNA.bytes_comp_array.view(np.uint8)
_COMP_DNA = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)


# ---- gvl's current awkward implementation (verbatim shape) -------------------
import numba as nb  # noqa: E402


@nb.vectorize(["u1(u1)"], nopython=True)
def _ufunc_comp_dna(seq):
    return _COMP_DNA[seq]


def _ak_comp_dna_helper(layout, **kwargs):
    if layout.is_numpy:
        return NumpyArray(_ufunc_comp_dna(layout.data), parameters=layout.parameters)


def _rc_awkward(arr):
    """gvl _ragged.reverse_complement: to_packed -> complement transform -> str.reverse."""
    og_type = type(arr)
    arr = ak.to_packed(arr)
    arr = ak_str.reverse(ak.transform(_ak_comp_dna_helper, arr))
    return og_type(arr)


def gvl_reverse_complement_ragged(rag: Ragged, to_rc: np.ndarray) -> Ragged:
    """gvl _query.reverse_complement_ragged bytes branch, verbatim."""
    return Ragged(ak.to_packed(ak.where(to_rc, _rc_awkward(rag), rag)))


# ---- data -------------------------------------------------------------------
def make_batch(batch: int, mean_len: int, jitter: int, seed: int):
    rng = np.random.default_rng(seed)
    lengths = rng.integers(mean_len - jitter, mean_len + jitter + 1, size=batch)
    total = int(lengths.sum())
    data = rng.integers(0, 4, size=total, dtype=np.uint8)
    data = np.array([65, 67, 71, 84], np.uint8)[data].view("S1")  # ACGT bytes
    offsets = lengths_to_offsets(lengths)
    rag = Ragged.from_offsets(data, (batch, None), offsets)
    to_rc = rng.random(batch) < 0.5
    return rag, to_rc


def timeit(fn, iters: int) -> float:
    # warmup (also triggers numba JIT)
    fn()
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/call


def peak_mb(fn) -> float:
    fn()  # warmup/JIT outside measurement
    tracemalloc.start()
    fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def main():
    BATCH, MEAN_LEN, JITTER, ITERS = 32, 16384, 400, 500
    rag, to_rc = make_batch(BATCH, MEAN_LEN, JITTER, seed=0)
    print(
        f"batch={BATCH}  mean_len={MEAN_LEN}  jitter=±{JITTER}  "
        f"total={int(rag.lengths.sum())} bytes  rc_rows={int(to_rc.sum())}/{BATCH}"
    )

    # --- correctness: flat kernel must match the awkward path exactly ---
    expected = gvl_reverse_complement_ragged(rag, to_rc)
    got = reverse_complement(rag, COMP_LUT, mask=to_rc, copy=True)
    exp_np = ak.to_packed(expected).to_numpy()
    got_np = ak.to_packed(got).to_numpy()
    assert np.array_equal(exp_np, got_np), "MISMATCH vs awkward baseline!"
    # also: original buffer untouched with copy=True
    assert reverse_complement(rag, COMP_LUT, mask=to_rc, copy=True) is not rag
    print("correctness: flat kernel == awkward baseline ✓")

    # --- timing ---
    t_ak = timeit(lambda: gvl_reverse_complement_ragged(rag, to_rc), ITERS)
    t_flat_copy = timeit(
        lambda: reverse_complement(rag, COMP_LUT, mask=to_rc, copy=True), ITERS
    )

    # in-place (copy=False): isolate kernel cost on owned buffers (no per-call copy).
    # Pre-stage fresh buffers so the mutation never compounds.
    offsets = np.ascontiguousarray(rag.offsets, np.int64)
    base = np.ascontiguousarray(rag.data).view(np.uint8)
    bufs = [base.copy() for _ in range(ITERS + 2)]
    _reverse_complement_ragged(bufs[0], offsets, COMP_LUT, to_rc)  # JIT warmup
    _reverse_complement_ragged(bufs[1], offsets, COMP_LUT, to_rc)
    t0 = time.perf_counter()
    for k in range(ITERS):
        _reverse_complement_ragged(bufs[k + 2], offsets, COMP_LUT, to_rc)
    t_flat_inplace = (time.perf_counter() - t0) / ITERS * 1e3

    # --- peak allocation for one call ---
    m_ak = peak_mb(lambda: gvl_reverse_complement_ragged(rag, to_rc))
    m_flat_copy = peak_mb(
        lambda: reverse_complement(rag, COMP_LUT, mask=to_rc, copy=True)
    )

    print()
    print(f"{'impl':<34}{'ms/call':>10}{'speedup':>10}{'peak MB':>10}")
    print(f"{'awkward (gvl current)':<34}{t_ak:>10.4f}{1.0:>9.1f}x{m_ak:>10.3f}")
    print(
        f"{'flat numba (copy=True)':<34}{t_flat_copy:>10.4f}"
        f"{t_ak / t_flat_copy:>9.1f}x{m_flat_copy:>10.3f}"
    )
    print(
        f"{'flat numba kernel (in-place)':<34}{t_flat_inplace:>10.4f}"
        f"{t_ak / t_flat_inplace:>9.1f}x{0.0:>10.3f}"
    )


if __name__ == "__main__":
    main()
