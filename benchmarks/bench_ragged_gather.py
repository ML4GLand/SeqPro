"""Benchmark + correctness oracle for Ragged integer-array fancy-indexing (issue #69).

Proves the O(k) index-resolution candidate (a) matches the current O(n)
arange-based reference on every representative + edge case and (b) is flat in n
while the reference grows ~proportionally to n. Transitional harness (cf.
benchmarks/bench_ragged_backends.py); the permanent guard is the pytest cases
in tests/test_ragged_core.py. See
docs/superpowers/specs/2026-07-21-ragged-gather-ok-design.md.

Run: pixi run -e dev python benchmarks/bench_ragged_gather.py
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

import numpy as np

from seqpro.rag._core import Ragged


def reference_resolve(where: Any, n: int) -> np.ndarray:
    """Current arange-based integer-array resolution — the O(n) reference/oracle."""
    idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
    idx = np.where(idx < 0, idx + n, idx)
    return idx


def candidate_resolve(where: Any, n: int) -> np.ndarray:
    """Proposed O(k) integer-array resolution. Body is copied verbatim into
    Ragged._gather_indices in Task 3."""
    arr = np.asarray(where)
    if arr.dtype.kind not in "iu":
        if arr.size == 0 and not isinstance(where, np.ndarray):
            arr = arr.astype(np.int64)
        else:
            raise IndexError(
                "only integers, slices (`:`), and integer arrays are valid indices"
            )
    idx = np.atleast_1d(arr).astype(np.int64, copy=False)
    neg = idx < 0
    if neg.any():
        idx = np.where(neg, idx + n, idx)
    oob = (idx < 0) | (idx >= n)
    if oob.any():
        raise IndexError(
            f"index {int(idx[oob][0])} is out of bounds for axis 0 with size {n}"
        )
    return idx


# ── Correctness oracle ────────────────────────────────────────────────────────


def _check_oracle() -> None:
    n = 32
    # (where) cases that must produce identical normalized indices.
    equal_cases = [
        np.array([0, 2, 5, 31]),  # in-range positives
        3,  # scalar int
        np.array([-1, -2, -32]),  # negatives -> normalized
        np.array([], dtype=np.int64),  # empty
        [],  # bare empty python list -> empty selection
        [0, 2, 4],  # python list
        np.array([1, 1, 1, 1]),  # repeated (k can exceed n)
        np.arange(64) % n,  # k > n
    ]
    for where in equal_cases:
        r = reference_resolve(where, n)
        c = candidate_resolve(where, n)
        assert r.dtype == c.dtype == np.int64, (where, r.dtype, c.dtype)
        np.testing.assert_array_equal(r, c, err_msg=f"mismatch for {where!r}")

    # cases where BOTH must raise IndexError
    raise_cases = [
        np.array([0, n]),  # OOB positive
        np.array([-n - 1]),  # OOB negative (< -n)
        np.array([0.0, 1.0]),  # float indices
    ]
    for where in raise_cases:
        for name, fn in (
            ("reference", reference_resolve),
            ("candidate", candidate_resolve),
        ):
            try:
                fn(where, n)
            except IndexError:
                continue
            raise AssertionError(f"{name} did not raise IndexError for {where!r}")
    print("oracle: OK (reference == candidate on all cases)")


# ── Timing helpers ────────────────────────────────────────────────────────────


def _time(
    fn: Callable[[], Any], *, repeats: int = 7, min_batch_s: float = 0.02
) -> float:
    """Seconds per call: warm up, autoscale a batch past min_batch_s, min of repeats."""
    for _ in range(3):
        fn()
    iters = 1
    while True:
        t0 = perf_counter()
        for _ in range(iters):
            fn()
        if perf_counter() - t0 >= min_batch_s:
            break
        iters *= 2
    best = float("inf")
    for _ in range(repeats):
        t0 = perf_counter()
        for _ in range(iters):
            fn()
        best = min(best, (perf_counter() - t0) / iters)
    return best


# ── Sweeps ────────────────────────────────────────────────────────────────────

K = 256
N_SWEEP = [4_096, 16_384, 65_536, 262_144, 1_048_576, 4_194_304]


def _resolve_sweep() -> None:
    """Isolated index-resolution: reference (O(n)) vs candidate (O(k))."""
    print(f"\nindex resolution, k={K} (us/call):")
    print(f"{'n':>10} {'reference':>12} {'candidate':>12} {'speedup':>9}")
    rng = np.random.default_rng(0)
    for n in N_SWEEP:
        where = rng.integers(0, n, size=K)
        t_ref = _time(lambda: reference_resolve(where, n))
        t_cand = _time(lambda: candidate_resolve(where, n))
        print(
            f"{n:>10} {t_ref * 1e6:>12.3f} {t_cand * 1e6:>12.3f} {t_ref / t_cand:>8.1f}x"
        )


def _end_to_end_sweep() -> None:
    """Full public-API gather Ragged.__getitem__(idx). Reflects whichever branch
    is currently compiled into _core.py (reference before Task 3, candidate after)."""
    print(f"\nend-to-end rag[idx], k={K} (us/call):")
    print(f"{'n':>10} {'rag[idx]':>12}")
    rng = np.random.default_rng(1)
    for n in N_SWEEP:
        rag = Ragged.from_lengths(
            np.arange(n, dtype=np.int32), np.ones(n, dtype=np.int64)
        )
        idx = rng.integers(0, n, size=K)
        t = _time(lambda: rag[idx])
        print(f"{n:>10} {t * 1e6:>12.3f}")


if __name__ == "__main__":
    _check_oracle()
    _resolve_sweep()
    _end_to_end_sweep()
    # Baseline number for the spec (n=65536, k=256):
    rng = np.random.default_rng(2)
    where = rng.integers(0, 65_536, size=K)
    print(
        f"\nbaseline @ n=65536 k=256: "
        f"reference={_time(lambda: reference_resolve(where, 65_536)) * 1e6:.2f} us  "
        f"candidate={_time(lambda: candidate_resolve(where, 65_536)) * 1e6:.2f} us"
    )
