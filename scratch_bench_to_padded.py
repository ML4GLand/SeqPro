"""Microbench: flat-buffer to_padded vs the awkward rpad/pad_none idiom."""

import time
import tracemalloc

import awkward.operations.str as ak_str
import numpy as np

from seqpro.rag import Ragged, to_padded


def _naive_pad_bytes(rag, pad_value, length):
    return Ragged(ak_str.rpad(rag, length, pad_value)).to_numpy()


def main():
    rng = np.random.default_rng(0)
    n, max_len = 1024, 4096
    lengths = rng.integers(max_len // 2, max_len, size=n).astype(np.uint32)
    total = int(lengths.sum())
    data = np.frombuffer(
        b"".join(rng.choice([b"A", b"C", b"G", b"T"], size=total)), dtype="S1"
    )
    rag = Ragged.from_lengths(data, lengths)
    L = int(lengths.max())

    # warmup (jit)
    to_padded(rag, b"N")
    _naive_pad_bytes(rag, b"N", L)

    # correctness: flat kernel must match the awkward baseline byte-for-byte
    assert np.array_equal(to_padded(rag, b"N"), _naive_pad_bytes(rag, b"N", L)), (
        "MISMATCH vs awkward baseline!"
    )

    def bench(fn, *a, rep=20):
        ts = []
        for _ in range(rep):
            t = time.perf_counter()
            fn(*a)
            ts.append(time.perf_counter() - t)
        return min(ts) * 1e3

    def peak(fn, *a):
        tracemalloc.start()
        fn(*a)
        p = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return p / 1e6

    t_old = bench(_naive_pad_bytes, rag, b"N", L)
    t_new = bench(to_padded, rag, b"N")
    print(f"batch {n} rows x ~{max_len} b ({total / 1e6:.1f} MB), pad to {L}")
    print(
        f"awkward (old):    {t_old:.3f} ms/call   peak +{peak(_naive_pad_bytes, rag, b'N', L):.2f} MB"
    )
    print(
        f"flat numba (new): {t_new:.3f} ms/call   peak +{peak(to_padded, rag, b'N'):.2f} MB"
    )
    print(f"speedup: {t_old / t_new:.1f}x")


if __name__ == "__main__":
    main()
