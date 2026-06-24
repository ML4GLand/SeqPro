"""Standalone: re-run the getitem hot-path bench vs the design gates.
Run: pixi run -e dev python tests/bench_ragged_getitem.py
Gates: slice within ~1.2x of a bare numpy view-slice baseline;
       to_numpy(validate=False) within ~1.2x of a bare reshape;
       from_offsets within ~1.1x of a bare dataclass wrap."""

import time
import numpy as np
from seqpro.rag import Ragged
from seqpro.rag._utils import lengths_to_offsets


def bench(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6


def make(B=128, P=2, L=2000):
    rng = np.random.default_rng(0)
    lengths = rng.integers(L // 2, L, size=B * P).astype(np.int64)
    off = lengths_to_offsets(lengths)
    data = rng.integers(0, 1000, size=int(off[-1])).astype(np.int32)
    return data, off, (B, P, None)


def make_uniform(B=128, P=2, L=2000):
    off = np.arange(B * P + 1, dtype=np.int64) * L
    data = np.arange(B * P * L, dtype=np.int32)
    return data, off, (B, P, None)


N = 20000
data, off, shape = make()
r = Ragged.from_offsets(data, shape, off)
t_from_offsets = bench(lambda: Ragged.from_offsets(data, shape, off), N)
t_slice = bench(lambda: r[16:112], N)
slice_contiguous = r[16:112].is_contiguous
print(f"from_offsets : {t_from_offsets:.3f} us")
print(f"slice [16:112]: {t_slice:.3f} us  contiguous={slice_contiguous}")
data, off, shape = make_uniform()
ru = Ragged.from_offsets(data, shape, off)
t_to_numpy_f = bench(lambda: ru.to_numpy(validate=False), N)
t_to_numpy_t = bench(lambda: ru.to_numpy(), N)
print(f"to_numpy(v=F): {t_to_numpy_f:.3f} us")
print(f"to_numpy(v=T): {t_to_numpy_t:.3f} us")
