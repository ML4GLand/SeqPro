"""Microbenchmark: seqpro.rag.to_packed vs ak.to_packed vs NumPy gather.

Run in the bench env:
    pixi run -e bench python benchmarks/bench_to_packed.py --out bench_to_packed

Outputs <out>.csv and <out>_*.png. Sweeps n_rows, mean length, length
distribution, source (RAM vs memmap), dtype itemsize, and thread count, and
prints an effect-size ranking of which axes most affect the speedup ratio.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def make_lengths(rng, n_rows, mean_len, distribution):
    if distribution == "uniform":
        lo, hi = max(1, mean_len // 2), mean_len * 2
        return rng.integers(lo, hi + 1, size=n_rows)
    # long-tailed: most rows short, a few very long (load imbalance)
    lengths = rng.integers(1, max(2, mean_len // 4), size=n_rows)
    n_big = max(1, n_rows // 100)
    lengths[rng.choice(n_rows, n_big, replace=False)] = mean_len * 50
    return lengths


def build(rng, n_rows, mean_len, distribution, dtype, source, tmpdir):
    import seqpro.rag as spr

    lengths = make_lengths(rng, n_rows, mean_len, distribution).astype(np.int64)
    total = int(lengths.sum())
    if source == "memmap":
        path = Path(tmpdir) / "data.dat"
        data = np.memmap(path, dtype=dtype, mode="w+", shape=(total,))
        data[:] = rng.integers(0, 255, size=total).astype(dtype)
    else:
        data = rng.integers(0, 255, size=total).astype(dtype)
    rag = spr.Ragged.from_lengths(data, lengths)
    # reorder so the layout is an (unpacked) ListArray, forcing a real gather
    return rag[rng.permutation(n_rows)], total * np.dtype(dtype).itemsize


def numpy_gather(rag):
    import seqpro.rag as spr

    parts = rag._parts
    offs = parts.offsets
    starts, stops = (offs[0], offs[1]) if offs.ndim == 2 else (offs[:-1], offs[1:])
    lengths = stops - starts
    out_off = np.empty(lengths.size + 1, dtype=np.int64)
    out_off[0] = 0
    np.cumsum(lengths, out=out_off[1:])
    idx = np.repeat(starts - out_off[:-1], lengths) + np.arange(int(out_off[-1]))
    return spr.Ragged.from_offsets(parts.data[idx], parts.shape, out_off)


def timeit(fn, arg, repeats):
    fn(arg)  # warm up (JIT, caches)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="bench_to_packed")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--n-rows", type=int, nargs="+", default=[1000, 10000, 100000, 1000000]
    )
    p.add_argument("--mean-len", type=int, nargs="+", default=[16, 256, 4096])
    p.add_argument("--distributions", nargs="+", default=["uniform", "longtail"])
    p.add_argument("--sources", nargs="+", default=["ram", "memmap"])
    p.add_argument("--dtypes", nargs="+", default=["uint8", "float64"])
    p.add_argument("--threads", type=int, nargs="+", default=[int(os.cpu_count() or 1)])
    args = p.parse_args()

    import awkward as ak
    import pandas as pd
    import seqpro.rag as spr  # noqa: F401  (ensures kernel import)
    import tempfile

    rng = np.random.default_rng(0)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for nthreads in args.threads:
            import numba

            numba.set_num_threads(nthreads)
            impls = {
                "seqpro": lambda r: r.to_packed(),
                "ak": lambda r: spr.Ragged(ak.to_packed(r)),
                "numpy": numpy_gather,
            }
            for n_rows in args.n_rows:
                for mean_len in args.mean_len:
                    for dist in args.distributions:
                        for src in args.sources:
                            for dt in args.dtypes:
                                rag, nbytes = build(
                                    rng, n_rows, mean_len, dist, dt, src, tmp
                                )
                                for name, fn in impls.items():
                                    t = timeit(fn, rag, args.repeats)
                                    rows.append(
                                        dict(
                                            impl=name,
                                            n_rows=n_rows,
                                            mean_len=mean_len,
                                            distribution=dist,
                                            source=src,
                                            dtype=dt,
                                            threads=nthreads,
                                            seconds=t,
                                            gbps=nbytes / t / 1e9,
                                        )
                                    )

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False)
    _plots_and_summary(df, args.out)


def _plots_and_summary(df, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # throughput vs n_rows, faceted by mean_len x dtype, hue impl
    g = sns.relplot(
        data=df,
        x="n_rows",
        y="gbps",
        hue="impl",
        style="source",
        col="mean_len",
        row="dtype",
        kind="line",
        marker="o",
        facet_kws=dict(sharey=False),
    )
    g.set(xscale="log")
    g.savefig(f"{out}_throughput.png", dpi=120)

    # thread scaling for seqpro
    if df["threads"].nunique() > 1:
        sub = df[df.impl == "seqpro"]
        gt = sns.relplot(
            data=sub, x="threads", y="gbps", hue="mean_len", kind="line", marker="o"
        )
        gt.savefig(f"{out}_threads.png", dpi=120)
    plt.close("all")

    # effect-size ranking on speedup (seqpro / ak) — goal 2
    wide = df.pivot_table(
        index=["n_rows", "mean_len", "distribution", "source", "dtype", "threads"],
        columns="impl",
        values="gbps",
    ).reset_index()
    wide["speedup"] = wide["seqpro"] / wide["ak"]
    print("\n=== Speedup (seqpro / ak) summary ===")
    print(wide["speedup"].describe())
    print("\n=== Mean speedup by axis (effect size) ===")
    axes = ["n_rows", "mean_len", "distribution", "source", "dtype", "threads"]
    spreads = {}
    for ax in axes:
        means = wide.groupby(ax)["speedup"].mean()
        spreads[ax] = means.max() - means.min()
        print(f"\n{ax}:")
        print(means)
    print("\n=== Axes ranked by speedup spread (most influential first) ===")
    for ax, s in sorted(spreads.items(), key=lambda kv: -kv[1]):
        print(f"  {ax:14s} spread={s:.2f}")


if __name__ == "__main__":
    main()
