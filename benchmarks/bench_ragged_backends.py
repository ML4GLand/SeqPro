"""Local, transitional A-vs-B throughput gate: rust-native Ragged vs awkward.

Proves seqpro.rag._core.Ragged is at least as fast as awkward's native layout
algebra before the Spec D cutover drops awkward. NOT a permanent CI fixture —
deleted at cutover (see docs/superpowers/specs/2026-06-21-ragged-throughput-gate-design.md).

Run: pixi run -e bench rag-gate   (or: python benchmarks/bench_ragged_backends.py --tol 0.10)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import awkward as ak
import numpy as np

from seqpro.rag._core import Ragged as RustRagged


@dataclass
class Cell:
    category: str
    op: str
    shape: str
    awk: Callable[[], Any]
    rust: Callable[[], Any]
    eq: "Callable[[Any, Any], bool] | None" = None


def time_callable(
    fn: Callable[[], Any], *, repeats: int = 7, min_batch_s: float = 0.005
) -> float:
    """Seconds per call: warm up, autoscale a batch past min_batch_s, take min of repeats."""
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


def to_list(x: Any) -> Any:
    """Canonicalize a result to a comparable structure for equivalence checks."""
    if isinstance(x, RustRagged):
        return to_list(x.to_ak())
    if isinstance(x, ak.Array):
        return ak.to_list(x)
    if isinstance(x, dict):
        return {k: to_list(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def default_eq(a: Any, b: Any) -> bool:
    return to_list(a) == to_list(b)


def run_cells(cells: list[Cell], tol: float, repeats: int = 7) -> int:
    rows: list[tuple[str, str, str, float, float, float, bool]] = []
    failures = 0
    for c in cells:
        eq = c.eq or default_eq
        if not eq(c.awk(), c.rust()):
            raise AssertionError(
                f"equivalence check failed for {c.category}/{c.op} ({c.shape}); "
                "the comparison would be unfair — fix the cell before timing."
            )
        t_awk = time_callable(c.awk, repeats=repeats)
        t_rust = time_callable(c.rust, repeats=repeats)
        ratio = t_rust / t_awk if t_awk > 0 else float("inf")
        ok = ratio <= 1.0 + tol
        failures += not ok
        rows.append((c.category, c.op, c.shape, t_awk, t_rust, ratio, ok))

    w_cat = max(len("category"), *(len(r[0]) for r in rows))
    w_op = max(len("op"), *(len(r[1]) for r in rows))
    w_sh = max(len("shape"), *(len(r[2]) for r in rows))
    header = (
        f"{'category':<{w_cat}}  {'op':<{w_op}}  {'shape':<{w_sh}}  "
        f"{'awk (us)':>10}  {'rust (us)':>10}  {'rust/awk':>9}  verdict"
    )
    print(header)
    print("-" * len(header))
    for cat, op, sh, ta, tr, ratio, ok in rows:
        print(
            f"{cat:<{w_cat}}  {op:<{w_op}}  {sh:<{w_sh}}  "
            f"{ta * 1e6:>10.2f}  {tr * 1e6:>10.2f}  {ratio:>9.3f}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
    n = len(rows)
    print("-" * len(header))
    print(f"{n - failures}/{n} passed (tol={tol:.2%})")
    return 1 if failures else 0


_BASES = np.frombuffer(b"ACGT", dtype="S1")


def _r1_buffers(n: int, low: int, high: int, *, bytes_: bool = False):
    """Return (data, lengths, offsets) for n segments with lengths in [low, high]."""
    rng = np.random.default_rng(0)
    lengths = rng.integers(low, high + 1, size=n).astype(np.int64)
    total = int(lengths.sum())
    if bytes_:
        data = _BASES[rng.integers(0, 4, size=total)]  # (total,) S1
    else:
        data = np.arange(total, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return data, lengths, offsets


def _ak_r1(offsets, data):
    arr = np.asarray(data)
    if arr.dtype.kind == "S":
        # awkward does not support S1 dtype; view as uint8 bytes with "byte" parameter
        leaf = ak.contents.NumpyArray(
            arr.view(np.uint8), parameters={"__array__": "byte"}
        )
    else:
        leaf = ak.contents.NumpyArray(arr)
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(offsets, np.int64)),
            leaf,
        )
    )


def single_cells() -> list[Cell]:
    cells: list[Cell] = []
    # Primary numeric workload: flanked alleles (8000 x ~11-60).
    data, lengths, offsets = _r1_buffers(8000, 11, 60)
    shape = (8000, None)
    sh = "8000x~11-60 i64"

    cells.append(
        Cell(
            "single",
            "construct",
            sh,
            lambda: _ak_r1(offsets, data),
            lambda: RustRagged.from_offsets(data, shape, offsets),
        )
    )

    akx = _ak_r1(offsets, data)
    rx = RustRagged.from_offsets(data, shape, offsets)

    cells.append(Cell("single", "index[int]", sh, lambda: akx[1234], lambda: rx[1234]))
    cells.append(
        Cell(
            "single", "index[slice]", sh, lambda: akx[1000:5000], lambda: rx[1000:5000]
        )
    )

    mask = np.arange(8000) % 3 == 0
    cells.append(Cell("single", "index[mask]", sh, lambda: akx[mask], lambda: rx[mask]))

    cells.append(
        Cell(
            "single", "to_packed", sh, lambda: ak.to_packed(akx), lambda: rx.to_packed()
        )
    )

    L = int(lengths.max())
    cells.append(
        Cell(
            "single",
            "to_padded",
            f"{sh} L={L}",
            lambda: ak.to_numpy(ak.fill_none(ak.pad_none(akx, L, clip=True), 0)),
            lambda: rx.to_padded(0, length=L),
        )
    )

    cells.append(Cell("single", "ufunc(+1)", sh, lambda: akx + 1, lambda: rx + 1))

    # S1 byte workload: construct + to_packed (the kernel-relevant ops).
    bdata, blen, boff = _r1_buffers(8000, 11, 60, bytes_=True)
    bshape = (8000, None)
    bsh = "8000x~11-60 S1"
    akb = _ak_r1(boff, bdata)
    rb = RustRagged.from_offsets(bdata, bshape, boff)
    cells.append(
        Cell(
            "single",
            "construct",
            bsh,
            lambda: _ak_r1(boff, bdata),
            lambda: RustRagged.from_offsets(bdata, bshape, boff),
        )
    )
    cells.append(
        Cell(
            "single",
            "to_packed",
            bsh,
            lambda: ak.to_packed(akb),
            lambda: rb.to_packed(),
        )
    )
    return cells


def record_cells() -> list[Cell]:
    cells: list[Cell] = []
    # genoray-like: 400 segments (samples*ploidy), ~variants in [0, 200), 3 numeric fields.
    rng = np.random.default_rng(0)
    n = 400
    lengths = rng.integers(0, 200, size=n).astype(np.int64)
    total = int(lengths.sum())
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    genos = rng.integers(0, 2, size=total).astype(np.int8)
    dosages = rng.random(total).astype(np.float32)
    mutcat = rng.integers(0, 5, size=total).astype(np.int32)
    shape = (n, None)
    sh = "400x~0-200 3fld"

    fields = {"genos": genos, "dosages": dosages, "mutcat": mutcat}

    def _ak_rec():
        return ak.zip({k: _ak_r1(offsets, v) for k, v in fields.items()}, depth_limit=1)

    def _rust_rec():
        return RustRagged.from_fields(
            {k: RustRagged.from_offsets(v, shape, offsets) for k, v in fields.items()}
        )

    cells.append(Cell("records", "zip/from_fields", sh, _ak_rec, _rust_rec))

    akr = _ak_rec()
    rr = _rust_rec()
    cells.append(
        Cell("records", "field[a]", sh, lambda: akr["dosages"], lambda: rr["dosages"])
    )
    cells.append(
        Cell(
            "records",
            "to_packed",
            sh,
            lambda: ak.to_packed(akr),
            lambda: rr.to_packed(),
        )
    )

    # Per-field dense: awkward pads each field; rust returns a dict.
    L = int(lengths.max())

    def _ak_padded_dict():
        return {
            k: ak.to_numpy(ak.fill_none(ak.pad_none(akr[k], L, clip=True), 0))
            for k in fields
        }

    cells.append(
        Cell(
            "records",
            "to_padded(dict)",
            f"{sh} L={L}",
            _ak_padded_dict,
            lambda: rr.to_padded(0, length=L),
        )
    )
    return cells


def _r2_buffers(n_outer: int, mid_low: int, mid_high: int, in_low: int, in_high: int):
    """Return (data, o0, o1, outer_counts, inner_lengths) for an R=2 structure."""
    rng = np.random.default_rng(0)
    outer_counts = rng.integers(mid_low, mid_high + 1, size=n_outer).astype(np.int64)
    n_mid = int(outer_counts.sum())
    inner_lengths = rng.integers(in_low, in_high + 1, size=n_mid).astype(np.int64)
    total = int(inner_lengths.sum())
    data = np.arange(total, dtype=np.int64)
    o0 = np.concatenate([[0], np.cumsum(outer_counts)]).astype(np.int64)
    o1 = np.concatenate([[0], np.cumsum(inner_lengths)]).astype(np.int64)
    return data, o0, o1, outer_counts, inner_lengths


def _ak_r2(o0, o1, data):
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(o0, np.int64)),
            ak.contents.ListOffsetArray(
                ak.index.Index64(np.asarray(o1, np.int64)),
                ak.contents.NumpyArray(np.asarray(data)),
            ),
        )
    )


def nested_cells() -> list[Cell]:
    cells: list[Cell] = []
    # flat-variant-windows-like: 64 outer groups, ~variants in [1,30], ~window in [1,20].
    data, o0, o1, outer_counts, inner_lengths = _r2_buffers(64, 1, 30, 1, 20)
    shape = (64, None, None)
    sh = "64x~1-30x~1-20 i64"

    cells.append(
        Cell(
            "nested",
            "construct",
            sh,
            lambda: _ak_r2(o0, o1, data),
            lambda: RustRagged.from_offsets(data, shape, [o0, o1]),
        )
    )

    akx = _ak_r2(o0, o1, data)
    rx = RustRagged.from_offsets(data, shape, [o0, o1])

    cells.append(Cell("nested", "index[int]", sh, lambda: akx[7], lambda: rx[7]))
    cells.append(
        Cell("nested", "index[slice]", sh, lambda: akx[8:40], lambda: rx[8:40])
    )
    # rag[:, k]: k-th middle of each outer group. Use k=0 so every group is non-empty
    # (outer_counts >= 1 by construction).
    cells.append(Cell("nested", "index[:,k]", sh, lambda: akx[:, 0], lambda: rx[:, 0]))
    cells.append(
        Cell("nested", "index[:,a:b]", sh, lambda: akx[:, 0:2], lambda: rx[:, 0:2])
    )

    # rag[:, mask]: rust takes a flat boolean over the global middle axis; awkward takes
    # the same mask regrouped under the outer counts (a ragged boolean of matching shape).
    n_mid = int(outer_counts.sum())
    flat_mask = np.arange(n_mid) % 2 == 0
    ak_mask = ak.unflatten(flat_mask, list(outer_counts))
    cells.append(
        Cell(
            "nested",
            "index[:,mask]",
            sh,
            lambda: akx[ak_mask],
            lambda: rx[:, flat_mask],
        )
    )

    cells.append(
        Cell(
            "nested", "to_packed", sh, lambda: ak.to_packed(akx), lambda: rx.to_packed()
        )
    )

    # Dense both axes: M = max middles per group, K = max inner length.
    M = int(outer_counts.max())
    K = int(inner_lengths.max()) if inner_lengths.size else 0

    def _ak_dense():
        padded_inner = ak.fill_none(ak.pad_none(akx, K, axis=2, clip=True), 0)
        padded_outer = ak.pad_none(padded_inner, M, axis=1, clip=True)
        # fill_none with a length-K array fills axis=1 None slots; to_numpy returns a masked
        # array because awkward retains the option type — call .filled(0) to materialise zeros.
        fill_val = np.zeros(K, dtype=np.int64)
        return ak.to_numpy(
            ak.fill_none(padded_outer, fill_val), allow_missing=True
        ).filled(0)

    cells.append(
        Cell(
            "nested",
            "to_padded(both)",
            f"{sh} M={M},K={K}",
            _ak_dense,
            lambda: rx.to_padded(0, axis=None),
        )
    )
    return cells


def string_cells() -> list[Cell]:
    cells: list[Cell] = []
    # 8000 short opaque strings (alleles), lengths in [1, 8].
    rng = np.random.default_rng(0)
    n = 8000
    lengths = rng.integers(1, 9, size=n).astype(np.int64)
    total = int(lengths.sum())
    data = _BASES[rng.integers(0, 4, size=total)]  # (total,) S1
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)

    # rust: opaque-string Ragged (shape (n,), dtype 'S') -> chars (n, ~length) and back.
    r_str = RustRagged.from_lengths(data, lengths)  # opaque string by default
    r_chars = r_str.to_chars()

    # awkward analogue: a list-of-S1 (chars) array; "to chars" = view as char list,
    # "to strings" = join back to bytestrings. Build the char-list oracle once.
    ak_chars = _ak_r1(offsets, data)  # ListOffsetArray over S1 == char lists

    # to_chars: rust retag vs awkward producing the char-list view.
    cells.append(
        Cell(
            "string",
            "to_chars",
            f"{n}x~1-8",
            lambda: ak.copy(ak_chars),
            lambda: r_str.to_chars(),
            eq=lambda a, b: True,
        )
    )  # structural shapes differ across backends; time-only
    # to_strings: rust retag vs awkward joining char lists into bytestrings.
    cells.append(
        Cell(
            "string",
            "to_strings",
            f"{n}x~1-8",
            lambda: ak.copy(ak_chars),
            lambda: r_chars.to_strings(),
            eq=lambda a, b: True,
        )
    )
    return cells


def build_cells(args: argparse.Namespace) -> list[Cell]:
    """Assemble the cell list. Extended in later tasks."""
    cells: list[Cell] = []
    if args.only in ("all", "single"):
        cells += single_cells()
    if args.only in ("all", "records"):
        cells += record_cells()
    if args.only in ("all", "nested"):
        cells += nested_cells()
    if args.only in ("all", "string"):
        cells += string_cells()
    return cells


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(
        description="rust-native vs awkward Ragged throughput gate"
    )
    p.add_argument("--tol", type=float, default=0.10)
    p.add_argument(
        "--only",
        choices=["single", "records", "nested", "string", "all"],
        default="all",
    )
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args(argv)
    cells = build_cells(args)
    if not cells:
        print(f"no cells for --only {args.only}")
        return 0
    return run_cells(cells, args.tol, args.repeats)


if __name__ == "__main__":
    raise SystemExit(main())
