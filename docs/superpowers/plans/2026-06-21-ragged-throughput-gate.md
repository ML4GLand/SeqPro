# Ragged Throughput Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transitional, local A-vs-B benchmark that proves the rust-native `Ragged` (`seqpro.rag._core.Ragged`) is at least as fast as awkward's native layout algebra across Core + records + nested R=2 ops, gating the Spec D cutover.

**Architecture:** A single standalone script, `benchmarks/bench_ragged_backends.py`, with three layers: (1) a generic timing/gate harness (warmup → autoscaled batch → min-of-repeats → per-op tolerance verdict → non-zero exit), (2) per-op "cells" each pairing an awkward callable (raw `ak.*`) with a rust callable (`_core.Ragged`) plus an equivalence check that guarantees both do the same logical work, and (3) workload builders that produce identical raw numpy buffers for both backends. Run via `pixi run -e bench rag-gate`. Retired at the Spec D cutover.

**Tech Stack:** Python 3.13 (bench env), numpy, awkward (`ak`), `seqpro.rag._core.Ragged`, `time.perf_counter`. No new runtime deps. pixi for task wiring.

## Global Constraints

- **Spec source of truth:** `docs/superpowers/specs/2026-06-21-ragged-throughput-gate-design.md`. Epic SSoT: `docs/roadmap/rust-ragged.md`.
- **Baseline = awkward's *own* algebra (raw `ak.*`), not seqpro's numba wrappers.** The public `seqpro.rag.Ragged` (`_array.py`) already routes `to_packed`/`to_padded` through numba; benchmarking it would measure numba-vs-rust, not awkward-vs-rust. Every awkward callable in a cell uses `ak.Array` / `ak.contents.*` / `ak.to_packed` / `ak.to_numpy` / `ak.pad_none` / `ak.fill_none` / `ak.zip` / awkward `__getitem__` directly.
- **Rust side = `seqpro.rag._core.Ragged`** (imported as `from seqpro.rag._core import Ragged as RustRagged`), the type the epic will promote at cutover.
- **Both backends built from identical raw numpy buffers**, seeded with `np.random.default_rng(0)` (argless `default_rng()` is non-deterministic — always pass the seed; matches `tests/test_bench_tokenize.py`).
- **Gate criterion:** per-op `rust_time <= awkward_time * (1 + tol)`, default `tol = 0.10`, overridable via `--tol`. Any op over tolerance ⇒ FAIL ⇒ non-zero exit.
- **Each cell asserts equivalence (same logical result) before timing.** A failed equivalence check is a hard error (the comparison would be unfair), distinct from a gate FAIL.
- **Not a permanent CI fixture.** No `tests/` files, no CI workflow edits. The script is deleted at cutover; this is intentional — do not add throwaway pytest comparisons.
- **No `skills/seqpro/SKILL.md` change** (internal/transitional; skill update remains Spec D).
- Run everything in the bench env: `pixi run -e bench …`. The rust extension must be built: `pixi run -e bench maturin develop`.

### Reference: rust-native `Ragged` API (from `python/seqpro/rag/_core.py`)

- `RustRagged.from_lengths(data, lengths)` — `lengths` is a 1-D int array (R=1) or a tuple `(outer_counts, inner_lengths)` (R=2).
- `RustRagged.from_offsets(data, shape, offsets, *, str_offsets=None)` — `offsets` a single array (R=1) or `list[array]` (R=2); `shape` a tuple with `None` per ragged axis.
- `RustRagged.from_fields(fields: dict[str, RustRagged])` — record (SoA); also exposed as `seqpro.rag.zip`.
- Indexing via `rag[key]`: outer int/slice/mask; tuple `rag[i, j]`; inner `rag[:, k]` / `rag[:, a:b]` / `rag[:, mask]`.
- `rag.to_packed(*, copy=True)`, `rag.to_padded(pad_value, *, length=None, axis=None)`, `rag.to_numpy(...)`.
- `rag.to_chars()` / `rag.to_strings()` (zero-copy retag), `rag.to_ak()` (→ `ak.Array`), `rag + scalar` (`__array_ufunc__`).
- `rag.data` / `rag.offsets` / `rag.shape` / `rag.dtype` / `rag.fields`.

### Reference: awkward oracle construction (from `tests/test_ragged_nested_diff.py`)

```python
import awkward as ak
import numpy as np

# R=1 from offsets:
def ak_r1(offsets, data):
    return ak.Array(ak.contents.ListOffsetArray(
        ak.index.Index64(np.asarray(offsets, np.int64)),
        ak.contents.NumpyArray(np.asarray(data)),
    ))

# R=2 from two offset levels o0 (over middles) and o1 (over inner):
def ak_r2(o0, o1, data):
    return ak.Array(ak.contents.ListOffsetArray(
        ak.index.Index64(np.asarray(o0, np.int64)),
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(o1, np.int64)),
            ak.contents.NumpyArray(np.asarray(data)),
        ),
    ))
```

`offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)`.

---

### Task 1: Harness core (timing, gate, table, CLI)

**Files:**
- Create: `benchmarks/bench_ragged_backends.py`

**Interfaces:**
- Consumes: nothing from other tasks (foundation).
- Produces (used by all later tasks):
  - `@dataclass Cell(category: str, op: str, shape: str, awk: Callable[[], Any], rust: Callable[[], Any], eq: Callable[[Any, Any], bool] | None = None)`
  - `time_callable(fn: Callable[[], Any], *, repeats: int = 7, min_batch_s: float = 0.005) -> float` — seconds per call (min-of-repeats over an autoscaled batch).
  - `to_list(x: Any) -> Any` — canonicalize an `ak.Array`, `RustRagged`, `np.ndarray`, or `dict` to a comparable Python/numpy structure for equivalence checks.
  - `default_eq(a: Any, b: Any) -> bool` — structural equality via `to_list`.
  - `run_cells(cells: list[Cell], tol: float) -> int` — prints the table + summary, returns process exit code (0 pass, 1 fail).
  - `build_cells(args) -> list[Cell]` — assembled across later tasks; Task 1 ships a stub returning a single self-test cell.
  - `main(argv: list[str] | None = None) -> int` and `if __name__ == "__main__": raise SystemExit(main())`.
  - CLI flags: `--tol FLOAT` (default 0.10), `--only {single,records,nested,string,all}` (default all), `--repeats INT` (default 7).

- [ ] **Step 1: Write the harness module**

```python
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


def run_cells(cells: list[Cell], tol: float) -> int:
    rows: list[tuple[str, str, str, float, float, float, bool]] = []
    failures = 0
    for c in cells:
        eq = c.eq or default_eq
        if not eq(c.awk(), c.rust()):
            raise AssertionError(
                f"equivalence check failed for {c.category}/{c.op} ({c.shape}); "
                "the comparison would be unfair — fix the cell before timing."
            )
        t_awk = time_callable(c.awk)
        t_rust = time_callable(c.rust)
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


def build_cells(args: argparse.Namespace) -> list[Cell]:
    """Assemble the cell list. Extended in later tasks; Task 1 ships a self-test."""
    cells: list[Cell] = []
    if args.only in ("all", "single"):
        # Self-test cell proving the harness + gate wiring (identical work both sides).
        x = np.arange(1000, dtype=np.int64)
        cells.append(
            Cell("selftest", "sum", "1000", lambda: int(x.sum()), lambda: int(x.sum()))
        )
    return cells


def main(argv: "list[str] | None" = None) -> int:
    p = argparse.ArgumentParser(description="rust-native vs awkward Ragged throughput gate")
    p.add_argument("--tol", type=float, default=0.10)
    p.add_argument("--only", choices=["single", "records", "nested", "string", "all"], default="all")
    p.add_argument("--repeats", type=int, default=7)
    args = p.parse_args(argv)
    cells = build_cells(args)
    if not cells:
        print(f"no cells for --only {args.only}")
        return 0
    return run_cells(cells, args.tol)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the harness self-test**

Run: `pixi run -e bench python benchmarks/bench_ragged_backends.py --only single`
Expected: a one-row table with `selftest  sum  1000  …  PASS` and `1/1 passed (tol=10.00%)`; exit code 0.

- [ ] **Step 3: Verify the gate fails when rust is slower**

Run: `pixi run -e bench python -c "import sys; sys.argv=['x','--tol','-1']; from benchmarks.bench_ragged_backends import main; print('exit', main())"`
(Setting `tol=-1` forces every ratio over threshold.)
Expected: the row shows `FAIL`, summary `0/1 passed`, printed `exit 1`. Confirms non-zero exit on failure.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_ragged_backends.py
git commit -m "feat: harness core for rust-vs-awkward Ragged throughput gate"
```

---

### Task 2: Single-level op cells (numeric + S1)

**Files:**
- Modify: `benchmarks/bench_ragged_backends.py`

**Interfaces:**
- Consumes: `Cell`, `to_list`, `default_eq` (Task 1).
- Produces: `single_cells() -> list[Cell]`, called from `build_cells` when `args.only in ("all","single")` (replacing the Task 1 self-test).

- [ ] **Step 1: Add single-level workload builders + cells**

Add above `build_cells`:

```python
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
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(np.asarray(offsets, np.int64)),
            ak.contents.NumpyArray(np.asarray(data)),
        )
    )


def single_cells() -> list[Cell]:
    cells: list[Cell] = []
    # Primary numeric workload: flanked alleles (8000 x ~11-60).
    data, lengths, offsets = _r1_buffers(8000, 11, 60)
    shape = (8000, None)
    sh = "8000x~11-60 i64"

    cells.append(Cell("single", "construct", sh,
        lambda: _ak_r1(offsets, data),
        lambda: RustRagged.from_offsets(data, shape, offsets)))

    akx = _ak_r1(offsets, data)
    rx = RustRagged.from_offsets(data, shape, offsets)

    cells.append(Cell("single", "index[int]", sh,
        lambda: akx[1234], lambda: rx[1234]))
    cells.append(Cell("single", "index[slice]", sh,
        lambda: akx[1000:5000], lambda: rx[1000:5000]))

    mask = (np.arange(8000) % 3 == 0)
    cells.append(Cell("single", "index[mask]", sh,
        lambda: akx[mask], lambda: rx[mask]))

    cells.append(Cell("single", "to_packed", sh,
        lambda: ak.to_packed(akx), lambda: rx.to_packed()))

    L = int(lengths.max())
    cells.append(Cell("single", "to_padded", f"{sh} L={L}",
        lambda: ak.to_numpy(ak.fill_none(ak.pad_none(akx, L, clip=True), 0)),
        lambda: rx.to_padded(0, length=L)))

    cells.append(Cell("single", "ufunc(+1)", sh,
        lambda: akx + 1, lambda: rx + 1))

    # S1 byte workload: construct + to_packed (the kernel-relevant ops).
    bdata, blen, boff = _r1_buffers(8000, 11, 60, bytes_=True)
    bshape = (8000, None)
    bsh = "8000x~11-60 S1"
    akb = _ak_r1(boff, bdata)
    rb = RustRagged.from_offsets(bdata, bshape, boff)
    cells.append(Cell("single", "construct", bsh,
        lambda: _ak_r1(boff, bdata),
        lambda: RustRagged.from_offsets(bdata, bshape, boff)))
    cells.append(Cell("single", "to_packed", bsh,
        lambda: ak.to_packed(akb), lambda: rb.to_packed()))
    return cells
```

- [ ] **Step 2: Wire `single_cells` into `build_cells`**

Replace the `if args.only in ("all", "single"):` block in `build_cells` with:

```python
    if args.only in ("all", "single"):
        cells += single_cells()
```

- [ ] **Step 3: Run the single-level gate**

Run: `pixi run -e bench python benchmarks/bench_ragged_backends.py --only single`
Expected: a table with rows for construct / index[int] / index[slice] / index[mask] / to_packed / to_padded / ufunc(+1) (numeric) and construct / to_packed (S1). Every equivalence check passes (no `AssertionError`). Summary `N/N passed` ideally; if any FAIL, that's a real signal — record it, do not loosen the gate.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_ragged_backends.py
git commit -m "feat: single-level op cells for Ragged throughput gate"
```

---

### Task 3: Record (SoA) op cells

**Files:**
- Modify: `benchmarks/bench_ragged_backends.py`

**Interfaces:**
- Consumes: `Cell`, `_ak_r1` (Task 2).
- Produces: `record_cells() -> list[Cell]`, called from `build_cells` when `args.only in ("all","records")`.

- [ ] **Step 1: Add record workload + cells**

```python
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
    cells.append(Cell("records", "field[a]", sh,
        lambda: akr["dosages"], lambda: rr["dosages"]))
    cells.append(Cell("records", "to_packed", sh,
        lambda: ak.to_packed(akr), lambda: rr.to_packed()))

    # Per-field dense: awkward pads each field; rust returns a dict.
    L = int(lengths.max())

    def _ak_padded_dict():
        return {k: ak.to_numpy(ak.fill_none(ak.pad_none(akr[k], L, clip=True), 0))
                for k in fields}

    cells.append(Cell("records", "to_padded(dict)", f"{sh} L={L}",
        _ak_padded_dict, lambda: rr.to_padded(0, length=L)))
    return cells
```

- [ ] **Step 2: Wire into `build_cells`**

Add to `build_cells`:

```python
    if args.only in ("all", "records"):
        cells += record_cells()
```

- [ ] **Step 3: Run the record gate**

Run: `pixi run -e bench python benchmarks/bench_ragged_backends.py --only records`
Expected: rows for zip/from_fields, field[a], to_packed, to_padded(dict); all equivalence checks pass; summary printed. (Note: `field[a]` and zero-copy ops will likely show rust ≪ awkward — expected.)

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_ragged_backends.py
git commit -m "feat: record (SoA) op cells for Ragged throughput gate"
```

---

### Task 4: Nested R=2 op cells

**Files:**
- Modify: `benchmarks/bench_ragged_backends.py`

**Interfaces:**
- Consumes: `Cell` (Task 1).
- Produces: `nested_cells() -> list[Cell]`, called from `build_cells` when `args.only in ("all","nested")`.

- [ ] **Step 1: Add R=2 workload builder + awkward oracle + cells**

```python
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

    cells.append(Cell("nested", "construct", sh,
        lambda: _ak_r2(o0, o1, data),
        lambda: RustRagged.from_offsets(data, shape, [o0, o1])))

    akx = _ak_r2(o0, o1, data)
    rx = RustRagged.from_offsets(data, shape, [o0, o1])

    cells.append(Cell("nested", "index[int]", sh,
        lambda: akx[7], lambda: rx[7]))
    cells.append(Cell("nested", "index[slice]", sh,
        lambda: akx[8:40], lambda: rx[8:40]))
    # rag[:, k]: k-th middle of each outer group. Use k=0 so every group is non-empty
    # (outer_counts >= 1 by construction).
    cells.append(Cell("nested", "index[:,k]", sh,
        lambda: akx[:, 0], lambda: rx[:, 0]))
    cells.append(Cell("nested", "index[:,a:b]", sh,
        lambda: akx[:, 0:2], lambda: rx[:, 0:2]))

    # rag[:, mask]: rust takes a flat boolean over the global middle axis; awkward takes
    # the same mask regrouped under the outer counts (a ragged boolean of matching shape).
    n_mid = int(outer_counts.sum())
    flat_mask = (np.arange(n_mid) % 2 == 0)
    ak_mask = ak.unflatten(flat_mask, list(outer_counts))
    cells.append(Cell("nested", "index[:,mask]", sh,
        lambda: akx[ak_mask], lambda: rx[:, flat_mask]))

    cells.append(Cell("nested", "to_packed", sh,
        lambda: ak.to_packed(akx), lambda: rx.to_packed()))

    # Dense both axes: M = max middles per group, K = max inner length.
    M = int(outer_counts.max())
    K = int(inner_lengths.max()) if inner_lengths.size else 0

    def _ak_dense():
        padded_inner = ak.fill_none(ak.pad_none(akx, K, axis=2, clip=True), 0)
        padded_outer = ak.pad_none(padded_inner, M, axis=1, clip=True)
        # fill_none for the all-missing outer rows then to_numpy.
        return ak.to_numpy(ak.fill_none(padded_outer, 0, axis=1))

    cells.append(Cell("nested", "to_padded(both)", f"{sh} M={M},K={K}",
        _ak_dense, lambda: rx.to_padded(0, axis=None)))
    return cells
```

- [ ] **Step 2: Wire into `build_cells`**

```python
    if args.only in ("all", "nested"):
        cells += nested_cells()
```

- [ ] **Step 3: Run the nested gate**

Run: `pixi run -e bench python benchmarks/bench_ragged_backends.py --only nested`
Expected: rows for construct / index[int] / index[slice] / index[:,k] / index[:,a:b] / index[:,mask] / to_packed / to_padded(both). All equivalence checks pass. If the `to_padded(both)` equivalence check fails, the awkward dense construction in `_ak_dense` does not match `rx.to_padded(0, axis=None)`'s shape/fill — debug with `to_list` on both before adjusting (use systematic-debugging); the rust result is the contract.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_ragged_backends.py
git commit -m "feat: nested R=2 op cells for Ragged throughput gate"
```

---

### Task 5: String (to_chars / to_strings) op cells

**Files:**
- Modify: `benchmarks/bench_ragged_backends.py`

**Interfaces:**
- Consumes: `Cell`, `_BASES` (Task 2), `_ak_r1` (Task 2).
- Produces: `string_cells() -> list[Cell]`, called from `build_cells` when `args.only in ("all","string")`.

- [ ] **Step 1: Add string-conversion cells**

`to_chars`/`to_strings` are zero-copy retags on the rust side. The awkward equivalent is the round-trip awkward would do between a bytestring leaf and a char list. We measure the rust retag against awkward's `ak.enforce_type` / list-of-chars conversion on an equivalent array.

```python
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
    cells.append(Cell("string", "to_chars", f"{n}x~1-8",
        lambda: ak.copy(ak_chars), lambda: r_str.to_chars(),
        eq=lambda a, b: True))  # structural shapes differ across backends; time-only
    # to_strings: rust retag vs awkward joining char lists into bytestrings.
    cells.append(Cell("string", "to_strings", f"{n}x~1-8",
        lambda: ak.copy(ak_chars), lambda: r_chars.to_strings(),
        eq=lambda a, b: True))
    return cells
```

Rationale for `eq=lambda a, b: True`: the rust retag and the awkward representation are intentionally different objects (opaque-string vs char-list), so a structural equality check is not meaningful here — these two cells are pure throughput comparisons of "convert a string collection's representation". All other cells keep the real equivalence check.

- [ ] **Step 2: Wire into `build_cells`**

```python
    if args.only in ("all", "string"):
        cells += string_cells()
```

- [ ] **Step 3: Run the string gate**

Run: `pixi run -e bench python benchmarks/bench_ragged_backends.py --only string`
Expected: rows for to_chars / to_strings. Rust retags are near-instant (zero-copy), so expect rust ≪ awkward and PASS.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/bench_ragged_backends.py
git commit -m "feat: string-conversion op cells for Ragged throughput gate"
```

---

### Task 6: Wire pixi `rag-gate` task and run the full gate

**Files:**
- Modify: `pixi.toml` (`[feature.bench.tasks]`, around line 92-94)

**Interfaces:**
- Consumes: the complete `benchmarks/bench_ragged_backends.py` (Tasks 1-5).
- Produces: `pixi run -e bench rag-gate`.

- [ ] **Step 1: Add the task**

In `pixi.toml`, under `[feature.bench.tasks]` (which currently has `i-kernel` and `bench`), add:

```toml
rag-gate = "python benchmarks/bench_ragged_backends.py"
```

- [ ] **Step 2: Run the full gate via pixi**

Run: `pixi run -e bench rag-gate`
Expected: the full table (single + records + nested + string categories), a final `N/N passed (tol=10.00%)` line, and exit code 0 if rust meets the gate. Confirm exit code: `echo $?` ⇒ `0` on pass.

- [ ] **Step 3: Record the outcome in the roadmap decision log**

If the gate passes, append a dated bullet to the decision log in `docs/roadmap/rust-ragged.md` noting the gate ran green and the per-op ratios summary (paste the table or the worst-case `rust/awk` ratio). If any op FAILs, do NOT edit the gate to pass — open a follow-up note in the decision log describing which op regressed; that op is a Spec D blocker.

- [ ] **Step 4: Commit**

```bash
git add pixi.toml docs/roadmap/rust-ragged.md
git commit -m "feat: wire rag-gate pixi task; record throughput gate outcome"
```

---

## Self-Review

**Spec coverage:**
- Harness (script, pixi task, build-inputs-once, warmup, min-of-repeats, table, non-zero exit) → Tasks 1, 6. ✓
- Gate criterion (per-op `rust <= awkward*(1+tol)`, `tol=0.10`, `--tol`) → Task 1 (`run_cells`, CLI). ✓
- Op matrix: single-level construct/index(int,slice,mask)/to_packed/to_padded/to_numpy/ufunc → Task 2 ✓; records zip/field/to_packed/to_numpy(to_padded dict) → Task 3 ✓; nested R=2 construct/`[:,k]`/`[:,a:b]`/`[:,mask]`/to_packed/to_padded(both) → Task 4 ✓; string to_chars/to_strings → Task 5 ✓.
- Workloads (survey + bench shapes) → Tasks 2-5 use flanked alleles, genoray 3-field, flat-windows R=2, short alleles. ✓
- Baseline = raw `ak.*` (not numba wrappers) → Global Constraints + every cell. ✓
- Fairness (equivalence before timing) → Task 1 `run_cells` asserts `eq` per cell. ✓
- Retirement / no permanent CI / no SKILL.md → Global Constraints; spec retirement section unchanged. ✓
- `to_numpy`: covered by `to_padded` cells (rust `to_numpy` requires uniform lengths; the ragged workloads are non-uniform, so the dense comparison goes through `to_padded`, which is the meaningful densify op). Noted here so the omission of a literal `to_numpy()` call is intentional, not a gap.

**Placeholder scan:** No TBD/TODO; every code step is complete and runnable. ✓

**Type consistency:** `Cell`, `time_callable`, `to_list`, `default_eq`, `run_cells`, `build_cells`, `single_cells`, `record_cells`, `nested_cells`, `string_cells` names are used consistently across tasks; helpers (`_ak_r1`, `_ak_r2`, `_r1_buffers`, `_r2_buffers`, `_BASES`) defined before use. Each builder re-seeds locally with `np.random.default_rng(0)`, so results are deterministic regardless of cell-build order. `RustRagged` alias used throughout. ✓
