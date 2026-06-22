# Spec: awkward-vs-rust `Ragged` throughput gate

**Status:** Designed (awaiting implementation plan)
**Date:** 2026-06-21
**Epic:** Rust-native `Ragged` (see `docs/roadmap/rust-ragged.md` — SSoT)
**Relation:** Concretizes the "benchmark vs the old awkward path" line in **Spec D**;
this gate must pass *before* the Spec D cutover removes `awkward`.

## Purpose

A **transitional, local milestone gate** proving the rust-native `Ragged`
(`seqpro.rag._core.Ragged`) is at least as fast as the awkward-backed
implementation across the operations consumers rely on, *before* Spec D cuts over
and drops `awkward` from the dependency tree.

It runs while both backings still coexist (today: public `seqpro.rag.Ragged` is
awkward-backed via `_array.py`; the rust-native type lives in `_core.py`). Once
the gate is green:

1. Spec D proceeds with the cutover.
2. The gate script and **all** awkward-comparison code are **retired**.
3. The rust-side op timings fold into the existing CodSpeed bench
   (`tests/test_bench_*`) for forward regression tracking.

This is explicitly **not** a permanent CI fixture. CodSpeed (instruction-count,
regression-over-time) remains the forward-looking guard; this gate is a one-time
A-vs-B comparison that exists only during the coexistence window.

## Non-goals

- No permanent CI integration. The gate is run locally, by hand, once.
- No CodSpeed A/B gating (CodSpeed's model tracks single-benchmark regression, not
  A-vs-B; not a fit for this comparison).
- No public API change; no `skills/seqpro/SKILL.md` update (internal/transitional,
  consistent with Spec B/C — the skill update remains Spec D).

## Harness

A standalone script: **`benchmarks/bench_ragged_backends.py`**, invoked via a new
`[feature.bench.tasks]` entry, e.g. `pixi run -e bench rag-gate`.

Chosen over a pytest test because the gate is explicitly transitional: a script
reads top-to-bottom as a report, runs by hand trivially, and is deleted wholesale
at cutover without leaving throwaway awkward-vs-rust comparisons in the permanent
suite.

For each `(category, op, workload)` cell:

1. **Build inputs once, outside timing**, from identical raw numpy buffers with a
   pinned `np.random.default_rng(0)` (matches existing bench convention). Awkward
   inputs are built with `ak.Array` / `ak.contents.*` exactly as the differential
   test oracles do (`tests/test_ragged_nested_diff.py`); rust inputs via
   `_core.Ragged` constructors (or the `_ingest.layout_from_ak` bridge). Both
   backends therefore measure the **same logical work** on the **same buffers**.
2. **Warm up** with a few untimed calls (numba / rust-kernel / cache warm).
3. **Measure** wall-clock via `time.perf_counter`: autoscale inner iteration count
   so each timed batch clears a floor (~a few ms), run `R` repeats, report the
   **min** (least-noise estimator for "best achievable").
4. **Report** a table row: `category | op | shape | awkward | rust | ratio
   (rust/awk) | verdict`.

After all cells: print a summary (`X/Y passed`) and **exit non-zero** if any op
fails the gate.

## Gate criterion

Per-op, wall-clock:

```
rust_time <= awkward_time * (1 + tol)
```

- Default `tol = 0.10` (10%), overridable via `--tol` on the CLI.
- Ratios (`rust/awk`) are always printed so near-ties are visible.
- Any op over tolerance ⇒ FAIL ⇒ non-zero exit code.

Per-op (not aggregate) so a single regressed op cannot be masked by faster ones.
Tolerance (vs strict) absorbs wall-clock jitter on near-ties while still catching
real regressions.

## Operation matrix (Core + records + nested R=2)

**Single-level (numeric + `S1`):**
- construct-from-buffers
- index: int row
- index: slice
- index: boolean mask (gather)
- `to_packed`
- `to_padded` / `to_numpy`
- ufunc (elementwise, numeric — e.g. add scalar; awkward broadcasts, rust applies
  numpy to `.data`)

**Records (SoA, shared offsets):**
- `from_fields` / `zip` construct
- field access (`rag["a"]`)
- record `to_packed`
- record `to_numpy` / `to_padded` (per-field SoA dicts)

**Nested R=2:**
- nested construct
- `rag[:, k]` (uniform int — k-th middle of each group)
- `rag[:, a:b]` (per-group slice)
- `rag[:, mask]` (mask over global middle axis)
- nested `to_packed`
- nested `to_padded(axis=None)` / `to_numpy`

**String:**
- `to_chars` / `to_strings` (zero-copy retag), single-level and nested.

### Fairness note

The public awkward `Ragged` wrapper is restricted to **one** ragged level, so the
records and R=2 rows compare **awkward-native ops (`ak.*`) vs rust-native** — the
comparison acknowledged as acceptable during design (these patterns live *outside*
the awkward `Ragged` wrapper in real consumer code today anyway). Zero-copy rust
ops (field access, `to_chars`/`to_strings`) will be lopsidedly fast versus the
awkward equivalent; that is a true reflection of the new design, not a measurement
artifact. Each timed callable must perform the **equivalent logical work** on both
sides (e.g. awkward `to_packed` ⇒ `ak.to_packed`; awkward dense ⇒ `ak.to_numpy` /
`ak.pad_none`+fill).

## Workloads

Representative sizes drawn from the consumer shape survey (roadmap) and the
existing `tests/test_bench_tokenize.py` shapes:

| Category | Workload | Shape |
|---|---|---|
| Single-level | short alleles | 8000 × ~1–4 bp |
| Single-level | flanked alleles | 8000 × ~11–60 bp |
| Single-level | CREs | 500 × ~100–200 bp |
| Records | genoray-like 3-field SoA | n ≈ 8000 at a `~variants` level |
| Nested R=2 | alleles (string-under-axis) | `(batch, ploidy, ~variants, ~allele_len)` |
| Nested R=2 | flat variant windows | `(batch, ploidy, ~variants, ~window)` |
| Nested R=2 | codon-annotation record | `(regions, ~genes, ~codons)` |

One primary workload per op to keep total runtime in the seconds range; exact
sizes finalized in the implementation plan.

## Output artifact

- Console table + summary (above).
- Non-zero exit on any per-op failure (the gate signal).
- Optional: write the table to a markdown/JSON file under `benchmarks/` for the
  record (decided in the plan; not required for the gate to function).

## Retirement plan (Spec D)

At the Spec D cutover:

1. Delete `benchmarks/bench_ragged_backends.py` and the `rag-gate` pixi task.
2. Migrate the rust-side op timings into the CodSpeed bench (`tests/test_bench_*`)
   for forward regression tracking.
3. Log the gate outcome and its retirement in the roadmap decision log.

The gate must be run **before** `import awkward` is removed (it depends on both
backings being importable).

## Risks / mitigations

- **Wall-clock noise** → warmup + autoscaled batches + min-of-repeats + 10%
  tolerance; run locally on a quiet machine; re-run on a suspicious near-tie.
- **Unfair callable asymmetry** (one side doing less work) → every cell pins both
  callables to the same logical output, mirroring the differential-test oracles.
- **Awkward import removed too early** → spec explicitly orders the gate before the
  Spec D dependency drop; roadmap SSoT updated to record this ordering.
