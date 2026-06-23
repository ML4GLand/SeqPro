# Rust-Ragged Consumer Audit — Design

**Date:** 2026-06-21
**Status:** Approved (design); pending implementation plan

## Goal & scope

Validate that the three downstream consumers — **genoray**, **GenVarLoader** (gvl),
and **genvarformer** (gvf) — work against seqpro 0.16's **Rust-backed** `Ragged`
(`seqpro.rag._core.Ragged`) before that class retires the awkward-backed one
(`seqpro.rag._array.Ragged`).

**End state (definition of done):** all three consumer test suites pass against the
Rust `Ragged`, with every fix landed in the right place and its rationale recorded.

- gvf coverage is the **CPU-runnable subset** on this macOS arm64 box; its GPU/CUDA
  tests (torch+cu126, flash-attn, flash-linear-attention — all linux-64 only) are
  explicitly out of scope here and flagged for a Linux-GPU follow-up.
- Fix locus is decided **case-by-case** with documented rationale: a clear regression
  in the Rust class is fixed in seqpro; an intentionally cleaner/different behavior is
  adapted to in the consumer.

## Background / current state

- `SeqPro/python/seqpro/rag/__init__.py` currently exports:
  - `Ragged` ← `_array.py` (**awkward-backed**, imports `awkward as ak`)
  - `_CoreRagged` ← `_core.py` (**Rust-backed**, numpy-only)
  - `zip()` already targets `_core` (`_CoreRagged.from_fields`).
- Local seqpro is **0.16.0**; the Rust extension is already compiled
  (`python/seqpro/seqpro.abi3.so`). `cargo 1.96` is available.
- Dependency chain: **genvarformer → genvarloader (≥0.31) + genoray (≥2.9.2) + seqpro (≥0.16)**;
  **genvarloader → genoray (≥2.9.2) + seqpro (≥0.15.1)**; **genoray → seqpro (≥0.11)**.
- seqpro is a **conda** dependency in genoray/gvl pixi (`==0.11.*` / `==0.15.1`) and a
  **pypi** dependency in gvf. Pointing to local means an **editable path** build.
- Consumers import via `from seqpro.rag import Ragged`. Observed API surface across the
  three consumers (from source grep):
  - constructors: `from_offsets`, `from_lengths`, `from_fields`, `empty`
  - methods: `to_packed`, `to_padded`, `to_numpy`, `to_ak`, `view`, `squeeze`, `reshape`
  - attrs: `data`, `offsets`, `lengths`, `shape`, `dtype`, `ndim`, `parts`
  - dunders: `__getitem__`, `__setitem__`, `__array__`
  - `_core.Ragged` is confirmed to expose all of the above **except `parts`**, which
    must be verified in the pre-flight smoke check.

## The seqpro swap (single source of truth)

The backend swap is one edit in `SeqPro/python/seqpro/rag/__init__.py`:

- **Awkward baseline (Phase A):** `Ragged` ← `_array` (unchanged from today).
- **Rust swap (Phase B):** `Ragged` ← `_core`.

Because consumer envs install seqpro as an **editable path**, this Python-only flip is
picked up live with no rebuild. SeqPro is a clean git repo, so the flip is revertable
for returning to the awkward baseline between phases. Only Rust-source fixes require a
rebuild; `__init__.py` / `_core.py` Python edits do not.

## Chosen approach: sequential, dependency-order, two-phase (Approach A)

Work **genoray → gvl → gvf** so a fix in an upstream lib is in place before the
downstream lib is tested. Each consumer runs a two-phase test to separate version drift
from the backend swap.

### Pre-flight smoke check (before genoray)

A focused check that `_core.Ragged` provides the consumer API surface listed above.
Catches obvious gaps (notably `parts`) before they appear as noise in consumer runs.
Resolve any gap (add to `_core` or confirm the consumer no longer needs it) before
proceeding.

### Per-consumer procedure (on a dedicated branch per consumer)

1. **Repoint pixi → local seqpro.** Remove the conda `seqpro` pin (genoray/gvl) and add
   an editable path dep under `[pypi-dependencies]`:
   `seqpro = { path = "../SeqPro", editable = true }`. gvf already takes seqpro via
   pypi; add the path override. Run `pixi install`.
2. **Phase A — version-bump baseline.** With `rag/__init__` on awkward, run the suite.
   Record pass/fail. Failures here are seqpro version drift (0.11/0.15.1 → 0.16), **not**
   the swap.
3. **Phase B — Rust swap.** Flip `rag/__init__` to `_core`. Re-run the suite.
4. **Attribute & fix.** Tests that passed in A but fail in B = **swap-caused** → fix
   case-by-case (seqpro `_core` bug vs consumer adaptation), recording rationale.
   Pre-existing A failures = version drift; fix only if they block reaching green, but
   note them regardless.
5. **Green gate.** Suite passes (gvf: CPU subset; GPU/CUDA tests marked deferred).
   Commit fixes on the branch.

## Error handling / diagnosis

- Apply systematic-debugging to each swap-caused failure: reproduce minimally, find root
  cause, then decide fix locus.
- Maintain a **breakage ledger**: `consumer | test | phase-A status | phase-B status |
  root cause | fix locus | rationale`.
- Triage build/env failures (cargo, maturin/editable build, dependency resolution)
  separately from test failures.

## Testing & verification

- genoray: `pixi run test` → `pytest tests` (has a `gen` data-prep dependency).
- gvl: `pytest tests` (the suite's `cargo test --release` half is run only if relevant
  to the swap).
- gvf: targeted CPU subset via test paths / `-k` / markers; GPU/CUDA tests excluded and
  recorded as deferred.
- No "green" claim without the actual pytest summary line as evidence
  (verification-before-completion).

## Deliverables

- Consumer fixes committed per-branch (genoray, gvl, gvf) + any seqpro `_core` / Rust
  fixes on a SeqPro branch.
- A short **audit report**: the breakage ledger plus a ship-readiness verdict, including
  the gvf GPU-tests-deferred caveat.
- The seqpro `rag/__init__` flip to `_core` left in place as the intended final swap.

## Risks & open items

- **`parts` attribute** on `_core.Ragged` is unconfirmed — first item of the pre-flight
  smoke check.
- **Editable Rust build** in each consumer env needs cargo on PATH (present) and maturin
  via build isolation; resolution snags are an env-triage item, not a test failure.
- **genoray double-jump** (0.11 → 0.16) may surface version-drift failures unrelated to
  the swap; the Phase A baseline is what disambiguates them.
- **gvf GPU coverage gap** on macOS is accepted; full validation deferred to Linux GPU.
