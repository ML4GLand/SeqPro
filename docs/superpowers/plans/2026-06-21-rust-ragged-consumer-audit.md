# Rust-Ragged Consumer Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Confirm genoray, GenVarLoader (gvl), and genvarformer (gvf) pass their test suites against seqpro 0.16's Rust-backed `Ragged`, fixing every breakage, so the awkward-backed `Ragged` can be retired.

**Architecture:** Flip `seqpro.rag.Ragged` from the awkward backend (`_array.Ragged`) to the Rust backend (`_core.Ragged`) via a one-line edit in `rag/__init__.py`. Point each consumer's pixi env at the local seqpro via an editable path so the flip is picked up live. Audit consumers in dependency order (genoray → gvl → gvf), each in two phases — awkward baseline (isolates seqpro 0.11/0.15.1→0.16 version drift) then Rust swap (isolates swap-caused failures) — fixing case-by-case until green.

**Tech Stack:** Python, pixi (conda+pypi envs), maturin/PyO3 Rust extension (seqpro), pytest, numpy, awkward (being retired).

## Global Constraints

- Local seqpro version: **0.16.0**; Rust extension already compiled (`SeqPro/python/seqpro/seqpro.abi3.so`); `cargo 1.96` available; `maturin` resolved via build isolation.
- The backend flip lives **only** in `SeqPro/python/seqpro/rag/__init__.py`. Phase A = `Ragged` from `_array`; Phase B = `Ragged` from `_core`. `zip()` already targets `_core` — leave it.
- Editable install means Python edits to `rag/__init__.py` / `_core.py` need **no** rebuild; only Rust-source edits do.
- Dependency order is mandatory: **genoray → gvl → gvf**. A fix in an upstream lib must be committed before the downstream lib is tested.
- Fix locus is **case-by-case**: clear Rust-`Ragged` regression → fix in seqpro `_core`; intentionally cleaner/different behavior → adapt the consumer. Record the rationale for every fix in the ledger.
- gvf coverage = **CPU-runnable subset on macOS arm64**. GPU/CUDA tests (torch+cu126, flash-attn, flash-linear-attention; linux-64 only) are out of scope here and recorded as deferred-to-Linux-GPU.
- No "green" claim without the actual pytest summary line as evidence.
- Each consumer is worked on a dedicated branch off `main`. seqpro fixes go on a dedicated SeqPro branch.
- Maintain a breakage ledger at `SeqPro/docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md` with columns: `consumer | test | phase-A | phase-B | root cause | fix locus | rationale`.

---

### Task 1: Pre-flight — `_core.Ragged` API surface smoke check

Verify the Rust `Ragged` exposes the surface consumers use before touching any consumer, so missing methods surface as one clear failure rather than noise across three suites.

**Files:**
- Create: `SeqPro/tests/test_core_ragged_surface.py`
- Reference: `SeqPro/python/seqpro/rag/_core.py`, `SeqPro/python/seqpro/rag/__init__.py`

**Interfaces:**
- Consumes: `seqpro.rag._core.Ragged`.
- Produces: confirmation (or gap list) for constructors `from_offsets`, `from_lengths`, `from_fields`, `empty`; methods `to_packed`, `to_padded`, `to_numpy`, `to_ak`, `view`, `squeeze`, `reshape`; attrs `data`, `offsets`, `lengths`, `shape`, `dtype`, `ndim`, `parts`; dunders `__getitem__`, `__setitem__`, `__array__`.

- [ ] **Step 1: Write the surface test**

```python
# SeqPro/tests/test_core_ragged_surface.py
"""Smoke check: the Rust _core.Ragged exposes the API surface consumers use.

Surface derived from grepping genoray/GenVarLoader/genvarformer source.
"""
import numpy as np
import pytest
from seqpro.rag._core import Ragged

CONSTRUCTORS = ["from_offsets", "from_lengths", "from_fields", "empty"]
METHODS = ["to_packed", "to_padded", "to_numpy", "to_ak", "view", "squeeze", "reshape"]
ATTRS = ["data", "offsets", "lengths", "shape", "dtype", "ndim", "parts"]
DUNDERS = ["__getitem__", "__setitem__", "__array__"]


@pytest.mark.parametrize("name", CONSTRUCTORS + METHODS)
def test_callable_present(name):
    assert callable(getattr(Ragged, name, None)), f"Ragged.{name} missing"


def test_instance_attrs_present():
    # 2 rows of lengths [3, 2] over data 0..4
    r = Ragged.from_lengths(np.arange(5), np.array([3, 2]))
    missing = [a for a in ATTRS if not hasattr(r, a)]
    assert not missing, f"missing instance attrs: {missing}"


@pytest.mark.parametrize("name", DUNDERS)
def test_dunder_present(name):
    assert hasattr(Ragged, name), f"Ragged.{name} missing"
```

- [ ] **Step 2: Run the test**

Run: `cd SeqPro && pixi run -e dev python -m pytest tests/test_core_ragged_surface.py -v`
Expected: PASS, or a precise list of missing members (e.g. `parts`, which is unconfirmed on `_core`).

- [ ] **Step 3: Resolve any gap**

For each missing member: if a consumer genuinely needs it (confirm against the grep in the spec), add it to `_core.Ragged` (Python edit in `_core.py`, picked up live; or Rust edit + rebuild if it must be native). If no consumer needs it, delete it from this test's surface lists and note why. Re-run Step 2 until green.

- [ ] **Step 4: Commit**

```bash
cd SeqPro && git checkout -b rust-ragged-audit-fixes 2>/dev/null || git checkout rust-ragged-audit-fixes
git add tests/test_core_ragged_surface.py python/seqpro/rag/_core.py
git commit -m "test: add _core.Ragged consumer API surface smoke check"
```

---

### Task 2: Create the breakage ledger

A single living document that records every breakage and its disposition. Created empty-but-structured now; appended to throughout Tasks 4–9.

**Files:**
- Create: `SeqPro/docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md`

- [ ] **Step 1: Write the ledger skeleton**

```markdown
# Rust-Ragged Consumer Audit — Breakage Ledger

Records every test breakage observed during the audit and its disposition.

Status legend: PASS / FAIL / SKIP(deferred) / N/A.

| consumer | test (nodeid) | phase-A | phase-B | root cause | fix locus | rationale |
|----------|---------------|---------|---------|-----------|-----------|-----------|
| _example_ | _tests/x::test_y_ | PASS | FAIL | _short cause_ | seqpro `_core` / consumer | _why here_ |

## Ship-readiness verdict

_Filled in Task 10._
```

- [ ] **Step 2: Commit**

```bash
cd SeqPro && git add docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md
git commit -m "docs: add rust-ragged audit breakage ledger skeleton"
```

---

### Task 3: Branch each consumer

Create the audit branch in each consumer repo so all subsequent edits are isolated and revertable.

**Files:**
- Modify: git state of `genoray`, `GenVarLoader`, `genvarformer` (branch only).

- [ ] **Step 1: Branch all three**

```bash
cd /Users/david/projects
for l in genoray GenVarLoader genvarformer; do
  git -C "$l" checkout -b rust-ragged-audit 2>/dev/null || git -C "$l" checkout rust-ragged-audit
  echo "$l -> $(git -C "$l" rev-parse --abbrev-ref HEAD)"
done
```

Expected: each prints `<lib> -> rust-ragged-audit`.

- [ ] **Step 2: Confirm clean working trees**

Run: `for l in genoray GenVarLoader genvarformer; do echo "== $l =="; git -C /Users/david/projects/$l status --short; done`
Expected: no output under each header (clean), or only expected local artifacts.

---

### Task 4: genoray — repoint pixi to local seqpro & Phase A baseline

Point genoray at local seqpro 0.16 with the **awkward** backend still wired, and capture the version-bump baseline.

**Files:**
- Modify: `genoray/pixi.toml` (remove conda `seqpro = "==0.11.*"` at line 79; add editable path dep under `[pypi-dependencies]`)
- Reference: `SeqPro/python/seqpro/rag/__init__.py` (must be on Phase A / `_array`)

**Interfaces:**
- Consumes: local `SeqPro` editable build.
- Produces: genoray Phase-A pass/fail set (the version-drift baseline) recorded in the ledger.

- [ ] **Step 1: Confirm seqpro is on the awkward backend (Phase A)**

Run: `grep -n "import.*Ragged" /Users/david/projects/SeqPro/python/seqpro/rag/__init__.py`
Expected: `Ragged` imported from `._array` (line 3). If it already points at `_core`, revert that line to `_array` before continuing.

- [ ] **Step 2: Repoint genoray pixi to local seqpro**

In `genoray/pixi.toml`: remove the conda line `seqpro = "==0.11.*"` (line 79). Under `[pypi-dependencies]` (line 36) add:

```toml
seqpro = { path = "../SeqPro", editable = true }
```

- [ ] **Step 3: Install and verify the local seqpro is active**

Run:
```bash
cd /Users/david/projects/genoray && pixi install \
  && pixi run python -c "import seqpro, seqpro.rag as r; print(seqpro.__version__, r.Ragged.__module__)"
```
Expected: `0.16.0 seqpro.rag._array` (local version, awkward backend). If the build fails, triage as an env issue (cargo on PATH, maturin build isolation) — not a test failure.

- [ ] **Step 4: Run the Phase-A suite**

Run: `cd /Users/david/projects/genoray && pixi run test`
Expected: a pytest summary line. Record every FAIL as a Phase-A (version-drift) entry in the ledger. Do NOT fix these yet unless Step 4 cannot produce a summary at all.

- [ ] **Step 5: Commit the pixi repoint**

```bash
cd /Users/david/projects/genoray
git add pixi.toml pixi.lock
git commit -m "build: point pixi at local seqpro (editable) for rust-ragged audit"
```

---

### Task 5: genoray — Phase B (Rust swap), fix to green

Flip seqpro to the Rust backend, re-run genoray, and fix swap-caused failures until the suite is green.

**Files:**
- Modify: `SeqPro/python/seqpro/rag/__init__.py` (Phase B flip)
- Modify (as needed): `genoray/genoray/_svar.py` and/or `SeqPro/python/seqpro/rag/_core.py`
- Append: ledger

**Interfaces:**
- Consumes: genoray Phase-A baseline (Task 4).
- Produces: green genoray suite against `_core.Ragged`; fixes committed.

- [ ] **Step 1: Flip seqpro to the Rust backend (Phase B)**

In `SeqPro/python/seqpro/rag/__init__.py`, change the `Ragged` import to come from `._core`:

```python
from ._core import Ragged, _CoreRagged  # adjust to match _core's exported names
```

Keep `_array` imports only for symbols still sourced from it (`DTYPE_co`, `RDTYPE_co`, `is_rag_dtype`) unless those also exist in `_core`. Verify:

Run: `cd /Users/david/projects/genoray && pixi run python -c "import seqpro.rag as r; print(r.Ragged.__module__)"`
Expected: `seqpro.rag._core`.

- [ ] **Step 2: Re-run the genoray suite (Phase B)**

Run: `cd /Users/david/projects/genoray && pixi run test`
Expected: a pytest summary. Compare against Phase A: tests that passed in A but fail in B are **swap-caused**. Record each in the ledger with phase-A=PASS, phase-B=FAIL.

- [ ] **Step 3: Fix each swap-caused failure (loop)**

For each swap-caused failure, apply superpowers:systematic-debugging: reproduce the single test (`pixi run python -m pytest <nodeid> -x -vv`), find root cause, decide fix locus per the case-by-case rule, apply the fix (in `genoray/genoray/_svar.py` for consumer adaptation, or `SeqPro/python/seqpro/rag/_core.py` for a Rust-`Ragged` regression), and record consumer/test/cause/locus/rationale in the ledger. Re-run the single test to confirm it passes.

- [ ] **Step 4: Run the full genoray suite to green**

Run: `cd /Users/david/projects/genoray && pixi run test`
Expected: pytest summary shows 0 failures (pre-existing Phase-A/version-drift failures, if any remain, must be explicitly listed in the ledger as accepted-out-of-scope, not silently ignored).

- [ ] **Step 5: Commit fixes**

```bash
# consumer fixes
cd /Users/david/projects/genoray && git add -A \
  && git commit -m "fix: adapt genoray to rust-backed seqpro Ragged"
# any seqpro fixes
cd /Users/david/projects/SeqPro && git add -A \
  && git commit -m "fix(rag): _core.Ragged fixes found via genoray audit"
```
(Skip whichever commit has no changes.)

---

### Task 6: GenVarLoader — repoint pixi & Phase A baseline

Same pattern as Task 4, for gvl. genoray fixes from Task 5 are already committed on its branch and consumed via the path dep, so gvl tests against the fixed upstream.

**Files:**
- Modify: `GenVarLoader/pixi.toml` (remove conda `seqpro = "==0.15.1"` at line 91; add editable path dep under `[pypi-dependencies]` at line 49)
- Reference: `SeqPro/python/seqpro/rag/__init__.py`

**Interfaces:**
- Consumes: local SeqPro editable build; genoray on its `rust-ragged-audit` branch.
- Produces: gvl Phase-A baseline in the ledger.

- [ ] **Step 1: Set seqpro back to the awkward backend for the baseline**

In `SeqPro/python/seqpro/rag/__init__.py`, temporarily revert the `Ragged` import to `._array`. Verify:
Run: `cd /Users/david/projects/GenVarLoader && grep -n Ragged ../SeqPro/python/seqpro/rag/__init__.py | head -1`
Expected: import from `._array`.

- [ ] **Step 2: Repoint gvl pixi to local seqpro**

In `GenVarLoader/pixi.toml`: remove `seqpro = "==0.15.1"` (line 91). Under `[pypi-dependencies]` (line 49) add:

```toml
seqpro = { path = "../SeqPro", editable = true }
```

Confirm genoray also resolves to local source if gvl pins it; if gvl uses published genoray, add `genoray = { path = "../genoray", editable = true }` too so the audited genoray is what's tested.

- [ ] **Step 3: Install and verify**

Run:
```bash
cd /Users/david/projects/GenVarLoader && pixi install \
  && pixi run python -c "import seqpro, seqpro.rag as r; import genoray; print(seqpro.__version__, r.Ragged.__module__, genoray.__file__)"
```
Expected: `0.16.0 seqpro.rag._array` and a `genoray.__file__` under `/Users/david/projects/genoray`.

- [ ] **Step 4: Run the Phase-A suite**

Run: `cd /Users/david/projects/GenVarLoader && pixi run python -m pytest tests`
Expected: pytest summary. Record FAILs as gvl Phase-A (version-drift) entries. (Use `pytest tests` directly rather than `pixi run test` to skip the `cargo test --release` half unless a Rust change is in play.)

- [ ] **Step 5: Commit the pixi repoint**

```bash
cd /Users/david/projects/GenVarLoader && git add pixi.toml pixi.lock \
  && git commit -m "build: point pixi at local seqpro (editable) for rust-ragged audit"
```

---

### Task 7: GenVarLoader — Phase B (Rust swap), fix to green

**Files:**
- Modify: `SeqPro/python/seqpro/rag/__init__.py` (Phase B flip)
- Modify (as needed): GenVarLoader source under `GenVarLoader/python/` or `GenVarLoader/genvarloader/` and/or `SeqPro/python/seqpro/rag/_core.py`
- Append: ledger

**Interfaces:**
- Consumes: gvl Phase-A baseline (Task 6).
- Produces: green gvl suite against `_core.Ragged`; fixes committed.

- [ ] **Step 1: Flip seqpro to the Rust backend (Phase B)**

Re-apply the `_core` import in `SeqPro/python/seqpro/rag/__init__.py`. Verify:
Run: `cd /Users/david/projects/GenVarLoader && pixi run python -c "import seqpro.rag as r; print(r.Ragged.__module__)"`
Expected: `seqpro.rag._core`.

- [ ] **Step 2: Re-run the gvl suite (Phase B)**

Run: `cd /Users/david/projects/GenVarLoader && pixi run python -m pytest tests`
Expected: pytest summary. Diff against Phase A; record swap-caused failures (note gvl-specific usages: `to_padded`, `to_ak`, `from_fields`, `parts`).

- [ ] **Step 3: Fix each swap-caused failure (loop)**

Per failure: superpowers:systematic-debugging on the single nodeid (`pixi run python -m pytest <nodeid> -x -vv`), root cause, case-by-case fix locus, apply, record in ledger, re-run the single test green. A fix landing in `_core.py` benefits genoray too — if it touches a behavior genoray exercised, re-run genoray's suite to confirm no regression.

- [ ] **Step 4: Run the full gvl suite to green**

Run: `cd /Users/david/projects/GenVarLoader && pixi run python -m pytest tests`
Expected: pytest summary, 0 failures (accepted version-drift exceptions explicitly listed in ledger).

- [ ] **Step 5: Commit fixes**

```bash
cd /Users/david/projects/GenVarLoader && git add -A \
  && git commit -m "fix: adapt GenVarLoader to rust-backed seqpro Ragged"
cd /Users/david/projects/SeqPro && git add -A \
  && git commit -m "fix(rag): _core.Ragged fixes found via GenVarLoader audit"
```
(Skip empty commits. If a `_core` fix changed genoray behavior, also re-run + re-commit genoray.)

---

### Task 8: genvarformer — repoint pixi & identify the CPU-runnable subset

gvf's full suite needs CUDA/flash-attn (linux-64). On macOS we cover the CPU-runnable, Ragged-touching subset and record the rest as deferred.

**Files:**
- Modify: `genvarformer/pixi.toml` (add editable path override for seqpro under `[pypi-dependencies]` at line 46; add path overrides for genoray and genvarloader so audited upstreams are tested)
- Reference: `SeqPro/python/seqpro/rag/__init__.py`

**Interfaces:**
- Consumes: local SeqPro + audited genoray + audited gvl (all on `rust-ragged-audit` branches / path deps).
- Produces: the chosen gvf CPU test selection (paths/`-k`/markers) and a deferred-tests list, both recorded in the ledger.

- [ ] **Step 1: Set seqpro to awkward backend for the baseline**

Revert `SeqPro/python/seqpro/rag/__init__.py` `Ragged` import to `._array`.

- [ ] **Step 2: Repoint gvf pixi to local upstreams**

In `genvarformer/pixi.toml` under `[pypi-dependencies]` (line 46) add:

```toml
seqpro = { path = "../SeqPro", editable = true }
genoray = { path = "../genoray", editable = true }
genvarloader = { path = "../GenVarLoader", editable = true }
```

- [ ] **Step 3: Install and verify on the macOS (osx-arm64) env**

Run:
```bash
cd /Users/david/projects/genvarformer && pixi install \
  && pixi run python -c "import seqpro, seqpro.rag as r; print(seqpro.__version__, r.Ragged.__module__)"
```
Expected: `0.16.0 seqpro.rag._array`. If install pulls linux-only CUDA deps and fails on macOS, restrict to the osx-arm64 env/feature that genvarformer defines and note any unavoidable gaps.

- [ ] **Step 4: Enumerate the CPU-runnable, Ragged-touching subset**

Run: `cd /Users/david/projects/genvarformer && pixi run python -m pytest --collect-only -q 2>&1 | head -50`
Then select tests that (a) collect/import without CUDA and (b) exercise seqpro `Ragged` (data Sources, set/pooling/aggregation, encoder output handling). Record the chosen selection expression and the explicitly-deferred GPU tests in the ledger.

- [ ] **Step 5: Run the Phase-A subset**

Run: `cd /Users/david/projects/genvarformer && pixi run python -m pytest <selection> -v`
Expected: pytest summary for the subset. Record Phase-A FAILs.

- [ ] **Step 6: Commit the pixi repoint**

```bash
cd /Users/david/projects/genvarformer && git add pixi.toml pixi.lock \
  && git commit -m "build: point pixi at local seqpro/genoray/gvl for rust-ragged audit"
```

---

### Task 9: genvarformer — Phase B (Rust swap), fix CPU subset to green

**Files:**
- Modify: `SeqPro/python/seqpro/rag/__init__.py` (Phase B flip)
- Modify (as needed): genvarformer source under `genvarformer/` and/or `SeqPro/python/seqpro/rag/_core.py`
- Append: ledger

**Interfaces:**
- Consumes: gvf Phase-A subset baseline (Task 8).
- Produces: green gvf CPU subset against `_core.Ragged`; fixes committed; deferred-GPU list finalized.

- [ ] **Step 1: Flip seqpro to the Rust backend (Phase B)**

Re-apply the `_core` import in `SeqPro/python/seqpro/rag/__init__.py`. Verify:
Run: `cd /Users/david/projects/genvarformer && pixi run python -c "import seqpro.rag as r; print(r.Ragged.__module__)"`
Expected: `seqpro.rag._core`.

- [ ] **Step 2: Re-run the gvf subset (Phase B)**

Run: `cd /Users/david/projects/genvarformer && pixi run python -m pytest <selection> -v`
Expected: pytest summary. Diff vs Phase A; record swap-caused failures.

- [ ] **Step 3: Fix each swap-caused failure (loop)**

Per failure: superpowers:systematic-debugging on the single nodeid, root cause, case-by-case fix locus, apply, record in ledger, re-run green. Any `_core.py` fix must be re-validated against genoray and gvl suites (re-run both) since it is shared.

- [ ] **Step 4: Run the full gvf subset to green**

Run: `cd /Users/david/projects/genvarformer && pixi run python -m pytest <selection> -v`
Expected: pytest summary, 0 failures across the CPU subset.

- [ ] **Step 5: Commit fixes**

```bash
cd /Users/david/projects/genvarformer && git add -A \
  && git commit -m "fix: adapt genvarformer to rust-backed seqpro Ragged"
cd /Users/david/projects/SeqPro && git add -A \
  && git commit -m "fix(rag): _core.Ragged fixes found via genvarformer audit"
```
(Skip empty commits.)

---

### Task 10: Finalize — verdict & leave the swap in place

Confirm the end state, write the ship-readiness verdict, and ensure seqpro is left on the Rust backend as the intended final swap.

**Files:**
- Modify: `SeqPro/docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md` (verdict section)
- Confirm: `SeqPro/python/seqpro/rag/__init__.py` on `_core`

- [ ] **Step 1: Re-confirm all three suites green (single sweep)**

Run:
```bash
cd /Users/david/projects/genoray && pixi run test
cd /Users/david/projects/GenVarLoader && pixi run python -m pytest tests
cd /Users/david/projects/genvarformer && pixi run python -m pytest <selection> -v
```
Expected: three pytest summaries, 0 failures (modulo ledger-documented version-drift / deferred-GPU exceptions). Paste the summary lines into the ledger.

- [ ] **Step 2: Confirm seqpro is left on the Rust backend**

Run: `grep -n "import.*Ragged" /Users/david/projects/SeqPro/python/seqpro/rag/__init__.py`
Expected: `Ragged` from `._core`.

- [ ] **Step 3: Write the ship-readiness verdict**

In the ledger's verdict section, state: total swap-caused breakages, how many fixed in seqpro vs consumers, any remaining accepted exceptions, the deferred gvf-GPU test list, and a clear GO / NO-GO on retiring the awkward backend.

- [ ] **Step 4: Commit the verdict**

```bash
cd /Users/david/projects/SeqPro && git add docs/superpowers/specs/2026-06-21-rust-ragged-audit-ledger.md \
  && git commit -m "docs: rust-ragged audit verdict — ship-readiness recorded"
```

---

## Self-Review

**Spec coverage:**
- Pre-flight smoke check → Task 1. Ledger → Task 2. Branching → Task 3.
- genoray two-phase + fix → Tasks 4–5. gvl → Tasks 6–7. gvf (CPU subset) → Tasks 8–9.
- Editable path repoint, version-drift isolation, case-by-case fix locus, shared-`_core` re-validation → embedded in each phase task and Global Constraints.
- gvf GPU deferral, verdict/report, leave-swap-in-place → Tasks 8/9/10.
- All spec sections map to a task.

**Placeholder note:** `<selection>` in Tasks 8–10 is a deliberate output of Task 8 Step 4 (the CPU test selection cannot be enumerated until collection runs), not an unfilled placeholder — it is defined once and referenced consistently.

**Type/name consistency:** backend modules referenced consistently as `_array` (awkward) and `_core` (Rust); the flip is always the single `rag/__init__.py` `Ragged` import; ledger path and columns identical across Tasks 2–10.
