# Tokenize LUT Optimization + CodSpeed Microbenchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `tokenize`'s per-character linear-scan kernel with a 256-entry NumPy lookup-table gather, and add `pytest-codspeed` microbenchmarks wired into a CodSpeed CI workflow.

**Architecture:** `tokenize` builds a 256-entry `int32` LUT (`unknown_token` fill, scattered with the token map) and tokenizes via `np.take(lut, seqs.view(uint8), out=out)` for both dense and Ragged inputs. `gufunc_tokenize` is left untouched because `decode_tokens` (out of scope) still uses it. Benchmarks live in a `test_*` file collected by the normal suite and timed under `--codspeed`. A precomputed-DNA LUT fast path is added only if a benchmark proves it faster.

**Tech Stack:** Python, NumPy, Numba (existing, untouched), pytest, pytest-codspeed, pixi, GitHub Actions, CodSpeed.

**Spec:** `docs/superpowers/specs/2026-06-12-tokenize-lut-codspeed-design.md`

---

## File Structure

- `python/seqpro/_encoders.py` — modify `tokenize` only (the LUT change). `decode_tokens` and all overloads/signatures unchanged.
- `python/seqpro/_numba.py` — unchanged (`gufunc_tokenize` stays for `decode_tokens`).
- `tests/test_tokenize.py` — add an equivalence/characterization test guarding the refactor.
- `tests/test_bench_tokenize.py` — new; `pytest-codspeed` benchmarks (dense + 3 ragged profiles) + the DNA-fast-path comparison.
- `pixi.toml` — add `pytest-codspeed` to the `bench` feature and a `bench` task.
- `.github/workflows/bench.yaml` — new CodSpeed CI workflow.

---

## Task 1: Characterization test locking tokenize output (guards the refactor)

This test pins current `tokenize` output to a direct `gufunc_tokenize` reference. It passes against the current implementation AND must keep passing after the LUT change — that is the safety net.

**Files:**
- Test: `tests/test_tokenize.py` (append)

- [ ] **Step 1: Write the test**

Append to `tests/test_tokenize.py`:

```python
def test_tokenize_matches_gufunc_reference():
    """LUT output must be byte-for-byte identical to the linear-scan gufunc."""
    from seqpro._numba import gufunc_tokenize

    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    unknown_token = 4

    def reference(cast_seq):
        source = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
        target = np.array(list(token_map.values()), dtype=np.int32)
        return gufunc_tokenize(
            cast_seq.view(np.uint8), source, target, np.int32(unknown_token)
        )

    # Dense 2-D, with known + unknown ("N", "x") characters.
    seqs = ["ACGTN", "TTxAC", "GGGGG"]
    cast = sp.cast_seqs(seqs)  # (3, 5) S1
    expected = reference(cast)
    result = sp.tokenize(cast, token_map, unknown_token=unknown_token)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int32

    # out= path: result written in place, equals expected, returns same buffer.
    out = np.empty(cast.shape, dtype=np.int32)
    returned = sp.tokenize(cast, token_map, unknown_token=unknown_token, out=out)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(returned, expected)

    # Ragged path.
    rag_seqs = ["ACGTN", "TTxAC", "GGGGG"]
    data = np.frombuffer("".join(rag_seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in rag_seqs])
    rag = Ragged.from_lengths(data, lengths)
    rag_result = sp.tokenize(rag, token_map, unknown_token=unknown_token)
    flat_expected = reference(np.frombuffer(b"".join(s.encode() for s in rag_seqs), dtype="S1"))
    np.testing.assert_array_equal(rag_result.data, flat_expected)
    np.testing.assert_array_equal(rag_result.lengths.ravel(), lengths)
```

- [ ] **Step 2: Run the test to verify it passes against the CURRENT implementation**

Run: `pixi run -e dev pytest tests/test_tokenize.py::test_tokenize_matches_gufunc_reference -v`
Expected: PASS (current code already produces this output — this confirms the reference is correct).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tokenize.py
git commit -m "test: characterize tokenize output against gufunc reference"
```

---

## Task 2: Replace tokenize's kernel with a 256-entry LUT gather

**Files:**
- Modify: `python/seqpro/_encoders.py` (function `tokenize`, lines ~239-276 — the body after the docstring only)
- Test: `tests/test_tokenize.py` (the Task 1 test + existing tests)

- [ ] **Step 1: Verify the existing + characterization tests pass (baseline green)**

Run: `pixi run -e dev pytest tests/test_tokenize.py -v`
Expected: PASS (all tokenize tests, including `test_tokenize_matches_gufunc_reference`).

- [ ] **Step 2: Rewrite the `tokenize` body**

In `python/seqpro/_encoders.py`, replace the body of `tokenize` BELOW the docstring (currently lines ~264-276, starting at `source = np.array(...)`) with:

```python
    # Build a 256-entry lookup table: lut[byte] -> token. Input is uint8 (0-255)
    # after cast_seqs, so a single gather replaces a per-character linear scan.
    keys = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    vals = np.array(list(token_map.values()), dtype=np.int32)
    lut = np.full(256, np.int32(unknown_token), dtype=np.int32)
    lut[keys] = vals

    if isinstance(seqs, Ragged):
        seqs = seqs.to_packed()
        n = len(seqs.lengths.ravel())
        trailing = seqs.data.shape[1:]
        flat = np.take(lut, seqs.data.view(np.uint8))
        return Ragged.from_offsets(flat, (n, None, *trailing), seqs.offsets)

    _seqs = cast_seqs(seqs)
    return np.take(lut, _seqs.view(np.uint8), out=out)
```

Do NOT modify the docstring, the `@overload` signatures, or `decode_tokens`. Leave the `gufunc_tokenize` import in place (still used by `decode_tokens`).

- [ ] **Step 3: Run the characterization + existing tokenize tests**

Run: `pixi run -e dev pytest tests/test_tokenize.py -v`
Expected: PASS (identical output to the reference; roundtrip and ragged tests still green).

- [ ] **Step 4: Run the full suite to catch downstream callers (e.g. transforms, ohe roundtrips)**

Run: `pixi run -e dev pytest tests/ -q`
Expected: PASS (no regressions).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_encoders.py
git commit -m "perf(tokenize): use 256-entry LUT gather instead of linear scan"
```

---

## Task 3: Add pytest-codspeed dependency and a bench task

**Files:**
- Modify: `pixi.toml` (`[feature.bench.dependencies]` and `[feature.bench.tasks]`)

- [ ] **Step 1: Add the dependency**

In `pixi.toml`, under `[feature.bench.dependencies]` (currently `marimo`, `seaborn`, `statsmodels`), add:

```toml
pytest-codspeed = "*"
```

- [ ] **Step 2: Add the bench task**

In `pixi.toml`, under `[feature.bench.tasks]` (currently has `i-kernel`), add:

```toml
bench = "pytest tests/test_bench_tokenize.py --codspeed"
```

- [ ] **Step 3: Install the bench environment**

Run: `pixi install -e bench`
Expected: resolves and installs `pytest-codspeed` (and its `pytest-benchmark`-compatible API) into the `bench` env.

- [ ] **Step 4: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build(bench): add pytest-codspeed and bench task"
```

---

## Task 4: Write the tokenize microbenchmarks (dense + 3 ragged profiles)

`pytest-codspeed` provides a `benchmark` fixture (pytest-benchmark compatible). Named `test_*` so the default suite collects them — under plain pytest the body runs once (cheap); under `--codspeed` it is instrumented and timed.

**Files:**
- Create: `tests/test_bench_tokenize.py`

- [ ] **Step 1: Write the benchmark file**

Create `tests/test_bench_tokenize.py`:

```python
"""Microbenchmarks for ``seqpro.tokenize`` (pytest-codspeed).

Collected by the normal test suite (runs each body once, ~free) and timed under
``pytest --codspeed`` (see the ``bench`` pixi task / bench.yaml CI workflow).
"""

from __future__ import annotations

import numpy as np
import pytest
import seqpro as sp
from seqpro.rag import Ragged

DNA_TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
UNKNOWN_TOKEN = 4
_BASES = np.frombuffer(b"ACGT", dtype="S1")


def _rng():
    # Argless default_rng would be nondeterministic; pin the seed.
    return np.random.default_rng(0)


def _dense(batch: int, length: int) -> np.ndarray:
    rng = _rng()
    idx = rng.integers(0, 4, size=(batch, length))
    return _BASES[idx]  # (batch, length) S1


def _ragged(n: int, low: int, high: int) -> Ragged:
    rng = _rng()
    lengths = rng.integers(low, high + 1, size=n).astype(np.int64)
    total = int(lengths.sum())
    data = _BASES[rng.integers(0, 4, size=total)]
    return Ragged.from_lengths(data, lengths)


def test_bench_dense_batch(benchmark):
    """Realistic training batch (512, 1024) DNA."""
    seqs = _dense(512, 1024)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_short_alleles(benchmark):
    """Thousands of very short sequences (both alleles)."""
    seqs = _ragged(8000, 1, 4)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_flanked_alleles(benchmark):
    """Thousands of >10 bp sequences (alleles with flank nucleotides)."""
    seqs = _ragged(8000, 11, 60)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))


def test_bench_ragged_cres(benchmark):
    """Hundreds of 100-200 bp sequences (CREs)."""
    seqs = _ragged(500, 100, 200)
    benchmark(lambda: sp.tokenize(seqs, DNA_TOKEN_MAP, unknown_token=UNKNOWN_TOKEN))
```

- [ ] **Step 2: Verify the benchmarks are collected and pass as plain tests**

Run: `pixi run -e bench pytest tests/test_bench_tokenize.py -v`
Expected: PASS — 4 tests collected and run (the `benchmark` fixture executes each callable once without `--codspeed`).

- [ ] **Step 3: Verify they run under codspeed instrumentation locally**

Run: `pixi run -e bench pytest tests/test_bench_tokenize.py --codspeed -v`
Expected: PASS — codspeed reports timing for the 4 benchmarks (a "running in walltime mode / not in CI" notice is fine locally).

- [ ] **Step 4: Confirm the default test suite still passes (benchmarks run as plain tests there too)**

Run: `pixi run -e dev pytest tests/test_bench_tokenize.py -q`
Expected: PASS (the `benchmark` fixture is provided by pytest-codspeed; if the `dev` env lacks it this file is skipped/errors — if so, the bench file should only be collected in the bench env: see note).

Note: if `dev` env errors on the missing `benchmark` fixture, add `tests/test_bench_tokenize.py` to a `--ignore` for the default `test` task, or guard the import with `pytest.importorskip("pytest_codspeed")` at module top. Prefer `pytest.importorskip` so the file self-skips cleanly:

```python
pytest.importorskip("pytest_codspeed")
```

(Place it directly after the imports.)

- [ ] **Step 5: Commit**

```bash
git add tests/test_bench_tokenize.py
git commit -m "test(bench): add tokenize microbenchmarks (dense + ragged profiles)"
```

---

## Task 5: DNA fast-path comparison + conditional integration

Add a benchmark comparing the per-call generic LUT against a **precomputed module-level DNA LUT** (built once at import, reused when `token_map` matches canonical DNA). Integrate the precomputed branch into `tokenize` ONLY if the benchmark shows a real, repeatable win; otherwise document the non-win and stop.

**Files:**
- Modify: `tests/test_bench_tokenize.py` (add comparison benchmarks)
- Modify (CONDITIONAL on benchmark result): `python/seqpro/_encoders.py`

- [ ] **Step 1: Add the comparison benchmarks**

Append to `tests/test_bench_tokenize.py`:

```python
# Candidate DNA fast path: a LUT built once at import, reused across calls,
# avoiding the per-call np.full(256)+scatter. Compared head-to-head below.
_DNA_LUT = np.full(256, np.int32(UNKNOWN_TOKEN), dtype=np.int32)
_DNA_LUT[np.frombuffer(b"ACGT", dtype="S1").view(np.uint8)] = np.arange(4, dtype=np.int32)


def _generic_tokenize(u8: np.ndarray) -> np.ndarray:
    keys = np.array([c.encode("ascii") for c in DNA_TOKEN_MAP]).view(np.uint8)
    vals = np.array(list(DNA_TOKEN_MAP.values()), dtype=np.int32)
    lut = np.full(256, np.int32(UNKNOWN_TOKEN), dtype=np.int32)
    lut[keys] = vals
    return np.take(lut, u8)


def test_bench_dna_generic_lut(benchmark):
    u8 = _dense(512, 1024).view(np.uint8)
    benchmark(lambda: _generic_tokenize(u8))


def test_bench_dna_precomputed_lut(benchmark):
    u8 = _dense(512, 1024).view(np.uint8)
    benchmark(lambda: np.take(_DNA_LUT, u8))
```

- [ ] **Step 2: Run the comparison and record the numbers**

Run: `pixi run -e bench pytest tests/test_bench_tokenize.py -k "dna" --codspeed -v`
Expected: PASS. Record the two timings. The precomputed LUT only saves the O(256) build; on a (512,1024) batch the gather dominates, so a win is expected to be negligible.

- [ ] **Step 3: Decision gate**

- If `test_bench_dna_precomputed_lut` is **NOT meaningfully faster** (e.g. <5% and within noise): STOP integration. The generic LUT stands. Add a one-line note to the spec's "DNA-specific fast path" section recording the measured non-win, then commit only the benchmark additions:

  ```bash
  git add tests/test_bench_tokenize.py docs/superpowers/specs/2026-06-12-tokenize-lut-codspeed-design.md
  git commit -m "test(bench): compare generic vs precomputed DNA LUT (no meaningful win)"
  ```

- If it **IS meaningfully faster** (repeatable, well outside noise): proceed to Step 4.

- [ ] **Step 4 (CONDITIONAL): Integrate the precomputed DNA LUT**

Only if Step 3 found a real win. In `python/seqpro/_encoders.py`, add a module-level constant near the top (after imports):

```python
# Precomputed LUT for the canonical DNA token map (fast path for tokenize).
_DNA_TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
_DNA_LUT = np.full(256, np.int32(4), dtype=np.int32)
_DNA_LUT[np.frombuffer(b"ACGT", dtype="S1").view(np.uint8)] = np.arange(4, dtype=np.int32)
```

Then in `tokenize`, replace the LUT-build block with a fast-path check:

```python
    if token_map == _DNA_TOKEN_MAP and unknown_token == 4:
        lut = _DNA_LUT
    else:
        keys = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
        vals = np.array(list(token_map.values()), dtype=np.int32)
        lut = np.full(256, np.int32(unknown_token), dtype=np.int32)
        lut[keys] = vals
```

- [ ] **Step 5 (CONDITIONAL): Verify correctness unchanged**

Run: `pixi run -e dev pytest tests/test_tokenize.py -v`
Expected: PASS (`test_tokenize_matches_gufunc_reference` confirms the fast path matches the reference, since it uses the canonical DNA map + unknown_token=4).

- [ ] **Step 6 (CONDITIONAL): Commit**

```bash
git add python/seqpro/_encoders.py tests/test_bench_tokenize.py
git commit -m "perf(tokenize): precomputed LUT fast path for canonical DNA token map"
```

---

## Task 6: CodSpeed CI workflow

**Files:**
- Create: `.github/workflows/bench.yaml`

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/bench.yaml`:

```yaml
name: Benchmarks

on:
  pull_request:
    branches:
      - main
  workflow_dispatch:

jobs:
  benchmarks:
    runs-on: ubuntu-latest
    name: "Run CodSpeed microbenchmarks"
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.67.2
          cache: true
          environments: bench
          locked: false
      - name: Build extension
        run: pixi run -e bench maturin develop
      - name: Run benchmarks
        uses: CodSpeedHQ/action@v3
        with:
          token: ${{ secrets.CODSPEED_TOKEN }}
          run: pixi run -e bench bench
```

- [ ] **Step 2: Validate the workflow YAML syntax**

Run: `python -c "import yaml,sys; yaml.safe_load(open('.github/workflows/bench.yaml')); print('valid')"`
Expected: prints `valid`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/bench.yaml
git commit -m "ci(bench): add CodSpeed benchmark workflow"
```

- [ ] **Step 4: Manual handoff note (out of repo — user action)**

After merge, the user must: install the **CodSpeed GitHub App** on the repo and add a **`CODSPEED_TOKEN`** repository secret (from codspeed.io). Until then the workflow runs but cannot upload results / comment on PRs. This is not a code step — surface it in the PR description.

---

## Self-Review

- **Spec coverage:**
  - LUT optimization of `tokenize` only → Task 2. ✓
  - `gufunc_tokenize`/`decode_tokens` untouched → Task 2 Step 2 explicitly preserves them. ✓
  - Equivalence/characterization test (dense, ragged, `out=`) → Task 1. ✓
  - DNA fast path, benchmarked & gated on a real win → Task 5. ✓
  - Microbenchmarks in `tests/test_bench_tokenize.py` (dense + 3 ragged profiles) → Task 4. ✓
  - `pytest-codspeed` dep + `bench` task in pixi → Task 3. ✓
  - `bench.yaml` CodSpeed CI workflow + maturin build → Task 6. ✓
  - User wires App + `CODSPEED_TOKEN` → Task 6 Step 4. ✓
  - No `SKILL.md` change (signature/behavior unchanged) → confirmed; no task needed. ✓
- **Placeholder scan:** No TBD/TODO; all code steps contain full code. The only conditional content (Task 5 Steps 4-6) is explicitly gated and fully specified. ✓
- **Type/name consistency:** `DNA_TOKEN_MAP`/`UNKNOWN_TOKEN` (bench module) vs `_DNA_TOKEN_MAP`/`_DNA_LUT` (encoders module) are intentionally distinct namespaces; the canonical map `{"A":0,"C":1,"G":2,"T":3}` and `unknown_token == 4` are consistent across Tasks 4-5. `np.take(lut, ..., out=out)` signature consistent with the `out` param of `tokenize`. ✓
