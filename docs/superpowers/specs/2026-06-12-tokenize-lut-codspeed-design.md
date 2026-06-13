# Design: Optimize `tokenize` with a LUT + CodSpeed microbenchmarks

Date: 2026-06-12
Status: Approved (pending implementation)

## Problem

`seqpro.tokenize` is bottlenecking model training. The forward path runs through
`gufunc_tokenize` (`python/seqpro/_numba.py`), which tokenizes each character with a
**per-character linear scan** over the token map's keys:

```python
res[0] = unknown_token
for i in range(len(source)):
    if seq == source[i]:
        res[0] = target[i]
        break
```

This is O(n_chars × alphabet_size) with a branch per comparison. Sequence input is always
`uint8` bytes (0–255) after `cast_seqs`, so the per-character cost can be made O(1) with a
fixed 256-entry lookup table.

There is also no microbenchmark guarding tokenize performance, so regressions are invisible
in CI.

## Scope

- **In scope:** optimize `tokenize` (forward path only); add equivalence test; add
  `pytest-codspeed` microbenchmarks; add the `bench` tooling; add a CodSpeed CI workflow.
- **Out of scope:** `decode_tokens`. It reuses `gufunc_tokenize` in reverse (int32 → bytes);
  that kernel stays exactly as-is and `decode_tokens` is unchanged.

## Approach

**256-entry int32 LUT with dual dispatch** (small inputs: `np.take`; large inputs: parallel
Numba `lut_gather`).

Build a 256-entry int32 table once per call, then dispatch on element count:

```python
lut = np.full(256, unknown_token, dtype=np.int32)
lut[source_bytes] = target          # source_bytes: uint8 ASCII codes of token_map keys

if u8.size < _TOKENIZE_PARALLEL_THRESHOLD:   # ~40k elements
    out = np.take(lut, u8, out=out)          # single-threaded, zero dispatch overhead
else:
    lut_gather(u8.reshape(-1), lut, out.reshape(-1))  # parallel Numba kernel
```

- The 1 KB LUT sits in L1; the gather itself is memory-bandwidth bound for large inputs.
- `np.take` wins for small inputs: its zero thread-dispatch overhead beats the parallel
  kernel's ~96 µs launch floor below ~40k elements.
- For large inputs (`_TOKENIZE_PARALLEL_THRESHOLD ≈ 40_000`), a single-threaded `np.take`
  was measured ~20–28% **slower** than the original `gufunc_tokenize` (parallel gufunc) on
  dense (512×1024) and flanked-allele profiles. A pure `np.take` implementation therefore
  **regressed** vs the original kernel on the inputs that matter most for training.
- `lut_gather` (`@nb.njit(parallel=True)` over a flat 1-D buffer) restores and improves on
  the original: measured ~2.2–5.7× faster than `gufunc_tokenize` across dense and ragged
  profiles, while the small-input `np.take` path retains its win where thread overhead
  dominates.
- `np.take` natively supports `out=`, preserving the zero-alloc dense training path for
  small inputs; the parallel path allocates its own output buffer.
- The crossover (`_TOKENIZE_PARALLEL_THRESHOLD = 40_000`) is measured on a 14-core machine;
  the performance curve is flat near the crossover so the exact value is not sensitive.

Note on the earlier rejected alternative: an initial prototype used **only** `np.take` (no
parallel fallback), described here as "pure-NumPy LUT gather". That prototype was correct and
beat the old gufunc on small inputs, but was measured ~20–28% slower on large inputs. The
dual-dispatch design above supersedes it.

### DNA-specific fast path (benchmarked, adopted only if faster)

Consider a specialized path for the common DNA case mapping `A,C,G,T → 0,1,2,3` and everything
else → `4` (unknown). The benchmark (component 3) will compare it head-to-head against the
generic LUT, and we adopt it **only if measurably faster** on the dense and ragged profiles.

Expectation (to be confirmed by data, not assumed): it likely will **not** beat the generic
LUT, because both are bound by the **int32 output write** (4 bytes written per character vs a
1-byte read), and the generic LUT's input gather is already a single L1-resident table lookup.
An arithmetic scheme like `(byte >> 1) & 3` is a bijection only on `{A,C,G,T}`, produces the
wrong code order, and still needs a range check to route non-canonical bytes to `4` — adding
branches to a path whose cost is dominated by the write. If the benchmark nonetheless shows a
win, the DNA path is selected automatically when `token_map` matches the canonical DNA mapping
(`{"A":0,"C":1,"G":2,"T":3}` with `unknown_token == 4`), falling back to the generic LUT
otherwise; the public signature is unchanged either way. If it does not win, the generic LUT
stands alone and the comparison benchmark is dropped or left as a documented non-win.

**Measured result (2026-06-12, macOS, dense (512,1024) input):** No meaningful win. Over 3
runs the "best time" was 38–41µs for both paths and the winner flipped between runs (run 1:
generic=41.16µs, precomputed=38.09µs; run 2: generic=38.97µs, precomputed=40.49µs; run 3:
generic=40.36µs, precomputed=38.87µs). The 2–7% swings are within the reported 5–8% relative
StdDev — entirely noise. The gather dominates; skipping the O(256) LUT build saves nothing
measurable. **Decision: generic LUT stands; DNA fast path not integrated.**

### Preserved assumptions & semantics

- **Single-character ASCII keys** in `token_map`. The existing code already assumes this
  (`np.array([c.encode("ascii") for c in token_map]).view(np.uint8)` feeding a 1-D `(n)`
  gufunc signature), so the LUT — indexed by a single byte — preserves it.
- **Public signature unchanged:** `tokenize(seqs, token_map, unknown_token, out=None)`.
- **Output dtype** stays `np.int32`.
- **Unknown chars** map to `unknown_token` (the LUT fill value).
- **`out=`** supported for dense input only (as today); `np.take(..., out=out)` writes in place.
- **Ragged path:** same gather over `seqs.to_packed().data`, rebuilt via
  `Ragged.from_offsets(flat, (n, None, *trailing), seqs.offsets)`.

### LUT rebuild cost

The LUT is rebuilt every call (O(256) fill + O(alphabet) scatter), negligible versus the
gather over B×L characters. No caching — `token_map` is an unhashable dict and the cost is
not measurable against real batches (YAGNI).

## Components

### 1. Core change — `python/seqpro/_encoders.py`, `tokenize` only

Replace the `source`/`target`/`gufunc_tokenize` body with LUT construction + `np.take` for
both the dense and Ragged branches. Do not touch `_numba.gufunc_tokenize` (still used by
`decode_tokens`) or any overloads/signatures.

### 2. Equivalence test — `tests/test_tokenize.py`

A test asserting the new LUT output is **byte-for-byte identical** to the previous
`gufunc_tokenize` path. To avoid depending on deleted code, compute the reference inline
(direct `gufunc_tokenize` call, still importable from `seqpro._numba`) and compare. Cover:

- known characters in the alphabet,
- unknown characters → `unknown_token`,
- dense 2-D input,
- ragged input,
- dense input with a preallocated `out=` array.

This locks correctness independent of the existing roundtrip tests.

### 3. Microbenchmarks — `tests/test_bench_tokenize.py`

`pytest-codspeed` benchmarks using `@pytest.mark.benchmark`. Named `test_*` so the default
`pixi run test` collects them (running each body once as a plain test — effectively free) and
`--codspeed` instruments/times them. Inputs built once per parametrization with a fixed-seed
`np.random.default_rng`:

- **Dense batch:** `(512, 1024)` DNA (`B × L`) — the shape bottlenecking training.
- **Ragged — short alleles:** thousands of very short sequences (both alleles).
- **Ragged — flanked alleles:** thousands of >10 bp sequences (alleles with flank nucleotides).
- **Ragged — CREs:** hundreds of 100–200 bp sequences.

Each benchmark calls `sp.tokenize(...)` with a DNA token map and a fixed `unknown_token`.

**DNA-fast-path comparison:** include a paired benchmark that times the generic LUT against a
candidate DNA-specific path on the same inputs, so the data decides whether the specialized
path is worth keeping (see "DNA-specific fast path" above). Only the winning implementation
ships in `tokenize`.

**gufunc baseline:** four additional `test_bench_baseline_*` benchmarks time the old
`gufunc_tokenize` kernel directly on the same inputs (dense (512×1024) and the three ragged
profiles). This gives CodSpeed a head-to-head baseline so any future regression of the
LUT-gather impl vs the original kernel surfaces automatically in CI.

### 4. Tooling — `pixi.toml`

- Add `pytest-codspeed` to the `bench` feature dependencies.
- Add a `bench` task under `[feature.bench.tasks]`, e.g.:
  `bench = "pytest tests/test_bench_tokenize.py --codspeed"`.

The `--codspeed` flag only times `@pytest.mark.benchmark` tests; plain `pixi run test`
continues to pass (benchmarks run as ordinary, cheap tests).

### 5. CI — `.github/workflows/bench.yaml`

New workflow, mirroring `test.yaml`'s pixi setup:

- Triggers: `pull_request` to `main` + `workflow_dispatch`.
- Single job on `ubuntu-latest`, pixi `bench` environment (py313).
- Steps: checkout → setup-pixi (bench env) → `maturin develop` (build the extension, since the
  editable install needs the compiled `.so`) → run benchmarks via `CodSpeedHQ/action` invoking
  the pixi `bench` task.
- Uses `secrets.CODSPEED_TOKEN`.

**User action (out of repo):** install the CodSpeed GitHub App on the repo and add the
`CODSPEED_TOKEN` repository secret. Until then the workflow runs but cannot upload results.

### 6. Skill — no change

`skills/seqpro/SKILL.md` is untouched: the public signature and observable behavior of
`tokenize` are unchanged; this is an internal performance change only.

## Testing strategy

- **Correctness:** the new equivalence test + the existing `test_tokenize.py` roundtrip tests
  must pass on all Python matrix versions via `pixi run test`.
- **Performance:** CodSpeed reports per-benchmark timings and flags regressions on PRs once the
  App/secret are configured.

## Risks

- **`np.take` with multi-dim index + `out=`:** verify `np.take(lut, idx, out=out)` accepts a
  multi-dimensional `idx` with a matching-shape `out` (it does; out shape must equal idx shape).
  Covered by the `out=` equivalence test case.
- **CodSpeed CI without secret:** workflow is harmless but non-reporting until the user wires
  the App + secret; documented above.
- **Numba extension build in CI:** the bench job must build the Rust/Numba-backed package
  (`maturin develop`) before running, matching how other envs install the editable package.
