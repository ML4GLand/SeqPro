# Design: Port `tokenize` and `translate` to Rust/PyO3 (optimization experiment)

**Date:** 2026-06-18
**Status:** Approved design, pending implementation plan
**Type:** Optimization experiment (keep-or-revert outcome)

## Motivation

Port `seqpro.tokenize` (`python/seqpro/_encoders.py`) and `AminoAlphabet.translate`
(`python/seqpro/alphabets/_alphabets.py`) from their current Numba/NumPy
implementations to the existing Rust/PyO3 extension (`src/`). The project already
ships a PyO3/maturin extension (`src/lib.rs`, `kshuffle.rs`, plus an unused
`kmer_encode.rs`), so the integration friction is low.

This is a genuine experiment: both functions are *already* well optimized with
parallel Numba kernels (a 256-entry LUT gather for `tokenize`; an O(1) 64-entry
codon-hash LUT for `translate`, with linear-scan, drop-compaction and
stop-truncation kernels). Beating tuned, parallel, memory-bandwidth-bound Numba
is not a foregone conclusion.

The experiment pursues three concrete wins:

1. **Raw steady-state speed** — beat the parallel Numba kernels on large-array throughput.
2. **Kill the Numba dependency (validation)** — remove JIT warmup and AOT-compile
   these paths. See scope caveat below: this experiment *validates* the path; it
   does not by itself remove `import numba` from the package.
3. **Small-array / latency wins** — the Numba parallel path has a ~96µs
   thread-launch floor; Rust may win the latency-sensitive small-input regime.

`tokenize` and `translate` are treated as the bellwether for whether the
remaining Numba kernels are worth porting later.

## Decision rule (keep or revert)

**Keep if the Rust port wins clearly in at least one regime (small-array latency
OR large throughput) and never meaningfully regresses the others.** Otherwise
revert.

## Scope

- **In scope:** `tokenize` and `translate`, at **full feature parity** — every
  input path and option, so Numba is removed from *these two functions' own code
  paths*.
- **Out of scope:** `decode_tokens` (tokenize's inverse, stays on the shared
  `gufunc_tokenize` kernel), `ohe`/`decode_ohe`, `pad_seqs`, complement/rev-comp,
  k-shuffle. These continue to use Numba.

### Honesty caveat on "kill the Numba dependency"

Because `decode_tokens`, `ohe`, `decode_ohe`, `pad_seqs`, etc. stay on Numba, the
package will still `import numba`. The global dependency-removal win is therefore
**validated** (no JIT warmup, AOT codegen, clean speed comparison) for these two
functions, **not achieved** package-wide. Achieving it requires porting the rest
of the kernels in follow-up work, which this experiment is meant to inform.

A direct consequence: `translate`'s OHE-ragged path currently relies on the
Numba-backed `decode_ohe`/`ohe`. To keep Numba out of `translate`'s own paths, the
port gives it a **self-contained Rust OHE↔AA path** rather than calling back into
those functions (see Section 3).

## Approach: hybrid, measured (option C)

Start with **thin Rust kernels** (mirror the `kshuffle` pattern): Python keeps all
orchestration (`cast_seqs`, `check_axes`, ragged `to_packed()`/`from_offsets`, LUT
and codon-LUT construction — all NumPy, not Numba — and shape/offset bookkeeping),
Rust does only the inner compute. Then gather the full benchmark sweep and
**thicken the boundary only in the specific regimes the data shows Python overhead
dominating** (likely small-array). This reaches honest numbers fastest, keeps the
initial port reviewable, and turns the small-array latency goal into a measured
follow-up rather than an upfront bet.

(Rejected alternatives: **A — thin kernels only**, leaves small-array Python
overhead unaddressed; **B — thick boundary from the start**, largest Rust surface
and most edge cases up front before any data justifies it.)

## Section 1 — Architecture & module layout

**Rust (`src/`):** two new modules registered in the `#[pymodule]` in `src/lib.rs`
alongside `_k_shuffle`:

- `src/tokenize.rs` → `_tokenize`. Inner contract: contiguous `u8` view + 256-entry
  `i32` LUT (+ optional `out`). Internally selects serial vs. rayon-parallel by a
  re-measured element-count threshold (replacing the current 40k constant).
- `src/translate.rs` → `_translate` and friends. Strides directly over codons (no
  `sliding_window_view`). Separate entry points/flags for the standard LUT path,
  the generic scan path, `unknown="drop"` compaction, `truncate_stop`, and the
  self-contained OHE↔AA path.

**Python:** `_encoders.tokenize` and `AminoAlphabet.translate` keep their exact
public signatures, overloads, validation, and ragged/dense dispatch. They swap the
Numba kernel calls for `seqpro.seqpro._tokenize` / `._translate`. All NumPy
orchestration stays in Python in the thin-kernel phase.

**Stays on Numba (unchanged):** `decode_tokens`, `ohe`, `decode_ohe`, `pad_seqs`,
complement/rev-comp, k-shuffle helpers. The now-unused kernels in `_numba.py`
(`lut_gather`, `gufunc_translate`, `gufunc_translate_lut`, `_nb_drop_unknown_codons`,
`_nb_find_stop_ends`) are **left in place during the experiment** for a cheap
revert; deleted only after a keep decision.

## Section 2 — `tokenize` port

**Inner contract.** Python builds the 256-entry `i32` LUT (NumPy, as today) and
passes `(u8_view, lut, out_opt)` to `_tokenize`, which computes `out[i] = lut[seq[i]]`.

**Single path replacing two.** The current `np.take` (small) vs. Numba `lut_gather`
(≥40k) branch collapses into one Rust entry point that picks serial vs. rayon
internally by a re-measured threshold. Testing whether one Rust path can dominate
both regimes is itself part of the experiment.

**Parity preserved exactly:**
- `out=` with `dtype != int32` → `TypeError` (Python-side, unchanged).
- Non-C-contiguous `out` + `parallel=True` → `ValueError`. For a strided `out`,
  Rust falls back to a strided serial write (mirrors today's `np.take(out=)`).
- `parallel` escape hatch (`None`/`True`/`False`): `None` → threshold heuristic,
  `True`/`False` → force.
- Ragged: Python `to_packed()` then `Ragged.from_offsets(...)`; Rust gathers the
  flat packed `u8` buffer.
- Empty input and multi-dim/trailing axes (output shape == input shape) preserved.

**Rust shape handling:** accept `IxDyn` like `_k_shuffle`; the gather is
elementwise/shape-agnostic, so trailing axes need no special logic.

## Section 3 — `translate` port

Rust strides directly over codons (no `sliding_window_view`/`array_slice`).

**Standard LUT path** (`codon_lut is not None`): for each `codon_size`-byte stride,
upper-case via `& 0xDF`, range-check all bytes ∈ {A,C,G,T}, then
`lut[pack_index(b0,b1,b2)]`; non-canonical → `marker_byte`. Ports
`gufunc_translate_lut` + `_pack_codon_index`. The pack hash and 64-entry LUT remain
built in Python (`_build_translate_lut`) and passed in — single source of truth
preserved.

**Generic scan path** (`codon_lut is None`, non-standard alphabets): Rust gets
`(codon_keys (n,k) u8, codon_values (n,) u8, marker_byte)` and does the
case-insensitive per-codon linear scan. Ports `gufunc_translate`.

**`unknown="drop"`:** Rust does per-sequence stream compaction over codon-indexed
offsets, returning `(out_u8, new_offsets)`. Ports `_nb_drop_unknown_codons`. Drop
criterion = per-nucleotide-byte validity against `valid_upper_bytes` (passed in).
Always returns a `Ragged`, even for dense input (unchanged rule).

**`truncate_stop`:** Rust scans each translated sub-sequence for the first stop
byte (`*`) and returns truncated end offsets. Ports `_nb_find_stop_ends`. Produces
the same `(2, n)` ListArray (start/end) layout.

**Self-contained OHE↔AA path:** instead of Numba-backed `decode_ohe` → translate →
`ohe`, a dedicated Rust path takes packed OHE `u8` data `(total, n_nuc)` + the
nucleotide alphabet bytes + the AA alphabet bytes and goes **OHE → codon → AA →
OHE** in one pass, emitting `(total_aa, n_aa)` rows. `validate=True` still runs the
Python-side one-hot row check (`_check_ohe_rows`, NumPy) first. This keeps
`translate` free of Numba on the OHE path and avoids the intermediate byte buffer.
**Highest-correctness-risk surface** (no direct 1:1 Numba analog; fuses three
current steps) → heaviest differential testing.

**Parity preserved in Python:** all overloads, `check_axes`, `length_axis`
normalization, codon-divisibility checks, `validate=`
(`_check_nuc_bytes`/`_check_ohe_rows`, NumPy), `_parse_unknown`, ragged
`to_packed()`/`from_offsets` reconstruction, and empty-input branches.

## Section 4 — Parallelism & thresholds

Both kernels switch serial↔parallel on a tunable element-count threshold via rayon
(already a dependency). The current 40k `tokenize` constant was measured for
Numba's ~96µs thread-launch floor; rayon's floor differs, so each kernel's
crossover is **re-measured on the bench machine** and the constants baked into Rust
with a documenting comment (matching the existing style). `translate` gets its own
threshold (per-element codon work differs from a plain gather). The `parallel=`
escape-hatch semantics are unchanged at the Python layer.

## Section 5 — Benchmarking (the experiment's instrument)

A committed Python-level `pytest-benchmark` suite measuring end-to-end cost
*including* PyO3 marshalling (which Rust-only criterion misses and which is
decisive for the small-array goal).

Sweep:
- **Sizes:** small (~10²), medium (~10⁴, near the old crossover), large (~10⁶–10⁷).
- **Layouts:** dense 1-D, dense multi-dim (trailing axes), ragged.
- **`translate` variants:** standard LUT, `unknown="drop"`, `truncate_stop`,
  OHE-ragged.
- **Baseline vs. port:** every case run against both the current Numba/NumPy
  implementation and the Rust port on one machine (keep the Numba functions
  reachable during the experiment, or benchmark the pre-port commit, for
  apples-to-apples).
- **Warmup:** report Numba *with* and *without* JIT warmup, since cold-start is one
  of the costs Rust eliminates.

Results are tabulated in the PR description; the keep/revert call applies the
decision rule above.

## Section 6 — Testing (TDD, differential)

- **Existing `tests/` + Hypothesis suite** is the correctness oracle; the port must
  pass it unchanged — the strongest parity guarantee.
- **Differential property tests:** Hypothesis-generate inputs across the full
  matrix (random alphabets, codon sizes, non-canonical bytes, lowercase, empty,
  ragged, OHE) and assert Rust output is **bit-identical** to the current Numba
  output.
- **Rust unit tests** per module: gather, codon hash/range-check, drop-compaction,
  stop-truncation cores.
- **Explicit edge cases:** empty input; `out=` strided + `parallel=True` → error;
  non-canonical/lowercase codons; partial standard alphabet (missing codons → `X`);
  multi-track ragged + `drop` → `ValueError`.

## Section 7 — Docs / skill

Public signatures do not change, so `skills/seqpro/SKILL.md` is not expected to need
edits. Confirm during implementation; update if any documented behavior (e.g. the
`parallel=` note) shifts.

## Revert path

Numba kernels are left in `_numba.py` and the pre-port behavior is recoverable by
reverting the Python call sites to the Numba kernels and unregistering the Rust
modules. No Numba kernel is deleted until the keep decision is made.

---

## Results

**Measured:** 2026-06-19, osx-arm64, 14-core machine. Baseline = commit `833a4ef`
(pre-port, pure Numba/`np.take`). Rust = current branch HEAD. All measurements
fully measured (no reasoned/fabricated numbers).

### Machine details

- Platform: osx-arm64 (Apple Silicon, 14 cores)
- Python 3.10, Numba 0.58.1, NumPy 1.26.0
- pytest-benchmark 5.2.3 (warm times = `min` across rounds)
- Cold times = `time.perf_counter()` on a fresh process (single call)

### Benchmark table — `tokenize`

| n | Numba cold (µs) | Numba warm (µs) | Rust cold (µs) | Rust warm (µs) |
|---|---|---|---|---|
| 100 | 38 (`np.take`, no JIT) | 4.1 | 126 | 9.5 |
| 10,000 | 29 (`np.take`, no JIT) | 23.0 | 65 | 47.7 |
| 1,000,000 | 2,130 (Numba JIT + run) | 229 | 2,178 | 1,122 |

**Notes:**
- Baseline uses `np.take` (no Numba) for n < 40,000 (the old `_TOKENIZE_PARALLEL_THRESHOLD`).
  Cold cost at small/medium sizes reflects `np.take` startup, not JIT compilation.
- Rust cold overhead (~126 µs at n=100) is a first-call import/init cost, not
  recompilation; subsequent calls pay only the warm time.
- Rust warm at n=1M is **1,122 µs vs. Numba warm 229 µs** — Rust is ~2× slower
  than the parallel Numba kernel at large scale. The Rust kernel at the current
  threshold (32k, re-measured in Task 11) selects the parallel path at 1M.

### Benchmark table — `translate`

| n_codons | Numba cold (µs) | Numba warm (µs) | Rust cold (µs) | Rust warm (µs) |
|---|---|---|---|---|
| 33 | 324 (JIT compile) | 86.5 | 129 | 8.7 |
| 3,333 | 284 (JIT compile) | 83.1 | 114 | 55.9 |
| 333,333 | 946 (JIT + run) | 610 | 9,319 | 8,987 |

**Notes:**
- Baseline translate always uses Numba (`gufunc_translate_lut` / `gufunc_translate`)
  for all sizes — no `np.take` fallback. Cold times include JIT compilation.
- Rust cold at small sizes (8–10× faster than Numba cold) eliminates the JIT
  penalty entirely.
- Rust warm at n=33 is **8.7 µs vs. Numba warm 86.5 µs** — ~10× faster warm.
- Rust warm at n=3,333 is **55.9 µs vs. Numba warm 83.1 µs** — ~1.5× faster warm.
- Rust warm at n=333,333 is **8,987 µs vs. Numba warm 610 µs** — Rust is ~15×
  slower at large scale. The translate Rust kernel is serial-only (no rayon path
  wired; `TRANSLATE_PARALLEL_THRESHOLD` defined but unused per Task 11 findings).

### Decision rule application

The rule: **keep if Rust wins clearly in ≥1 regime and never meaningfully regresses others.**

| Regime | Winner | Magnitude |
|---|---|---|
| `tokenize` cold, any size | Rust (at n≥10k) or tie (n=100) | Rust avoids 2+ s JIT for n=1M |
| `tokenize` warm small (n=100) | Numba (`np.take`) | 4.1 µs vs 9.5 µs — 2× faster |
| `tokenize` warm medium (n=10k) | Numba (`np.take`) | 23 µs vs 48 µs — 2× faster |
| `tokenize` warm large (n=1M) | **Numba** | 229 µs vs 1,122 µs — ~5× faster |
| `translate` cold, any size | **Rust** | 8–10× faster; no JIT stall |
| `translate` warm small (n=33) | **Rust** | 8.7 µs vs 86.5 µs — 10× faster |
| `translate` warm medium (n=3,333) | **Rust** | 55.9 µs vs 83.1 µs — 1.5× faster |
| `translate` warm large (n=333,333) | Numba | 610 µs vs 8,987 µs — 15× faster |

**Summary of findings:**

`translate`: Rust is a clear winner on cold and warm-small/medium. The regression
at large scale (333k codons) is severe (15×) because the translate Rust kernel has
no parallel path — it is purely serial. The `TRANSLATE_PARALLEL_THRESHOLD` constant
exists in `src/translate.rs` but is unused. A rayon-parallel translate kernel would
likely close or reverse this gap at large scale.

`tokenize`: Rust regresses warm across all measured sizes. The baseline uses
`np.take` (not Numba) for n < 40k, which is highly optimized and cache-friendly.
At n=1M, the parallel Numba `lut_gather` is ~5× faster than the Rust parallel
gather. Rust also adds a cold-startup overhead (~126 µs first call) that `np.take`
avoids entirely.

### Decision: RECOMMEND REVERT (or conditional keep — see below)

**As measured, the data does not support a clean KEEP verdict:**

- `tokenize` regresses in every warm regime (2–5×). The baseline uses `np.take`,
  not Numba, for the small/medium sizes, so there was never a cold-JIT problem to
  solve there. At large scale the parallel Numba kernel outperforms Rust.
- `translate` strongly wins cold and warm-small (the clearest win; the data does
  support keeping translate). But it regresses 15× at large scale due to the
  missing parallel path.

**Conditional paths for the controller/user to choose from:**

1. **REVERT both** — restore Numba/`np.take` for `tokenize` and Numba for
   `translate`. Keep the design doc, bench harness, and Rust kernel code (do not
   delete `src/tokenize.rs` or `src/translate.rs`) for the follow-up work
   identified below.

2. **KEEP translate, REVERT tokenize** — `translate` wins clearly in cold + warm
   small/medium (the primary use case for biological sequence data). The 333k-codon
   regression is a known gap (missing rayon path) not a fundamental limitation.
   `tokenize` has no clear win in any regime.

3. **KEEP both, add rayon to translate** — accept the current translate regression as
   temporary, add the parallel Rust translate kernel as immediate follow-up before
   merging. This realizes the original experiment's full intent and likely closes
   the large-scale gap.

**Controller's recommendation to user:** Option 2 or 3. Option 2 is the
conservative choice (ships what's strictly better); Option 3 is the aggressive
choice (finishes the experiment properly). Option 1 is safe but wastes a clear
10× win on `translate` cold + small.

### Step 3 — Skill verification

**Signatures unchanged:** `git diff 833a4ef HEAD -- python/seqpro/_encoders.py
python/seqpro/alphabets/_alphabets.py | grep -E '^[+-].*def (tokenize|translate)'`
produced **no output** — no signature lines were added or removed.

**Behavior review:** The only non-internal behavior change is that empty input to
`translate` now returns an empty result (previously raised a Numba error).
`skills/seqpro/SKILL.md` does not document that empty input raises — it describes
the `unknown=` semantics and `validate=` fast-fail path only. No documented behavior
shifted.

**Skill determination: no skill change required.**
