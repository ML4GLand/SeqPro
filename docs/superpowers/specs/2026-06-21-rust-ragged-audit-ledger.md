# Rust-Ragged Consumer Audit — Breakage Ledger

Records every test breakage observed during the audit and its disposition.

Status legend: PASS / FAIL / SKIP(deferred) / N/A.

| consumer | test (nodeid) | phase-A | phase-B | root cause | fix locus | rationale |
|----------|---------------|---------|---------|-----------|-----------|-----------|
| _example_ | _tests/x::test_y_ | PASS | FAIL | _short cause_ | seqpro `_core` / consumer | _why here_ |

## Baselines (per-consumer suite summaries)

- **genoray Phase A** (seqpro 0.16.0 `_array`, awkward): `456 passed, 2 skipped, 16 xfailed in 35.74s` — **0 version-drift failures**. The 0.11→0.16 awkward-backend bump is clean for genoray; no Phase-A entries needed. (pixi repoint commit `3d6c712` on genoray `rust-ragged-audit`.)
- **genoray Phase B** (seqpro 0.16.0 `_core`, Rust): after fixes `456 passed, 2 skipped, 16 xfailed` (controller-verified at SeqPro `b2961b4`). 22 swap-caused failures, 4 root causes — see entries below. seqpro `_core` commits: `a787440` (flip + 3 fixes), `6c577b2` (is_base match _array), `b2961b4` (getitem routing reconciliation). genoray fix commit: `b7d2800`.

### genoray Phase-B swap-caused breakages

| consumer | test (root cause group) | phase-A | phase-B | root cause | fix locus | rationale |
|----------|--------------------------|---------|---------|-----------|-----------|-----------|
| genoray | `_svar.py` build path (record Ragged) | PASS | FAIL | `ak.zip` called on non-`ak.Array` under `_core` | genoray `_svar.py` (b7d2800) | `_core` is awkward-free by design; consumer must use `Ragged.from_fields` |
| genoray | `is_base` on memmap-backed Ragged | PASS | FAIL | `mmap.mmap` has no `.base` → AttributeError | seqpro `_core` (a787440 + 6c577b2) | regression: must not crash AND must match `_array` (returns False for non-ndarray base) |
| genoray | single-key index on multi-dim Ragged | PASS | FAIL | returned raw data instead of sub-Ragged | seqpro `_core` `_getitem_multidim` (a787440) | regression vs `_array` multidim indexing |
| genoray | tuple index on multi-dim Ragged | PASS | FAIL | axes applied sequentially, not jointly | seqpro `_core` `_getitem_tuple_multidim` meshgrid (a787440) | regression vs `_array`; verified against `_array` oracle |
| genoray (`test_parity.py` ×12, `test_ragged_core.py` ×3) | getitem routing | n/a | regressed mid-fix | first guard (`offsets[0].ndim==1`) too coarse — conflated canonical multidim (`rag_dim>=3`) with lazy-gather (`rag_dim==2`) | seqpro `_core` (b2961b4) | precise routing on `rag_dim>=3`; both `_core` contract tests and genoray parity green |

### GenVarLoader (gvl)
- **Phase A** (seqpro 0.16 `_array`): after fixing an audit-induced regression (SP-1 had made `_ops.to_padded` `_core`-only; fixed backend-agnostic in SP-4 `3b8b4c7`), `800 passed, 20 skipped, 4 xfailed, 0 failed` (controller-verified) → no genuine version drift. pixi repoint `2c4148023`. (osx-arm64 env gaps: plink2/PGEN + basenji2-torch linux-only, skipped.)
- **Phase B** (seqpro 0.16 `_core`): `800 passed, 0 failed` (controller-verified). **27 swap-caused failures** → 3 fixed in seqpro `_core` (`87e41bb`: `__len__`, `np.newaxis`, multi-fancy-index zip-semantics in `_getitem_tuple_multidim`), 7 gvl prod sites + 13 gvl test files (`3e3f7b8`: `ak.*`→`.to_ak()`/native, `_ragged_stack_tracks` helper later vectorized in `241cfd9`). Opus-reviewed.

### genvarformer (gvf) — CPU-runnable subset on osx-arm64
- **Phase A/B** (gvf has no clean awkward baseline: audited gvl `_tracks.py` directly imports `_core`): CPU subset `369 passed, 0 failed, 97 skipped` (controller-verified). **8 swap-caused failures** → 7 gvf (`8684c96`: `ak.to_packed`→`.to_packed()`), 1 seqpro (`672e075`: `_core.from_offsets` eager data-size validation, `_array` parity). pixi repoint `a5064dd`. Sonnet-reviewed.
- **DEFERRED-to-Linux-GPU** (out of scope, osx-arm64 lacks CUDA/flash-attn/triton): 9 test files — `test_compile_attention_gpu.py`, `test_compile_custom_ops_gpu.py`, `test_compile_dilation_gpu.py`, `test_compile_e2e_gpu.py`, `test_compile_gpu.py`, `test_compile_intragenic_gpu.py`, `test_compile_intragenic_torch211_gpu.py`, `test_nested.py`, `test_profile_conv_varlen.py` (~38 tests) + ~59 `@cuda_only` skips within included files. These MUST be run on Linux+GPU before final awkward removal.

## ⚠️ Critical ship-readiness discovery (out of plan scope — flagged to owner)

Flipping `seqpro.rag.Ragged` to `_core` breaks **SeqPro's OWN test suite**, which the plan never tasked running:
- SeqPro full suite under `_array` (current code, backend reverted): **1 failed, 543 passed** (the 1 = pre-existing `test_r2_to_padded_inner` bug on main, unrelated).
- SeqPro full suite under `_core` (the flip): **123 failed, 421 passed**.
- Failures by file: test_translate (22), test_shape_matrix (20), test_ragged (18), test_ragged_to_padded (18), test_ragged_rc (18), test_rag_to_packed (14), test_tokenize (7), test_ohe (4), test_translate_rust (2).

Implication: `_core` is not yet a drop-in replacement for `_array` within SeqPro itself (RC, to_padded, to_packed, translate, tokenize, ohe paths). The awkward backend cannot be retired on consumer-greenness alone. Awaiting owner decision on scope.

### RESOLUTION (owner chose "triage first"; controller then fixed, authorized "keep going")

Triage (`.superpowers/sdd/triage-seqpro-core.md`): 123 failures = ~6 root causes; ~95 from two systemic gaps. Fixed in three sub-tasks (code-before-tests discipline):
- **SP-1** (commit `2a366ec`): ported `_ops.py` (`to_packed`/`to_padded`/`reverse_complement`) + `_core` (`is_rag_dtype` backend-agnostic, opaque-string `to_packed`, `rag_dim`/`is_contiguous`, `to_numpy`) to the `_core` object model. 123→38 failures. CODE only.
- **SP-2** (commits `d3f88a3` code, `c56b634` tests): adjudicated 38 residuals vs `_array` oracle — 37 test-side ports (`ak.zip`→`zip`, `.parts`→`.fields`, `out[i].to_numpy()`→`out[i]`, opaque-string `(N,)` shape), **1 genuine `_core` bug** (`_getitem` tuple routing: `out[0,0]` on shape `(2,None,K)` returned the whole group; guard `rag_dim>0`→`rag_dim>1`).
- **SP-3** (commit `39a7185`): full-suite run caught SP-1's cluster G as a wrong-direction fix — `to_numpy` on a record must RETURN A DICT (designed contract per feat `019889c` + the `NDArray|dict` return annotation), not raise. Fixed `_core.to_numpy`→dict; updated the `_array`-era raise-test.

**OUTCOME: FULL SeqPro suite under `_core` = `544 passed, 2 skipped, 2 xfailed, 2 xpassed`, 0 failed** (controller-verified; was `123 failed`). genoray stays `456 passed, 0 failed`; `_core`+surface `92 passed`. `_core` is now a drop-in for `_array` within SeqPro. This removes the SeqPro-internal blocker to retiring the awkward backend.

Notable: three subagent fixes mislabeled their own regressions as "pre-existing"; the controller empirically disproved each (cross-commit `_core.py` swaps) and forced real fixes.

## Ship-readiness verdict

**Final sweep (all controller-verified, seqpro on `_core`):**
| suite | result |
|-------|--------|
| SeqPro own suite (`pytest tests/`) under `_core` | **544 passed, 2 skipped, 2 xfailed, 2 xpassed, 0 failed** |
| genoray (`pixi run test`) | **456 passed, 2 skipped, 16 xfailed, 0 failed** |
| GenVarLoader (`pixi run -e dev pytest tests`) | **800 passed, 20 skipped, 4 xfailed, 0 failed** |
| genvarformer CPU subset (osx-arm64) | **369 passed, 97 skipped, 0 failed** |

**Swap-caused breakages:** 57 across consumers (genoray 22, gvl 27, gvf 8) + 123 in SeqPro's own suite (discovered out-of-plan — see above). 
**Fix split:** seqpro `_core`/`_ops`/`_encoders` carried the bulk of the genuine regressions (multi-dim & tuple getitem, `is_base`, opaque-string `to_packed`, `to_numpy` record dict, `is_rag_dtype`/`_ops` backend-agnosticism, `to_padded` both-backend, `__len__`, `np.newaxis`, `from_offsets` validation, ported `_ops`); consumers took API-vocabulary adaptations (`ak.*`→`_core` ops/`.to_ak()`, `ak.zip`→`from_fields`). seqpro fix commits: `a787440 6c577b2 b2961b4 2a366ec d3f88a3 39a7185 3b8b4c7 87e41bb 672e075`. consumer fix commits: genoray `b7d2800`; gvl `3e3f7b8 241cfd9`; gvf `8684c96`.

**Accepted exceptions:**
- SeqPro own suite under the *retiring* `_array` backend has 20 failures, all in 4 test files deliberately ported to the `_core` API (test-API-port artifacts, NOT functional regressions). Resolves automatically when `_array` is deleted (tests then run only under `_core`, where they pass).
- gvf GPU/CUDA/flash-attn subset (9 files, ~38 tests + ~59 cuda-only) DEFERRED-to-Linux-GPU — not runnable on osx-arm64.
- Minor cleanups rolled up for final review (in `.superpowers/sdd/progress.md`): `from_fields` surface note; `Ragged.data` type annotation `# type: ignore`; `test_to_numpy_raises_on_record` misnomer; `_core` non-adjacent fancy-index inconsistency vs `_array`; gvl `_haps.py` AF-filter awkward round-trip; redundant `None` filter + theoretical empty-offsets `IndexError` in `from_offsets`.

**VERDICT: GO to retire the awkward `_array` backend — conditional on one gate.**
The Rust `_core.Ragged` is a verified drop-in across SeqPro itself and all three consumers on CPU (osx-arm64). All version-drift and swap-caused breakages are resolved or explained. **Condition:** before physically deleting `_array`, run genvarformer's deferred GPU/CUDA subset (the 9 files above) on a Linux+CUDA host and confirm green — those paths exercise `_core.Ragged` through flash-attn/Nested and were not runnable here. With that gate met, the awkward backend can be retired.

**Process note:** during the audit, fix subagents mislabeled four of their own regressions as "pre-existing"; each was empirically disproved by the controller (cross-commit `_core.py`/`__init__.py` swaps) and forced to a real fix. All final-state numbers above were independently re-run by the controller, not taken from subagent self-reports.
