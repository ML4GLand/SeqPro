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

_Filled in Task 10._
