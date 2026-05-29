# Design: codon→AA LUT completion + optional input validation

Date: 2026-05-28
Status: Approved (pending written-spec review)
Scope: hardening and completing PR #35 (`perf(translate): O(1) LUT codon→AA lookup`)

## Problem

PR #35 added an O(1) lookup-table (LUT) path for `AminoAlphabet.translate` on the
standard genetic code. Review surfaced four issues:

1. **Silent mistranslation of non-ACGT input.** The LUT index hash
   `(byte >> 1) & 3` is a bijection only on `{A, C, G, T}`. Every other byte
   (lowercase `acgt`, `N`, IUPAC ambiguity codes, arbitrary ASCII) aliases onto
   one of the four buckets and is silently translated as if it were a standard
   nucleotide. The old linear-scan path left such cases as uninitialized output
   buffer (garbage). Neither is correct; there is no way for a caller to get a
   guarantee that input was valid.
2. **The Ragged path was not optimized.** Only the dense/non-Ragged branch was
   routed through the LUT. The motivating workload (genome-scale, variable-length
   sequences) flows through the Ragged branch, which still used the slow scan.
3. **Maintainability / clarity smells.** The bit-pack logic is duplicated in
   Python (`_build_translate_lut`) and Numba (`gufunc_translate_lut`) and must be
   kept in sync by hand; LUT-building used control-flow-by-exception; docstrings
   carried hardware-specific speedup numbers; the `'X'` sentinel comment was
   misleading; a benchmark file was named `test_*` and so was collected by pytest.

## Goals

- Give callers a way to guarantee valid input: a `validate` flag that raises on
  any out-of-alphabet input, so that **if `validate=True` returns without
  raising, the translation is exact**.
- Route the Ragged path through the LUT for standard alphabets.
- Remove the duplication and clarity smells.
- Keep the default path zero-overhead (no behavior change for existing valid
  ACGT callers).

## Non-goals

- Changing translation semantics for valid ACGT k=3 input (output stays
  bit-for-bit identical).
- Auto-uppercasing or coercing input. `validate` reports problems; it does not
  silently fix them.
- Validating codon-table *completeness* for arbitrary custom alphabets.
  `validate` checks nucleotide membership; for the standard genetic code (the
  only complete-by-construction case) that is sufficient for exactness.

## Versioning

Adding `validate` is a new public capability → conventional-commit type is
`feat(translate):` → **MINOR** bump under the project's pre-1.0 semver
(`0.x` feat → `0.(x+1).0`). This was explicitly chosen over splitting `validate`
into a separate PR to keep the change in one place.

## Design

### 1. `validate: bool = False` on `AminoAlphabet.translate`

- Add as a keyword-only argument to every `translate` overload and the
  implementation. Default `False` preserves the zero-overhead fast path.
- At `AminoAlphabet.__init__`, precompute once:
  - `self._valid_nuc_bytes`: the unique nucleotide bytes across all `codons`
    (for standard AA → `{65, 67, 71, 84}` = `ACGT`), as a small `uint8` array.
- When `validate=True`, run a pre-check **before** either kernel, independent of
  whether the LUT or fallback runs:
  - **Byte input (dense and Ragged-bytes):** `np.isin(buf.view(np.uint8),
    self._valid_nuc_bytes)`; if any element is out of set, raise `ValueError`
    naming the offending unique characters (decoded for readability).
  - **OHE Ragged input:** a structural one-hot check on `seqs.data` over the
    alphabet axis (axis 1 in the packed flat buffer): every row must have
    `sum == 1` (for non-negative `uint8` this is exactly one-hot), and the
    alphabet-axis width must equal the nucleotide alphabet size. Raise
    `ValueError` otherwise. This is required because `gufunc_ohe_char_idx`
    (`_numba.py:39`) only maps all-zero rows to the unknown sentinel; a
    **multi-hot** row silently resolves to a real nucleotide, so a
    decode-then-membership check would not catch it.
- **Guarantee:** `validate=True` returning normally ⇒ every nucleotide is in the
  alphabet ⇒ on the standard genetic code the LUT result is exact. `validate=False`
  (default) ⇒ behavior unchanged from PR #35; invalid input is undefined on the
  LUT path (documented).

### 2. Route the Ragged path through the LUT

In the Ragged branch (`_alphabets.py:384`, after any OHE decode), mirror the dense
branch: when `self.codon_lut is not None`, call `gufunc_translate_lut(codons.view(
np.uint8), self.codon_lut, axes=[-1, -1, ()])`; otherwise fall back to
`gufunc_translate`.

### 3. Single source of truth for the bit-pack

Add a module-level `@nb.njit` helper:

```python
def _pack_codon_index(b0, b1, b2):
    return (((b0 >> 1) & 3) << 4) | (((b1 >> 1) & 3) << 2) | ((b2 >> 1) & 3)
```

Call it from both `gufunc_translate_lut` (Numba) and `_build_translate_lut`
(Python — njit functions are callable from the interpreter). The runtime lookup
and the table construction can no longer drift out of sync.

### 4. Predicate instead of try/except; honest `'X'` sentinel

- Replace the `try/except ValueError` in `__init__` with an explicit predicate
  `_can_build_lut(codons)` → `all(len(c) == 3 and set(c) <= set("ACGT") for c in
  codons)`. If true, build the LUT; else `self.codon_lut = None`.
- `_build_translate_lut` no longer raises (the predicate gates it).
- Keep the `np.full(64, ord("X"))` initialization: it is a genuine safety net for
  a *partial* ACGT/k=3 alphabet (a missing codon → `'X'`, not garbage). Fix the
  comment, which wrongly implied the sentinel is load-bearing for the complete
  64-codon standard alphabet (where every slot is overwritten).

### 5. Docstrings carry no speed claims

Remove every concrete multiplier from `gufunc_translate_lut`, `_build_translate_lut`,
and the benchmark module docstring. Docstrings describe behavior and guide usage
(the LUT path is selected automatically by `AminoAlphabet.translate` on the
standard genetic code; the linear scan is the fallback for non-standard
alphabets). Speedups are data- and hardware-dependent and belong only in the
benchmark's runtime output.

### 6. Benchmark file

Rename `tests/test_translate_lut_bench.py` → `tests/bench_translate_lut.py` so
pytest's default `test_*` collection skips it. It remains runnable as
`python tests/bench_translate_lut.py`. (No `testpaths` override exists in
`pyproject.toml`, so renaming is sufficient to exclude it.)

### 7. SKILL.md

Per CLAUDE.md, a public signature change requires a same-PR skill update. Add a
short note for `AminoAlphabet.translate(..., validate=False)` documenting the flag
and its guarantee.

## Testing

- **Retain** the existing equality, hypothesis fuzz, and BioPython cross-validation
  tests; they now also exercise the Ragged LUT path.
- `validate=True` raises on lowercase / `N` / non-ACGT input — dense and Ragged.
- `validate=True` passes on valid ACGT input; `validate=False` never raises.
- OHE Ragged: `validate=True` raises on a multi-hot row and on an all-zero row;
  passes on valid one-hot.
- Ragged LUT output == Ragged fallback output == dense output (bitwise).
- A non-standard alphabet (k≠3 or non-ACGT codon) → `codon_lut is None`, fallback
  path used, results still correct.

## Commit

`feat(translate): O(1) codon→AA LUT for dense and Ragged paths + optional input validation`
