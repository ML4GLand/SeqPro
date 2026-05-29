# Translate LUT Completion + Input Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `AminoAlphabet.translate` safe and complete — add an opt-in `validate` flag that guarantees in-alphabet input, route the Ragged path through the O(1) codon LUT, and remove the maintainability/clarity smells left by PR #35.

**Architecture:** Build on commit `779bd97` (PR #35's LUT work, already in this branch). The bit-pack hash gets a single njit source of truth; LUT construction is gated by an explicit predicate instead of try/except; a `validate` flag runs a vectorized in-alphabet check (byte membership for string input, structural one-hot check for OHE input) before any kernel; the Ragged path gains the same `codon_lut`-vs-fallback branch the dense path already has.

**Tech Stack:** Python, NumPy, Numba (`@nb.njit` / `@nb.guvectorize`), awkward-array (`Ragged`), pytest + hypothesis + pytest-cases, BioPython (test ground truth), pixi (env), commitizen (conventional commits).

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `python/seqpro/_numba.py` | Numba kernels. Add `_pack_codon_index` njit helper (single source of truth for the codon→index hash); `gufunc_translate_lut` calls it. Strip speed claims from docstrings. | Modify |
| `python/seqpro/alphabets/_alphabets.py` | `AminoAlphabet`. Add `_can_build_lut` predicate, `_valid_nuc_bytes`, `_check_nuc_bytes`, `_check_ohe_rows`, `validate` kwarg on `translate`, Ragged LUT branch. `_build_translate_lut` no longer raises. | Modify |
| `tests/test_translate.py` | New tests for predicate, validation (byte + OHE), Ragged LUT, fallback. | Modify |
| `tests/bench_translate_lut.py` | Benchmark (renamed from `tests/test_translate_lut_bench.py` so pytest stops collecting it). | Rename + edit |
| `skills/seqpro/SKILL.md` | Document the `validate` flag (required by CLAUDE.md for public-signature changes). | Modify |

**Environment note:** All commands run inside the dev environment. Prefix with `pixi run -e dev`. No `maturin develop` is needed — only Python (`_numba.py` is Numba-JIT, not the Rust extension). Numba recompiles changed kernels automatically.

**Baseline check (run once before starting):**

```bash
pixi run -e dev pytest tests/test_translate.py -q
```
Expected: all pass (this is the regression baseline the refactors must preserve).

---

### Task 1: Single source of truth for the codon→index hash

**Files:**
- Modify: `python/seqpro/_numba.py` (add `_pack_codon_index` before `gufunc_translate_lut` at line 145; update `gufunc_translate_lut` body at lines 191-195)
- Modify: `python/seqpro/alphabets/_alphabets.py` (import + use in `_build_translate_lut`, lines 10-16 and 254-267)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_translate.py` (after the existing `# --- LUT path tests ---` block):

```python
def test_pack_codon_index_bijective_over_acgt():
    """The 64 ACGT codons map to 64 distinct indices covering exactly [0, 63]."""
    from seqpro._numba import _pack_codon_index

    idxs = set()
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                idxs.add(int(_pack_codon_index(ord(a), ord(b), ord(c))))
    assert idxs == set(range(64))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/test_translate.py::test_pack_codon_index_bijective_over_acgt -v`
Expected: FAIL with `ImportError: cannot import name '_pack_codon_index'`

- [ ] **Step 3: Add the njit helper in `_numba.py`**

Insert immediately before the `@nb.guvectorize` decorator of `gufunc_translate_lut` (currently line 145):

```python
@nb.njit(cache=True)
def _pack_codon_index(b0, b1, b2):
    """Pack a 3-codon's ASCII bytes into a 6-bit LUT index in ``[0, 63]``.

    Uses the 2-bit-per-nucleotide hash ``(byte >> 1) & 3``, a bijection on
    ``{A, C, G, T}``. This is the single source of truth for the codon→index
    mapping: both the runtime lookup (:func:`gufunc_translate_lut`) and the
    table builder (``_build_translate_lut``) call it, so they cannot drift
    out of sync.
    """
    n0 = (b0 >> 1) & 3
    n1 = (b1 >> 1) & 3
    n2 = (b2 >> 1) & 3
    return (n0 << 4) | (n1 << 2) | n2
```

- [ ] **Step 4: Use the helper in `gufunc_translate_lut`**

Replace the body of `gufunc_translate_lut` (currently lines 191-195):

```python
    n0 = (seq_kmers[0] >> 1) & 3
    n1 = (seq_kmers[1] >> 1) & 3
    n2 = (seq_kmers[2] >> 1) & 3
    idx = (n0 << 4) | (n1 << 2) | n2
    res[0] = codon_lut[idx]
```

with:

```python
    res[0] = codon_lut[_pack_codon_index(seq_kmers[0], seq_kmers[1], seq_kmers[2])]
```

- [ ] **Step 5: Use the helper in `_build_translate_lut`**

In `python/seqpro/alphabets/_alphabets.py`, add `_pack_codon_index` to the `_numba` import block (currently lines 10-16):

```python
from .._numba import (
    _nb_find_stop_ends,
    _pack_codon_index,
    gufunc_complement_bytes,
    gufunc_translate,
    gufunc_translate_lut,
    ufunc_comp_dna,
)
```

Then in `_build_translate_lut`, replace the index computation (currently lines 262-265):

```python
        n0 = (n0_byte >> 1) & 3
        n1 = (n1_byte >> 1) & 3
        n2 = (n2_byte >> 1) & 3
        idx = (n0 << 4) | (n1 << 2) | n2
```

with:

```python
        idx = _pack_codon_index(n0_byte, n1_byte, n2_byte)
```

(Leave the surrounding `for codon, aa in zip(...)` loop and validation untouched in this task — that is cleaned up in Task 2.)

- [ ] **Step 6: Run the new test and the full translate suite**

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS — the new test passes and all pre-existing translate tests stay green (the refactor is behavior-preserving).

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/_numba.py python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "refactor(translate): single source of truth for codon LUT index"
```

---

### Task 2: Gate LUT build with a predicate; drop exception control-flow

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (`_build_translate_lut` lines 223-267; new `_can_build_lut`; `__init__` lines 317-323)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translate.py`:

```python
def test_can_build_lut_predicate():
    from seqpro.alphabets._alphabets import _can_build_lut

    assert _can_build_lut(["ATG", "AAA"]) is True
    assert _can_build_lut(["AUG"]) is False  # U is not in ACGT
    assert _can_build_lut(["AT", "ATG"]) is False  # not all length-3
    assert _can_build_lut(["AT"]) is False


def test_nonstandard_alphabet_has_no_lut():
    """A non-ACGT alphabet falls back: codon_lut is None."""
    alpha = sp.AminoAlphabet(["AUG", "UAA"], ["M", "*"])
    assert alpha.codon_lut is None


def test_partial_acgt_alphabet_fills_unknown_with_X():
    """A k=3 ACGT alphabet missing codons still builds a LUT; absent codons
    resolve to the 'X' sentinel rather than garbage."""
    from seqpro._numba import _pack_codon_index

    alpha = sp.AminoAlphabet(["ATG"], ["M"])
    assert alpha.codon_lut is not None
    idx_atg = int(_pack_codon_index(ord("A"), ord("T"), ord("G")))
    assert chr(alpha.codon_lut[idx_atg]) == "M"
    idx_aaa = int(_pack_codon_index(ord("A"), ord("A"), ord("A")))
    assert chr(alpha.codon_lut[idx_aaa]) == "X"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_translate.py::test_can_build_lut_predicate -v`
Expected: FAIL with `ImportError: cannot import name '_can_build_lut'`

- [ ] **Step 3: Add the `_can_build_lut` predicate**

In `python/seqpro/alphabets/_alphabets.py`, immediately before `def _build_translate_lut(` (currently line 223):

```python
def _can_build_lut(codons: list[str]) -> bool:
    """True when the standard O(1) LUT path applies: every codon is length-3
    and uses only the four standard nucleotides A, C, G, T. Non-standard
    alphabets (different codon size or extended/IUPAC characters) use the
    generic :func:`gufunc_translate` scan instead."""
    return all(len(c) == 3 and set(c) <= set("ACGT") for c in codons)
```

- [ ] **Step 4: Make `_build_translate_lut` non-raising and fix its docstring**

Replace the entire `_build_translate_lut` function (currently lines 223-267) with:

```python
def _build_translate_lut(
    codons: list[str], amino_acids: list[str]
) -> NDArray[np.uint8]:
    """Build the 64-entry codon→AA lookup table consumed by ``gufunc_translate_lut``.

    The packed index for each codon is computed by ``_pack_codon_index`` — the
    same hash the runtime uses — so the table and the lookup cannot drift.

    Callers must gate construction with ``_can_build_lut`` (length-3, ACGT-only
    codons); this function does not re-validate. The table is initialised to the
    ``'X'`` (unknown amino acid) byte so that a *partial* standard alphabet — one
    that is ACGT/k=3 but omits some of the 64 codons — resolves missing codons to
    ``'X'`` rather than uninitialised memory. For the complete standard genetic
    code all 64 slots are overwritten.

    Parameters
    ----------
    codons
        List of length-3 DNA strings (uppercase ACGT).
    amino_acids
        List of single-character amino-acid strings, aligned with ``codons``.

    Returns
    -------
    NDArray[np.uint8]
        Shape ``(64,)``; ``lut[idx]`` is the AA byte for the codon at packed
        index ``idx``.
    """
    lut = np.full(64, ord("X"), dtype=np.uint8)
    for codon, aa in zip(codons, amino_acids, strict=True):
        idx = _pack_codon_index(ord(codon[0]), ord(codon[1]), ord(codon[2]))
        lut[idx] = ord(aa)
    return lut
```

- [ ] **Step 5: Replace try/except with the predicate in `__init__`**

In `AminoAlphabet.__init__`, replace the LUT-build block (currently lines 317-323):

```python
        # Build the 64-entry O(1) lookup table when the alphabet is the
        # standard ACGT × k=3 case; fall back to the generic linear-scan
        # path for non-standard alphabets.
        try:
            self.codon_lut = _build_translate_lut(codons, amino_acids)
        except ValueError:
            self.codon_lut = None
```

with:

```python
        # Build the 64-entry O(1) lookup table only for the standard ACGT × k=3
        # case; non-standard alphabets use the generic linear-scan path.
        if _can_build_lut(codons):
            self.codon_lut = _build_translate_lut(codons, amino_acids)
        else:
            self.codon_lut = None
```

- [ ] **Step 6: Run tests**

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS — new predicate/fallback/partial tests pass; existing tests stay green.

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "refactor(translate): gate LUT build with predicate, drop exception control-flow"
```

---

### Task 3: Add `validate` flag — byte-membership path

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (class attr near line 276; `__init__` add `_valid_nuc_bytes`; new `_check_nuc_bytes`; `validate` on 3 overloads + impl; dense + Ragged-bytes call sites)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translate.py`:

```python
def test_translate_validate_passes_on_valid_acgt():
    # Should not raise.
    sp.AA.translate("ATGAAA", length_axis=-1, validate=True)


def test_translate_validate_raises_on_lowercase():
    with pytest.raises(ValueError, match="outside the alphabet"):
        sp.AA.translate("atgAAA", length_axis=-1, validate=True)


def test_translate_validate_raises_on_N():
    with pytest.raises(ValueError, match="outside the alphabet"):
        sp.AA.translate("ATGNNN", length_axis=-1, validate=True)


def test_translate_validate_false_does_not_raise_on_invalid():
    # Default path performs no validation and must not raise.
    out = sp.AA.translate("ATGNNN", length_axis=-1, validate=False)
    assert out.shape[-1] == 2


def test_translate_ragged_bytes_validate_raises():
    rag = _make_ragged_bytes("ATGNNN")
    with pytest.raises(ValueError, match="outside the alphabet"):
        sp.AA.translate(rag, validate=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_translate.py::test_translate_validate_raises_on_N -v`
Expected: FAIL with `TypeError: translate() got an unexpected keyword argument 'validate'`

- [ ] **Step 3: Declare the attribute and precompute the valid-byte set**

In `python/seqpro/alphabets/_alphabets.py`, add a class attribute after the `codon_lut` declaration/docstring (currently ends line 280):

```python
    _valid_nuc_bytes: NDArray[np.uint8]
    """Unique nucleotide bytes across all codons (e.g. ``ACGT`` for the standard
    alphabet), used by ``translate(validate=True)`` to check string input."""
```

In `__init__`, after the `self.codon_lut = ...` block from Task 2 (the new if/else), add:

```python
        # Unique nucleotide bytes across all codons, for translate(validate=True).
        nuc_chars = sorted({c for codon in codons for c in codon})
        self._valid_nuc_bytes = np.array(
            [ord(c) for c in nuc_chars], dtype=np.uint8
        )
```

- [ ] **Step 4: Add the `_check_nuc_bytes` helper**

Add as a method on `AminoAlphabet` (place it just before the `@overload` block for `translate`, currently line 325):

```python
    def _check_nuc_bytes(self, buf: NDArray[np.uint8]) -> None:
        """Raise ``ValueError`` if any byte in ``buf`` is outside the alphabet's
        nucleotides. Used by ``translate(validate=True)`` for string input."""
        ok = np.isin(buf, self._valid_nuc_bytes)
        if not bool(ok.all()):
            bad = np.unique(buf[~ok]).astype(np.uint8).tobytes().decode(
                "ascii", "replace"
            )
            allowed = self._valid_nuc_bytes.tobytes().decode("ascii")
            raise ValueError(
                f"translate(validate=True): input contains characters outside "
                f"the alphabet {{{allowed}}}: found {bad!r}."
            )
```

- [ ] **Step 5: Add `validate` to all three overloads and the implementation**

In each of the three `@overload def translate(...)` signatures and the implementation `def translate(...)` (currently lines 325-359), add a keyword-only parameter after `truncate_stop: bool = False,`:

```python
        validate: bool = False,
```

So each signature's keyword-only block reads:

```python
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
```

(The third overload keeps `nuc_alphabet: NucleotideAlphabet` without a default — only append `validate: bool = False` to it.)

Add a `validate` entry to the implementation's docstring Parameters section (after `truncate_stop`):

```python
        validate
            When True, raise ValueError if any input nucleotide is outside this
            alphabet (e.g. lowercase, ``N``, or other non-ACGT bytes; for OHE
            input, any row that is not exactly one-hot). When validation passes,
            the translation is guaranteed exact. Default False (no checking).
```

- [ ] **Step 6: Call the check in the dense path**

In the dense branch, immediately after `seqs = cast_seqs(seqs)` (currently line 383):

```python
            if validate:
                self._check_nuc_bytes(seqs.view(np.uint8))
```

- [ ] **Step 7: Call the check in the Ragged-bytes path**

In the Ragged branch, the `else` arm that sets `nuc_bytes_flat = seqs.data` (currently lines 436-437) becomes:

```python
        else:
            nuc_bytes_flat = seqs.data
            if validate:
                self._check_nuc_bytes(nuc_bytes_flat.view(np.uint8))
```

- [ ] **Step 8: Run tests**

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS — the five new `validate` byte tests pass; existing tests stay green.

- [ ] **Step 9: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "feat(translate): add validate flag for nucleotide input checking"
```

---

### Task 4: `validate` for OHE Ragged input (structural one-hot check)

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (new `_check_ohe_rows`; call in the `is_ohe` branch)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translate.py`:

```python
def test_translate_ohe_validate_passes_valid():
    ohe = sp.DNA.ohe(sp.cast_seqs("ATCGAT"))
    rag = Ragged.from_lengths(ohe, np.array([6]))
    # Should not raise.
    sp.AA.translate(rag, nuc_alphabet=sp.DNA, validate=True)


def test_translate_ohe_validate_raises_multihot():
    ohe = sp.DNA.ohe(sp.cast_seqs("ATCGAT")).copy()
    ohe[0, :] = 1  # multi-hot row (sum == 4)
    rag = Ragged.from_lengths(ohe, np.array([6]))
    with pytest.raises(ValueError, match="one-hot"):
        sp.AA.translate(rag, nuc_alphabet=sp.DNA, validate=True)


def test_translate_ohe_validate_raises_allzero():
    ohe = sp.DNA.ohe(sp.cast_seqs("ATCGAT")).copy()
    ohe[0, :] = 0  # all-zero row (sum == 0)
    rag = Ragged.from_lengths(ohe, np.array([6]))
    with pytest.raises(ValueError, match="one-hot"):
        sp.AA.translate(rag, nuc_alphabet=sp.DNA, validate=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/test_translate.py::test_translate_ohe_validate_raises_multihot -v`
Expected: FAIL — currently no OHE validation, so `translate` returns instead of raising (the multi-hot row decodes silently).

- [ ] **Step 3: Add the `_check_ohe_rows` helper**

Add as a method on `AminoAlphabet`, next to `_check_nuc_bytes`:

```python
    def _check_ohe_rows(self, data: NDArray[np.uint8], n_nuc: int) -> None:
        """Raise ``ValueError`` unless every row of OHE ``data`` (alphabet axis
        is axis 1 of the packed flat buffer) is exactly one-hot over ``n_nuc``
        nucleotides. Used by ``translate(validate=True)`` for OHE Ragged input.

        Required because decoding maps an all-zero row to the unknown sentinel
        but a *multi-hot* row silently resolves to a real nucleotide — so a
        decode-then-membership check would not catch it.
        """
        if data.shape[1] != n_nuc:
            raise ValueError(
                f"translate(validate=True): OHE width {data.shape[1]} does not "
                f"match nucleotide alphabet size {n_nuc}."
            )
        sums = data.sum(axis=1, dtype=np.int64)
        if not bool((sums == 1).all()):
            raise ValueError(
                "translate(validate=True): every OHE row must be one-hot "
                "(exactly one 1 per nucleotide position)."
            )
```

- [ ] **Step 4: Call the check in the `is_ohe` branch**

In the Ragged branch, the `is_ohe` arm (currently lines 430-435) becomes:

```python
        if is_ohe:
            if nuc_alphabet is None:
                raise ValueError("nuc_alphabet is required for OHE Ragged input.")
            if validate:
                self._check_ohe_rows(seqs.data, len(nuc_alphabet.array))
            nuc_bytes_flat: NDArray[np.bytes_] = nuc_alphabet.decode_ohe(  # type: ignore[union-attr]
                seqs.data, ohe_axis=-1
            )
```

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS — the three OHE validation tests pass; existing tests stay green.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "feat(translate): validate OHE inputs are one-hot"
```

---

### Task 5: Route the Ragged path through the LUT

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (Ragged kernel call, currently lines 444-458)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_translate.py`:

```python
def test_translate_ragged_uses_lut_matches_biopython():
    """Ragged path (now LUT-routed for the standard alphabet) matches BioPython."""
    rng = np.random.default_rng(7)
    seq = "".join(rng.choice(list("ACGT"), size=300))
    rag = _make_ragged_bytes(seq)
    out = sp.AA.translate(rag)
    expected = sp.cast_seqs(str(translate(seq)))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), expected)


def test_translate_ragged_fallback_for_nonstandard_alphabet():
    """Non-ACGT alphabet has codon_lut=None → Ragged path uses the scan fallback."""
    alpha = sp.AminoAlphabet(["AUG", "UAA"], ["M", "*"])
    assert alpha.codon_lut is None
    data = np.array(list("AUGUAA"), dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6]))
    out = alpha.translate(rag)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("M*"))
```

- [ ] **Step 2: Run tests to verify status**

Run: `pixi run -e dev pytest tests/test_translate.py::test_translate_ragged_fallback_for_nonstandard_alphabet -v`
Expected: PASS already (fallback path exists). The LUT-routing change is verified by the full suite in Step 4 — the `match_biopython` test passes before and after, but only exercises the LUT branch after the change. Both tests are committed now so the behavior is locked in.

- [ ] **Step 3: Add the LUT branch to the Ragged kernel call**

Replace the `if total > 0:` block (currently lines 444-458):

```python
        if total > 0:
            codons = np.lib.stride_tricks.sliding_window_view(
                nuc_bytes_flat, codon_size, axis=0
            )
            # sliding_window_view appends window axis at end → slice axis 0 by codon_size
            codons = codons[::codon_size]
            # codons shape: (num_codons, *trailing, codon_size); codon axis is last
            translated_flat: NDArray[np.bytes_] = gufunc_translate(
                codons.view(np.uint8),
                self.codon_array.view(np.uint8),
                self.aa_array.view(np.uint8),
                axes=[-1, (-2, -1), (-1), ()],  # type: ignore
            ).view("S1")  # (num_codons, *trailing)
        else:
            translated_flat = np.empty((0, *trailing), dtype="S1")
```

with:

```python
        if total > 0:
            codons = np.lib.stride_tricks.sliding_window_view(
                nuc_bytes_flat, codon_size, axis=0
            )
            # sliding_window_view appends window axis at end → slice axis 0 by codon_size
            codons = codons[::codon_size]
            # codons shape: (num_codons, *trailing, codon_size); codon axis is last
            translated_flat: NDArray[np.bytes_]
            if self.codon_lut is not None:
                translated_flat = gufunc_translate_lut(
                    codons.view(np.uint8),
                    self.codon_lut,
                    axes=[-1, -1, ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
            else:
                translated_flat = gufunc_translate(
                    codons.view(np.uint8),
                    self.codon_array.view(np.uint8),
                    self.aa_array.view(np.uint8),
                    axes=[-1, (-2, -1), (-1), ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
        else:
            translated_flat = np.empty((0, *trailing), dtype="S1")
```

- [ ] **Step 4: Run the full translate suite**

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS — all Ragged tests (bytes basic, truncate-stop variants, OHE) and the two new tests pass, confirming the LUT path produces identical results to the former scan path.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "perf(translate): route Ragged path through codon LUT"
```

---

### Task 6: Remove hardware-specific speed claims from docstrings; rename the bench file

**Files:**
- Modify: `python/seqpro/_numba.py` (`gufunc_translate` docstring ~line 121; `gufunc_translate_lut` docstring ~lines 156-189)
- Rename + Modify: `tests/test_translate_lut_bench.py` → `tests/bench_translate_lut.py`

- [ ] **Step 1: Soften the `gufunc_translate` docstring**

In `python/seqpro/_numba.py`, the `gufunc_translate` docstring opening (added by PR #35) currently reads:

```python
    """Translate k-mers into amino acids via O(n) linear scan.

    Generic fallback for non-standard alphabets where the codon length
    differs from 3. For standard genetic-code translation (k=3), prefer
    :func:`gufunc_translate_lut` — orders of magnitude faster via O(1)
    table lookup.
```

Replace with:

```python
    """Translate k-mers into amino acids via an O(n) linear scan.

    Generic fallback for non-standard alphabets (codon length other than 3,
    or extended/IUPAC characters). For the standard genetic code (k=3, ACGT),
    ``AminoAlphabet.translate`` automatically uses the O(1)
    :func:`gufunc_translate_lut` path instead.
```

- [ ] **Step 2: Rewrite the `gufunc_translate_lut` docstring without speed claims**

Replace the `gufunc_translate_lut` docstring (currently lines 156-189, the triple-quoted block) with:

```python
    """Translate a 3-codon to its amino acid via an O(1) lookup table.

    Selected automatically by ``AminoAlphabet.translate`` for the standard
    genetic code (k=3, ACGT); non-standard alphabets use
    :func:`gufunc_translate` instead.

    The LUT is indexed by ``_pack_codon_index``, which hashes each nucleotide's
    ASCII byte with ``(byte >> 1) & 3`` — a bijection on ``{A, C, G, T}`` — and
    packs the three 2-bit codes into a 6-bit index in ``[0, 63]``. The
    64-element ``codon_lut`` (built by ``AminoAlphabet`` at construction) returns
    the amino-acid byte for that index.

    Parameters
    ----------
    seq_kmers
        A 3-codon as ASCII bytes (e.g. ``[65, 84, 71]`` = ``"ATG"``).
    codon_lut
        64-byte LUT, built by ``AminoAlphabet`` at construction time.
    res
        Output buffer.
    """
```

- [ ] **Step 3: Rename the benchmark file and strip its speed claim**

Run:

```bash
git mv tests/test_translate_lut_bench.py tests/bench_translate_lut.py
```

Then in `tests/bench_translate_lut.py`, replace the module docstring (currently the opening triple-quoted block) with:

```python
"""Microbench: O(64) linear scan vs O(1) LUT in ``AminoAlphabet.translate``.

Not collected by pytest (filename is not ``test_*``). Run explicitly to measure
the speedup on your hardware::

    python tests/bench_translate_lut.py

Speedups are data- and hardware-dependent; this script reports measured numbers
rather than asserting a fixed multiplier.
"""
```

- [ ] **Step 4: Verify pytest no longer collects the bench and the suite passes**

Run: `pixi run -e dev pytest tests/ -q --collect-only -k bench_translate_lut`
Expected: no tests collected (0 items) — the bench is excluded.

Run: `pixi run -e dev pytest tests/test_translate.py -q`
Expected: PASS.

Run (sanity-check the bench still executes as a script): `pixi run -e dev python tests/bench_translate_lut.py`
Expected: prints timing lines and exits 0 (its `assert n_diff == 0` holds).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_numba.py tests/bench_translate_lut.py
git commit -m "docs(translate): remove hardware-specific speedup claims; rename bench"
```

---

### Task 7: Document the `validate` flag in the seqpro skill

**Files:**
- Modify: `skills/seqpro/SKILL.md`

- [ ] **Step 1: Add a note about `validate`**

In `skills/seqpro/SKILL.md`, directly after the alphabets bullet (line 25, the `Alphabets are singletons` line), add:

```markdown
- **`AminoAlphabet.translate(seqs, ..., validate=False)`**: translates nucleotides → amino acids. Pass `validate=True` to raise on any input outside the alphabet — non-ACGT bytes (lowercase, `N`, IUPAC codes) for string/byte input, or any non-one-hot row for OHE input. When `validate=True` returns without raising, the translation is guaranteed exact; the default `validate=False` skips the check for speed and treats out-of-alphabet input as undefined.
```

- [ ] **Step 2: Verify the skill mentions match the implemented signature**

Run: `grep -n "validate" skills/seqpro/SKILL.md`
Expected: shows the new line referencing `validate=False`, matching the `translate` signature from Task 3.

- [ ] **Step 3: Commit**

```bash
git add skills/seqpro/SKILL.md
git commit -m "docs: document translate validate flag in seqpro skill"
```

---

### Task 8: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the entire test suite**

Run: `pixi run -e dev pytest tests/ -q`
Expected: all tests pass (the PR #35 baseline of 260 passed + the new tests added here), no errors, no unexpected collection of the bench file.

- [ ] **Step 2: Lint and format**

Run:
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
```
Expected: no lint errors; formatting clean (run `ruff format python/ tests/` to fix if needed, then re-commit).

- [ ] **Step 3: Confirm the branch is ready**

Run: `git log --oneline 779bd97..HEAD`
Expected: the spec commit plus the seven task commits, all conventional-commit formatted.

---

## Self-Review

**Spec coverage:**
- `validate` flag (byte + OHE) → Tasks 3, 4. ✓
- Ragged path through LUT → Task 5. ✓
- Single source of truth for bit-pack → Task 1. ✓
- Predicate instead of try/except + honest `'X'` sentinel → Task 2. ✓
- Remove speed claims from docstrings → Task 6. ✓
- Rename bench out of pytest collection → Task 6. ✓
- SKILL.md update → Task 7. ✓
- Default path zero-overhead (validate defaults False) → Task 3. ✓
- `feat:` overall (commit types: refactor/feat/perf/docs across tasks; PR-level is feat) → consistent with versioning section of spec. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. ✓

**Type/name consistency:** `_pack_codon_index` (Task 1) used in Tasks 1, 2, test code; `_can_build_lut` (Task 2); `_valid_nuc_bytes` / `_check_nuc_bytes` (Task 3); `_check_ohe_rows` (Task 4) — names identical across all references. `validate` keyword added to all three overloads + impl (Task 3). ✓
