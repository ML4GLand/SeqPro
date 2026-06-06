# Translate Unknown-Codon Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine PR #40 (kernel corruption fix) and PR #41 (case-insensitivity + unknown-codon policy) into one non-breaking PR with a single new `translate(..., unknown="X")` parameter.

**Architecture:** Two Numba translate kernels gain a configurable `marker_byte` sentinel and unconditional `& 0xDF` case-folding. A new `@njit` compaction kernel implements `unknown="drop"`. The `AminoAlphabet.translate` wrapper parses one new string parameter `unknown` (single char ⇒ pad with it; `"drop"` ⇒ drop non-canonical codons, returning a `Ragged`). `validate=True` becomes case-insensitive and remains the single fast-fail path; the `error`/`collapse` modes and `unknown_marker` parameter from #41 are not implemented.

**Tech Stack:** Python, NumPy, Numba (`guvectorize`/`njit`), `seqpro.rag.Ragged`, pytest, pixi, ruff.

**Spec:** `docs/superpowers/specs/2026-06-05-translate-unknown-codon-policy-design.md`

**Environment note:** All commands run under pixi. Prefix test runs with `pixi run` or run inside `pixi shell -e dev`. No Rust changes here — `maturin develop` is NOT required; Numba recompiles kernels automatically when signatures change.

---

## File Structure

- `python/seqpro/_numba.py` — modify `gufunc_translate` and `gufunc_translate_lut` (add `marker_byte` + case-folding); add `_nb_drop_unknown_codons` compaction kernel.
- `python/seqpro/alphabets/_alphabets.py` — add `_valid_upper_bytes`; make `_check_nuc_bytes` case-insensitive; rewrite `translate` wrapper (parse `unknown`, pad/drop paths, 4 overloads).
- `tests/test_translate.py` — new tests for kernels, drop kernel, wrapper behaviors; port retained #40/#41 tests; delete removed-feature tests.
- `skills/seqpro/SKILL.md` — document `unknown=`.
- `CHANGELOG.md` — combined `fix` + `feat` entry.
- `CLAUDE.md` — two new convention rules.

---

## Task 1: Set up combined branch

**Files:** none (git only)

- [ ] **Step 1: Create branch off main**

```bash
cd /Users/david/projects/SeqPro
git checkout main
git pull --ff-only
git checkout -b feat/translate-unknown-codon-policy
```

- [ ] **Step 2: Confirm clean baseline test run**

Run: `pixi run test -q 2>&1 | tail -5`
Expected: existing suite passes (e.g. `... passed ...`), no errors. Records the pre-change baseline.

---

## Task 2: Add `marker_byte` + case-folding to `gufunc_translate`

**Files:**
- Modify: `python/seqpro/_numba.py` (`gufunc_translate`, lines ~109-142)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_translate.py`:

```python
import numpy as np
import seqpro as sp
from seqpro._numba import gufunc_translate


def test_gufunc_translate_marker_on_no_match():
    kmer_keys = sp.AA.codon_array.view(np.uint8)
    kmer_values = sp.AA.aa_array.view(np.uint8)
    seq = np.frombuffer(b"\x00\x00\x00", dtype=np.uint8).copy()
    out = gufunc_translate(seq, kmer_keys, kmer_values, np.uint8(ord("X")))
    assert int(out) == ord("X")
    out2 = gufunc_translate(seq, kmer_keys, kmer_values, np.uint8(ord("-")))
    assert int(out2) == ord("-")


def test_gufunc_translate_case_insensitive():
    kmer_keys = sp.AA.codon_array.view(np.uint8)
    kmer_values = sp.AA.aa_array.view(np.uint8)
    upper = gufunc_translate(
        np.frombuffer(b"ATG", np.uint8).copy(), kmer_keys, kmer_values, np.uint8(ord("X"))
    )
    lower = gufunc_translate(
        np.frombuffer(b"atg", np.uint8).copy(), kmer_keys, kmer_values, np.uint8(ord("X"))
    )
    assert int(upper) == int(lower) == ord("M")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_gufunc_translate_marker_on_no_match -v`
Expected: FAIL — `gufunc_translate` currently takes 3 args, not 4 (TypeError / wrong number of arguments).

- [ ] **Step 3: Implement the kernel change**

Replace the `gufunc_translate` decorator signature and body in `python/seqpro/_numba.py`. The decorator gufunc signature gains a scalar `u1` input:

```python
@nb.guvectorize(
    ["(u1[:], u1[:, :], u1[:], u1, u1[:])"],
    "(k),(j,k),(j),()->()",
    target="parallel",
    cache=True,
)
def gufunc_translate(
    seq_kmers: NDArray[np.uint8],
    kmer_keys: NDArray[np.uint8],
    kmer_values: NDArray[np.uint8],
    marker_byte: np.uint8,
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate k-mers into amino acids via an O(n) linear scan.

    Generic fallback for non-standard alphabets (codon length other than 3,
    or extended/IUPAC characters). For the standard genetic code (k=3, ACGT),
    ``AminoAlphabet.translate`` automatically uses the O(1)
    :func:`gufunc_translate_lut` path instead.

    A k-mer that does not match any entry in ``kmer_keys`` resolves to the
    caller-supplied ``marker_byte`` rather than leaving ``res[0]``
    uninitialised. ``guvectorize`` allocates output buffers via ``np.empty``,
    so without this sentinel a missing-codon match would emit whatever byte
    happened to be on the page — typically NUL on fresh pages, producing
    silently corrupt AA sequences downstream.

    Case-insensitivity: ASCII letters in ``seq_kmers`` and ``kmer_keys`` are
    upper-cased on the fly via ``b & 0xDF`` (the bit-5 flip is a no-op on
    uppercase ASCII alphas and turns lowercase into uppercase), so soft-masked
    / mixed-case input still translates normally.

    Parameters
    ----------
    seq_kmers
        A k-mer.
    kmer_keys
        All unique k-mers as an (n, k) array.
    kmer_values
        Values corresponding to each k-mer, in corresponding order.
    marker_byte
        ASCII byte emitted when no kmer in ``kmer_keys`` matches the input
        (i.e. an unknown codon). The Python wrapper validates this is a
        single byte.
    res
        Array to save the result in, by default None
    """
    res[0] = marker_byte  # type: ignore
    k = len(seq_kmers)
    for i in range(len(kmer_keys)):
        match = True
        for j in range(k):
            if (seq_kmers[j] & 0xDF) != (kmer_keys[i, j] & 0xDF):
                match = False
                break
        if match:
            res[0] = kmer_values[i]  # type: ignore
            break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_translate.py::test_gufunc_translate_marker_on_no_match tests/test_translate.py::test_gufunc_translate_case_insensitive -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_numba.py tests/test_translate.py
git commit -m "fix(translate): marker sentinel + case-folding in gufunc_translate"
```

---

## Task 3: Add `marker_byte` + case-folding to `gufunc_translate_lut`

**Files:**
- Modify: `python/seqpro/_numba.py` (`gufunc_translate_lut`, lines ~165-198)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

```python
from seqpro._numba import gufunc_translate_lut


def test_gufunc_translate_lut_marker_on_noncanonical():
    # Pre-fix collisions: NNN -> T, \x00\x00\x00 -> K. Post-fix: marker.
    for codon in (b"NNN", b"\x00\x00\x00"):
        seq = np.frombuffer(codon, dtype=np.uint8).copy()
        out = gufunc_translate_lut(seq, sp.AA.codon_lut, np.uint8(ord("X")))
        assert int(out) == ord("X"), codon


def test_gufunc_translate_lut_case_insensitive():
    upper = gufunc_translate_lut(
        np.frombuffer(b"ATG", np.uint8).copy(), sp.AA.codon_lut, np.uint8(ord("X"))
    )
    lower = gufunc_translate_lut(
        np.frombuffer(b"atg", np.uint8).copy(), sp.AA.codon_lut, np.uint8(ord("X"))
    )
    assert int(upper) == int(lower) == ord("M")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_gufunc_translate_lut_marker_on_noncanonical -v`
Expected: FAIL — `gufunc_translate_lut` currently takes 2 args, not 3.

- [ ] **Step 3: Implement the kernel change**

Replace `gufunc_translate_lut` decorator and body in `python/seqpro/_numba.py`:

```python
@nb.guvectorize(
    ["(u1[:], u1[:], u1, u1[:])"],
    "(k),(m),()->()",
    target="parallel",
    cache=True,
)
def gufunc_translate_lut(
    seq_kmers: NDArray[np.uint8],
    codon_lut: NDArray[np.uint8],
    marker_byte: np.uint8,
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate a 3-codon to its amino acid via an O(1) lookup table.

    Selected automatically by ``AminoAlphabet.translate`` for the standard
    genetic code (k=3, ACGT); non-standard alphabets use
    :func:`gufunc_translate` instead.

    The ``(byte >> 1) & 3`` hash is **not** a bijection outside ``{A, C, G, T}``:
    e.g. ``N`` (0x4E) and ``NUL`` (0x00) both collide onto valid LUT slots and
    would silently yield biologically wrong AAs (``NNN -> T``, ``\\x00\\x00\\x00 ->
    K``). Every codon byte is range-checked against ``{A, C, G, T}`` before the
    LUT lookup; any non-canonical byte short-circuits to the caller-supplied
    ``marker_byte`` sentinel.

    Case-insensitivity: each input byte is upper-cased via ``b & 0xDF`` before
    the range check and the LUT-index hash, so lowercase nucleotides (e.g.
    soft-masked ``acg``) translate identically to their uppercase forms.

    Parameters
    ----------
    seq_kmers
        A 3-codon as ASCII bytes (e.g. ``[65, 84, 71]`` = ``"ATG"``).
    codon_lut
        64-byte LUT, built by ``AminoAlphabet`` at construction time.
    marker_byte
        ASCII byte emitted when any codon byte is non-canonical (i.e. not in
        ``{A, C, G, T, a, c, g, t}``). The Python wrapper validates this is a
        single byte.
    res
        Output buffer.
    """
    b0 = seq_kmers[0] & 0xDF
    b1 = seq_kmers[1] & 0xDF
    b2 = seq_kmers[2] & 0xDF
    if (
        (b0 == 65 or b0 == 67 or b0 == 71 or b0 == 84)
        and (b1 == 65 or b1 == 67 or b1 == 71 or b1 == 84)
        and (b2 == 65 or b2 == 67 or b2 == 71 or b2 == 84)
    ):
        res[0] = codon_lut[_pack_codon_index(b0, b1, b2)]
    else:
        res[0] = marker_byte
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_translate.py::test_gufunc_translate_lut_marker_on_noncanonical tests/test_translate.py::test_gufunc_translate_lut_case_insensitive -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_numba.py tests/test_translate.py
git commit -m "fix(translate): marker sentinel + case-folding in gufunc_translate_lut"
```

---

## Task 4: Add the `_nb_drop_unknown_codons` compaction kernel

**Files:**
- Modify: `python/seqpro/_numba.py` (add new `@nb.njit` function near the translate kernels)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

```python
from seqpro._numba import _nb_drop_unknown_codons


def test_nb_drop_unknown_codons_compacts_per_sequence():
    # 2 sequences of 3 codons each (codon_size=3).
    # seq0 codons: ATG (keep), NNN (drop), GGG (keep)
    # seq1 codons: AAA (keep), CCC (keep), T?T (drop -- '?' non-canonical)
    codons = np.stack([
        np.frombuffer(b"ATG", np.uint8),
        np.frombuffer(b"NNN", np.uint8),
        np.frombuffer(b"GGG", np.uint8),
        np.frombuffer(b"AAA", np.uint8),
        np.frombuffer(b"CCC", np.uint8),
        np.frombuffer(b"T?T", np.uint8),
    ]).copy()
    translated = np.frombuffer(b"MXGKPX", np.uint8).copy()  # markers at dropped slots
    offsets = np.array([0, 3, 6], dtype=np.int64)
    valid_upper = np.frombuffer(b"ACGT", np.uint8).copy()

    out, new_offsets = _nb_drop_unknown_codons(translated, codons, offsets, valid_upper)
    assert out.tobytes() == b"MGKP"
    assert new_offsets.tolist() == [0, 2, 4]


def test_nb_drop_unknown_codons_case_insensitive_keep():
    # lowercase canonical codon must be KEPT (not dropped).
    codons = np.stack([
        np.frombuffer(b"atg", np.uint8),
        np.frombuffer(b"nnn", np.uint8),
    ]).copy()
    translated = np.frombuffer(b"MX", np.uint8).copy()
    offsets = np.array([0, 2], dtype=np.int64)
    valid_upper = np.frombuffer(b"ACGT", np.uint8).copy()
    out, new_offsets = _nb_drop_unknown_codons(translated, codons, offsets, valid_upper)
    assert out.tobytes() == b"M"
    assert new_offsets.tolist() == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_nb_drop_unknown_codons_compacts_per_sequence -v`
Expected: FAIL — `ImportError: cannot import name '_nb_drop_unknown_codons'`.

- [ ] **Step 3: Implement the kernel**

Add to `python/seqpro/_numba.py` (after `gufunc_translate_lut`):

```python
@nb.njit(cache=True)
def _nb_drop_unknown_codons(translated, codons, offsets, valid_upper):
    """Compact a flat translated AA buffer, dropping non-canonical codons.

    Single-pass per-sequence stream compaction for ``translate(unknown="drop")``.
    A codon is dropped iff any of its bytes — after upper-casing via ``& 0xDF``
    — is not in ``valid_upper``. Offsets are codon-indexed into ``translated``.

    Parameters
    ----------
    translated
        (num_codons,) uint8 AA bytes (S1 viewed as u1).
    codons
        (num_codons, k) uint8 input codon bytes.
    offsets
        (n+1,) int64 codon-indexed offsets into ``translated``.
    valid_upper
        (v,) uint8 upper-cased valid nucleotide bytes (e.g. ord("ACGT")).

    Returns
    -------
    (out, new_offsets)
        ``out`` is (num_kept,) uint8; ``new_offsets`` is (n+1,) int64, monotonic.
    """
    n = len(offsets) - 1
    num_codons = translated.shape[0]
    k = codons.shape[1]
    v = len(valid_upper)
    out = np.empty(num_codons, dtype=np.uint8)
    new_offsets = np.empty(n + 1, dtype=np.int64)
    new_offsets[0] = 0
    w = 0
    for s in range(n):
        start = offsets[s]
        end = offsets[s + 1]
        for c in range(start, end):
            keep = True
            for j in range(k):
                b = codons[c, j] & 0xDF
                ok = False
                for t in range(v):
                    if b == valid_upper[t]:
                        ok = True
                        break
                if not ok:
                    keep = False
                    break
            if keep:
                out[w] = translated[c]
                w += 1
        new_offsets[s + 1] = w
    return out[:w].copy(), new_offsets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_translate.py -k nb_drop_unknown -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_numba.py tests/test_translate.py
git commit -m "feat(translate): add _nb_drop_unknown_codons compaction kernel"
```

---

## Task 5: Case-insensitive `validate` + `_valid_upper_bytes`

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (`AminoAlphabet.__init__` ~line 323-325; `_check_nuc_bytes` ~line 327-338)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing test**

```python
import pytest


def test_validate_accepts_lowercase_rejects_N():
    # lowercase canonical must pass validate=True (translates exactly now)
    sp.AA.translate(np.frombuffer(b"atgaaa", "S1").reshape(1, -1), length_axis=-1,
                    validate=True)
    # N must still raise
    with pytest.raises(ValueError):
        sp.AA.translate(np.frombuffer(b"atgNNN", "S1").reshape(1, -1), length_axis=-1,
                        validate=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_validate_accepts_lowercase_rejects_N -v`
Expected: FAIL — `validate=True` currently rejects lowercase `atg` (raises on the first assertion's translate call).

- [ ] **Step 3: Implement**

In `AminoAlphabet.__init__`, right after `self._valid_nuc_bytes = ...`:

```python
        # Upper-cased view (bit-5 flip) for case-insensitive validation and
        # drop-codon detection. Identical to _valid_nuc_bytes when the
        # alphabet's nucleotides are already uppercase (standard DNA case).
        self._valid_upper_bytes = self._valid_nuc_bytes & np.uint8(0xDF)
```

Rewrite `_check_nuc_bytes` to compare upper-cased bytes:

```python
    def _check_nuc_bytes(self, buf: NDArray[np.uint8]) -> None:
        """Raise ``ValueError`` if any byte in ``buf`` is outside the alphabet's
        nucleotides. Case-insensitive (``& 0xDF``), matching ``translate``'s
        unconditional case-folding. Used by ``translate(validate=True)``."""
        ok = np.isin(buf & np.uint8(0xDF), self._valid_upper_bytes)
        if not bool(ok.all()):
            bad = np.unique(buf[~ok]).tobytes().decode("ascii", "replace")
            allowed = self._valid_nuc_bytes.tobytes().decode("ascii")
            raise ValueError(
                f"translate(validate=True): input contains characters outside "
                f"the alphabet {{{allowed}}}: found {bad!r}."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run pytest tests/test_translate.py::test_validate_accepts_lowercase_rejects_N -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "feat(translate): case-insensitive validate=True"
```

---

## Task 6: Wrapper — parse `unknown`, pad path, overloads

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (imports; overloads ~line 365-393; `translate` body ~line 395-465)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_translate_pad_default_is_X():
    seqs = np.frombuffer(b"ATGNNN", "S1").reshape(1, -1)
    out = sp.AA.translate(seqs, length_axis=-1)  # default unknown="X"
    assert out.view("S1").tobytes() == b"MX"


def test_translate_pad_custom_marker():
    seqs = np.frombuffer(b"ATGNNN", "S1").reshape(1, -1)
    out = sp.AA.translate(seqs, length_axis=-1, unknown="-")
    assert out.view("S1").tobytes() == b"M-"


def test_translate_pad_lowercase_translates():
    seqs = np.frombuffer(b"atgaaa", "S1").reshape(1, -1)
    out = sp.AA.translate(seqs, length_axis=-1)
    assert out.view("S1").tobytes() == b"MK"


@pytest.mark.parametrize("bad", ["xy", "", "ZZ", "drops"])
def test_translate_invalid_unknown_raises(bad):
    seqs = np.frombuffer(b"ATG", "S1").reshape(1, -1)
    with pytest.raises(ValueError):
        sp.AA.translate(seqs, length_axis=-1, unknown=bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_translate_pad_custom_marker -v`
Expected: FAIL — `translate()` has no `unknown` parameter (TypeError: unexpected keyword argument).

- [ ] **Step 3: Implement**

In `_alphabets.py`, update the typing import:

```python
from typing import Literal, cast, overload
```

Add a module-level helper above `class NucleotideAlphabet` (or above `AminoAlphabet`):

```python
def _parse_unknown(unknown: str) -> tuple[bool, np.uint8]:
    """Parse the ``unknown`` policy string for ``AminoAlphabet.translate``.

    Returns ``(is_drop, marker_byte)``. ``"drop"`` -> ``(True, 0)``; a single
    ASCII char -> ``(False, ord(char))``; anything else raises ``ValueError``.
    """
    if unknown == "drop":
        return True, np.uint8(0)
    if isinstance(unknown, str) and len(unknown) == 1 and ord(unknown) <= 0xFF:
        return False, np.uint8(ord(unknown))
    raise ValueError(
        f"unknown must be a single ASCII character (pad) or the literal "
        f'"drop"; got {unknown!r}.'
    )
```

Replace the three `translate` overloads with four (note `unknown` is keyword-only, after `*`):

```python
    @overload
    def translate(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: Literal["drop"],
    ) -> Ragged[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: StrSeqType,
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
    ) -> NDArray[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.bytes_],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
    ) -> Ragged[np.bytes_]: ...
    @overload
    def translate(
        self,
        seqs: Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
    ) -> Ragged[np.uint8]: ...
```

Update the real signature and the dense path. Replace from `def translate(self, seqs: ... ) -> ...:` through the end of the dense (`if not isinstance(seqs, Ragged):`) block. The new dense block (pad returns dense; drop reshapes to Ragged):

```python
    def translate(
        self,
        seqs: StrSeqType | Ragged[np.bytes_] | Ragged[np.uint8],
        length_axis: int | None = None,
        *,
        nuc_alphabet: NucleotideAlphabet | None = None,
        truncate_stop: bool = False,
        validate: bool = False,
        unknown: str = "X",
    ) -> NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]:
        """Translate nucleotide sequences to amino acids.

        Parameters
        ----------
        seqs
            Nucleotide sequences. Ragged inputs must have all lengths divisible by
            the codon size. For OHE Ragged (uint8), nuc_alphabet is required.
        length_axis
            Only used for non-Ragged array input.
        nuc_alphabet
            Required when seqs is a Ragged OHE (uint8) array, to decode OHE -> bytes.
        truncate_stop
            When True, each output sequence is truncated at the first stop codon
            (inclusive). Only valid for Ragged input. Default False.
        validate
            When True, raise ValueError if any input nucleotide is outside this
            alphabet (case-insensitive: lowercase ``acgt`` pass, ``N`` and other
            non-ACGT bytes raise; for OHE input, any non-one-hot row raises).
            When validation passes, the translation is guaranteed exact. This is
            the single fast-fail path — there is no separate ``error`` policy.
            Default False.
        unknown
            Policy for codons containing a byte outside ``{A, C, G, T, a, c, g,
            t}``. Either a single ASCII character (default ``"X"``) emitted once
            per non-canonical codon ("pad"), or the literal ``"drop"`` which
            removes non-canonical codons entirely. Because ``"drop"`` changes
            per-sequence length, it always returns a ``Ragged`` (even for dense
            input). Case-insensitivity is unconditional and independent of this
            parameter.

        Returns
        -------
        NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]
            Translated amino acids. Dense input returns a dense array unless
            ``unknown="drop"``, which returns a Ragged.
        """
        is_drop, marker_byte = _parse_unknown(unknown)

        if not isinstance(seqs, Ragged):
            check_axes(seqs, length_axis, False)
            seqs = cast_seqs(seqs)
            if validate:
                self._check_nuc_bytes(seqs.view(np.uint8))
            codon_size = self.codon_array.shape[-1]
            if length_axis is None:
                length_axis = -1
            if length_axis < 0:
                length_axis = seqs.ndim + length_axis
            if seqs.shape[length_axis] % codon_size != 0:
                raise ValueError(
                    "Sequence length is not evenly divisible by codon length."
                )

            if is_drop:
                # Normalize to (n_seq, L): move length axis last, flatten the
                # rest into the batch dim. Each row becomes one Ragged sequence.
                norm = np.moveaxis(seqs, length_axis, -1)
                seq_len = norm.shape[-1]
                norm = np.ascontiguousarray(norm.reshape(-1, seq_len))
                n_seq = norm.shape[0]
                n_codons = seq_len // codon_size
                codons = np.lib.stride_tricks.sliding_window_view(
                    norm, window_shape=codon_size, axis=-1
                )[:, ::codon_size]  # (n_seq, n_codons, codon_size)
                codons_u1 = np.ascontiguousarray(codons.view(np.uint8))
                if self.codon_lut is not None:
                    translated = gufunc_translate_lut(
                        codons_u1, self.codon_lut, marker_byte,
                        axes=[-1, -1, (), ()],  # type: ignore
                    )
                else:
                    translated = gufunc_translate(
                        codons_u1,
                        self.codon_array.view(np.uint8),
                        self.aa_array.view(np.uint8),
                        marker_byte,
                        axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
                    )
                translated_flat = np.ascontiguousarray(translated.reshape(-1))
                codons_flat = codons_u1.reshape(-1, codon_size)
                offsets = np.arange(n_seq + 1, dtype=np.int64) * n_codons
                out_u1, new_offsets = _nb_drop_unknown_codons(
                    translated_flat, codons_flat, offsets, self._valid_upper_bytes
                )
                return Ragged.from_offsets(
                    out_u1.view("S1"), (n_seq, None), new_offsets
                )

            # pad: shape-preserving dense output
            codons = np.lib.stride_tricks.sliding_window_view(
                seqs, window_shape=codon_size, axis=length_axis
            )
            codons = array_slice(codons, length_axis, slice(None, None, codon_size))
            if self.codon_lut is not None:
                return gufunc_translate_lut(
                    codons.view(np.uint8),
                    self.codon_lut,
                    marker_byte,
                    axes=[-1, -1, (), ()],  # type: ignore
                ).view("S1")
            return gufunc_translate(
                codons.view(np.uint8),
                self.codon_array.view(np.uint8),
                self.aa_array.view(np.uint8),
                marker_byte,
                axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
            ).view("S1")
```

> NOTE: the Ragged path below this block is updated in Task 7. After this task the Ragged-input branch still calls the kernels with the OLD (no-`marker_byte`) signatures and will fail for Ragged input — that is expected and fixed in Task 7. The dense tests in this task do not exercise the Ragged branch.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_translate.py -k "translate_pad or invalid_unknown" -v`
Expected: PASS (pad default, custom marker, lowercase, and all invalid-unknown params).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "feat(translate): add unknown= param (pad path) and overloads"
```

---

## Task 7: Wrapper — Ragged path (pad + drop)

**Files:**
- Modify: `python/seqpro/alphabets/_alphabets.py` (Ragged path, ~line 465-530, the block that calls the kernels and computes `new_offsets`)
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the failing tests**

```python
import seqpro as sp
from seqpro.rag import Ragged


def _rag_bytes(seq_list):
    data = np.frombuffer(b"".join(seq_list), "S1")
    lengths = np.array([len(s) for s in seq_list], dtype=np.int64)
    return Ragged.from_lengths(data, lengths)


def test_translate_ragged_pad():
    rag = _rag_bytes([b"ATGNNN", b"AAACCC"])
    out = sp.AA.translate(rag, unknown="X")
    flat = out.to_packed().data.view("S1").tobytes()
    assert flat == b"MXKP"  # ATG=M NNN=X AAA=K CCC=P


def test_translate_ragged_drop():
    rag = _rag_bytes([b"ATGNNN", b"AAACCC"])
    out = sp.AA.translate(rag, unknown="drop")
    p = out.to_packed()
    assert p.data.view("S1").tobytes() == b"MKP"  # NNN dropped from seq0
    assert p.lengths.ravel().tolist() == [1, 2]


def test_translate_ragged_drop_lowercase_kept():
    rag = _rag_bytes([b"atgnnn"])
    out = sp.AA.translate(rag, unknown="drop")
    p = out.to_packed()
    assert p.data.view("S1").tobytes() == b"M"
    assert p.lengths.ravel().tolist() == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run pytest tests/test_translate.py::test_translate_ragged_pad -v`
Expected: FAIL — Ragged branch still calls kernels without `marker_byte` (TypeError / wrong arg count).

- [ ] **Step 3: Implement**

In the Ragged path, locate the block that computes `translated_flat` (the `if total > 0:` / `else:` block) and the line `new_offsets = offsets // codon_size`. Replace the kernel calls to pass `marker_byte`, capture `codons_u1`, and insert the drop compaction. The updated block:

```python
        total = nuc_bytes_flat.shape[0]
        trailing = nuc_bytes_flat.shape[1:]
        if total > 0:
            codons = np.lib.stride_tricks.sliding_window_view(
                nuc_bytes_flat, codon_size, axis=0
            )
            codons = codons[::codon_size]
            # codons shape: (num_codons, *trailing, codon_size); codon axis last
            codons_u1 = codons.view(np.uint8)
            translated_flat: NDArray[np.bytes_]
            if self.codon_lut is not None:
                translated_flat = gufunc_translate_lut(
                    codons_u1,
                    self.codon_lut,
                    marker_byte,
                    axes=[-1, -1, (), ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
            else:
                translated_flat = gufunc_translate(
                    codons_u1,
                    self.codon_array.view(np.uint8),
                    self.aa_array.view(np.uint8),
                    marker_byte,
                    axes=[-1, (-2, -1), (-1), (), ()],  # type: ignore
                ).view("S1")  # (num_codons, *trailing)
        else:
            codons_u1 = np.empty((0, codon_size), dtype=np.uint8)
            translated_flat = np.empty((0, *trailing), dtype="S1")

        new_offsets = offsets // codon_size  # (n+1,) codon-indexed in translated_flat

        if is_drop:
            # For OHE Ragged the nucleotide-alphabet trailing axis was consumed
            # by decode_ohe, so translated_flat / codons_u1 are 1-D along the
            # codon axis. (Dense drop is handled in the non-Ragged branch.)
            if trailing:
                raise ValueError(
                    "unknown='drop' is not supported for dense-trailing Ragged "
                    "input (e.g. multi-track). Use a single-track Ragged."
                )
            if total > 0:
                out_u1, new_offsets = _nb_drop_unknown_codons(
                    translated_flat.view(np.uint8),
                    np.ascontiguousarray(codons_u1),
                    new_offsets.astype(np.int64),
                    self._valid_upper_bytes,
                )
                translated_flat = out_u1.view("S1")
            else:
                translated_flat = np.empty((0,), dtype="S1")
```

Add the `_nb_drop_unknown_codons` import to the existing `from .._numba import (...)` block at the top of the file.

> The `truncate_stop` block and the `is_ohe` re-encode block below remain unchanged — they consume `translated_flat` and `new_offsets`, which now reflect the drop.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run pytest tests/test_translate.py -k "translate_ragged" -v`
Expected: PASS (pad, drop, drop-lowercase).

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/alphabets/_alphabets.py tests/test_translate.py
git commit -m "feat(translate): Ragged pad + drop paths via unknown="
```

---

## Task 8: OHE Ragged drop + dense-drop reshape coverage

**Files:**
- Test: `tests/test_translate.py`

- [ ] **Step 1: Write the tests**

```python
def test_translate_dense_drop_returns_ragged_2d():
    # Batch of 2 dense sequences; one has a non-canonical codon.
    seqs = np.frombuffer(b"ATGNNNAAACCC", "S1").reshape(2, 6)  # rows: ATGNNN, AAACCC
    out = sp.AA.translate(seqs, length_axis=-1, unknown="drop")
    p = out.to_packed()
    assert p.data.view("S1").tobytes() == b"MKP"
    assert p.lengths.ravel().tolist() == [1, 2]


def test_translate_dense_drop_single_sequence():
    seqs = np.frombuffer(b"ATGNNN", "S1")  # 1-D
    out = sp.AA.translate(seqs, length_axis=-1, unknown="drop")
    p = out.to_packed()
    assert p.data.view("S1").tobytes() == b"M"
    assert p.lengths.ravel().tolist() == [1]


def test_translate_ohe_ragged_drop():
    rag = _rag_bytes([b"ATGNNN"])
    ohe = sp.DNA.ohe(rag)  # Ragged[uint8]
    out = sp.AA.translate(ohe, nuc_alphabet=sp.DNA, unknown="drop")
    # OHE in -> OHE out; decode to compare
    aa = sp.AA.decode_ohe(out.to_packed().data, ohe_axis=-1)
    assert aa.view("S1").tobytes() == b"M"
```

> If `sp.DNA.ohe` does not accept a Ragged directly, build the OHE Ragged from the decoded path used elsewhere in `tests/test_translate.py` (search the existing file for how OHE Ragged inputs are constructed and mirror it). Keep the assertion: `unknown="drop"` on OHE Ragged drops `NNN` and yields `M`.

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `pixi run pytest tests/test_translate.py -k "dense_drop or ohe_ragged_drop" -v`
Expected: PASS if Tasks 6-7 are correct. If `test_translate_ohe_ragged_drop` errors on OHE construction, fix the test per the note (this is a test-only adjustment, not a code change).

- [ ] **Step 3: Commit**

```bash
git add tests/test_translate.py
git commit -m "test(translate): dense-drop reshape + OHE Ragged drop coverage"
```

---

## Task 9: Port retained #40/#41 tests; delete removed-feature tests

**Files:**
- Test: `tests/test_translate.py`

- [ ] **Step 1: Inspect the existing #40/#41 test coverage**

Run:
```bash
git show pr41:tests/test_translate.py > /tmp/pr41_test_translate.py
grep -n "def test_" /tmp/pr41_test_translate.py
```
Expected: a list of test functions. Identify three buckets:
- **Keep/port:** canonical-codon correctness (every codon + stop codon still maps), pad back-compat (`unknown="X"` == #40 output), case-insensitive end-to-end, dense/Ragged/OHE happy paths.
- **Delete:** anything referencing `on_unknown="error"`, `on_unknown="collapse"`, `collapse`, `unknown_marker`, `_validate_on_unknown`, `_validate_unknown_marker`, `_codon_unknown_mask`, `_shrink_ragged_unknowns`, or the back-compat test named `test_translate_pad_back_compat_matches_pre_policy_behavior` (re-create as below).
- **Translate API rename:** any test using `on_unknown=` must become `unknown=`; `on_unknown="pad", unknown_marker=c` becomes `unknown=c`; `on_unknown="shorten"` becomes `unknown="drop"`.

- [ ] **Step 2: Port the canonical-correctness regression test**

Add (covers that every standard codon and stop codon still translates correctly through whichever kernel is selected):

```python
def test_all_canonical_codons_unchanged():
    codons = [c.decode() for c in sp.AA.codons]  # e.g. "ATG", ...
    joined = "".join(codons)
    seqs = np.frombuffer(joined.encode(), "S1").reshape(1, -1)
    out = sp.AA.translate(seqs, length_axis=-1).view("S1").tobytes().decode()
    expected = "".join(sp.AA.codon_to_aa[c] for c in codons)
    assert out == expected
```

- [ ] **Step 3: Add the pad back-compat pin (replaces #41's renamed test)**

```python
def test_translate_pad_X_matches_bugfix_behavior():
    # unknown="X" pads every non-canonical codon with 'X' — the #40 contract.
    seqs = np.frombuffer(b"ATGNNNaaaXYZ", "S1").reshape(1, -1)  # XYZ non-canonical
    out = sp.AA.translate(seqs, length_axis=-1, unknown="X").view("S1").tobytes()
    assert out == b"MXKX"  # ATG=M, NNN=X, aaa=K, XYZ=X
```

- [ ] **Step 4: Delete removed-feature tests**

Remove every test in `tests/test_translate.py` matching the "Delete" bucket from Step 1. Verify none remain:

Run: `grep -nE "on_unknown|collapse|unknown_marker|_shrink_ragged|_codon_unknown_mask" tests/test_translate.py`
Expected: no output (exit 1 / empty).

- [ ] **Step 5: Run the full translate test module**

Run: `pixi run pytest tests/test_translate.py -v 2>&1 | tail -20`
Expected: all pass, no errors, no references to removed symbols.

- [ ] **Step 6: Commit**

```bash
git add tests/test_translate.py
git commit -m "test(translate): port retained coverage, drop error/collapse/marker tests"
```

---

## Task 10: Docs — SKILL.md, CHANGELOG.md, CLAUDE.md

**Files:**
- Modify: `skills/seqpro/SKILL.md` (the `AminoAlphabet.translate` bullet, ~line 26)
- Modify: `CHANGELOG.md` (new top entry)
- Modify: `CLAUDE.md` (Key Conventions section)

- [ ] **Step 1: Update SKILL.md**

Replace the existing `AminoAlphabet.translate` bullet with:

```markdown
- **`AminoAlphabet.translate(seqs, ..., validate=False, unknown="X")`**: translates nucleotides → amino acids. Case-insensitive: lowercase/soft-masked `acgt` always translate. `unknown=` controls non-canonical codons (anything outside `{A,C,G,T}`): a single character (default `"X"`) pads one marker per bad codon; the literal `"drop"` removes bad codons and returns a `Ragged` (even for dense input, since lengths then vary). `validate=True` is the single fast-fail path — it raises (case-insensitively) on `N`/IUPAC/non-one-hot input and, when it returns, guarantees exact translation. There is no separate `error` mode.
```

- [ ] **Step 2: Update CHANGELOG.md**

Add at the very top of `CHANGELOG.md`, above the current `## 0.14.0` heading (use an `## Unreleased` block; commitizen will fold it into the next release):

```markdown
## Unreleased

### Feat

- **translate**: `unknown=` parameter on `AminoAlphabet.translate` — single char pads non-canonical codons (default `"X"`), `"drop"` removes them (returns Ragged). Translation is now unconditionally case-insensitive (soft-masked `acgt` translate). `validate=True` is now case-insensitive.

### Fix

- **translate**: non-canonical codons (e.g. `N`, NUL, symbolic ALT bytes) no longer silently corrupt output in either translate kernel; they resolve to the `unknown` marker. Fixes `np.empty` NUL leak in `gufunc_translate` and `(byte>>1)&3` hash collisions (`NNN→T`, `\x00\x00\x00→K`) in `gufunc_translate_lut`.
```

- [ ] **Step 3: Update CLAUDE.md Key Conventions**

In `CLAUDE.md`, under the `## Key Conventions` section, append two bullets:

```markdown
- **Validation is opt-in and front-loaded.** Add fast-fail/input validation via a `validate=` flag (or equivalent single opt-in), not per-feature `error` modes. There must be one obvious way to ask "is this input clean?" — don't duplicate the check across parameters.
- **No naive NumPy in hot paths.** Never use raw Python loops or naive NumPy (e.g. per-segment `np.concatenate`, Python `for` over sequences) where a Numba kernel is faster and leaner — unless the NumPy version is *verifiably* comparable in time and memory. When Numba is a poor fit (graph algorithms like k-shuffle), use the Rust/PyO3 extension (`src/`).
```

- [ ] **Step 4: Commit**

```bash
git add skills/seqpro/SKILL.md CHANGELOG.md CLAUDE.md
git commit -m "docs: document translate unknown= policy; add validation + perf conventions"
```

---

## Task 11: Full verification

**Files:** none

- [ ] **Step 1: Lint and format**

Run: `pixi run ruff check python/ tests/ && pixi run ruff format --check python/ tests/`
Expected: no errors. If format check fails, run `pixi run ruff format python/ tests/` and amend the relevant commit.

- [ ] **Step 2: Full test suite**

Run: `pixi run test 2>&1 | tail -10`
Expected: all pass (count ≥ the Task 1 baseline, since net tests were added). Specifically no failures, no errors, no collection errors from removed symbols.

- [ ] **Step 3: Confirm no dangling references to removed #41 symbols**

Run:
```bash
grep -rnE "on_unknown|unknown_marker|_shrink_ragged_unknowns|_codon_unknown_mask|_validate_on_unknown|_validate_unknown_marker|OnUnknown|collapse" python/ tests/
```
Expected: no output (all removed-feature code and references gone).

- [ ] **Step 4: Verify the public signature via a smoke import**

Run:
```bash
pixi run python -c "
import numpy as np, seqpro as sp
print(sp.AA.translate(np.frombuffer(b'ATGNNN','S1').reshape(1,-1), length_axis=-1).view('S1').tobytes())
print(sp.AA.translate(np.frombuffer(b'ATGNNN','S1').reshape(1,-1), length_axis=-1, unknown='drop').to_packed().data.view('S1').tobytes())
"
```
Expected: `b'MX'` then `b'M'`.

---

## Task 12: PR mechanics

**Files:** none (git/gh)

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feat/translate-unknown-codon-policy
```

- [ ] **Step 2: Open the unified PR**

```bash
gh pr create --base main --head feat/translate-unknown-codon-policy \
  --title "feat(translate): case-insensitive + unknown codon policy" \
  --body "Combines #40 (kernel corruption fix) and #41 (case-insensitivity + unknown policy) into one non-breaking PR. Adds a single \`unknown=\` parameter (single char = pad, default \"X\" = #40 behavior; \"drop\" = drop non-canonical codons, returns Ragged). Drops #41's \`error\`/\`collapse\` modes and \`unknown_marker\` param: \`validate=True\` (now case-insensitive) is the single fast-fail path. Closes #41.

Design: docs/superpowers/specs/2026-06-05-translate-unknown-codon-policy-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 3: Close PR #41 and #40 pointing at the unified PR**

```bash
gh pr comment 41 --body "Superseded by the unified PR (combines #40 + #41 with a simplified single-parameter API). Closing."
gh pr close 41
gh pr comment 40 --body "Superseded by the unified PR (folds this kernel fix in). Closing in favor of the combined branch."
gh pr close 40
```

> Confirm with the user before closing #40/#41 if they want to keep either open for review history.

---

## Self-Review Notes (for the implementer)

- The `unknown` parameter is keyword-only (after `*`) in every overload and the real signature — callers must write `unknown="drop"`, never positionally.
- After Task 6, the Ragged branch is temporarily broken (old kernel arity); Task 7 fixes it. Run Ragged tests only from Task 7 onward.
- `_nb_drop_unknown_codons` is the only new public-ish kernel name; it is referenced in Tasks 4, 6, 7 with the same signature `(translated, codons, offsets, valid_upper)`.
- Dense `"drop"` with a trailing (multi-axis) layout flattens non-length axes into the batch dim; Ragged `"drop"` with a dense trailing axis (multi-track) raises (Task 7) — both documented.
