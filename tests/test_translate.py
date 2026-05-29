import hypothesis.strategies as st
import numpy as np
import pytest
import seqpro as sp
from Bio.Seq import translate
from hypothesis import given
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro._numba import gufunc_translate
from seqpro.rag import Ragged


def nb_1_kmer():
    seq_kmers = sp.cast_seqs("ATC").view(np.uint8)
    desired = sp.cast_seqs("I").view(np.uint8)
    return seq_kmers, desired


def nb_2_kmers():
    seq_kmers = sp.cast_seqs(["ATC", "GAT"]).view(np.uint8)
    desired = sp.cast_seqs("ID").view(np.uint8)
    return seq_kmers, desired


@parametrize_with_cases("seq_kmers, desired", cases=".", prefix="nb_")
def test_gufunc_translate(
    seq_kmers: NDArray[np.uint8],
    desired: NDArray[np.uint8],
):
    kmer_keys = sp.AA.codon_array.view(np.uint8)
    kmer_values = sp.AA.aa_array.view(np.uint8)

    actual = gufunc_translate(seq_kmers, kmer_keys, kmer_values)
    np.testing.assert_array_equal(actual, desired)


# --- LUT path tests ---


def test_aa_alphabet_has_lut():
    """Standard ACGT × k=3 alphabet builds the 64-entry LUT at __init__."""
    assert sp.AA.codon_lut is not None
    assert sp.AA.codon_lut.shape == (64,)
    assert sp.AA.codon_lut.dtype == np.uint8


def test_lut_translates_every_standard_codon():
    """All 64 standard codons translate to the same AA via LUT as via the codon table."""
    from seqpro._numba import gufunc_translate_lut

    for codon, expected_aa in zip(sp.AA.codons, sp.AA.amino_acids, strict=True):
        seq_kmer = sp.cast_seqs(codon).view(np.uint8)
        actual = gufunc_translate_lut(seq_kmer, sp.AA.codon_lut)
        assert int(actual) == ord(expected_aa), (
            f"codon {codon!r} → {chr(int(actual))!r}, expected {expected_aa!r}"
        )


def test_lut_translates_stop_codons():
    """The three stop codons TAA, TAG, TGA map to '*'."""
    from seqpro._numba import gufunc_translate_lut

    for stop in ("TAA", "TAG", "TGA"):
        seq_kmer = sp.cast_seqs(stop).view(np.uint8)
        actual = gufunc_translate_lut(seq_kmer, sp.AA.codon_lut)
        assert chr(int(actual)) == "*", f"{stop} → {chr(int(actual))}, expected '*'"


def test_lut_translates_start_codon():
    """ATG → M."""
    from seqpro._numba import gufunc_translate_lut

    seq_kmer = sp.cast_seqs("ATG").view(np.uint8)
    actual = gufunc_translate_lut(seq_kmer, sp.AA.codon_lut)
    assert chr(int(actual)) == "M"


def test_lut_and_linear_scan_produce_identical_output():
    """Random-codon fuzz: LUT path == linear-scan path bitwise."""
    from seqpro._numba import gufunc_translate_lut

    rng = np.random.default_rng(0)
    n = 5000
    codons_idx = rng.choice(4, size=(n, 3))  # 0..3 per position
    # Map index back to ASCII byte using the alphabet order (ACGT).
    byte_for_idx = np.array([ord(c) for c in "ACGT"], dtype=np.uint8)
    codons_bytes = byte_for_idx[codons_idx]  # (n, 3) uint8

    kmer_keys = sp.AA.codon_array.view(np.uint8)
    kmer_values = sp.AA.aa_array.view(np.uint8)
    expected = gufunc_translate(codons_bytes, kmer_keys, kmer_values)
    actual = gufunc_translate_lut(codons_bytes, sp.AA.codon_lut)
    np.testing.assert_array_equal(actual, expected)


def test_pack_codon_index_bijective_over_acgt():
    """The 64 ACGT codons map to 64 distinct indices covering exactly [0, 63]."""
    from seqpro._numba import _pack_codon_index

    idxs = set()
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                idxs.add(int(_pack_codon_index(ord(a), ord(b), ord(c))))
    assert idxs == set(range(64))


def test_alphabet_translate_uses_lut_by_default():
    """``sp.AA.translate`` should now go through the LUT path on standard
    AA. Cross-validate against BioPython to confirm the LUT is correct."""
    rng = np.random.default_rng(1)
    nucs = list("ACGT")
    seq = "".join(rng.choice(nucs, size=300, replace=True))
    bio_aa = str(translate(seq))
    sp_aa = sp.AA.translate(seq, length_axis=-1).view("S1")
    np.testing.assert_array_equal(sp_aa, sp.cast_seqs(bio_aa))


def case_1d():
    seqs = sp.cast_seqs("ATCGAT")
    desired = sp.cast_seqs("ID")
    return seqs, desired


def case_2d():
    seqs = sp.cast_seqs(["ATCGAT", "GATCAT"])
    desired = sp.cast_seqs(["ID", "DH"])
    return seqs, desired


def case_3d():
    seqs = sp.cast_seqs([["ATCGAT", "GATCAT"], ["GATCAT", "ATCGAT"]])
    desired = sp.cast_seqs([["ID", "DH"], ["DH", "ID"]])
    return seqs, desired


def case_full_codon_table():
    seqs = sp.AA.codon_array
    desired = sp.AA.aa_array[:, None]
    return seqs, desired


@parametrize_with_cases("seqs, desired", cases=".", prefix="case_")
def test_translate(seqs, desired):
    actual = sp.AA.translate(seqs, length_axis=-1)
    np.testing.assert_array_equal(actual, desired)


cds_dna_st = st.lists(
    st.text(list(sp.DNA.alphabet), min_size=3, max_size=3), min_size=1
).map(lambda c: "".join(c))


@given(cds_dna_st)
def test_translate_fuzzing(seq: str):
    actual = sp.AA.translate(seq, length_axis=-1)
    desired = sp.cast_seqs(str(translate(seq)))
    np.testing.assert_array_equal(actual, desired)


# --- Ragged tests ---


def _make_ragged_bytes(*seqs: str) -> Ragged:
    data = np.array(list("".join(seqs)), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    return Ragged.from_lengths(data, lengths)


def _rag_bytes_to_array(element) -> NDArray[np.bytes_]:
    """Convert a bytes Ragged element (Python bytes) to a 1D S1 numpy array."""
    return np.frombuffer(element, dtype="S1")


def test_translate_ragged_bytes_basic():
    # ATCGAT → I D  |  GATCAT → D H
    rag = _make_ragged_bytes("ATCGAT", "GATCAT")
    out = sp.AA.translate(rag)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("ID"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[1]), sp.cast_seqs("DH"))


def test_translate_ragged_bytes_truncate_stop_present():
    # ATGAAATAA = M K *  |  ATGGGG = M G (no stop)
    rag = _make_ragged_bytes("ATGAAATAA", "ATGGGG")
    out = sp.AA.translate(rag, truncate_stop=True)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("MK*"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[1]), sp.cast_seqs("MG"))


def test_translate_ragged_bytes_truncate_stop_absent():
    # No stop codon → full sequence returned
    rag = _make_ragged_bytes("ATGGGG", "ATCGAT")
    out = sp.AA.translate(rag, truncate_stop=True)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("MG"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[1]), sp.cast_seqs("ID"))


def test_translate_ragged_bytes_mixed_stop():
    # seq0: stop present  |  seq1: no stop  |  seq2: stop at last codon
    rag = _make_ragged_bytes("ATGAAATAAGGG", "ATCGAT", "ATGTAA")
    out = sp.AA.translate(rag, truncate_stop=True)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("MK*"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[1]), sp.cast_seqs("ID"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[2]), sp.cast_seqs("M*"))


def test_translate_ragged_ohe_basic():
    dna_seqs = [sp.cast_seqs(s) for s in ["ATCGAT", "GATCAT"]]
    ohe_data = np.concatenate([sp.DNA.ohe(s) for s in dna_seqs])  # (12, 4)
    lengths = np.array([6, 6])
    rag_ohe = Ragged.from_lengths(ohe_data, lengths)

    out = sp.AA.translate(rag_ohe, nuc_alphabet=sp.DNA)
    assert out.dtype == np.uint8
    # Decode output OHE to check values
    aa0 = sp.AA.decode_ohe(out[0].to_numpy(), ohe_axis=-1)
    aa1 = sp.AA.decode_ohe(out[1].to_numpy(), ohe_axis=-1)
    np.testing.assert_array_equal(aa0, sp.cast_seqs("ID"))
    np.testing.assert_array_equal(aa1, sp.cast_seqs("DH"))


def test_translate_ragged_ohe_truncate_stop():
    # ATGAAATAA = M K *  (stop truncated inclusive)
    dna_seq = sp.cast_seqs("ATGAAATAAATGGGG")
    ohe_data = sp.DNA.ohe(dna_seq)  # (15, 4)
    lengths = np.array([9, 6])
    rag_ohe = Ragged.from_lengths(ohe_data, lengths)

    out = sp.AA.translate(rag_ohe, nuc_alphabet=sp.DNA, truncate_stop=True)
    aa0 = sp.AA.decode_ohe(out[0].to_numpy(), ohe_axis=-1)
    aa1 = sp.AA.decode_ohe(out[1].to_numpy(), ohe_axis=-1)
    np.testing.assert_array_equal(aa0, sp.cast_seqs("MK*"))
    np.testing.assert_array_equal(aa1, sp.cast_seqs("MG"))


def test_translate_ragged_error_no_nuc_alphabet():
    dna_seq = sp.cast_seqs("ATCGAT")
    ohe_data = sp.DNA.ohe(dna_seq)
    lengths = np.array([6])
    rag_ohe = Ragged.from_lengths(ohe_data, lengths)
    with pytest.raises(ValueError, match="nuc_alphabet"):
        sp.AA.translate(rag_ohe)


def test_translate_ragged_error_bad_length():
    rag = _make_ragged_bytes("ATCG")  # length 4, not divisible by 3
    with pytest.raises(ValueError, match="divisible by codon"):
        sp.AA.translate(rag)


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


# --- validate flag tests ---


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


def test_check_ohe_rows_raises_on_1d_data():
    """A 1-D buffer (not (total, n_nuc)) yields a clear ValueError, not IndexError."""
    with pytest.raises(ValueError, match="must be 2-D"):
        sp.AA._check_ohe_rows(np.zeros(6, dtype=np.uint8), 4)


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


# --- path-activation tests ---
# These guard that the LUT optimization is actually *used* for the standard
# alphabet (and conversely that the scan is used for non-standard ones). Without
# them, a regression that silently routed every call through the linear scan
# would still pass every correctness test, because the scan is correct, just slow.


def test_dense_standard_alphabet_uses_lut_not_scan(monkeypatch):
    """Standard-alphabet dense path must hit the LUT kernel, never the scan."""
    import seqpro.alphabets._alphabets as alpha_mod

    def _boom(*args, **kwargs):
        raise AssertionError("linear scan was used instead of the LUT")

    monkeypatch.setattr(alpha_mod, "gufunc_translate", _boom)
    out = sp.AA.translate("ATGAAA", length_axis=-1).view("S1")
    np.testing.assert_array_equal(out, sp.cast_seqs(str(translate("ATGAAA"))))


def test_ragged_standard_alphabet_uses_lut_not_scan(monkeypatch):
    """Standard-alphabet Ragged path must hit the LUT kernel, never the scan."""
    import seqpro.alphabets._alphabets as alpha_mod

    def _boom(*args, **kwargs):
        raise AssertionError("linear scan was used instead of the LUT")

    monkeypatch.setattr(alpha_mod, "gufunc_translate", _boom)
    rag = _make_ragged_bytes("ATGAAA")
    out = sp.AA.translate(rag)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("MK"))


def test_nonstandard_alphabet_uses_scan_not_lut(monkeypatch):
    """A non-standard (codon_lut=None) alphabet must use the scan, never the LUT."""
    import seqpro.alphabets._alphabets as alpha_mod

    def _boom(*args, **kwargs):
        raise AssertionError("LUT kernel was used for a non-standard alphabet")

    monkeypatch.setattr(alpha_mod, "gufunc_translate_lut", _boom)
    alpha = sp.AminoAlphabet(["AUG", "UAA"], ["M", "*"])
    data = np.array(list("AUGUAA"), dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6]))
    out = alpha.translate(rag)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("M*"))


# --- additional validate coverage ---


def test_translate_ragged_bytes_validate_passes_on_valid():
    """validate=True on clean multi-sequence Ragged bytes must not raise."""
    rag = _make_ragged_bytes("ATGAAA", "GGGTTT")
    out = sp.AA.translate(rag, validate=True)
    np.testing.assert_array_equal(_rag_bytes_to_array(out[0]), sp.cast_seqs("MK"))
    np.testing.assert_array_equal(_rag_bytes_to_array(out[1]), sp.cast_seqs("GF"))


def test_translate_validate_passes_on_2d_valid():
    """validate=True over a 2-D dense array of valid sequences must not raise."""
    seqs = sp.cast_seqs(["ATGAAA", "GGGTTT"])
    out = sp.AA.translate(seqs, length_axis=-1, validate=True)
    np.testing.assert_array_equal(out, sp.cast_seqs(["MK", "GF"]))


def test_translate_validate_raises_on_2d_with_invalid():
    """A non-ACGT byte anywhere in a 2-D dense array is caught by validate=True."""
    seqs = sp.cast_seqs(["ATGAAA", "ATGNNN"])  # second row has N
    with pytest.raises(ValueError, match="outside the alphabet"):
        sp.AA.translate(seqs, length_axis=-1, validate=True)


def test_check_ohe_rows_raises_on_width_mismatch():
    """OHE width not matching the nucleotide alphabet size raises a clear error."""
    data = np.zeros((6, 5), dtype=np.uint8)  # width 5 != DNA's 4
    with pytest.raises(ValueError, match="width"):
        sp.AA._check_ohe_rows(data, 4)


def test_translate_ragged_multiseq_matches_biopython():
    """Differential check over a multi-sequence, variable-length Ragged batch."""
    rng = np.random.default_rng(11)
    seqs = [
        "".join(rng.choice(list("ACGT"), size=3 * int(rng.integers(2, 40))))
        for _ in range(5)
    ]
    rag = _make_ragged_bytes(*seqs)
    out = sp.AA.translate(rag)
    for i, s in enumerate(seqs):
        expected = sp.cast_seqs(str(translate(s)))
        np.testing.assert_array_equal(_rag_bytes_to_array(out[i]), expected)
