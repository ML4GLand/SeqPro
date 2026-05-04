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
