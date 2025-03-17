import hypothesis.strategies as st
import numpy as np
import seqpro as sp
from Bio.Seq import translate
from hypothesis import given
from numpy.typing import NDArray
from pytest_cases import parametrize_with_cases
from seqpro._numba import gufunc_translate


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
