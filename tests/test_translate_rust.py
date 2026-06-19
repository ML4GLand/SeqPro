import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import seqpro as sp
from seqpro import AA  # standard AminoAlphabet
from seqpro.rag import Ragged


def _bio_like_translate(seq_str: str) -> str:
    """Reference using AA.codon_to_aa directly (independent of the impl)."""
    out = []
    for i in range(0, len(seq_str), 3):
        out.append(AA.codon_to_aa.get(seq_str[i : i + 3], "X"))
    return "".join(out)


@pytest.mark.parametrize(
    "seq",
    ["ATGAAATAA", "atgaaataa", "ATGNNNTAA", ""],
)
def test_translate_dense_pad_matches_reference(seq):
    arr = np.frombuffer(seq.encode("ascii"), "S1")
    got = sp.AA.translate(arr, length_axis=0)
    exp = _bio_like_translate(seq.upper())
    assert got.tobytes().decode() == exp


def test_translate_ragged_bytes_pad():
    # Two sequences, lengths 9 and 6 (both divisible by 3).
    data = np.frombuffer(b"ATGAAATAAAAATAA", "S1")
    offsets = np.array([0, 9, 15], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.AA.translate(rag)
    assert got.data.tobytes().decode() == _bio_like_translate("ATGAAATAAAAATAA")
    np.testing.assert_array_equal(got.offsets, offsets // 3)


def test_translate_ragged_bytes_pad_multitrack():
    # Two sequences (lengths 9 and 6, both divisible by 3), each with 2 track
    # columns holding DIFFERENT nucleotide sequences. Data shape is (15, 2).
    #
    # col0: ATGAAATAA (seq0) + AAAAAA (seq1)  ->  MK*  KK
    # col1: GGTCCCTTT (seq0) + GGTGGT (seq1)  ->  GPF  GG
    #
    # The two columns translate to completely different AAs, so any transposition
    # or column-swap in the multi-track moveaxis path is immediately detected.
    col0 = np.frombuffer(b"ATGAAATAAAAAAAA", "S1")  # 15 nt, col 0
    col1 = np.frombuffer(b"GGTCCCTTTGGTGGT", "S1")  # 15 nt, col 1
    data = np.stack([col0, col1], axis=1)  # (15, 2) S1
    offsets = np.array([0, 9, 15], dtype=np.int64)

    rag = Ragged.from_offsets(data, (2, None, 2), offsets)
    got = sp.AA.translate(rag)

    # Output shape: (5 codons total, 2 tracks); offsets reduced by codon_size=3.
    assert got.data.shape == (5, 2)
    np.testing.assert_array_equal(got.offsets, offsets // 3)

    # Each track column must match an independent reference translation.
    assert got.data[:, 0].tobytes().decode() == _bio_like_translate("ATGAAATAAAAAAAA")
    assert got.data[:, 1].tobytes().decode() == _bio_like_translate("GGTCCCTTTGGTGGT")


def test_translate_dense_drop_removes_noncanonical():
    # ATG (M), NNN (drop), TAA (*). Drop returns a Ragged even for dense input.
    arr = np.frombuffer(b"ATGNNNTAA", "S1")
    got = sp.AA.translate(arr, unknown="drop", length_axis=0)
    assert got.data.tobytes().decode() == "M*"
    np.testing.assert_array_equal(got.offsets, np.array([0, 2], dtype=np.int64))


def test_translate_ragged_truncate_stop():
    data = np.frombuffer(b"ATGTAAAAAAAAAAA", "S1")  # seq0: M * K ; seq1: K K
    offsets = np.array([0, 9, 15], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.AA.translate(rag, truncate_stop=True)
    # seq0 truncates after '*' -> "M*"; seq1 has no stop -> "KK"
    # got[i] returns Python bytes for a 1D Ragged element
    assert np.frombuffer(got[0], "S1").tobytes().decode() == "M*"
    assert np.frombuffer(got[1], "S1").tobytes().decode() == "KK"


def test_translate_ohe_ragged_roundtrips():
    # Build OHE ragged DNA for "ATGTAA" (len 6, 2 codons -> "M*").
    seq = np.frombuffer(b"ATGTAA", "S1")
    ohe = sp.DNA.ohe(seq)  # (6, 4)
    offsets = np.array([0, 6], dtype=np.int64)
    rag = Ragged.from_offsets(ohe, (1, None, 4), offsets)
    got = sp.AA.translate(rag, nuc_alphabet=sp.DNA)
    # Decode AA OHE back to bytes for assertion.
    aa_bytes = sp.AA.decode_ohe(got.data, ohe_axis=-1)
    assert aa_bytes.tobytes().decode() == _bio_like_translate("ATGTAA")


def test_translate_ohe_ragged_drop_removes_noncanonical():
    # "ATGNNNTAA": ATG=M, NNN=drop (all-zero OHE rows), TAA=*  -> "M*"
    seq = np.frombuffer(b"ATGNNNTAA", "S1")
    ohe = sp.DNA.ohe(seq)  # (9, 4); N rows are all-zero
    offsets = np.array([0, 9], dtype=np.int64)
    rag = Ragged.from_offsets(ohe, (1, None, 4), offsets)
    got = sp.AA.translate(rag, nuc_alphabet=sp.DNA, unknown="drop")
    aa_bytes = sp.AA.decode_ohe(got.data, ohe_axis=-1)
    assert aa_bytes.tobytes().decode() == "M*"
    np.testing.assert_array_equal(got.offsets, np.array([0, 2], dtype=np.int64))


@settings(max_examples=200, deadline=None)
@given(
    n_codons=st.integers(min_value=0, max_value=30),
    data=st.data(),
    unknown=st.sampled_from(["X", "drop"]),
)
def test_translate_dense_differential(n_codons, data, unknown):
    chars = data.draw(
        st.lists(
            st.sampled_from(list("ACGTacgtN")),
            min_size=n_codons * 3,
            max_size=n_codons * 3,
        )
    )
    seq = "".join(chars)
    arr = np.frombuffer(seq.encode("ascii"), "S1")
    got = sp.AA.translate(arr, unknown=unknown, length_axis=0)
    if unknown == "drop":
        # Reference: translate then drop non-canonical codons.
        ref = []
        for i in range(0, len(seq), 3):
            cod = seq[i : i + 3].upper()
            if set(cod) <= set("ACGT"):
                ref.append(sp.AA.codon_to_aa.get(cod, "X"))
        assert got.data.tobytes().decode() == "".join(ref)
    else:
        assert got.tobytes().decode() == _bio_like_translate(seq.upper())
