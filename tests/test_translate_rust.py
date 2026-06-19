import numpy as np
import pytest

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
