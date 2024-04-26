from typing import List, Union

import numpy as np
import pytest
import seqpro as sp
from pytest_cases import parametrize_with_cases


def length_empty():
    return "", 0


def length_single():
    return "A", 1


def length_variable():
    return ["A", "AA", "AAA"], [1, 2, 3]


@parametrize_with_cases("seq, lengths", cases=".", prefix="length_")
def test_length(seq: Union[str, List[str]], lengths: Union[int, List[int]]):
    if len(seq) == 0:
        with pytest.raises(ValueError):
            sp.length(seq)
        return

    test_lengths = sp.length(seq)
    np.testing.assert_equal(test_lengths, lengths)


def gc_content_empty():
    return "", 0, 0


def gc_content_single():
    return "A", 0, 0


def gc_content_variable():
    seqs = ["A", "C", "G", "T", "ACGT", "ACGTACGT"]
    counts = np.array([0, 1, 1, 0, 2, 4], "i8")
    length = max([len(s) for s in seqs])
    proportions = counts / length
    return seqs, counts, proportions


def gc_content_ohe():
    seqs = sp.DNA.ohe(["A", "C", "G", "T", "ACGT", "ACGTACGT"])
    counts = np.array([0, 1, 1, 0, 2, 4], "i8")
    length = seqs.shape[-2]
    proportions = counts / length
    return seqs, counts, proportions


@parametrize_with_cases("seqs, counts, proportions", cases=".", prefix="gc_content_")
def test_gc_content(seqs, counts, proportions):
    if len(seqs) == 0:
        with pytest.raises(ValueError):
            sp.gc_content(seqs)
        return

    if isinstance(seqs, np.ndarray) and seqs.dtype == np.uint8:
        length_axis = -2
        ohe_axis = -1
    else:
        length_axis = None
        ohe_axis = None
    test_counts = sp.gc_content(
        seqs,
        normalize=False,
        length_axis=length_axis,
        ohe_axis=ohe_axis,
        alphabet=sp.DNA,
    )
    test_props = sp.gc_content(
        seqs,
        normalize=True,
        length_axis=length_axis,
        ohe_axis=ohe_axis,
        alphabet=sp.DNA,
    )
    np.testing.assert_equal(test_counts, counts)
    np.testing.assert_equal(test_props, proportions)


def nucleotide_content_empty():
    seqs = ""
    counts = np.array([0, 0, 0, 0], np.int64)
    proportions = counts.astype(np.float64)
    return seqs, counts, proportions


def nucleotide_content_single():
    seqs = "A"
    counts = np.array([1, 0, 0, 0], np.int64)
    proportions = counts.astype(np.float64)
    return seqs, counts, proportions


def nucleotide_content_variable():
    seqs = ["A", "C", "G", "T", "ACGT", "ACGTACGT"]
    counts = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]
    )
    length = max([len(s) for s in seqs])
    proportions = counts / length
    return seqs, counts, proportions


def nucleotide_content_ohe():
    seqs = sp.DNA.ohe(["A", "C", "G", "T", "ACGT", "ACGTACGT"])
    counts = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 2],
        ]
    )
    length = seqs.shape[-2]
    proportions = counts / length
    return seqs, counts, proportions


@parametrize_with_cases(
    "seqs, counts, proportions", cases=".", prefix="nucleotide_content_"
)
def test_nucleotide_content(seqs, counts, proportions):
    if len(seqs) == 0:
        with pytest.raises(ValueError):
            sp.nucleotide_content(seqs)
        return

    if isinstance(seqs, np.ndarray) and seqs.dtype == np.uint8:
        length_axis = -2
    else:
        length_axis = None
    test_counts = sp.nucleotide_content(
        seqs, normalize=False, length_axis=length_axis, alphabet=sp.DNA
    )
    test_props = sp.nucleotide_content(
        seqs, normalize=True, length_axis=length_axis, alphabet=sp.DNA
    )
    np.testing.assert_equal(test_counts, counts)
    np.testing.assert_equal(test_props, proportions)


def count_kmers_empty():
    return "", 1, None


def count_kmers_single():
    return "A", 1, {"A": 1}


def count_kmers_variable():
    return "ACGT", 2, {"AC": 1, "CG": 1, "GT": 1}


@parametrize_with_cases("seq, k, counts", cases=".", prefix="count_kmers_")
def test_count_kmers(seq: str, k: int, counts: dict):
    from seqpro._analyzers import count_kmers_seq

    if len(seq) < k:
        with pytest.raises(ValueError):
            count_kmers_seq(seq, k)
        return

    test_counts = count_kmers_seq(seq, k)
    assert test_counts == counts
