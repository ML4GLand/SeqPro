from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._utils import SeqType, cast_seqs, check_axes
from .alphabets._alphabets import NucleotideAlphabet


def length(seqs: SeqType, length_axis: int | None = None) -> NDArray[np.integer]:
    """Calculate the length of each sequence.

    Parameters
    ----------
    seqs
        Sequences. For arrays, ``length_axis`` selects which axis encodes sequence
        length; defaults to the last axis.
    length_axis
        Axis to count non-empty characters along. Defaults to the last axis.

    Returns
    -------
    NDArray[np.integer]
        Array containing the length of each sequence; ``length_axis`` is removed.
    """
    _seqs = cast_seqs(seqs)
    if length_axis is None:
        length_axis = -1
    return (_seqs != b"").sum(length_axis)


def gc_content(
    seqs: SeqType,
    normalize: bool = True,
    length_axis: int | None = None,
    alphabet: NucleotideAlphabet | None = None,
    ohe_axis: int | None = None,
) -> NDArray[np.integer | np.float64]:
    """Compute the number or proportion of G & C nucleotides.

    Parameters
    ----------
    seqs
    normalize
        True => return proportions
        False => return counts
    length_axis
        Needed if seqs is an array.
    alphabet
        Needed if seqs is OHE.
    ohe_axis
        Needed if seqs is OHE.

    Returns
    -------
    NDArray[np.integer | np.float64]
        Integers if unnormalized, otherwise floats.
    """
    check_axes(seqs, length_axis, ohe_axis)

    arr = cast_seqs(seqs)

    if length_axis is None:  # length axis after casting strings
        length_axis = arr.ndim - 1
    elif length_axis < 0:
        length_axis = arr.ndim + length_axis

    if arr.dtype == np.uint8:  # OHE
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze OHE sequences.")
        assert ohe_axis is not None

        gc_idx = np.flatnonzero(np.isin(alphabet.array, np.array([b"C", b"G"])))
        gc_content = cast(
            NDArray[np.integer],
            np.take(arr, gc_idx, ohe_axis).sum((length_axis, ohe_axis)),
        )
    else:
        gc_content = cast(
            NDArray[np.integer], np.isin(arr, np.array([b"C", b"G"])).sum(length_axis)
        )

    if normalize:
        gc_content = gc_content / arr.shape[length_axis]

    return gc_content


def nucleotide_content(
    seqs: SeqType,
    normalize: bool = True,
    length_axis: int | None = None,
    alphabet: NucleotideAlphabet | None = None,
) -> NDArray[np.integer | np.floating]:
    """Compute the number or proportion of each nucleotide.

    Parameters
    ----------
    seqs
    normalize
        True => return proportions
        False => return counts
    length_axis
        Needed if seqs is an array.

    Returns
    -------
    NDArray[np.integer | np.floating]
        Integers if unnormalized, otherwise floats.
    """
    check_axes(seqs, length_axis, False)

    arr = cast_seqs(seqs)

    if length_axis is None:
        length_axis = arr.ndim - 1
    elif length_axis < 0:
        length_axis = arr.ndim + length_axis

    if arr.dtype == np.uint8:  # OHE
        nuc_content = cast(NDArray[np.integer], arr.sum(length_axis))
    else:
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze string nucleotide content.")
        nuc_content = np.zeros(
            (*arr.shape[:length_axis], *arr.shape[length_axis + 1 :], len(alphabet)),
            dtype=np.int64,
        )
        for i, nuc in enumerate(alphabet.array):
            nuc_content[..., i] = (arr == nuc).sum(length_axis)

    if normalize:
        nuc_content = nuc_content / arr.shape[length_axis]

    return nuc_content


def count_kmers_seq(seq: str, k: int) -> dict[str, int]:
    """
    Count unique k-mers.

    Parameters
    ----------
    seq
        Nucleotide seq expressed as a string.
    k
        k value for k-mers (e.g. k=3 generates 3-mers).

    Returns
    -------
    dict
        k-mers (str) mapped to their counts (int).
    """
    if len(seq) < k:
        raise ValueError("Length of seq must be >= k.")

    _seq = np.frombuffer(seq.encode("ascii"), "S1")
    kmers = np.lib.stride_tricks.sliding_window_view(_seq, k).view(f"S{k}")
    kmers, counts = np.unique(kmers, return_counts=True)
    out = dict(zip(kmers.astype(str), counts))

    return out


# TODO: non-trivial to parallelize/SIMD this
def _count_kmers(
    seqs: SeqType,
    k: int,
    alphabet: NucleotideAlphabet,
    length_axis: int | None = None,
) -> NDArray[np.unsignedinteger]:
    """
    Count unique k-mers.

    Parameters
    ----------
    seqs
    k
        k value for k-mers (e.g. k=3 generates 3-mers).
    alphabet

    Returns
    -------
    NDArray[np.unsignedinteger]
        Counts of all possible k-mers for each input sequence.
    """
    raise NotImplementedError

    check_axes(seqs, length_axis, False)

    arr = cast_seqs(seqs)

    if length_axis is None:
        length_axis = -1

    length = arr.shape[length_axis]

    if length >= k:
        raise ValueError("k is larger than sequence length")

    if arr.dtype == np.uint8:
        raise NotImplementedError
    else:
        kmers = np.lib.stride_tricks.sliding_window_view(arr, k, -1)
        # can't use np.unique here because each sequence would return a potentially
        # different number of kmers -> ragged array
        uniq, counts = np.unique(kmers, axis=-2)

    raise NotImplementedError
