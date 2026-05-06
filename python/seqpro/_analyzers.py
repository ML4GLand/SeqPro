from typing import cast

import numpy as np
from numpy.typing import NDArray

from ._utils import SeqType, cast_seqs, check_axes
from .alphabets._alphabets import NucleotideAlphabet


def length(seqs: str | list[str]) -> NDArray[np.integer]:
    """Calculate the length of each sequence.

    Parameters
    ----------
    seqs
        List of sequences.

    Returns
    -------
    result
        Array containing the length of each sequence.
    """
    _seqs = cast_seqs(seqs)
    return (_seqs != b"").sum(-1)


def gc_content(
    seqs: SeqType,
    normalize=True,
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
    result
        Returns integers if unnormalized, otherwise floats.
    """
    check_axes(seqs, length_axis, ohe_axis)

    seqs = cast_seqs(seqs)

    if length_axis is None:  # length axis after casting strings
        length_axis = seqs.ndim - 1
    elif length_axis < 0:
        length_axis = seqs.ndim + length_axis

    if seqs.dtype == np.uint8:  # OHE
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze OHE sequences.")
        assert ohe_axis is not None

        gc_idx = np.flatnonzero(np.isin(alphabet.array, np.array([b"C", b"G"])))
        gc_content = cast(
            NDArray[np.integer],
            np.take(seqs, gc_idx, ohe_axis).sum((length_axis, ohe_axis)),
        )
    else:
        gc_content = cast(
            NDArray[np.integer], np.isin(seqs, np.array([b"C", b"G"])).sum(length_axis)
        )

    if normalize:
        gc_content = gc_content / seqs.shape[length_axis]

    return gc_content


def nucleotide_content(
    seqs: SeqType,
    normalize=True,
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
    result
        Returns integers if unnormalized, otherwise floats.
    """
    check_axes(seqs, length_axis, False)

    seqs = cast_seqs(seqs)

    if length_axis is None:
        length_axis = seqs.ndim - 1
    elif length_axis < 0:
        length_axis = seqs.ndim + length_axis

    if seqs.dtype == np.uint8:  # OHE
        nuc_content = cast(NDArray[np.integer], seqs.sum(length_axis))
    else:
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze string nucleotide content.")
        nuc_content = np.zeros(
            (*seqs.shape[:length_axis], *seqs.shape[length_axis + 1 :], len(alphabet)),
            dtype=np.int64,
        )
        for i, nuc in enumerate(alphabet.array):
            nuc_content[..., i] = (seqs == nuc).sum(length_axis)

    if normalize:
        nuc_content = nuc_content / seqs.shape[length_axis]

    return nuc_content


def count_kmers_seq(seq: str, k: int) -> dict:
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
    kmers
        k-mers and their counts expressed in a dictionary.
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
    kmers
        Array of all possible unique k-mers.
    counts
        Counts of all possible k-mers for each input sequence.
    """
    raise NotImplementedError

    check_axes(seqs, length_axis, False)

    seqs = cast_seqs(seqs)

    if length_axis is None:
        length_axis = -1

    length = seqs.shape[length_axis]

    if length >= k:
        raise ValueError("k is larger than sequence length")

    if seqs.dtype == np.uint8:
        raise NotImplementedError
    else:
        kmers = np.lib.stride_tricks.sliding_window_view(seqs, k, -1)
        # can't use np.unique here because each sequence would return a potentially
        # different number of kmers -> ragged array
        uniq, counts = np.unique(kmers, axis=-2)

    raise NotImplementedError
