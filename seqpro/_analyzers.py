from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from ._alphabets import NucleotideAlphabet
from ._utils import SeqType, _check_axes, cast_seqs


def length(seqs: Union[str, List[str]]) -> NDArray[np.integer]:
    """Calculate the length of each sequence.

    Parameters
    ----------
    seqs : str, list[str]
        List of sequences.

    Returns
    -------
    np.array
        Array containing the length of each sequence.
    """
    _seqs = cast_seqs(seqs)
    return (_seqs != b"").sum(-1)


def gc_content(
    seqs: SeqType,
    normalize=True,
    length_axis: Optional[int] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
    ohe_axis: Optional[int] = None,
) -> NDArray[np.integer]:
    """Compute the number or proportion of G & C nucleotides.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes, uint8]
    normalize : bool, default True
        True -> return proportions
        False -> return counts
    length_axis : Optional[int]
        Needed if seqs is an array.
    alphabet : Optional[SequenceAlphabet]
        Needed if seqs is OHE.
    ohe_axis : Optional[int]
        Needed if seqs is OHE.

    Returns
    -------
    NDArray[np.integer]
    """
    _check_axes(seqs, length_axis, ohe_axis)

    seqs = cast_seqs(seqs)

    if length_axis is None:  # length axis after casting strings
        length_axis = -1

    if seqs.dtype == np.uint8:  # OHE
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze OHE sequences.")
        assert ohe_axis is not None

        gc_idx = np.flatnonzero(np.isin(alphabet.array, np.array([b"C", b"G"])))
        gc_content = np.take(seqs, gc_idx, ohe_axis).sum((length_axis, ohe_axis))
    else:
        gc_content = np.isin(seqs, np.array([b"C", b"G"])).sum(length_axis)

    if normalize:
        gc_content /= seqs.shape[length_axis]

    return gc_content


def nucleotide_content(
    seqs: SeqType,
    normalize=True,
    length_axis: Optional[int] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
):
    """Compute the number or proportion of each nucleotide.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes, uint8]
    normalize : bool, default True
        True -> return proportions
        False -> return counts
    length_axis : Optional[int]
        Needed if seqs is an array.

    Returns
    -------
    NDArray[np.integer]
    """
    _check_axes(seqs, length_axis, False)

    seqs = cast_seqs(seqs)

    if length_axis is None:
        length_axis = -1

    if seqs.dtype == np.uint8:  # OHE
        nuc_content = seqs.sum(length_axis)
    else:
        if alphabet is None:
            raise ValueError("Need an alphabet to analyze string nucleotide content.")
        nuc_content = np.zeros(
            (*seqs.shape[:length_axis], *seqs.shape[length_axis + 1 :], len(alphabet))
        )
        for i, nuc in enumerate(alphabet.array):
            nuc_content[..., i] = (seqs == nuc).sum(length_axis)

    if normalize:
        nuc_content /= seqs.shape[length_axis]

    return nuc_content


def count_kmers_seq(seq: str, k: int) -> dict:
    """
    Count unique k-mers.

    Parameters
    ----------
    seq : str
        Nucleotide seq expressed as a string.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).

    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    assert len(seq) >= k, "Length of seq must be greater than that of k."

    _seq = np.array([seq], "S").view("S1")
    kmers = np.lib.stride_tricks.sliding_window_view(_seq, k)
    kmers, counts = np.unique(kmers, return_counts=True)
    kmers = (b"".join(kmer).decode("ascii") for kmer in kmers)
    out = dict(zip(kmers, counts))

    return out


# TODO: non-trivial to parallelize/SIMD this
def _count_kmers(
    seqs: SeqType,
    k: int,
    alphabet: NucleotideAlphabet,
    length_axis: Optional[int] = None,
) -> NDArray[np.unsignedinteger]:
    """
    Count unique k-mers.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes, uint8]
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).
    alphabet : SequenceAlphabet

    Returns
    -------
    kmers : ndarray[bytes]
        Array of all possible unique k-mers.
    counts : ndarray[int]
        Counts of all possible k-mers for each input sequence.
    """
    _check_axes(seqs, length_axis, False)

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
