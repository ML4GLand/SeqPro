from typing import Optional, Tuple, Union, cast

import numpy as np
import ushuffle
from numpy.typing import NDArray

from ._utils import SeqType, _check_axes, cast_seqs
from .alphabets._alphabets import NucleotideAlphabet


def reverse_complement(
    seqs: SeqType,
    alphabet: NucleotideAlphabet,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
):
    """Reverse complement a sequence.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes, uint8]
    alphabet : NucleotideAlphabet
    length_axis : Optional[int], optional
        Needed for array input. Length axis, by default None
    ohe_axis : Optional[int], optional
        Needed for OHE input. One hot encoding axis, by default None

    Returns
    -------
    ndarray[bytes, uint8]
        Array of bytes (S1) or uint8 for string or OHE input, respectively.
    """
    return alphabet.reverse_complement(seqs, length_axis, ohe_axis)


def k_shuffle(
    seqs: SeqType,
    k: int,
    *,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
    seed: Optional[int] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Shuffle sequences while preserving k-let frequencies.

    Parameters
    ----------
    seqs : SeqType
    k : int
        Size of k-lets to preserve frequencies of.
    length_axis : Optional[int], optional
        Needed for array input. Axis that corresponds to the length of sequences.
    ohe_axes : Optional[int], optional
        Needed for OHE input. Axis that corresponds to the one hot encoding, should be
        the same size as the length of the alphabet.
    seed : Optional[int], optional
        Seed for shuffling.
    alphabet : Optional[NucleotideAlphabet], optional
        Alphabet, needed for OHE sequence input.
    """

    _check_axes(seqs, length_axis, ohe_axis)

    seqs = cast_seqs(seqs)

    # only get here if seqs was str or list[str]
    if length_axis is None:
        length_axis = -1

    if k == 1:
        rng = np.random.default_rng(seed)
        return rng.permuted(seqs, axis=length_axis)

    if seed is not None:
        import importlib

        importlib.reload(ushuffle)
        ushuffle.set_seed(seed)

    seqs = seqs.copy()

    if seqs.dtype == np.uint8:
        assert ohe_axis is not None
        seqs = cast(NDArray[np.uint8], seqs)
        ohe = True
        if alphabet is None:
            raise ValueError("Need an alphabet to process OHE sequences.")
        seqs = alphabet.decode_ohe(seqs, ohe_axis=ohe_axis)
    else:
        ohe = False

    length = seqs.shape[length_axis]
    with np.nditer(seqs.view(f"S{length}").ravel(), op_flags=["readwrite"]) as it:  # type: ignore
        for seq in it:
            seq[...] = ushuffle.shuffle(seq.tobytes(), k)  # type: ignore

    if ohe:
        assert ohe_axis is not None
        assert alphabet is not None
        seqs = cast(NDArray[np.bytes_], seqs)
        seqs = alphabet.ohe(seqs).swapaxes(-1, ohe_axis)

    return seqs


def bin_coverage(
    coverage: NDArray[np.number],
    bin_width: int,
    length_axis: int,
    normalize=False,
) -> NDArray[np.number]:
    """Bin coverage by summing over non-overlapping windows.

    Parameters
    ----------
    coverage_array : NDArray
    bin_width : int
        Width of the windows to sum over. Must be an even divisor of the length
        of the coverage array. If not, raises an error. The length dimension is
        assumed to be the second dimension.
    length_axis: int
    normalize : bool, default False
        Whether to normalize by the length of the bin.

    Returns
    -------
    binned_coverage : NDArray
    """
    length = coverage.shape[length_axis]
    if length % bin_width != 0:
        raise ValueError("Bin width must evenly divide length.")
    binned_coverage = np.add.reduceat(
        coverage, np.arange(0, length, bin_width), axis=length_axis
    )
    if normalize:
        binned_coverage /= bin_width
    return binned_coverage


def random_seqs(
    shape: Union[int, Tuple[int, ...]],
    alphabet: NucleotideAlphabet,
    seed: Optional[int] = None,
):
    """Generate random nucleotide sequences.

    Parameters
    ----------
    shape : int, tuple[int]
        Shape of sequences to generate
    alphabet : NucleotideAlphabet
        Alphabet to sample nucleotides from.
    seed : int, optional
        Random seed.

    Returns
    -------
    ndarray
        Randomly generated sequences.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(alphabet.array, size=shape)
