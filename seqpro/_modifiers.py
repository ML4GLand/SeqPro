from typing import Optional, Union, cast

import numpy as np
import ushuffle
from numpy.typing import NDArray

from ._alphabets import NucleotideAlphabet
from ._utils import SeqType, _check_axes, cast_seqs


def reverse_complement(
    seqs: SeqType,
    alphabet: NucleotideAlphabet,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
):
    return alphabet.reverse_complement(seqs, length_axis, ohe_axis)


def shuffle(
    seqs: SeqType, length_axis: Optional[int] = None, *, seed: Optional[int] = None
) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Shuffle sequences.

    Parameters
    ----------
    seqs : SeqType
    length_axis : Optional[int], optional
        Axis that corresponds to the length of sequences.
    seed : Optional[int], optional
        Seed for shuffling.

    Returns
    -------
    NDArray[bytes | uint8]
        Shuffled sequences.

    Raises
    ------
    ValueError
        When not given a length axis and an array is passed in.
    """
    _check_axes(seqs, length_axis, False)

    if isinstance(seqs, np.ndarray) and length_axis is None:
        raise ValueError("Need a length axis to shuffle arrays.")
    elif length_axis is None:
        length_axis = -1

    seqs = cast_seqs(seqs)

    rng = np.random.default_rng(seed)
    return rng.permuted(seqs, axis=length_axis)


def k_shuffle(
    seqs: SeqType,
    k: int,
    length_axis: Optional[int] = None,
    seed: Optional[int] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
):
    """Shuffle sequences while preserving k-let frequencies.

    Parameters
    ----------
    seqs : SeqType
    k : int
        Size of k-lets to preserve frequencies of.
    length_axis : Optional[int], optional
        Axis that corresponds to the length of sequences.
    seed : Optional[int], optional
        Seed for shuffling. NOTE: this can only be set once and subsequent calls cannot
        change the seed!
    """
    # ! ushuffle.set_seed only works the first time it is called
    if seed is not None:
        ushuffle.set_seed(seed)

    _check_axes(seqs, length_axis, False)

    seqs = cast_seqs(seqs).copy()

    # only get here if seqs was str or list[str]
    if length_axis is None:
        length_axis = -1

    if seqs.dtype == np.uint8:
        seqs = cast(NDArray[np.uint8], seqs)
        ohe = True
        if alphabet is None:
            raise ValueError("Need an alphabet to process OHE sequences.")
        seqs = alphabet.ohe_to_bytes(seqs)
    else:
        ohe = False

    length = seqs.shape[length_axis]
    with np.nditer(seqs.view(f"S{length}").ravel(), op_flags=["readwrite"]) as it:  # type: ignore
        for seq in it:
            seq[...] = ushuffle.shuffle(seq.tobytes(), k)  # type: ignore

    if ohe:
        assert alphabet is not None
        seqs = cast(NDArray[np.bytes_], seqs)
        seqs = alphabet.bytes_to_ohe(seqs)

    return seqs
