from typing import List, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._numba import gufunc_char_idx, gufunc_pad_both, gufunc_pad_left
from ._utils import SeqType, StrSeqType, _check_axes, array_slice, cast_seqs
from .alphabets._alphabets import NucleotideAlphabet


def pad_seqs(
    seqs: SeqType,
    pad: Literal["left", "both", "right"],
    pad_value: Optional[str] = None,
    length: Optional[int] = None,
    length_axis: Optional[int] = None,
) -> NDArray[Union[np.bytes_, np.uint8]]:
    """_summary_

    Parameters
    ----------
    seqs : Iterable[str]
    pad : Literal["left", "both", "right"]
        How to pad. If padding on both sides and an odd amount of padding is needed, 1
        more pad value will be on the right side.
    pad_val : str, optional
        Single character to pad sequences with. Needed for string input. Ignored for OHE
        sequences.
    length : int, optional
        Needed for character or OHE array input. Length to pad or truncate sequences to.
        If not given, uses the length of longest sequence.
    length_axis : Optional[int]
        Needed for array input.

    Returns
    -------
    Array of padded or truncated sequences.
    """
    _check_axes(seqs, length_axis, False)

    string_input = (
        isinstance(seqs, (str, list))
        or (isinstance(seqs, np.ndarray) and seqs.dtype.kind == "U")
        or (isinstance(seqs, np.ndarray) and seqs.dtype.type == np.object_)
    )

    seqs = cast_seqs(seqs)

    if length_axis is None:
        length_axis = -1

    if string_input:
        if pad_value is None:
            raise ValueError("Need a pad value for plain string input.")

        if length is not None:
            seqs = seqs[..., :length]

        seqs = seqs.view(np.uint8)

        if pad == "left":
            seqs = gufunc_pad_left(seqs)
        elif pad == "both":
            seqs = gufunc_pad_both(seqs)

        # convert empty character '' to pad_val
        seqs[seqs == 0] = ord(pad_value)

        seqs = cast(NDArray[np.bytes_], seqs.view("S1"))
    else:
        if length is None:
            raise ValueError("Need a length for array input.")

        length_diff = seqs.shape[length_axis] - length

        if length_diff == 0:
            return seqs
        elif length_diff > 0:  # longer than needed, truncate
            seqs = array_slice(seqs, length_axis, 0, length)
        else:  # shorter than needed, pad
            pad_arr_shape = (
                *seqs.shape[:length_axis],
                -length_diff,
                *seqs.shape[length_axis + 1 :],
            )
            if seqs.dtype == np.uint8:
                pad_arr = np.zeros(pad_arr_shape, np.uint8)
            else:
                if pad_value is None:
                    raise ValueError("Need a pad value for byte array input.")
                pad_arr = np.full(pad_arr_shape, pad_value.encode("ascii"), dtype="S1")
            seqs = np.concatenate([seqs, pad_arr], axis=length_axis)

    return seqs


def ohe(seqs: StrSeqType, alphabet: NucleotideAlphabet) -> NDArray[np.uint8]:
    """One hot encode a nucleotide sequence.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes]
    alphabet : NucleotideAlphabet

    Returns
    -------
    NDArray[np.uint8]
        Ohe hot encoded nucleotide sequences.
    """
    return alphabet.ohe(seqs)


def decode_ohe(
    seqs: NDArray[np.uint8],
    ohe_axis: int,
    alphabet: NucleotideAlphabet,
    unknown_char: str = "N",
) -> NDArray[np.bytes_]:
    """Convert an OHE array to an S1 byte array.

    Parameters
    ----------
    seqs : NDArray[np.uint8]
    ohe_axis : int
    alphabet : NucleotideAlphabet
    unknown_char : str, optional
        Single character to use for unknown values, by default "N"

    Returns
    -------
    NDArray[np.bytes_]
    """
    return alphabet.decode_ohe(seqs=seqs, ohe_axis=ohe_axis, unknown_char=unknown_char)


def tokenize(seqs: StrSeqType, alphabet: NucleotideAlphabet) -> NDArray[np.integer]:
    """Tokenize nucleotides. Replaces each nucleotide with its index in the alphabet.
    Nucleotides not in the alphabet are replaced with -1.

    Parameters
    ----------
    seqs : StrSeqType
    alphabet : NucleotideAlphabet

    Returns
    -------
    NDArray[int]
        Sequences of tokens (integers)
    """
    seqs = cast_seqs(seqs)
    return gufunc_char_idx(seqs.view(np.uint8), alphabet.array.view(np.uint8))


def decode_tokens(
    tokens: NDArray[np.integer], alphabet: NucleotideAlphabet, unknown_char: str = "N"
) -> NDArray[np.bytes_]:
    """Untokenize nucleotides. Replaces each token/index with its corresponding
    nucleotide in the alphabet.


    Parameters
    ----------
    ids : NDArray[np.integer]
    alphabet : NucleotideAlphabet
    unknown_char : str, optional
        Character to replace unknown tokens with, by default 'N'


    Returns
    -------
    NDArray[bytes] aka S1
        Sequences of nucleotides
    """
    chars = cast_seqs(alphabet.alphabet + unknown_char)
    return chars[tokens]
