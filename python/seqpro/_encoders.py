from typing import Dict, Literal, Optional, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._numba import gufunc_pad_both, gufunc_pad_left, gufunc_tokenize
from ._utils import SeqType, StrSeqType, array_slice, cast_seqs, check_axes
from .alphabets._alphabets import AminoAlphabet, NucleotideAlphabet


def pad_seqs(
    seqs: SeqType,
    pad: Literal["left", "both", "right"],
    pad_value: Optional[str] = None,
    length: Optional[int] = None,
    length_axis: Optional[int] = None,
) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Pad (or truncate) sequences on either the left, right, or both sides.

    Parameters
    ----------
    seqs : Iterable[str]
    pad : Literal["left", "both", "right"]
        How to pad. If padding on both sides and an odd amount of padding is needed, 1
        more pad value will be on the right side. Similarly for truncating, if an odd
        amount length needs to be truncated, 1 more character will be truncated from the
        right side.
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
    check_axes(seqs, length_axis, False)

    string_input = (
        isinstance(seqs, (str, list))
        or (isinstance(seqs, np.ndarray) and seqs.dtype.kind == "U")
        or (isinstance(seqs, np.ndarray) and seqs.dtype.type == np.object_)
    )

    seqs = cast_seqs(seqs)

    if length_axis is None:
        length_axis = seqs.ndim - 1

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
            if pad == "left":
                seqs = array_slice(seqs, length_axis, slice(-length))
            elif pad == "both":
                seqs = array_slice(
                    seqs, length_axis, slice(length_diff // 2, -length_diff // 2)
                )
            else:
                seqs = array_slice(seqs, length_axis, slice(None, length))
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


def ohe(
    seqs: StrSeqType, alphabet: Union[NucleotideAlphabet, AminoAlphabet]
) -> NDArray[np.uint8]:
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
    alphabet: Union[NucleotideAlphabet, AminoAlphabet],
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


def tokenize(
    seqs: StrSeqType,
    token_map: Dict[str, int],
    unknown_token: int,
    out: Optional[NDArray[np.int32]] = None,
) -> NDArray[np.int32]:
    """Tokenize nucleotides. Replaces each nucleotide with its corresponding token, if provided. Otherwise, uses each
    nucleotide's index in the alphabet. Nucleotides not in the alphabet or list of tokens are replaced with -1.

    Parameters
    ----------
    seqs : StrSeqType
    alphabet : NucleotideAlphabet
    tokens : Optional[StrSeqType], optional
        List of tokens to use for each nucleotide, by default None

    Returns
    -------
    NDArray[int32]
        Sequences of tokens (integers)
    """
    seqs = cast_seqs(seqs)
    source = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    target = np.array(list(token_map.values()), dtype=np.int32)
    _unknown_token = np.int32(unknown_token)
    return gufunc_tokenize(seqs.view(np.uint8), source, target, _unknown_token, out)


def decode_tokens(
    seqs: NDArray[np.int32],
    token_map: Dict[str, int],
    unknown_char: str = "N",
) -> NDArray[np.bytes_]:
    """Untokenize nucleotides. Replaces each token/index with its corresponding
    nucleotide in the alphabet.


    Parameters
    ----------
    ids : NDArray[np.int32]
    alphabet : NucleotideAlphabet
    tokens : Optional[NDArray[np.int32]], optional
        List of tokens to use for each nucleotide, by default None
    unknown_char : str, optional
        Character to replace unknown tokens with, by default 'N'


    Returns
    -------
    NDArray[bytes] aka S1
        Sequences of nucleotides
    """
    target = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    source = np.array(list(token_map.values()), dtype=np.int32)
    _unk_char = np.uint8(ord(unknown_char))
    _seqs = gufunc_tokenize(seqs, source, target, _unk_char).view("S1")
    return _seqs
