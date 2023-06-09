from typing import Literal, Optional, Union, cast, List

import torch
import numpy as np
from numpy.typing import NDArray

from ._alphabets import NucleotideAlphabet
from ._numba import gufunc_ohe, gufunc_ohe_char_idx, gufunc_pad_both, gufunc_pad_left
from ._utils import SeqType, StrSeqType, _check_axes, array_slice, cast_seqs
from ._helpers import _get_vocab, _sequencize, _one_hot2token
from tqdm.auto import tqdm

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
        Single character to pad sequences with. Ignored for OHE sequences.
    length : int, optional
        Length to pad or truncate sequences to. If not given, uses length of longest
        sequence.

    Returns
    -------
    Array of padded or truncated sequences.
    """
    _check_axes(seqs, length_axis, False)

    string_input = isinstance(seqs, (str, list)) or (
        isinstance(seqs, np.ndarray) and seqs.dtype.kind == "U"
    ) or (
        isinstance(seqs, np.ndarray) and seqs.dtype.type == np.object_
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
        elif length_diff > 0:
            seqs = array_slice(seqs, length_axis, 0, length)
        else:
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
    seqs = cast_seqs(seqs)
    return gufunc_ohe(seqs.view(np.uint8), alphabet.array.view(np.uint8))


# TODO: test this
def ohe_to_bytes(
    ohe_arr: NDArray[np.uint8],
    ohe_axis: int,
    alphabet: NucleotideAlphabet,
    unknown_char: str = "N",
) -> NDArray[np.bytes_]:
    """Convert an OHE array to an S1 byte array.

    Parameters
    ----------
    ohe_arr : NDArray[np.uint8]
    ohe_axis : int
    alphabet : NucleotideAlphabet
    unknown_char : str, optional
        Single character to use for unknown values, by default "N"

    Returns
    -------
    NDArray[np.bytes_]
    """
    idx = gufunc_ohe_char_idx(ohe_arr, axis=ohe_axis)  # type: ignore

    if ohe_axis < 0:
        ohe_axis_idx = ohe_arr.ndim + ohe_axis
    else:
        ohe_axis_idx = ohe_axis

    shape = *ohe_arr.shape[:ohe_axis_idx], *ohe_arr.shape[ohe_axis_idx + 1 :]

    _alphabet = np.concatenate([alphabet.array, [unknown_char.encode("ascii")]])

    return _alphabet[idx].reshape(shape)
