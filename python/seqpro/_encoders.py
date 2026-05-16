from typing import Literal, cast, overload

import awkward as ak
import numpy as np
from numpy.typing import NDArray

from ._numba import (
    gufunc_ohe,
    gufunc_ohe_char_idx,
    gufunc_pad_both,
    gufunc_pad_left,
    gufunc_tokenize,
)
from ._utils import SeqType, StrSeqType, array_slice, cast_seqs, check_axes
from .alphabets._alphabets import AminoAlphabet, NucleotideAlphabet
from .rag import Ragged


def pad_seqs(
    seqs: SeqType,
    pad: Literal["left", "both", "right"],
    pad_value: str | None = None,
    length: int | None = None,
    length_axis: int | None = None,
) -> NDArray[np.bytes_ | np.uint8]:
    """Pad (or truncate) sequences on either the left, right, or both sides.

    Parameters
    ----------
    seqs
    pad
        How to pad. If padding on both sides and an odd amount of padding is needed, 1
        more pad value will be on the right side. Similarly for truncating, if an odd
        amount length needs to be truncated, 1 more character will be truncated from the
        right side.
    pad_val
        Single character to pad sequences with. Needed for string input. Ignored for OHE
        sequences.
    length
        Needed for character or OHE array input. Length to pad or truncate sequences to.
        If not given, uses the length of longest sequence.
    length_axis
        Needed for array input.

    Returns
    -------
    NDArray[np.bytes_ | np.uint8]
        Padded (or truncated) sequences as S1 bytes or uint8 for OHE input.
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


@overload
def ohe(
    seqs: StrSeqType, alphabet: "NucleotideAlphabet | AminoAlphabet"
) -> NDArray[np.uint8]: ...
@overload
def ohe(
    seqs: "Ragged[np.bytes_]", alphabet: "NucleotideAlphabet | AminoAlphabet"
) -> "Ragged[np.uint8]": ...
def ohe(
    seqs: "StrSeqType | Ragged[np.bytes_]",
    alphabet: "NucleotideAlphabet | AminoAlphabet",
) -> "NDArray[np.uint8] | Ragged[np.uint8]":
    """One hot encode sequences against an alphabet.

    Parameters
    ----------
    seqs
        Sequences to encode. Ragged input must have dtype np.bytes_ (S1).
    alphabet

    Returns
    -------
    NDArray[np.uint8] | Ragged[np.uint8]
        One-hot encoded sequences. Dense output has shape (..., length, alphabet_size).
        Ragged output has shape (n, ~L, A).
    """
    arr = (
        alphabet.array
        if isinstance(alphabet, NucleotideAlphabet)
        else alphabet.aa_array
    )

    if isinstance(seqs, Ragged):
        seqs = Ragged(ak.to_packed(seqs))
        n = len(seqs.lengths.ravel())
        A = arr.shape[0]
        trailing = seqs.data.shape[1:]
        flat = gufunc_ohe(seqs.data.view(np.uint8), arr.view(np.uint8))
        # gufunc appends A last: (..., *trailing, A) → move A to axis 1
        if trailing:
            flat = np.moveaxis(flat, -1, 1)
        return Ragged.from_offsets(flat, (n, None, A, *trailing), seqs.offsets)

    _seqs = cast_seqs(seqs)
    return gufunc_ohe(_seqs.view(np.uint8), arr.view(np.uint8))


@overload
def decode_ohe(
    seqs: NDArray[np.uint8],
    ohe_axis: int,
    alphabet: "NucleotideAlphabet | AminoAlphabet",
    unknown_char: str = "N",
) -> NDArray[np.bytes_]: ...
@overload
def decode_ohe(
    seqs: "Ragged[np.uint8]",
    ohe_axis: int,
    alphabet: "NucleotideAlphabet | AminoAlphabet",
    unknown_char: str = "N",
) -> "Ragged[np.bytes_]": ...
def decode_ohe(
    seqs: "NDArray[np.uint8] | Ragged[np.uint8]",
    ohe_axis: int,
    alphabet: "NucleotideAlphabet | AminoAlphabet",
    unknown_char: str = "N",
) -> "NDArray[np.bytes_] | Ragged[np.bytes_]":
    """Convert an OHE array to an S1 byte array.

    Parameters
    ----------
    seqs
        OHE array. Ragged input must have shape (n, ~L, A, ...) as produced by ohe().
    ohe_axis
        Axis of the one-hot dimension. Ignored for Ragged input (always axis 1 of flat data).
    alphabet
    unknown_char
        Single character to use for unknown values, by default "N"

    Returns
    -------
    NDArray[np.bytes_] | Ragged[np.bytes_]
        S1 byte array of decoded characters; ohe_axis is removed from the shape.
    """
    arr = (
        alphabet.array
        if isinstance(alphabet, NucleotideAlphabet)
        else alphabet.aa_array
    )
    _alphabet = np.concatenate([arr, [unknown_char.encode("ascii")]])

    if isinstance(seqs, Ragged):
        seqs = Ragged(ak.to_packed(seqs))
        n = len(seqs.lengths.ravel())
        # A is always at axis 1 in flat data produced by ohe()
        trailing = seqs.data.shape[2:]
        idx = gufunc_ohe_char_idx(seqs.data, axis=1)  # type: ignore
        flat = _alphabet[idx]
        return Ragged.from_offsets(flat, (n, None, *trailing), seqs.offsets)

    idx = gufunc_ohe_char_idx(seqs, axis=ohe_axis)  # type: ignore
    ohe_axis_idx = seqs.ndim + ohe_axis if ohe_axis < 0 else ohe_axis
    shape = (*seqs.shape[:ohe_axis_idx], *seqs.shape[ohe_axis_idx + 1 :])
    return _alphabet[idx].reshape(shape)


@overload
def tokenize(
    seqs: StrSeqType,
    token_map: dict[str, int],
    unknown_token: int,
    out: NDArray[np.int32] | None = None,
) -> NDArray[np.int32]: ...
@overload
def tokenize(
    seqs: Ragged[np.bytes_],
    token_map: dict[str, int],
    unknown_token: int,
    out: None = None,
) -> Ragged[np.int32]: ...
def tokenize(
    seqs: StrSeqType | Ragged[np.bytes_],
    token_map: dict[str, int],
    unknown_token: int,
    out: NDArray[np.int32] | None = None,
) -> NDArray[np.int32] | Ragged[np.int32]:
    """Tokenize sequences. Maps each character to its integer token.
    Characters absent from token_map are replaced with unknown_token.

    Parameters
    ----------
    seqs
        Sequences to tokenize. Ragged input must have dtype np.bytes_ (S1).
    token_map
        Mapping of characters to tokens.
    unknown_token
        Token to use for unknown values.
    out
        Output array to store the result in. Only valid for non-Ragged input.

    Returns
    -------
    NDArray[np.int32] | Ragged[np.int32]
        Integer token IDs with the same shape/layout as the input.
    """
    source = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    target = np.array(list(token_map.values()), dtype=np.int32)
    _unknown_token = np.int32(unknown_token)

    if isinstance(seqs, Ragged):
        seqs = Ragged(ak.to_packed(seqs))
        n = len(seqs.lengths.ravel())
        flat = gufunc_tokenize(seqs.data.view(np.uint8), source, target, _unknown_token)
        return Ragged.from_offsets(flat, (n, None), seqs.offsets)

    _seqs = cast_seqs(seqs)
    return gufunc_tokenize(_seqs.view(np.uint8), source, target, _unknown_token, out)


@overload
def decode_tokens(
    seqs: NDArray[np.int32],
    token_map: dict[str, int],
    unknown_char: str = "N",
) -> NDArray[np.bytes_]: ...
@overload
def decode_tokens(
    seqs: Ragged[np.int32],
    token_map: dict[str, int],
    unknown_char: str = "N",
) -> Ragged[np.bytes_]: ...
def decode_tokens(
    seqs: NDArray[np.int32] | Ragged[np.int32],
    token_map: dict[str, int],
    unknown_char: str = "N",
) -> NDArray[np.bytes_] | Ragged[np.bytes_]:
    """Untokenize sequences. Maps each integer token back to its character.
    Tokens absent from token_map are replaced with unknown_char.

    Parameters
    ----------
    seqs
        Token ID array. Ragged input must have dtype np.int32.
    token_map
        Mapping of characters to tokens (same map used for tokenization).
    unknown_char
        Character to replace unknown tokens with, by default 'N'.

    Returns
    -------
    NDArray[np.bytes_] | Ragged[np.bytes_]
        S1 byte array with the same shape/layout as the input.
    """
    target = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    source = np.array(list(token_map.values()), dtype=np.int32)
    _unk_char = np.uint8(ord(unknown_char))

    if isinstance(seqs, Ragged):
        seqs = Ragged(ak.to_packed(seqs))
        n = len(seqs.lengths.ravel())
        flat = gufunc_tokenize(seqs.data, source, target, _unk_char).view("S1")
        return Ragged.from_offsets(flat, (n, None), seqs.offsets)

    return gufunc_tokenize(seqs, source, target, _unk_char).view("S1")
