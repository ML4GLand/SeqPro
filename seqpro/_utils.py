from typing import List, Optional, TypeVar, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

StrSeqType = Union[str, List[str], NDArray[Union[np.str_, np.bytes_]]]
SeqType = Union[str, List[str], NDArray[Union[np.str_, np.bytes_, np.uint8]]]


@overload
def cast_seqs(seqs: NDArray[np.uint8]) -> NDArray[np.uint8]:
    ...


@overload
def cast_seqs(seqs: StrSeqType) -> NDArray[np.bytes_]:
    ...


@overload
def cast_seqs(seqs: SeqType) -> NDArray[Union[np.bytes_, np.uint8]]:
    ...


def cast_seqs(seqs: SeqType) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Cast any sequence type to be a NumPy array of ASCII characters (or left alone as
    8-bit integers if the input is OHE).

    Parameters
    ----------
    seqs : str, (nested) list[str], ndarray[str, bytes, uint8]

    Returns
    -------
    ndarray with dtype |S1 or uint8
    """
    if isinstance(seqs, str):
        return np.array([seqs], "S").view("S1")
    elif isinstance(seqs, list):
        return np.array(seqs, "S")[..., None].view("S1")
    elif seqs.dtype.itemsize > 1:  # dtype == U or bigger than S1
        return seqs.astype("S")[..., None].view("S1")
    else:
        return cast(NDArray[Union[np.bytes_, np.uint8]], seqs)


def _check_axes(
    seqs: SeqType,
    length_axis: Optional[Union[int, bool]] = None,
    ohe_axis: Optional[Union[int, bool]] = None,
):
    """Raise errors if length_axis or ohe_axis is missing when they're needed. Pass
    False to corresponding axis to not check for it.

    - Any ndarray => length axis required.
    - Any OHE array => length and OHE axis required.
    """
    # bytes or OHE
    if length_axis is None and isinstance(seqs, np.ndarray) and seqs.itemsize == 1:
        raise ValueError("Need a length axis to process an ndarray.")

    # OHE
    if ohe_axis is None and isinstance(seqs, np.ndarray) and seqs.dtype == np.uint8:
        raise ValueError("Need an one hot encoding axis to process OHE sequences.")

    # length_axis != ohe_axis
    if (
        isinstance(length_axis, int)
        and isinstance(ohe_axis, int)
        and (length_axis == ohe_axis)
    ):
        raise ValueError("Length and OHE axis must be different.")


DTYPE = TypeVar("DTYPE", bound=np.generic)


def array_slice(
    a: NDArray[DTYPE], axis: int, start: int, end: int, step=1
) -> NDArray[DTYPE]:
    """Slice an array from a dynamic axis."""
    return a[(slice(None),) * (axis % a.ndim) + (slice(start, end, step),)]


def random_base(alphabet=["A", "G", "C", "T"]):
    """Generate a random base from the AGCT alpahbet.

    Parameters
    ----------
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).

    Returns
    -------
    str
        Randomly chosen base.
    """
    return np.random.choice(alphabet)


def random_seq(seq_len, alphabet=["A", "G", "C", "T"]):
    """Generate a random sequence of length seq_len.

    Parameters
    ----------
    seq_len : int
        Length of sequence to return.
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).

    Returns
    -------
    str
        Randomly generated sequence.
    """
    return "".join([random_base(alphabet) for i in range(seq_len)])


def random_seqs(seq_num, seq_len, alphabet=["A", "G", "C", "T"]):
    """Generate seq_num random sequences of length seq_len.

    Parameters
    ----------
    seq_num (int):
        number of sequences to return
    seq_len (int):
        length of sequence to return
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).

    Returns
    -------
    numpy array
        Array of randomly generated sequences.
    """
    return np.array([random_seq(seq_len, alphabet) for i in range(seq_num)])