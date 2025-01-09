from typing import List, Optional, TypeVar, Union, cast, overload

import numpy as np
from numpy.typing import NDArray

NestedStr = Union[str, List["NestedStr"]]
"""String or nested list of strings"""

StrSeqType = Union[NestedStr, NDArray[Union[np.str_, np.object_, np.bytes_]]]
"""String sequence type (i.e. SeqType but not)"""

SeqType = Union[NestedStr, NDArray[Union[np.str_, np.object_, np.bytes_, np.uint8]]]


@overload
def cast_seqs(seqs: NDArray[np.uint8]) -> NDArray[np.uint8]: ...


@overload
def cast_seqs(seqs: StrSeqType) -> NDArray[np.bytes_]: ...


@overload
def cast_seqs(seqs: SeqType) -> NDArray[Union[np.bytes_, np.uint8]]: ...


def cast_seqs(seqs: SeqType) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Cast any sequence type to be a NumPy array of ASCII characters (or left alone as
    8-bit unsigned integers if the input is OHE).

    Parameters
    ----------
    seqs : str, (nested) list[str], ndarray[str, object (Python strings), bytes, uint8]

    Returns
    -------
    ndarray with dtype |S1 or uint8
    """
    if isinstance(seqs, str):
        if len(seqs) == 0:
            raise ValueError("Empty string cannot be cast to a sequence array.")
        return np.array([seqs], "S").view("S1")
    elif isinstance(seqs, list):
        return np.array(seqs, "S")[..., None].view("S1")
    elif seqs.dtype.itemsize > 1:  # dtype == U or bigger than S1
        return seqs.astype("S")[..., None].view("S1")
    else:
        return cast(NDArray[Union[np.bytes_, np.uint8]], seqs)


def check_axes(
    seqs: SeqType,
    length_axis: Optional[Union[int, bool]] = None,
    ohe_axis: Optional[Union[int, bool]] = None,
):
    """Raise errors if length_axis or ohe_axis is missing when they're needed. Pass
    False to corresponding axis to not check for it.

    - ndarray with itemsize == 1 => length axis required.
    - OHE array => length and OHE axis required.
    """
    # bytes or OHE
    if length_axis is None and isinstance(seqs, np.ndarray) and seqs.itemsize == 1:
        raise ValueError(
            "Need a length axis to process an ndarray with itemsize == 1 (S1, u1)."
        )

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


def array_slice(a: NDArray[DTYPE], axis: int, slice_: slice) -> NDArray[DTYPE]:
    """Slice an array from a dynamic axis."""
    return a[(slice(None),) * (axis % a.ndim) + (slice_,)]
