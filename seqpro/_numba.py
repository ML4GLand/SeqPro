from typing import Optional, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel")
def gufunc_pad_left(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum()
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel")
def gufunc_pad_both(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum() // 2
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1, u1[:], u1[:])"], "(),(n)->(n)", target="parallel")
def gufunc_ohe(
    char: Union[np.uint8, NDArray[np.uint8]],
    alphabet: NDArray[np.uint8],
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]:  # type: ignore
    for i in nb.prange(len(alphabet)):
        res[i] = np.uint8(alphabet[i] == char)  # type: ignore


@nb.guvectorize(["(u1[:], intp[:])"], "(n)->()", target="parallel")
def gufunc_ohe_char_idx(
    seq: NDArray[np.uint8],
    res: Optional[NDArray[np.intp]] = None,
) -> NDArray[np.intp]:  # type: ignore
    """Get the index of each character in an OHE array, leaving unknown as -1.

    For example, with an ACGT-coded OHE array, this maps an OHE array of [A, C, G, T, N]
    to [0, 1, 2, 3, -1].

    Parameters
    ----------
    seq : ndarray[np.uint8]
        A one-hot encoded array of sequence(s).
    res : ndarray[np.intp], optional

    Returns
    -------
    ndarray[np.intp]
    """
    res[0] = np.intp(-1)  # type: ignore
    for i in nb.prange(len(seq)):
        res[0] = i * seq[i]  # type: ignore
