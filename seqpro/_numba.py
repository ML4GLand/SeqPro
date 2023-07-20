from typing import Optional, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", nopython=True)
def gufunc_pad_left(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum()
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", nopython=True)
def gufunc_pad_both(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum() // 2
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1, u1[:], u1[:])"], "(),(n)->(n)", target="parallel", nopython=True)
def gufunc_ohe(
    char: Union[np.uint8, NDArray[np.uint8]],
    alphabet: NDArray[np.uint8],
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]:  # type: ignore
    for i in nb.prange(len(alphabet)):
        res[i] = np.uint8(alphabet[i] == char)  # type: ignore


@nb.guvectorize(["(u1[:], intp[:])"], "(n)->()", target="parallel", nopython=True)
def gufunc_ohe_char_idx(
    seq: NDArray[np.uint8],
    res: Optional[NDArray[np.intp]] = None,
) -> NDArray[np.intp]:  # type: ignore
    """Get the index of each character in an OHE array, leaving unknown as -1.

    For example, with an ACGT-coded OHE array, this maps an OHE array equal to
    [A, C, G, T, N] to [0, 1, 2, 3, -1].

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
        if seq[i] == 1:
            res[0] = i  # type: ignore


@nb.guvectorize(
    ["(u1, u1[:], intp[:])"], "(),(n)->()", target="parallel", nopython=True
)
def gufunc_char_idx(
    seq: NDArray[np.uint8],
    alphabet: NDArray[np.uint8],
    res: Optional[NDArray[np.intp]] = None,
) -> NDArray[np.intp]:  # type: ignore
    res[0] = np.intp(-1)  # type: ignore
    for i in nb.prange(len(alphabet)):
        if seq == alphabet[i]:
            res[0] = i  # type: ignore


@nb.guvectorize(
    ["(u1[:], u1[:, :], u1[:], u1[:])"],
    "(k),(j,k),(j)->()",
    target="parallel",
    nopython=True,
)
def gufunc_translate(
    seq: NDArray[np.uint8],
    codons: NDArray[np.uint8],
    aminos_acids: NDArray[np.uint8],
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate 3-mers into amino acids.

    Parameters
    ----------
    seq : NDArray[np.uint8]
        A k-mer or ndarray of k-mers.
    codons : NDArray[np.uint8]
        All unique k-mer codons as an (n, k) array.
    aminos_acids : NDArray[np.uint8]
        All amino acids corresponding to each codon, in matching order.
    res : Optional[NDArray[np.uint8]], optional
        Array to save the result in, by default None
    """
    for i in nb.prange(len(codons)):
        if (seq == codons[i]).all():
            res[0] = aminos_acids[i]  # type: ignore


@nb.guvectorize("(n),(),()->(n)", nopython=True)
def gufunc_jitter_helper(
    arr: NDArray,
    start: Union[int, NDArray[np.integer]],
    max_jitter: Union[int, NDArray[np.integer]],
    res: NDArray,
):
    new_length = len(arr) - 2 * max_jitter
    res[:new_length] = arr[start : start + new_length]
