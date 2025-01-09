from typing import Optional, Union, overload

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", cache=True)
def gufunc_pad_left(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum()
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", cache=True)
def gufunc_pad_both(
    seq: NDArray[np.uint8], res: Optional[NDArray[np.uint8]] = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum() // 2
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1, u1[:], u1[:])"], "(),(n)->(n)", target="parallel", cache=True)
def gufunc_ohe(
    char: Union[np.uint8, NDArray[np.uint8]],
    alphabet: NDArray[np.uint8],
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]:  # type: ignore
    for i in nb.prange(len(alphabet)):
        res[i] = np.uint8(alphabet[i] == char)  # type: ignore


@nb.guvectorize(["(u1[:], intp[:])"], "(n)->()", target="parallel", cache=True)
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


@overload
def gufunc_tokenize(
    seq: NDArray[np.uint8],
    source: NDArray[np.uint8],
    target: NDArray[np.int32],
    unknown_token: np.int32,
    res: Optional[NDArray[np.int32]] = None,
) -> NDArray[np.int32]: ...


@overload
def gufunc_tokenize(
    seq: NDArray[np.int32],
    source: NDArray[np.int32],
    target: NDArray[np.uint8],
    unknown_token: np.uint8,
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]: ...


@nb.guvectorize(
    ["(u1, u1[:], i4[:], i4, i4[:])", "(i4, i4[:], u1[:], u1, u1[:])"],
    "(),(n),(n),()->()",
    target="parallel",
    cache=True,
)
def gufunc_tokenize(
    seq: NDArray[Union[np.int32, np.uint8]],
    source: NDArray[Union[np.int32, np.uint8]],
    target: NDArray[Union[np.int32, np.uint8]],
    unknown_token: Union[np.int32, np.uint8],
    res: Optional[NDArray[Union[np.int32, np.uint8]]] = None,
) -> NDArray[Union[np.int32, np.uint8]]:  # type: ignore
    """Tokenize a sequence.

    Note: np.int32 is returned since token IDs are generally used as indices into an array of embeddings
    (a la torch.nn.Embedding)."""
    res[0] = unknown_token  # type: ignore
    for i in nb.prange(len(source)):
        if seq == source[i]:
            res[0] = target[i]  # type: ignore
            break


@nb.guvectorize(
    ["(u1[:], u1[:, :], u1[:], u1[:])"],
    "(k),(j,k),(j)->()",
    target="parallel",
    cache=True,
)
def gufunc_translate(
    seq_kmers: NDArray[np.uint8],
    kmer_keys: NDArray[np.uint8],
    kmer_values: NDArray[np.uint8],
    res: Optional[NDArray[np.uint8]] = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate 3-mers into amino acids.

    Parameters
    ----------
    seq_kmers : NDArray[np.uint8]
        A k-mer.
    kmer_keys : NDArray[np.uint8]
        All unique k-mers as an (n, k) array.
    kmer_values : NDArray[np.uint8]
        Values corresponding to each k-mer, in corresponding order.
    res : NDArray[np.uint8], optional
        Array to save the result in, by default None
    """
    for i in nb.prange(len(kmer_keys)):
        if (seq_kmers == kmer_keys[i]).all():
            res[0] = kmer_values[i]  # type: ignore
            break
