from __future__ import annotations

from typing import overload

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", cache=True)
def gufunc_pad_left(
    seq: NDArray[np.uint8], res: NDArray[np.uint8] | None = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum()
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", cache=True)
def gufunc_pad_both(
    seq: NDArray[np.uint8], res: NDArray[np.uint8] | None = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum() // 2
    for i in nb.prange(len(seq)):
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1, u1[:], u1[:])"], "(),(n)->(n)", target="parallel", cache=True)
def gufunc_ohe(
    char: np.uint8 | NDArray[np.uint8],
    alphabet: NDArray[np.uint8],
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    for i in nb.prange(len(alphabet)):
        res[i] = np.uint8(alphabet[i] == char)  # type: ignore


@nb.guvectorize(["(u1[:], intp[:])"], "(n)->()", target="parallel", cache=True)
def gufunc_ohe_char_idx(
    seq: NDArray[np.uint8],
    res: NDArray[np.intp] | None = None,
) -> NDArray[np.intp]:  # type: ignore
    """Get the index of each character in an OHE array, leaving unknown as -1.

    For example, with an ACGT-coded OHE array, this maps an OHE array equal to
    [A, C, G, T, N] to [0, 1, 2, 3, -1].

    Parameters
    ----------
    seq
        A one-hot encoded array of sequence(s).
    res

    Returns
    -------
    NDArray[np.intp]
        Index of the set bit in each OHE vector, or -1 for unknown characters.
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
    res: NDArray[np.int32] | None = None,
) -> NDArray[np.int32]: ...


@overload
def gufunc_tokenize(
    seq: NDArray[np.int32],
    source: NDArray[np.int32],
    target: NDArray[np.uint8],
    unknown_token: np.uint8,
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]: ...


@nb.guvectorize(
    ["(u1, u1[:], i4[:], i4, i4[:])", "(i4, i4[:], u1[:], u1, u1[:])"],
    "(),(n),(n),()->()",
    target="parallel",
    cache=True,
)
def gufunc_tokenize(
    seq: NDArray[np.int32 | np.uint8],
    source: NDArray[np.int32 | np.uint8],
    target: NDArray[np.int32 | np.uint8],
    unknown_token: np.int32 | np.uint8,
    res: NDArray[np.int32 | np.uint8] | None = None,
) -> NDArray[np.int32 | np.uint8]:  # type: ignore
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
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate k-mers into amino acids via an O(n) linear scan.

    Generic fallback for non-standard alphabets (codon length other than 3,
    or extended/IUPAC characters). For the standard genetic code (k=3, ACGT),
    ``AminoAlphabet.translate`` automatically uses the O(1)
    :func:`gufunc_translate_lut` path instead.

    Parameters
    ----------
    seq_kmers
        A k-mer.
    kmer_keys
        All unique k-mers as an (n, k) array.
    kmer_values
        Values corresponding to each k-mer, in corresponding order.
    res
        Array to save the result in, by default None
    """
    for i in nb.prange(len(kmer_keys)):
        if (seq_kmers == kmer_keys[i]).all():
            res[0] = kmer_values[i]  # type: ignore
            break


@nb.njit(cache=True)
def _pack_codon_index(b0, b1, b2):
    """Pack a 3-codon's ASCII bytes into a 6-bit LUT index in ``[0, 63]``.

    Uses the 2-bit-per-nucleotide hash ``(byte >> 1) & 3``, a bijection on
    ``{A, C, G, T}``. This is the single source of truth for the codon→index
    mapping: both the runtime lookup (:func:`gufunc_translate_lut`) and the
    table builder (``_build_translate_lut``) call it, so they cannot drift
    out of sync.
    """
    n0 = (b0 >> 1) & 3
    n1 = (b1 >> 1) & 3
    n2 = (b2 >> 1) & 3
    return (n0 << 4) | (n1 << 2) | n2


@nb.guvectorize(
    ["(u1[:], u1[:], u1[:])"],
    "(k),(m)->()",
    target="parallel",
    cache=True,
)
def gufunc_translate_lut(
    seq_kmers: NDArray[np.uint8],
    codon_lut: NDArray[np.uint8],
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate a 3-codon to its amino acid via an O(1) lookup table.

    Selected automatically by ``AminoAlphabet.translate`` for the standard
    genetic code (k=3, ACGT); non-standard alphabets use
    :func:`gufunc_translate` instead.

    The LUT is indexed by ``_pack_codon_index``, which hashes each nucleotide's
    ASCII byte with ``(byte >> 1) & 3`` — a bijection on ``{A, C, G, T}`` — and
    packs the three 2-bit codes into a 6-bit index in ``[0, 63]``. The
    64-element ``codon_lut`` (built by ``AminoAlphabet`` at construction) returns
    the amino-acid byte for that index.

    Parameters
    ----------
    seq_kmers
        A 3-codon as ASCII bytes (e.g. ``[65, 84, 71]`` = ``"ATG"``).
    codon_lut
        64-byte LUT, built by ``AminoAlphabet`` at construction time.
    res
        Output buffer.
    """
    res[0] = codon_lut[_pack_codon_index(seq_kmers[0], seq_kmers[1], seq_kmers[2])]


@nb.guvectorize(
    ["(u1, u1[:], u1[:])"],
    "(),(n)->()",
    nopython=True,
    cache=True,
)
def gufunc_complement_bytes(
    seq: NDArray[np.uint8],
    complement_map: NDArray[np.uint8],
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    res[0] = complement_map[seq]  # type: ignore


_COMP = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)


@nb.vectorize(["u1(u1)"], nopython=True, cache=True)
def ufunc_comp_dna(seq: NDArray[np.uint8]) -> NDArray[np.uint8]:
    return _COMP[seq]


@nb.njit(parallel=True, cache=True)
def _nb_find_stop_ends(
    data: NDArray[np.uint8],
    starts: NDArray[np.int64],
    full_ends: NDArray[np.int64],
    stop_char: np.uint8,
) -> NDArray[np.int64]:
    """Find per-sequence end positions in a flat translated AA buffer, truncating at the
    first occurrence of stop_char (inclusive). Runs in parallel across sequences.

    Parameters
    ----------
    data
        Flat translated AA buffer viewed as uint8.
    starts
        Start position of each sequence in data.
    full_ends
        Full (non-truncated) end position of each sequence in data.
    stop_char
        uint8 value of the stop codon character (ord('*') = 42).

    Returns
    -------
    NDArray[np.int64]
        Truncated end positions (exclusive), one per sequence.
    """
    n = len(starts)
    ends = np.empty(n, dtype=np.int64)
    for i in nb.prange(n):
        end = full_ends[i]
        for j in range(starts[i], full_ends[i]):
            if data[j] == stop_char:
                end = j + 1
                break
        ends[i] = end
    return ends
