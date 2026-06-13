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
    for i in nb.prange(len(seq)):  # type: ignore[not-iterable]
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1[:], u1[:])"], "(n)->(n)", target="parallel", cache=True)
def gufunc_pad_both(
    seq: NDArray[np.uint8], res: NDArray[np.uint8] | None = None
) -> NDArray[np.uint8]:  # type: ignore
    shift = (seq == 0).sum() // 2
    for i in nb.prange(len(seq)):  # type: ignore[not-iterable]
        res[(i + shift) % len(seq)] = seq[i]  # type: ignore


@nb.guvectorize(["(u1, u1[:], u1[:])"], "(),(n)->(n)", target="parallel", cache=True)
def gufunc_ohe(
    char: np.uint8 | NDArray[np.uint8],
    alphabet: NDArray[np.uint8],
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    for i in nb.prange(len(alphabet)):  # type: ignore[not-iterable]
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
    for i in nb.prange(len(seq)):  # type: ignore[not-iterable]
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
    for i in nb.prange(len(source)):  # type: ignore[not-iterable]
        if seq == source[i]:
            res[0] = target[i]  # type: ignore
            break


@nb.guvectorize(
    ["(u1[:], u1[:, :], u1[:], u1, u1[:])"],
    "(k),(j,k),(j),()->()",
    target="parallel",
    cache=True,
)
def gufunc_translate(
    seq_kmers: NDArray[np.uint8],
    kmer_keys: NDArray[np.uint8],
    kmer_values: NDArray[np.uint8],
    marker_byte: np.uint8,
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate k-mers into amino acids via an O(n) linear scan.

    Generic fallback for non-standard alphabets (codon length other than 3,
    or extended/IUPAC characters). For the standard genetic code (k=3, ACGT),
    ``AminoAlphabet.translate`` automatically uses the O(1)
    :func:`gufunc_translate_lut` path instead.

    A k-mer that does not match any entry in ``kmer_keys`` resolves to the
    caller-supplied ``marker_byte`` rather than leaving ``res[0]``
    uninitialised. ``guvectorize`` allocates output buffers via ``np.empty``,
    so without this sentinel a missing-codon match would emit whatever byte
    happened to be on the page — typically NUL on fresh pages, producing
    silently corrupt AA sequences downstream.

    Case-insensitivity: ASCII letters in ``seq_kmers`` and ``kmer_keys`` are
    upper-cased on the fly via ``b & 0xDF`` (the bit-5 flip is a no-op on
    uppercase ASCII alphas and turns lowercase into uppercase), so soft-masked
    / mixed-case input still translates normally.

    Parameters
    ----------
    seq_kmers
        A k-mer.
    kmer_keys
        All unique k-mers as an (n, k) array.
    kmer_values
        Values corresponding to each k-mer, in corresponding order.
    marker_byte
        ASCII byte emitted when no kmer in ``kmer_keys`` matches the input
        (i.e. an unknown codon). The Python wrapper validates this is a
        single byte.
    res
        Array to save the result in, by default None
    """
    res[0] = marker_byte  # type: ignore
    k = len(seq_kmers)
    for i in range(len(kmer_keys)):
        match = True
        for j in range(k):
            if (seq_kmers[j] & 0xDF) != (kmer_keys[i, j] & 0xDF):
                match = False
                break
        if match:
            res[0] = kmer_values[i]  # type: ignore
            break


@nb.njit(parallel=True, cache=True)
def lut_gather(
    seq: NDArray[np.uint8],
    lut: NDArray[np.int32],
    out: NDArray[np.int32],
) -> None:
    """Parallel LUT gather over a flat buffer: ``out[i] = lut[seq[i]]``.

    All three arrays must be 1-D and C-contiguous. Used by ``tokenize`` for
    inputs large enough to amortize thread-launch overhead.
    """
    for i in nb.prange(seq.shape[0]):  # type: ignore[not-iterable]
        out[i] = lut[seq[i]]


@nb.njit(cache=True)
def _pack_codon_index(b0: int, b1: int, b2: int) -> int:
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
    ["(u1[:], u1[:], u1, u1[:])"],
    "(k),(m),()->()",
    target="parallel",
    cache=True,
)
def gufunc_translate_lut(
    seq_kmers: NDArray[np.uint8],
    codon_lut: NDArray[np.uint8],
    marker_byte: np.uint8,
    res: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:  # type: ignore
    """Translate a 3-codon to its amino acid via an O(1) lookup table.

    Selected automatically by ``AminoAlphabet.translate`` for the standard
    genetic code (k=3, ACGT); non-standard alphabets use
    :func:`gufunc_translate` instead.

    The ``(byte >> 1) & 3`` hash is **not** a bijection outside ``{A, C, G, T}``:
    e.g. ``N`` (0x4E) and ``NUL`` (0x00) both collide onto valid LUT slots and
    would silently yield biologically wrong AAs (``NNN -> T``, ``\\x00\\x00\\x00 ->
    K``). Every codon byte is range-checked against ``{A, C, G, T}`` before the
    LUT lookup; any non-canonical byte short-circuits to the caller-supplied
    ``marker_byte`` sentinel.

    Case-insensitivity: each input byte is upper-cased via ``b & 0xDF`` before
    the range check and the LUT-index hash, so lowercase nucleotides (e.g.
    soft-masked ``acg``) translate identically to their uppercase forms.

    Parameters
    ----------
    seq_kmers
        A 3-codon as ASCII bytes (e.g. ``[65, 84, 71]`` = ``"ATG"``).
    codon_lut
        64-byte LUT, built by ``AminoAlphabet`` at construction time.
    marker_byte
        ASCII byte emitted when any codon byte is non-canonical (i.e. not in
        ``{A, C, G, T, a, c, g, t}``). The Python wrapper validates this is a
        single byte.
    res
        Output buffer.
    """
    b0 = seq_kmers[0] & 0xDF
    b1 = seq_kmers[1] & 0xDF
    b2 = seq_kmers[2] & 0xDF
    if (
        (b0 == 65 or b0 == 67 or b0 == 71 or b0 == 84)
        and (b1 == 65 or b1 == 67 or b1 == 71 or b1 == 84)
        and (b2 == 65 or b2 == 67 or b2 == 71 or b2 == 84)
    ):
        res[0] = codon_lut[_pack_codon_index(b0, b1, b2)]  # type: ignore[unsupported-operation]
    else:
        res[0] = marker_byte  # type: ignore[unsupported-operation]


@nb.njit(cache=True)
def _nb_drop_unknown_codons(
    translated: NDArray[np.uint8],
    codons: NDArray[np.uint8],
    offsets: NDArray[np.int64],
    valid_upper: NDArray[np.uint8],
):
    """Compact a flat translated AA buffer, dropping non-canonical codons.

    Single-pass per-sequence stream compaction for ``translate(unknown="drop")``.
    A codon is dropped iff any of its bytes — after upper-casing via ``& 0xDF``
    — is not in ``valid_upper``. Offsets are codon-indexed into ``translated``.

    Parameters
    ----------
    translated
        (num_codons,) uint8 AA bytes (S1 viewed as u1).
    codons
        (num_codons, k) uint8 input codon bytes.
    offsets
        (n+1,) int64 codon-indexed offsets into ``translated``.
    valid_upper
        (v,) uint8 upper-cased valid nucleotide bytes (e.g. ord("ACGT")).

    Returns
    -------
    (out, new_offsets)
        ``out`` is (num_kept,) uint8; ``new_offsets`` is (n+1,) int64, monotonic.
    """
    n = len(offsets) - 1
    num_codons = translated.shape[0]
    k = codons.shape[1]
    v = len(valid_upper)
    out = np.empty(num_codons, dtype=np.uint8)
    new_offsets = np.empty(n + 1, dtype=np.int64)
    new_offsets[0] = 0
    w = 0
    for s in range(n):
        start = offsets[s]
        end = offsets[s + 1]
        for c in range(start, end):
            keep = True
            for j in range(k):
                b = codons[c, j] & 0xDF
                ok = False
                for t in range(v):
                    if b == valid_upper[t]:
                        ok = True
                        break
                if not ok:
                    keep = False
                    break
            if keep:
                out[w] = translated[c]
                w += 1
        new_offsets[s + 1] = w
    return out[:w].copy(), new_offsets


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
    for i in nb.prange(n):  # type: ignore[not-iterable]
        end = full_ends[i]
        for j in range(starts[i], full_ends[i]):
            if data[j] == stop_char:
                end = j + 1
                break
        ends[i] = end
    return ends
