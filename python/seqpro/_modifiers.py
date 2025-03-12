from enum import Enum
from typing import List, Literal, Optional, Tuple, Union, cast

import numpy as np
from numpy.typing import NDArray

from ._utils import SeqType, cast_seqs, check_axes
from .alphabets._alphabets import NucleotideAlphabet
from .seqpro import _k_shuffle


def reverse_complement(
    seqs: SeqType,
    alphabet: NucleotideAlphabet,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
):
    """Reverse complement a sequence.

    Parameters
    ----------
    seqs : str, list[str], ndarray[str, bytes, uint8]
    alphabet : NucleotideAlphabet
    length_axis : Optional[int], optional
        Needed for array input. Length axis, by default None
    ohe_axis : Optional[int], optional
        Needed for OHE input. One hot encoding axis, by default None

    Returns
    -------
    ndarray[bytes, uint8]
        Array of bytes (S1) or uint8 for string or OHE input, respectively.
    """
    return alphabet.reverse_complement(seqs, length_axis, ohe_axis)


def k_shuffle(
    seqs: SeqType,
    k: int,
    *,
    length_axis: Optional[int] = None,
    ohe_axis: Optional[int] = None,
    seed: Optional[Union[int, np.random.Generator]] = None,
    alphabet: Optional[NucleotideAlphabet] = None,
) -> NDArray[Union[np.bytes_, np.uint8]]:
    """Shuffle sequences while preserving k-let frequencies.

    Parameters
    ----------
    seqs : SeqType
    k : int
        Size of k-lets to preserve frequencies of.
    length_axis : Optional[int], optional
        Needed for array input. Axis that corresponds to the length of sequences.
    ohe_axes : Optional[int], optional
        Needed for OHE input. Axis that corresponds to the one hot encoding, should be
        the same size as the length of the alphabet.
    seed : int, np.random.Generator, optional
        Seed or generator for shuffling.
    alphabet : Optional[NucleotideAlphabet], optional
        Alphabet, needed for OHE sequence input.
    """

    check_axes(seqs, length_axis, ohe_axis)

    if isinstance(seed, np.random.Generator):
        seed = seed.integers(0, np.iinfo(np.int32).max)

    seqs = cast_seqs(seqs)

    # only get here if seqs was str or list[str]
    if length_axis is None:
        length_axis = seqs.ndim - 1

    if seqs.dtype == np.uint8:
        assert ohe_axis is not None
        seqs = cast(NDArray[np.uint8], seqs)
        ohe = True
        if alphabet is None:
            raise ValueError("Need an alphabet to process OHE sequences.")
        seqs = alphabet.decode_ohe(seqs, ohe_axis=ohe_axis)
    else:
        ohe = False

    seqs = np.moveaxis(seqs, length_axis, -1)  # length must be final
    seqs = np.ascontiguousarray(seqs)  # must be contiguous

    shuffled = _k_shuffle(seqs.view("u1"), k, seed).view("S1")

    shuffled = np.moveaxis(shuffled, -1, length_axis)  # put length back where it was

    if ohe:
        assert ohe_axis is not None
        assert alphabet is not None
        shuffled = cast(NDArray[np.bytes_], shuffled)
        shuffled = alphabet.ohe(shuffled).swapaxes(-1, ohe_axis)

    return shuffled


def bin_coverage(
    coverage: NDArray[np.number],
    bin_width: int,
    length_axis: int,
    normalize=False,
) -> NDArray[np.number]:
    """Bin coverage by summing over non-overlapping windows.

    Parameters
    ----------
    coverage_array : NDArray
    bin_width : int
        Width of the windows to sum over. Must be an even divisor of the length
        of the coverage array. If not, raises an error.
    length_axis: int
    normalize : bool, default False
        Whether to normalize by the length of the bin.

    Returns
    -------
    binned_coverage : NDArray
    """
    length = coverage.shape[length_axis]
    if length % bin_width != 0:
        raise ValueError("Bin width must evenly divide length.")
    binned_coverage = np.add.reduceat(
        coverage, np.arange(0, length, bin_width), axis=length_axis
    )
    if normalize:
        binned_coverage /= bin_width
    return binned_coverage


def jitter(
    *arrays: NDArray,
    max_jitter: int,
    length_axis: int,
    jitter_axes: Union[int, Tuple[int, ...]],
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    """Randomly jitter data from arrays, using the same jitter across arrays.

    Parameters
    ----------
    *arrays : NDArray
        Arrays to be jittered. They must have the same sized jitter and length
        axes.
    max_jitter : int
        Maximum jitter amount.
    length_axis : int
    jitter_axes : Tuple[int, ...]
        Each slice along the jitter axes will be randomly jittered *independently*.
        Thus, if jitter_axes = 0, then every slice of data along axis 0 would be
        jittered independently. If jitter_axes = (0, 1), then each slice along axes 0
        and 1 would be randomly jittered independently.
    seed : int, np.random.Generator, optional
        Random seed or generator, by default None

    Returns
    -------
    arrays
        Jittered arrays. Each will have a new length equal to length - 2*max_jitter.

    Raises
    ------
    ValueError
        If any arrays have insufficient length to be jittered.
    """
    if isinstance(jitter_axes, int):
        jitter_axes = (jitter_axes,)

    # move jitter axes and length axis to back such that shape = (..., jitter, length)
    arrays, destination_axes = _align_axes(*arrays, axes=(*jitter_axes, length_axis))
    short_arrays = []
    for i, arr in enumerate(arrays):
        if arr.shape[-1] - 2 * max_jitter <= 0:
            short_arrays.append(i)
    if short_arrays:
        raise ValueError(
            f"Arrays {short_arrays} have insufficient length to be jittered with max_jitter={max_jitter}."
        )

    jitter_axes_shape = arrays[0].shape[-len(jitter_axes) - 1 : -1]
    if seed is None or isinstance(seed, int):
        rng = np.random.default_rng(seed)
    else:
        rng = seed
    starts = rng.integers(0, 2 * max_jitter + 1, jitter_axes_shape)

    sliced_arrs: List[NDArray] = []
    for arr in arrays:
        jittered_length = arr.shape[-1] - 2 * max_jitter
        sliced = _slice_kmers(arr, starts, jittered_length)
        sliced = np.moveaxis(sliced, destination_axes, [*jitter_axes, length_axis])
        sliced_arrs.append(sliced)

    return tuple(sliced_arrs)


def _align_axes(*arrays: NDArray, axes: Union[int, Tuple[int, ...]]):
    """Align axes of arrays, moving them to the back while preserving order.

    Parameters
    ----------
    *arrays : NDArray
        Arrays to align axes of.
    axes : Union[int, Tuple[int, ...]]
        Axes to align.

    Returns
    -------
    Tuple[NDArray]
        Aligned arrays.
    Tuple[int]
        Destination axes.

    Raises
    ------
    ValueError
        If axes cannot be aligned because they have different sizes.
    """
    if isinstance(axes, int):
        source_axes: Tuple[int, ...] = (axes,)
    else:
        source_axes = axes

    destination_axes = tuple(range(-len(source_axes), 0))
    arrays = tuple(np.moveaxis(a, source_axes, destination_axes) for a in arrays)

    first_axes_shape = arrays[0].shape[-len(destination_axes) : -1]
    for arr in arrays:
        if arr.shape[-len(destination_axes) : -1] != first_axes_shape:
            raise ValueError("Can't align axes with different sizes.")

    return arrays, destination_axes


def _slice_kmers(array: NDArray, starts: NDArray, k: int):
    """Get a view of an array sliced into k-mers, assuming starts align with the penultimate axes and length is the final axis.

    Parameters
    ----------
    array : NDArray
        Array to slice.
    starts : NDArray
        Start indices of k-mers.
    k : int
        Size of k-mers.

    Returns
    -------
    NDArray
        Sliced array.
    """
    n_axes_sliced = starts.ndim
    n_axes_not_sliced = array.ndim - n_axes_sliced - 1  # - 1 for length axis
    idx: List[Union[slice, NDArray]] = [slice(None)] * n_axes_not_sliced
    for i, size in enumerate(array.shape[-starts.ndim - 1 : -1]):
        shape = np.ones(starts.ndim, dtype=np.uint32)
        shape[i] = size
        idx.append(np.arange(size, dtype=np.intp).reshape(shape))
    idx.append(starts)

    windows = np.lib.stride_tricks.sliding_window_view(array, k, axis=-1)
    sliced = windows[tuple(idx)]
    return sliced


def random_seqs(
    shape: Union[int, Tuple[int, ...]],
    alphabet: NucleotideAlphabet,
    seed: Optional[Union[int, np.random.Generator]] = None,
):
    """Generate random nucleotide sequences.

    Parameters
    ----------
    shape : int, tuple[int]
        Shape of sequences to generate
    alphabet : NucleotideAlphabet
        Alphabet to sample nucleotides from.
    seed : int, np.random.Generator, optional
        Random seed or generator.

    Returns
    -------
    ndarray
        Randomly generated sequences.
    """
    if isinstance(seed, int) or seed is None:
        seed = np.random.default_rng(seed)
    return seed.choice(alphabet.array, size=shape)


class NormalizationMethod(str, Enum):
    CPM = "CPM"
    CPKM = "CPKM"


def normalize_coverage(
    coverage: NDArray,
    method: Literal["CPM", "CPKM"],
    total_counts: Union[int, NDArray],
    length_axis: int,
):
    """Normalize an array of coverage. Note that whether this corresponds to the
    conventional definitions of CPM, RPKM, and FPKM depends on how the underlying
    coverage was quantified. If the coverage is for reads, then CPKM = RPKM. If it is
    for fragments, then CPKM = FPKM.

    Parameters
    ----------
    coverage : NDArray
        Array of coverage.
    method : Union[Literal['CPM', 'CPKM'], NormalizationMethod]
        Normalization method, either CPM (counts per million) or CPKM (counts per
        kilobase per million).
    total_counts : Union[int, NDArray]
        The total number of reads or fragments from the experiment.
    length_axis : int

    Returns
    -------
    NDArray
        Array of normalized coverage.
    """
    length = coverage.shape[length_axis]

    if method == "CPM":
        coverage = coverage * 1e6 / total_counts
    elif method == "CPKM":
        coverage = coverage * 1e6 / total_counts / length
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return coverage
