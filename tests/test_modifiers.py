from collections import defaultdict

import numpy as np
import seqpro as sp
from seqpro._modifiers import _align_axes, _slice_kmers
from seqpro._utils import check_axes


def test_align_axes():
    arr1 = np.random.randint(0, 5, (3, 2, 7))
    arr2 = np.random.randint(0, 5, (3, 2, 4, 7))
    arr3 = np.random.randint(0, 5, (3, 2, 5, 6, 7))

    aligned, destination_axes = _align_axes(arr1, arr2, arr3, axes=(0, 1, -1))

    assert aligned[0].shape == (3, 2, 7)
    assert aligned[1].shape == (4, 3, 2, 7)
    assert aligned[2].shape == (5, 6, 3, 2, 7)
    assert destination_axes == (-3, -2, -1)


def naive_slice_kmers(arr, starts, k):
    """Slice kmers from an array of sequences.

    Parameters
    ----------
    arr : np.ndarray
        Array of sequences.
    starts : np.ndarray
        Array of starting indices for each sequence.
    k : int
        Length of each kmer.

    Returns
    -------
    np.ndarray
        Array of kmers.
    """
    kmers = np.empty(arr.shape[:-1] + (k,), dtype=arr.dtype)
    for i, j in np.ndindex(starts.shape):
        kmers[..., i, j, :] = arr[..., i, j, starts[i, j] : starts[i, j] + k]
    return kmers


def test_slice_kmers():
    rng = np.random.default_rng(0)
    arr1 = sp.random_seqs((3, 2, 7), sp.DNA)
    arr2 = rng.integers(0, 5, (4, 3, 2, 7))
    arr3 = rng.uniform(0, 5, (5, 6, 3, 2, 7)).astype(np.float16)

    starts = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0],
        ]
    )

    actual_sliced_arr1 = _slice_kmers(arr1, starts, 5)
    actual_sliced_arr2 = _slice_kmers(arr2, starts, 5)
    actual_sliced_arr3 = _slice_kmers(arr3, starts, 5)

    desired_sliced_arr1 = naive_slice_kmers(arr1, starts, 5)
    desired_sliced_arr2 = naive_slice_kmers(arr2, starts, 5)
    desired_sliced_arr3 = naive_slice_kmers(arr3, starts, 5)

    np.testing.assert_equal(actual_sliced_arr1, desired_sliced_arr1)
    np.testing.assert_equal(actual_sliced_arr2, desired_sliced_arr2)
    np.testing.assert_equal(actual_sliced_arr3, desired_sliced_arr3)


def test_jitter():
    rng = np.random.default_rng(0)
    arr1 = sp.random_seqs((2, 1, 4), sp.DNA)
    arr2 = rng.integers(0, 5, (2, 1, 4, 4))
    arr3 = rng.uniform(0, 5, (2, 1, 5, 6, 4)).astype(np.float16)

    max_jitter = 1
    jittered_length = 2
    jitter_axes = (0, 1)
    length_axis = -1
    seed = 0

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, 3, (2, 1))
    print(starts)

    jittered = sp.jitter(
        arr1,
        arr2,
        arr3,
        max_jitter=max_jitter,
        length_axis=length_axis,
        jitter_axes=jitter_axes,
        seed=seed,
    )

    des_jit_arr1 = _slice_kmers(arr1, starts, jittered_length)
    des_jit_arr2 = _slice_kmers(
        np.moveaxis(arr2, (0, 1), (1, 2)), starts, jittered_length
    )
    des_jit_arr2 = np.moveaxis(des_jit_arr2, (1, 2), (0, 1))
    des_jit_arr3 = _slice_kmers(
        np.moveaxis(arr3, (0, 1), (2, 3)), starts, jittered_length
    )
    des_jit_arr3 = np.moveaxis(des_jit_arr3, (2, 3), (0, 1))
    np.testing.assert_array_equal(jittered[0], des_jit_arr1, "Failed array 1")
    np.testing.assert_array_equal(jittered[1], des_jit_arr2, "Failed array 2")
    np.testing.assert_array_equal(jittered[2], des_jit_arr3, "Failed array 3")


def _count_kmers(seqs, k, length_axis):
    check_axes(seqs, length_axis, False)

    seqs = sp.cast_seqs(seqs)

    if length_axis is None:
        length_axis = -1

    length = seqs.shape[length_axis]

    if k > length:
        raise ValueError("k is larger than sequence length")

    if seqs.dtype == np.uint8:
        raise NotImplementedError
    else:
        counts = defaultdict(int)
        kmers = np.lib.stride_tricks.sliding_window_view(seqs, k, -1)
        for kmer in kmers:
            counts[kmer.tobytes()] += 1
        return counts


def test_k_shuffle():
    seqs = sp.random_seqs(20, sp.DNA)
    length_axis = -1
    seed = 0

    for k in range(2, 5):
        counts = _count_kmers(seqs, k, length_axis)
        shuffled = sp.k_shuffle(seqs, k, length_axis=length_axis, seed=seed)
        shuffled_counts = _count_kmers(shuffled, k, length_axis)

        assert counts == shuffled_counts
