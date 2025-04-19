import numpy as np
from pytest_cases import parametrize_with_cases
from seqpro._ragged import OFFSET_TYPE, Ragged, lengths_to_offsets


def case_int32():
    return Ragged.from_lengths(
        np.arange(10, dtype=np.int32), np.array([3, 2, 5], dtype=np.uint32)
    )


def case_s1():
    return Ragged.from_lengths(
        np.array([b"cathithere"]).view("S1"), np.array([3, 2, 5], dtype=np.uint32)
    )


def case_nested():
    return Ragged.from_lengths(
        np.arange(10),
        np.array(
            [
                [[1], [3]],
                [[2], [1]],
                [[1], [2]],
            ],
            dtype=np.uint32,
        ),
    )


@parametrize_with_cases("rag", cases=".")
def test_roundtrip(rag: Ragged):
    actual = Ragged.from_awkward(rag.to_awkward())
    np.testing.assert_equal(actual.data, rag.data)
    np.testing.assert_equal(actual.lengths, rag.lengths)
    np.testing.assert_equal(actual.offsets, rag.offsets)


def l2o_1d():
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    desired = np.array([0, 3, 5, 10], dtype=OFFSET_TYPE)
    return lengths, desired


def l2o_2d():
    lengths = np.array([[3, 2], [5, 4]], dtype=np.uint32)
    desired = np.array([0, 3, 5, 10, 14], dtype=OFFSET_TYPE)
    return lengths, desired


@parametrize_with_cases("lengths, desired", cases=".", prefix="l2o_")
def test_len_to_offsets(lengths, desired):
    actual = lengths_to_offsets(lengths)
    np.testing.assert_equal(actual, desired)
