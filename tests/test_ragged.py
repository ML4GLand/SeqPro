import awkward as ak
import numpy as np
from pytest_cases import parametrize_with_cases
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets


def case_int32():
    data = np.arange(10, dtype=np.int32)
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    offsets = lengths_to_offsets(lengths)
    shape = (3, None)
    content = Ragged.from_lengths(data, lengths)
    return content, data, shape, offsets


def case_s1():
    data = np.array([b"cathithere"]).view("S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    offsets = lengths_to_offsets(lengths)
    shape = (3, None)
    content = Ragged.from_lengths(data, lengths)
    return content, data, shape, offsets


def case_nested():
    data = np.arange(10)
    lengths = np.array(
        [
            [[1], [3]],
            [[2], [1]],
            [[1], [2]],
        ]
    )
    offsets = lengths_to_offsets(lengths)
    shape = (3, 2, 1, None)
    content = Ragged.from_lengths(data, lengths)
    return content, data, shape, offsets


def case_ak_params():
    content = ak.Array([[1, 2, 3], [4, 5]])
    content = ak.with_parameter(content, "__list__", None)
    data = np.arange(1, 6)
    shape = (2, None)
    offsets = np.array([0, 3, 5], dtype=OFFSET_TYPE)
    return content, data, shape, offsets


@parametrize_with_cases("content, data, shape, offsets", cases=".")
def test_init(content, data, shape, offsets):
    rag = Ragged(content)
    np.testing.assert_equal(rag.data, data)
    np.testing.assert_equal(rag.shape, shape)
    np.testing.assert_equal(rag.offsets, offsets)


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
