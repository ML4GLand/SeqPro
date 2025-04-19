import numpy as np
from pytest_cases import parametrize_with_cases
from seqpro._ragged import OFFSET_TYPE
from seqpro._utils import lengths_to_offsets


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
