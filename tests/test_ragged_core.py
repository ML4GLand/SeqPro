import numpy as np
import pytest
from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, validate_layout
from seqpro.rag._core import Ragged


def test_layout_numeric_basic():
    data = np.arange(10, dtype=np.int32)
    offsets = lengths_to_offsets(np.array([3, 2, 5], dtype=np.uint32))
    layout = RaggedLayout(data=data, offsets=[offsets], shape=(3, None))
    validate_layout(layout)
    assert layout.is_string is False
    assert layout.n_ragged == 1


def test_layout_string_flat_collection():
    # flat collection of sequences -> no ragged axis, just a string leaf
    data = np.frombuffer(b"cathithere", dtype="S1")
    str_offsets = np.array([0, 3, 5, 10], dtype=OFFSET_TYPE)
    layout = RaggedLayout(data=data, offsets=[], shape=(3,), str_offsets=str_offsets)
    validate_layout(layout)
    assert layout.is_string is True
    assert layout.n_ragged == 0


def test_layout_rejects_multiple_none():
    data = np.arange(6)
    with pytest.raises(NotImplementedError, match="Spec C"):
        validate_layout(
            RaggedLayout(data=data, offsets=[np.array([0, 6])], shape=(2, None, None))
        )


def test_layout_rejects_nonmonotonic_offsets():
    with pytest.raises(ValueError, match="monotonic"):
        validate_layout(
            RaggedLayout(
                data=np.arange(5),
                offsets=[np.array([0, 3, 2, 5], dtype=OFFSET_TYPE)],
                shape=(3, None),
            )
        )


def test_layout_rejects_segment_count_mismatch():
    with pytest.raises(ValueError, match="segment"):
        validate_layout(
            RaggedLayout(
                data=np.arange(10),
                offsets=[lengths_to_offsets(np.array([3, 2, 5]))],  # 3 segments
                shape=(4, None),  # claims 4
            )
        )


def test_from_lengths_numeric():
    data = np.arange(10, dtype=np.int32)
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3, None)
    assert rag.dtype == np.dtype(np.int32)
    np.testing.assert_array_equal(rag.data, data)
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5, 10]))
    np.testing.assert_array_equal(rag.lengths, np.array([3, 2, 5]))
    assert rag.rag_dim == 1


def test_from_lengths_nested_leading_dims():
    # case_nested from the legacy suite: leading (3,2,1), one ragged axis
    data = np.arange(10)
    lengths = np.array([[[1], [3]], [[2], [1]], [[1], [2]]])
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3, 2, 1, None)
    assert rag.rag_dim == 3
    np.testing.assert_array_equal(rag.offsets, lengths_to_offsets(lengths))


def test_from_lengths_string_collapses_to_leaf():
    # NEW string-leaf behavior: flat collection -> (N,), not (N, None)
    data = np.frombuffer(b"cathithere", dtype="S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    rag = Ragged.from_lengths(data, lengths)
    assert rag.shape == (3,)
    assert rag.dtype == np.dtype("S1")
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5, 10]))


def test_from_offsets_numeric_trailing_dim():
    data = np.zeros((6, 4), dtype=np.int32)
    rag = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6]))
    assert rag.shape == (2, None, 4)
    assert rag.data.shape == (6, 4)


def test_empty():
    rag = Ragged.empty((3, None), np.float64)
    assert rag.shape == (3, None)
    assert rag.data.size == 0
    np.testing.assert_array_equal(rag.offsets, np.zeros(4, dtype=np.int64))


def test_from_offsets_rejects_two_none():
    with pytest.raises(NotImplementedError, match="Spec C"):
        Ragged.from_offsets(np.arange(6), (2, None, None), np.array([0, 6]))


def test_state_predicates():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    assert rag.is_empty is False
    assert rag.is_contiguous is True
    assert rag.is_base is True
    empty = Ragged.empty((3, None), np.int32)
    assert empty.is_empty is True


def test_view_reinterprets_dtype_zero_copy():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
    v = rag.view(np.uint64)
    assert v.dtype == np.dtype(np.uint64)
    assert v.data.base is not None  # zero-copy view
    np.testing.assert_array_equal(v.data.view(np.int64), rag.data)
    assert v.offsets is rag.offsets  # offsets reused
