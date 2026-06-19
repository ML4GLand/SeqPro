import numpy as np
import pytest
from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, validate_layout


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
