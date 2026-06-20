import numpy as np
import pytest
from seqpro.rag._utils import lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, RecordLayout, validate_layout


def _two_field_record():
    shared = lengths_to_offsets(np.array([2, 1, 3], dtype=np.uint32))
    f0 = RaggedLayout(
        data=np.arange(6, dtype=np.int32), offsets=[shared], shape=(3, None)
    )
    f1 = RaggedLayout(
        data=np.arange(6, dtype=np.float64), offsets=[shared], shape=(3, None)
    )
    return RecordLayout(offsets=[shared], shape=(3, None), fields={"a": f0, "b": f1})


def test_record_layout_validates():
    validate_layout(_two_field_record())


def test_record_layout_rejects_empty_fields():
    with pytest.raises(ValueError, match="empty|at least one"):
        validate_layout(
            RecordLayout(offsets=[np.array([0])], shape=(0, None), fields={})
        )


def test_record_layout_rejects_unshared_offsets():
    a = lengths_to_offsets(np.array([2, 1, 3], dtype=np.uint32))
    b = a.copy()  # equal values, different object
    f0 = RaggedLayout(data=np.arange(6, dtype=np.int32), offsets=[a], shape=(3, None))
    f1 = RaggedLayout(data=np.arange(6, dtype=np.int32), offsets=[b], shape=(3, None))
    with pytest.raises(ValueError, match="shared|same offsets"):
        validate_layout(
            RecordLayout(offsets=[a], shape=(3, None), fields={"a": f0, "b": f1})
        )


def test_record_layout_rejects_opaque_field():
    shared = lengths_to_offsets(np.array([3, 3], dtype=np.uint32))
    opaque = RaggedLayout(
        data=np.frombuffer(b"catdog", "S1"), offsets=[], shape=(2,), str_offsets=shared
    )
    with pytest.raises(NotImplementedError, match="Spec C|opaque"):
        validate_layout(
            RecordLayout(offsets=[shared], shape=(2, None), fields={"s": opaque})
        )
