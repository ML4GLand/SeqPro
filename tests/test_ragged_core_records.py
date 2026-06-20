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


# ---------------------------------------------------------------------------
# Task 5: Ragged.from_fields + _is_record + rag.zip
# ---------------------------------------------------------------------------
from seqpro.rag._core import Ragged  # noqa: E402


def _record_ragged():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    return Ragged.from_fields({"a": a, "b": b})


def test_from_fields_builds_record():
    rag = _record_ragged()
    assert rag._is_record is True
    # .fields and field access land in Tasks 6/7; verify via layout internals instead
    assert list(rag._layout.fields) == ["a", "b"]


def test_from_fields_canonicalizes_shared_offsets():
    rag = _record_ragged()
    # All fields must share the SAME offsets object (identity, not just equality)
    assert rag._layout.fields["a"].offsets[0] is rag._layout.fields["b"].offsets[0]


def test_from_fields_rejects_empty():
    with pytest.raises(ValueError, match="empty|at least one"):
        Ragged.from_fields({})


def test_from_fields_rejects_offset_mismatch():
    a = Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32)
    )
    b = Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([3, 1, 2], np.uint32)
    )
    with pytest.raises(ValueError, match="offset|equal"):
        Ragged.from_fields({"a": a, "b": b})


def test_from_fields_rejects_opaque_field():
    lens = np.array([3, 3], dtype=np.uint32)
    s = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), lens)  # opaque
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    with pytest.raises(NotImplementedError, match="Spec C|opaque|chars"):
        Ragged.from_fields({"s": s, "n": n})


def test_zip_alias():
    import seqpro.rag as rag_mod

    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    rec = rag_mod.zip({"a": a, "b": b})
    assert rec._is_record is True


# ---------------------------------------------------------------------------
# Task 6: record-branch properties (data/dtype/offsets/shape/fields/state)
# ---------------------------------------------------------------------------


def test_record_data_dict_zero_copy():
    rag = _record_ragged()
    d = rag.data
    assert list(d.keys()) == ["a", "b"]
    np.testing.assert_array_equal(d["a"], np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(d["b"], np.arange(6, dtype=np.float64))
    # Zero-copy: the returned array IS the same object stored in the layout (no copy).
    # from_fields stores the original buffer directly (base is None on an owned array),
    # so we check object identity rather than .base is not None.
    # See task-6-report.md for the concern about the brief's .base assertion.
    assert d["a"] is rag._layout.fields["a"].data


def test_record_dtype_structured():
    rag = _record_ragged()
    assert rag.dtype == np.dtype([("a", np.int32), ("b", np.float64)])


def test_record_offsets_shape_fields_lengths():
    rag = _record_ragged()
    np.testing.assert_array_equal(rag.offsets, np.array([0, 2, 3, 6]))
    assert rag.shape == (3, None)
    assert rag.fields == ["a", "b"]
    np.testing.assert_array_equal(rag.lengths, np.array([2, 1, 3]))


def test_record_state_predicates():
    rag = _record_ragged()
    assert rag.is_empty is False
    assert rag.is_contiguous is True
    assert rag.is_base is True
