import awkward as ak
import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from seqpro.rag._core import Ragged
from seqpro.rag._layout import RaggedLayout, RecordLayout, validate_layout
from seqpro.rag._utils import lengths_to_offsets
from seqpro.rag._array import Ragged as AkRagged


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


# ---------------------------------------------------------------------------
# Task 7: record field access (key/attr) + __setitem__ mutation
# ---------------------------------------------------------------------------


def test_field_access_by_key_and_attr():
    rag = _record_ragged()
    np.testing.assert_array_equal(rag["a"].data, np.arange(6, dtype=np.int32))
    np.testing.assert_array_equal(rag.a.data, rag["a"].data)
    assert rag["a"].offsets is rag.offsets
    assert rag["a"].offsets is rag["b"].offsets


def test_field_access_unknown_raises():
    rag = _record_ragged()
    with pytest.raises(KeyError):
        rag["nope"]


def test_setitem_replace_field():
    rag = _record_ragged()
    rag["a"] = rag["a"].view(np.uint32)
    assert rag["a"].dtype == np.dtype(np.uint32)
    assert rag["a"].offsets is rag.offsets


def test_setitem_add_field():
    rag = _record_ragged()
    new = Ragged.from_offsets(np.arange(6, dtype=np.int16), (3, None), rag.offsets)
    rag["c"] = new
    assert rag.fields == ["a", "b", "c"]
    assert rag["c"].offsets is rag.offsets


def test_setitem_offset_mismatch_raises():
    rag = _record_ragged()
    bad = Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([3, 1, 2], np.uint32)
    )
    with pytest.raises(ValueError, match="offset|equal"):
        rag["d"] = bad


def test_getattr_unknown_raises_attribute_error():
    rag = _record_ragged()
    with pytest.raises(AttributeError):
        _ = rag.nonexistent


def test_getattr_single_level_unknown_raises_attribute_error():
    rag = Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32)
    )
    with pytest.raises(AttributeError):
        _ = rag.someunknownattr


# ---------------------------------------------------------------------------
# Task 8: record row-axis indexing (slice/mask -> record, int -> dict)
# ---------------------------------------------------------------------------


def test_record_row_slice_returns_record():
    rag = _record_ragged()
    sub = rag[1:3]
    assert sub._is_record is True
    np.testing.assert_array_equal(sub["a"][0], np.array([2]))  # row 1 of a
    np.testing.assert_array_equal(sub["b"][1], np.array([3.0, 4.0, 5.0]))  # row 2 of b
    assert sub["a"].offsets is sub["b"].offsets  # shared gather


def test_record_row_mask_returns_record():
    rag = _record_ragged()
    sub = rag[np.array([True, False, True])]
    np.testing.assert_array_equal(sub["a"][0], np.array([0, 1]))
    np.testing.assert_array_equal(sub["a"][1], np.array([3, 4, 5]))


def test_record_row_int_returns_dict():
    rag = _record_ragged()
    row = rag[0]
    assert set(row.keys()) == {"a", "b"}
    np.testing.assert_array_equal(row["a"], np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(row["b"], np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Task 9: squeeze / reshape on record Ragged (per-field)
# ---------------------------------------------------------------------------


def _record_with_leading_two():
    lens = np.array([2, 1, 3, 1, 2, 1], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(10, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(10, dtype=np.float64), lens)
    return Ragged.from_fields({"a": a, "b": b})


def test_record_reshape_leading():
    rag = _record_with_leading_two()
    re = rag.reshape(2, 3, None)
    assert re._is_record is True
    assert re.shape == (2, 3, None)
    assert re["a"].offsets is re["b"].offsets


def test_record_squeeze_trailing_one():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    a = Ragged.from_offsets(
        np.arange(6, dtype=np.int64).reshape(6, 1),
        (3, None, 1),
        lengths_to_offsets(lens),
    )
    b = Ragged.from_offsets(
        np.arange(6, dtype=np.float64).reshape(6, 1),
        (3, None, 1),
        lengths_to_offsets(lens),
    )
    rag = Ragged.from_fields({"a": a, "b": b})
    sq = rag.squeeze()
    assert sq.shape == (3, None)
    assert sq["a"].offsets is sq["b"].offsets


# ---------------------------------------------------------------------------
# Task 10: record-aware to_packed
# ---------------------------------------------------------------------------


def test_record_to_packed_after_slice():
    rag = _record_ragged()[::-1]  # gather -> (2, M) offsets
    packed = rag.to_packed()
    assert packed._is_record is True
    assert packed.is_base is True
    assert packed["a"].offsets is packed["b"].offsets
    np.testing.assert_array_equal(
        packed["a"].data, np.array([3, 4, 5, 2, 0, 1], dtype=np.int32)
    )
    np.testing.assert_array_equal(packed.offsets, np.array([0, 3, 4, 6]))


def test_record_to_packed_copy_false_passthrough():
    rag = _record_ragged()
    assert rag.to_packed(copy=False) is rag


def test_record_to_packed_copy_false_unpacked_raises():
    rag = _record_ragged()[::-1]
    with pytest.raises(ValueError, match="already-packed"):
        rag.to_packed(copy=False)


# ---------------------------------------------------------------------------
# Task 11: to_numpy / to_padded (per-field dict) + view / ufunc raise on records
# ---------------------------------------------------------------------------


def test_record_to_numpy_dict():
    lens = np.array([3, 3], dtype=np.uint32)
    a = Ragged.from_lengths(np.arange(6, dtype=np.int32), lens)
    b = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    rag = Ragged.from_fields({"a": a, "b": b})
    out = rag.to_numpy()
    np.testing.assert_array_equal(out["a"], np.arange(6, dtype=np.int32).reshape(2, 3))
    np.testing.assert_array_equal(
        out["b"], np.arange(6, dtype=np.float64).reshape(2, 3)
    )


def test_record_to_padded_dict():
    rag = _record_ragged()
    out = rag.to_padded(-1)
    assert set(out.keys()) == {"a", "b"}
    np.testing.assert_array_equal(out["a"][1], np.array([2, -1, -1], dtype=np.int32))


def test_record_view_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError):
        rag.view(np.uint32)


def test_record_ufunc_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError):
        rag + 1


def test_record_array_raises():
    rag = _record_ragged()
    with pytest.raises(TypeError, match="dense array"):
        np.asarray(rag)


# ---------------------------------------------------------------------------
# Task 12: _ingest bridge — record layout_from_ak + to_ak
# ---------------------------------------------------------------------------


def test_ingest_record_from_ak():
    arr = ak.Array(
        {"a": [[1, 2], [3], [4, 5, 6]], "b": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]}
    )
    rag = Ragged(arr)
    assert rag._is_record is True
    assert rag.fields == ["a", "b"]
    np.testing.assert_array_equal(rag["a"].data, np.array([1, 2, 3, 4, 5, 6]))
    assert rag["a"].offsets is rag["b"].offsets


def test_record_to_ak_roundtrips():
    rag = _record_ragged()
    out = rag.to_ak()
    assert set(ak.fields(out)) == {"a", "b"}
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(out["a"])), rag["a"].data)


# ---------------------------------------------------------------------------
# Task 13: Differential Hypothesis suite + char-record alignment
# ---------------------------------------------------------------------------


@st.composite
def _record_inputs(draw):
    n = draw(st.integers(1, 6))
    lengths = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n).map(np.array))
    total = int(lengths.sum())
    a = draw(arrays(np.int64, (total,), elements=st.integers(-50, 50)))
    b = draw(arrays(np.float64, (total,), elements=st.floats(-50, 50, allow_nan=False)))
    return lengths.astype(np.uint32), a, b


@given(_record_inputs())
def test_diff_record_properties(inp):
    lengths, a, b = inp
    new = Ragged.from_fields(
        {"a": Ragged.from_lengths(a, lengths), "b": Ragged.from_lengths(b, lengths)}
    )
    old = AkRagged(
        ak.zip({"a": ak.unflatten(a, lengths), "b": ak.unflatten(b, lengths)})
    )
    np.testing.assert_array_equal(new.offsets, old.offsets)
    assert new.shape == old.shape
    assert new.fields == list(ak.fields(old))
    np.testing.assert_array_equal(new["a"].data, old["a"].data)
    np.testing.assert_array_equal(new["b"].data, old["b"].data)


@given(_record_inputs())
def test_diff_record_to_packed_after_slice(inp):
    lengths, a, b = inp
    if len(lengths) < 2:
        return
    new = Ragged.from_fields(
        {"a": Ragged.from_lengths(a, lengths), "b": Ragged.from_lengths(b, lengths)}
    )[::-1].to_packed()
    old = AkRagged(
        ak.zip({"a": ak.unflatten(a, lengths), "b": ak.unflatten(b, lengths)})
    )[::-1].to_packed()
    np.testing.assert_array_equal(new["a"].data, old["a"].data)
    np.testing.assert_array_equal(new.offsets, old.offsets)


def test_char_field_record_aligns_on_length():
    # annotated-haplotypes shape: chars + per-base numeric, one shared offsets
    lens = np.array([3, 2], dtype=np.uint32)
    hap = Ragged.from_lengths(np.frombuffer(b"ATGCG", "S1"), lens).to_chars()
    annot = Ragged.from_lengths(np.arange(5, dtype=np.float32), lens)
    rec = Ragged.from_fields({"hap": hap, "annot": annot})
    assert rec.dtype == np.dtype([("hap", "S1"), ("annot", np.float32)])
    assert rec["hap"].offsets is rec["annot"].offsets
    np.testing.assert_array_equal(rec["hap"][0], np.frombuffer(b"ATG", "S1"))


# ---------------------------------------------------------------------------
# Task 13: Port legacy record tests (TestRecordRagged + TestToPackedRecord)
# ---------------------------------------------------------------------------


def _legacy_record():
    """Mirrors test_ragged.py::TestRecordRagged fixture: two fields [2,1,3]."""
    lengths = np.array([2, 1, 3], dtype=np.uint32)
    field0_data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    field1_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
    f0 = Ragged.from_lengths(field0_data, lengths)
    f1 = Ragged.from_lengths(field1_data, lengths)
    return Ragged.from_fields({"field0": f0, "field1": f1})


def test_legacy_record_offsets():
    rag = _legacy_record()
    expected = np.array([0, 2, 3, 6], dtype=np.uint32)
    np.testing.assert_array_equal(rag.offsets, expected)


def test_legacy_record_data_dict():
    rag = _legacy_record()
    d = rag.data
    assert isinstance(d, dict)
    assert list(d.keys()) == ["field0", "field1"]
    np.testing.assert_array_equal(d["field0"], np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(d["field1"], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_legacy_record_field_access_by_key():
    rag = _legacy_record()
    np.testing.assert_array_equal(rag["field0"].data, np.array([1, 2, 3, 4, 5, 6]))
    np.testing.assert_array_equal(
        rag["field1"].data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    )


def test_legacy_record_field_access_by_attr():
    rag = _legacy_record()
    np.testing.assert_array_equal(rag.field0.data, rag["field0"].data)
    np.testing.assert_array_equal(rag.field1.data, rag["field1"].data)


def test_legacy_record_offsets_shared_with_field():
    rag = _legacy_record()
    assert rag["field0"].offsets is rag.offsets


def test_legacy_record_offsets_shared_across_fields():
    rag = _legacy_record()
    assert rag["field0"].offsets is rag["field1"].offsets


def test_legacy_record_field_returns_ragged():
    rag = _legacy_record()
    assert isinstance(rag["field0"], Ragged)
    assert isinstance(rag["field1"], Ragged)


def test_legacy_record_dtype_structured():
    rag = _legacy_record()
    dt = rag.dtype
    assert isinstance(dt, np.dtype)
    assert dt.names == ("field0", "field1")
    assert dt["field0"] == np.dtype(np.int64)
    assert dt["field1"] == np.dtype(np.float64)


def test_legacy_record_dtype_field_order_preserved():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), lens)
    r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
    rag = Ragged.from_fields({"zeta": r1, "alpha": r2})
    assert list(rag.dtype.names) == ["zeta", "alpha"]


def test_legacy_record_field_setitem_roundtrip():
    rag = _legacy_record()
    original = rag["field0"].data.copy()
    rag["field0"] = rag["field0"].view(np.uint64)
    np.testing.assert_array_equal(rag["field0"].data.view(np.int64), original)


def test_legacy_record_squeeze():
    lens = np.array([2, 1, 3], dtype=np.uint32)
    f0 = Ragged.from_lengths(np.arange(6, dtype=np.int64).reshape(6, 1), lens)
    f1 = Ragged.from_lengths(np.arange(6, dtype=np.float64).reshape(6, 1), lens)
    rag = Ragged.from_fields({"a": f0, "b": f1})
    sq = rag.squeeze()
    assert isinstance(sq, Ragged)
    assert sq.shape == (3, None)
    np.testing.assert_array_equal(sq["a"].data, np.arange(6))
    assert sq["a"].offsets is sq.offsets


def test_legacy_record_reshape():
    lengths = np.array([2, 1, 3, 1, 2, 1], dtype=np.uint32)
    data_a = np.arange(10, dtype=np.int64)
    data_b = np.arange(10, dtype=np.float64)
    f0 = Ragged.from_lengths(data_a, lengths)
    f1 = Ragged.from_lengths(data_b, lengths)
    rag = Ragged.from_fields({"a": f0, "b": f1})
    re = rag.reshape(2, 3, None)
    assert isinstance(re, Ragged)
    assert re.shape == (2, 3, None)
    assert re["a"].offsets is re.offsets
    np.testing.assert_array_equal(re["a"].data, data_a)


# Ported from test_rag_to_packed.py::TestToPackedRecord


def _to_packed_record():
    """Mirrors test_rag_to_packed.py::TestToPackedRecord._record."""
    lengths = np.array([3, 2, 4], dtype=np.uint32)
    scores = np.arange(9, dtype=np.float64)
    flags = np.arange(9, dtype=np.int8)
    return Ragged.from_fields(
        {
            "score": Ragged.from_lengths(scores, lengths),
            "flag": Ragged.from_lengths(flags, lengths),
        }
    )


def test_legacy_record_to_packed_all_fields():
    rec = _to_packed_record()[::-1]  # reorder -> unpacked
    out = rec.to_packed()
    assert out.offsets.ndim == 1
    assert out.offsets[0] == 0
    # fields share one offsets object (zero-copy SoA contract)
    assert out["score"].offsets is out["flag"].offsets
    # reversed data: lengths [3,2,4] reversed -> [4,2,3]
    np.testing.assert_array_equal(out.offsets, np.array([0, 4, 6, 9]))
    np.testing.assert_array_equal(
        out["score"].data, np.array([5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 0.0, 1.0, 2.0])
    )
    np.testing.assert_array_equal(
        out["flag"].data, np.array([5, 6, 7, 8, 3, 4, 0, 1, 2], dtype=np.int8)
    )


def test_legacy_record_to_packed_shared_offsets():
    rec = _to_packed_record()[::-1]
    out = rec.to_packed()
    assert out["score"].offsets is out["flag"].offsets


def test_legacy_record_to_packed_copy_false_passthrough():
    rec = _to_packed_record()
    out = rec.to_packed(copy=False)
    assert out is rec


# ---------------------------------------------------------------------------
# -O-safe record guards on string-duality methods (is_string / to_chars / to_strings)
# ---------------------------------------------------------------------------


def test_record_is_string_is_false():
    rag = _record_ragged()
    assert rag.is_string is False


def test_record_to_chars_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError, match="to_chars"):
        rag.to_chars()


def test_record_to_strings_raises():
    rag = _record_ragged()
    with pytest.raises(NotImplementedError, match="to_strings"):
        rag.to_strings()
