import awkward as ak
import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets
from seqpro.rag._layout import RaggedLayout, validate_layout
from seqpro.rag._core import Ragged
from seqpro.rag._array import Ragged as AkRagged


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
    with pytest.raises(NotImplementedError, match="R >= 3|3 or more"):
        validate_layout(
            RaggedLayout(
                data=data,
                offsets=[np.array([0, 1]), np.array([0, 2]), np.array([0, 6])],
                shape=(2, None, None, None),
            )
        )


def test_layout_nested_r2_valid():
    data = np.arange(10, dtype=np.int32)
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)  # 2 outer rows -> 2,1 middles
    o1 = np.array([0, 3, 5, 10], dtype=OFFSET_TYPE)  # 3 middles -> data
    layout = RaggedLayout(data=data, offsets=[o0, o1], shape=(2, None, None))
    validate_layout(layout)
    assert layout.n_ragged == 2


def test_layout_nested_rejects_r3():
    with pytest.raises(NotImplementedError, match="3 or more|R >= 3|three"):
        validate_layout(
            RaggedLayout(
                data=np.arange(6),
                offsets=[np.array([0, 1]), np.array([0, 2]), np.array([0, 6])],
                shape=(1, None, None, None),
            )
        )


def test_layout_nested_rejects_inner_segment_mismatch():
    with pytest.raises(ValueError, match="segment|middle"):
        validate_layout(
            RaggedLayout(
                data=np.arange(10),
                offsets=[
                    np.array([0, 2, 3], dtype=OFFSET_TYPE),
                    np.array([0, 3, 5], dtype=OFFSET_TYPE),
                ],  # only 2 middles, O0 needs 3
                shape=(2, None, None),
            )
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
    assert rag.dtype == np.dtype("S")  # opaque string descriptor (string/char duality)
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


def test_from_offsets_nested_list():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data,
        (2, None, None),
        [np.array([0, 2, 3]), np.array([0, 3, 5, 10])],
    )
    assert rag.shape == (2, None, None)
    assert len(rag._layout.offsets) == 2
    np.testing.assert_array_equal(rag._layout.offsets[0], np.array([0, 2, 3]))
    np.testing.assert_array_equal(rag._layout.offsets[1], np.array([0, 3, 5, 10]))
    assert rag.data is data  # zero-copy
    # full peel/index verified in Task 4/5


def test_from_offsets_rejects_three_none():
    with pytest.raises(NotImplementedError):
        Ragged.from_offsets(
            np.arange(6),
            (1, None, None, None),
            [np.array([0, 1]), np.array([0, 2]), np.array([0, 6])],
        )


def test_from_lengths_nested_tuple():
    data = np.arange(10, dtype=np.int32)
    outer = np.array([2, 1])  # row0 has 2 middles, row1 has 1
    inner = np.array([3, 2, 5])  # the 3 middles' leaf lengths
    rag = Ragged.from_lengths(data, (outer, inner))
    assert rag.shape == (2, None, None)
    np.testing.assert_array_equal(rag._layout.offsets[0], np.array([0, 2, 3]))
    np.testing.assert_array_equal(rag._layout.offsets[1], np.array([0, 3, 5, 10]))
    assert rag.data is data  # zero-copy
    # to_ak() parity verified in Task 15/16


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


def test_getitem_int_returns_row():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    np.testing.assert_array_equal(rag[0], np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(rag[1], np.array([3, 4], dtype=np.int32))


def test_getitem_int_string_leaf():
    rag = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    assert rag[0] == b"cat"
    assert rag[2] == b"there"


def test_getitem_slice_returns_ragged():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[1:3]
    assert isinstance(sub, Ragged)
    assert sub.offsets.ndim == 2  # (2, M) start/stop gather
    np.testing.assert_array_equal(sub[0], np.array([3, 4]))
    np.testing.assert_array_equal(sub[1], np.array([5, 6, 7, 8, 9]))


def test_getitem_mask_returns_ragged():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[np.array([True, False, True])]
    np.testing.assert_array_equal(sub[0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1], np.array([5, 6, 7, 8, 9]))


def test_ufunc_scalar_mul():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    out = rag * 2.0
    assert isinstance(out, Ragged)
    np.testing.assert_array_equal(out.data, np.arange(6) * 2.0)
    assert out.offsets is rag.offsets


def test_ufunc_unary():
    rag = Ragged.from_lengths(np.arange(1, 7, dtype=np.float64), np.array([2, 1, 3]))
    out = np.log1p(rag)
    np.testing.assert_allclose(out.data, np.log1p(np.arange(1, 7)))


def test_ufunc_two_ragged_shared_offsets():
    a = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    b = a.view(np.float64)
    out = a + b
    np.testing.assert_array_equal(out.data, a.data * 2)


def test_ufunc_reduce_raises():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
    with pytest.raises(NotImplementedError):
        np.add.reduce(rag)


def test_squeeze_trailing_one():
    data = np.arange(6, dtype=np.int64).reshape(6, 1)
    rag = Ragged.from_offsets(
        data, (3, None, 1), lengths_to_offsets(np.array([2, 1, 3]))
    )
    sq = rag.squeeze()
    assert sq.shape == (3, None)
    np.testing.assert_array_equal(sq.data, np.arange(6))


def test_reshape_leading():
    rag = Ragged.from_lengths(
        np.arange(10, dtype=np.int64), np.array([2, 1, 3, 1, 2, 1])
    )
    re = rag.reshape(2, 3, None)
    assert re.shape == (2, 3, None)
    np.testing.assert_array_equal(re.data, np.arange(10))
    assert re.offsets is rag.offsets


def test_to_packed_from_slice():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    packed = rag[1:3].to_packed()
    assert packed.is_base is True
    np.testing.assert_array_equal(packed.data, np.array([3, 4, 5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(packed.offsets, np.array([0, 2, 7]))


def test_to_padded():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    out = rag.to_padded(-1)
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out[1], np.array([3, 4, -1, -1, -1]))


def test_to_numpy_equal_lengths():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    np.testing.assert_array_equal(rag.to_numpy(), np.arange(6).reshape(2, 3))


def test_to_numpy_jagged_raises():
    rag = Ragged.from_lengths(np.arange(5, dtype=np.int32), np.array([3, 2]))
    with pytest.raises(ValueError):
        rag.to_numpy()


def test_ingest_from_ak_numeric():
    arr = ak.Array([[1, 2, 3], [4, 5]])
    rag = Ragged(arr)
    assert rag.shape == (2, None)
    np.testing.assert_array_equal(rag.data, np.array([1, 2, 3, 4, 5]))
    np.testing.assert_array_equal(rag.offsets, np.array([0, 3, 5]))


def test_to_ak_roundtrips_values():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
    np.testing.assert_array_equal(ak.to_numpy(ak.flatten(rag.to_ak())), rag.data)


def test_ingest_record_from_ak_works():
    # Task 12: record ingest is now implemented (Spec B landed); verify basic round-trip
    arr = ak.Array({"a": [[1, 2], [3]], "b": [[1.0, 2.0], [3.0]]})
    rag = Ragged(arr)
    assert rag._is_record is True
    assert rag.fields == ["a", "b"]
    np.testing.assert_array_equal(rag["a"].data, np.array([1, 2, 3]))
    assert rag["a"].offsets is rag["b"].offsets


# ---------------------------------------------------------------------------
# Hypothesis differential tests vs the awkward oracle
# ---------------------------------------------------------------------------


@st.composite
def _ragged_inputs(draw):
    n = draw(st.integers(1, 6))
    lengths = draw(st.lists(st.integers(0, 5), min_size=n, max_size=n).map(np.array))
    total = int(lengths.sum())
    data = draw(arrays(np.int64, (total,), elements=st.integers(-100, 100)))
    return data, lengths


@given(_ragged_inputs())
def test_diff_numeric_properties(inp):
    data, lengths = inp
    new = Ragged.from_lengths(data, lengths.astype(np.uint32))
    old = AkRagged.from_lengths(data, lengths.astype(np.uint32))
    np.testing.assert_array_equal(new.data, old.data)
    np.testing.assert_array_equal(new.offsets, old.offsets)
    assert new.shape == old.shape
    np.testing.assert_array_equal(new.lengths, old.lengths)


@given(_ragged_inputs())
def test_diff_to_packed_after_slice(inp):
    data, lengths = inp
    if len(lengths) < 2:
        return
    new = Ragged.from_lengths(data, lengths.astype(np.uint32))[::2].to_packed()
    old = AkRagged.from_lengths(data, lengths.astype(np.uint32))[::2].to_packed()
    np.testing.assert_array_equal(new.data, old.data)
    np.testing.assert_array_equal(new.offsets, old.offsets)


@given(_ragged_inputs())
def test_diff_ufunc(inp):
    data, lengths = inp
    new = Ragged.from_lengths(data.astype(np.float64), lengths.astype(np.uint32))
    old = AkRagged.from_lengths(data.astype(np.float64), lengths.astype(np.uint32))
    np.testing.assert_allclose((new + 1.0).data, ak_flat(old + 1.0))


def ak_flat(ak_rag):
    import awkward as ak

    return ak.to_numpy(ak.flatten(ak_rag, axis=None))


def test_diff_string_shape_documented_change():
    # The one intentional divergence: bytes collection (N, None) -> (N,)
    data = np.frombuffer(b"cathithere", "S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    new = Ragged.from_lengths(data, lengths)
    old = AkRagged.from_lengths(data, lengths)
    assert new.shape == (3,)
    assert old.shape == (3, None)
    np.testing.assert_array_equal(new.offsets, old.offsets)  # same byte offsets
    np.testing.assert_array_equal(new.data, old.data)


def test_getitem_uses_rust_select_intarray():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    sub = rag[np.array([2, 0])]
    np.testing.assert_array_equal(sub[0], np.array([5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(sub[1], np.array([0, 1, 2]))
    # negative index: last row then first row
    sub_neg = rag[np.array([-1, 0])]
    np.testing.assert_array_equal(sub_neg[0], np.array([5, 6, 7, 8, 9]))
    np.testing.assert_array_equal(sub_neg[1], np.array([0, 1, 2]))


def test_opaque_string_dtype_is_flexible_bytes():
    rag = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    assert rag.dtype == np.dtype("S")
    assert rag.dtype.itemsize == 0
    # storage is still S1 bytes
    assert rag.data.dtype == np.dtype("S1")


def test_getitem_oversized_bool_mask_raises():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    # length-4 mask on 3-row Ragged must raise IndexError, not panic
    with pytest.raises(IndexError):
        rag[np.array([True, False, True, True])]


def test_getitem_undersized_bool_mask_raises():
    rag = Ragged.from_lengths(np.arange(10, dtype=np.int32), np.array([3, 2, 5]))
    # length-2 mask on 3-row Ragged must raise IndexError
    with pytest.raises(IndexError):
        rag[np.array([True, False])]


def test_is_string_predicate():
    s = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), np.array([3, 3]))
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    assert s.is_string is True
    assert n.is_string is False


def test_from_offsets_S1_with_none_is_chars_not_opaque():
    data = np.frombuffer(b"cathithere", "S1")
    offsets = np.array([0, 3, 5, 10])
    chars = Ragged.from_offsets(data, (3, None), offsets)
    assert chars.is_string is False  # counted axis => chars
    assert chars.dtype == np.dtype("S1")
    assert chars.shape == (3, None)


def test_from_offsets_S1_without_none_is_opaque():
    data = np.frombuffer(b"cathithere", "S1")
    str_offsets = np.array([0, 3, 5, 10])
    opaque = Ragged.from_offsets(data, (3,), str_offsets)
    assert opaque.is_string is True
    assert opaque.dtype == np.dtype("S")
    assert opaque.shape == (3,)


def test_to_chars_zero_copy_and_shape():
    s = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    c = s.to_chars()
    assert c.dtype == np.dtype("S1")
    assert c.shape == (3, None)
    assert c.is_string is False
    assert c.data is s.data  # zero-copy buffer
    np.testing.assert_array_equal(c.offsets, s.offsets)  # str_offsets promoted
    np.testing.assert_array_equal(c[0], np.frombuffer(b"cat", "S1"))


def test_to_strings_roundtrip():
    s = Ragged.from_lengths(np.frombuffer(b"cathithere", "S1"), np.array([3, 2, 5]))
    back = s.to_chars().to_strings()
    assert back.dtype == np.dtype("S")
    assert back.shape == (3,)
    assert back[0] == b"cat"
    assert back.data is s.data


def test_to_chars_raises_on_non_opaque():
    n = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    with pytest.raises(ValueError, match="opaque"):
        n.to_chars()


def test_to_strings_raises_on_trailing_dims():
    data = np.zeros((6, 4), dtype="S1")
    chars_with_trailing = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6]))
    with pytest.raises(ValueError, match="1-D|trailing"):
        chars_with_trailing.to_strings()


# ---------------------------------------------------------------------------
# Task 3: string-under-axis leaf + nested to_chars / to_strings
# ---------------------------------------------------------------------------


def test_string_under_axis_build_and_dtype():
    data = np.frombuffer(b"ACGTAC", dtype="S1")
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)  # 2 rows: 2 + 1 variants
    str_off = np.array([0, 1, 3, 6], dtype=OFFSET_TYPE)  # 3 variants' byte lens
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    assert rag.is_string is True
    assert rag.dtype == np.dtype("S")
    assert rag.shape == (2, None)


def test_string_under_axis_to_chars_to_strings_roundtrip():
    data = np.frombuffer(b"ACGTAC", dtype="S1")
    o0 = np.array([0, 2, 3], dtype=OFFSET_TYPE)
    str_off = np.array([0, 1, 3, 6], dtype=OFFSET_TYPE)
    rag = Ragged.from_offsets(data, (2, None), o0, str_offsets=str_off)
    chars = rag.to_chars()
    assert chars.dtype == np.dtype("S1")
    assert chars.shape == (2, None, None)
    assert len(chars._layout.offsets) == 2
    assert (
        chars._layout.offsets[1] is str_off
    )  # zero-copy: str_offsets became inner level
    assert chars.data is data
    back = chars.to_strings()
    assert back.dtype == np.dtype("S")
    assert back.shape == (2, None)
    assert back._layout.offsets[0] is o0  # outer level preserved


# ---------------------------------------------------------------------------
# Task 4: R=2 outer-row indexing (lazy gather, peel to 1-level)
# ---------------------------------------------------------------------------


def test_r2_outer_slice_preserves_nesting():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (3, None, None), [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])]
    )
    sub = rag[1:3]  # outer rows 1,2
    assert sub.shape == (2, None, None)
    assert len(sub._layout.offsets) == 2
    # row1 had 1 middle (data 5:8), row2 had 1 middle (data 8:10)
    np.testing.assert_array_equal(sub[0][0], np.array([5, 6, 7]))
    np.testing.assert_array_equal(sub[1][0], np.array([8, 9]))


# ---------------------------------------------------------------------------
# Task 5: Tuple indexing + leaf access via peel chaining
# ---------------------------------------------------------------------------


def test_r2_tuple_indexing_and_leaf():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (3, None, None), [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])]
    )
    np.testing.assert_array_equal(
        rag[0, 1], np.array([3, 4])
    )  # row0, middle1 -> data 3:5
    np.testing.assert_array_equal(rag[0][1], np.array([3, 4]))  # chaining equivalence
    assert isinstance(rag[2, 0], np.ndarray)


# ---------------------------------------------------------------------------
# Task 6: Per-group inner int / slice indexing (rag[:, k], rag[:, a:b])
# ---------------------------------------------------------------------------


def test_r2_per_group_inner_int():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (2, None, None), [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])]
    )
    got = rag[:, 0]  # 0th middle of each group
    assert got.shape == (2, None)
    np.testing.assert_array_equal(got[0], np.array([0, 1, 2]))  # group0 middle0 -> 0:3
    np.testing.assert_array_equal(got[1], np.array([5, 6, 7]))  # group1 middle0 -> 5:8


def test_r2_per_group_inner_int_out_of_range():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (2, None, None), [np.array([0, 1, 4]), np.array([0, 3, 5, 8, 10])]
    )
    with pytest.raises(IndexError):
        rag[:, 2]  # group0 has only 1 middle


def test_r2_per_group_inner_slice():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data, (2, None, None), [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])]
    )
    sub = rag[:, 0:1]  # first middle of each group, keep nesting
    assert sub.shape == (2, None, None)
    np.testing.assert_array_equal(sub[0, 0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1, 0], np.array([5, 6, 7]))


def test_r2_per_group_inner_slice_negative_raises():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data,
        (2, None, None),
        [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])],
    )
    with pytest.raises(NotImplementedError, match="negative"):
        rag[:, -2:]


# ---------------------------------------------------------------------------
# Task 8: Per-group inner mask / int-array indexing
# ---------------------------------------------------------------------------


def test_r2_inner_mask():
    data = np.arange(10, dtype=np.int32)
    rag = Ragged.from_offsets(
        data,
        (2, None, None),
        [np.array([0, 2, 4]), np.array([0, 3, 5, 8, 10])],
    )
    mask = np.array([True, False, True, True])  # over the 4 middles
    sub = rag[:, mask]
    assert sub.shape == (2, None, None)
    np.testing.assert_array_equal(
        sub.lengths.tolist(), [1, 2]
    )  # group counts after mask
    np.testing.assert_array_equal(sub[0, 0], np.array([0, 1, 2]))
    np.testing.assert_array_equal(sub[1, 0], np.array([5, 6, 7]))


def test_r2_inner_uniform_int_array():
    data = np.arange(12, dtype=np.int32)
    rag = Ragged.from_offsets(
        data,
        (2, None, None),
        [np.array([0, 2, 4]), np.array([0, 3, 6, 9, 12])],
    )
    sub = rag[:, np.array([0, 1])]  # middles 0 and 1 of each group
    assert sub.shape == (2, 2, None)
    # (L0, len(idx), None): leading dims flatten row-major to 4 segments:
    #   flat 0=(g0,i0), 1=(g0,i1), 2=(g1,i0), 3=(g1,i1)
    np.testing.assert_array_equal(
        sub[1], np.array([3, 4, 5])
    )  # (g0, idx1) -> data[3:6]
    np.testing.assert_array_equal(
        sub[2], np.array([6, 7, 8])
    )  # (g1, idx0) -> data[6:9]
