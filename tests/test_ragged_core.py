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


def test_ingest_record_raises_spec_b():
    arr = ak.Array({"a": [[1, 2], [3]], "b": [[1.0, 2.0], [3.0]]})
    with pytest.raises(NotImplementedError, match="Spec B"):
        Ragged(arr)


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
