import awkward as ak
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases
from seqpro.rag import OFFSET_TYPE, Ragged, is_rag_dtype, lengths_to_offsets
from seqpro.rag import zip as rag_zip


def case_int32():
    data = np.arange(10, dtype=np.int32)
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    offsets = lengths_to_offsets(lengths)
    shape = (3, None)
    content = Ragged.from_lengths(data, lengths)
    return content, data, shape, offsets


def case_s1():
    # _core.Ragged models a 1-D S1 collection built via from_lengths as an opaque
    # variable-width string leaf: the byte-length is NOT a counted axis, so the
    # shape is (3,) rather than (3, None). The flat data buffer and the (str_)offsets
    # are identical to the awkward-backed model.
    data = np.array([b"cathithere"]).view("S1")
    lengths = np.array([3, 2, 5], dtype=np.uint32)
    offsets = lengths_to_offsets(lengths)
    shape = (3,)
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


class TestRecordRagged:
    @pytest.fixture
    def rag(self):
        return Ragged(
            ak.Array(
                {
                    "field0": [[1, 2], [3], [4, 5, 6]],
                    "field1": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
                }
            )
        )

    def test_offsets(self, rag: Ragged):
        expected = np.array([0, 2, 3, 6], dtype=OFFSET_TYPE)
        np.testing.assert_array_equal(rag.offsets, expected)

    def test_data_dict(self, rag: Ragged):
        d = rag.data
        assert isinstance(d, dict)
        assert list(d.keys()) == ["field0", "field1"]
        np.testing.assert_array_equal(d["field0"], np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(
            d["field1"], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_fields_list(self, rag: Ragged):
        # _core.Ragged exposes record fields via .fields (a list of names) and
        # field access via rag[name]; the _array-only RagParts container does not
        # exist in the _core model.
        assert rag.fields == ["field0", "field1"]
        for name in rag.fields:
            assert isinstance(rag[name], Ragged)

    def test_fields_share_offsets(self, rag: Ragged):
        # Zero-copy SoA contract: every field's offsets IS the record's offsets.
        assert rag["field0"].offsets is rag.offsets
        assert rag["field1"].offsets is rag.offsets

    def test_data_dict_zero_copy(self, rag: Ragged):
        d = rag.data
        assert d["field0"].base is not None
        assert d["field1"].base is not None

    def test_field_access_by_key(self, rag: Ragged):
        np.testing.assert_array_equal(rag["field0"].data, np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_array_equal(
            rag["field1"].data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )

    def test_field_access_by_attr(self, rag: Ragged):
        np.testing.assert_array_equal(rag.field0.data, rag["field0"].data)
        np.testing.assert_array_equal(rag.field1.data, rag["field1"].data)

    def test_offsets_shared_with_field(self, rag: Ragged):
        assert rag["field0"].offsets is rag.offsets

    def test_offsets_shared_across_fields(self, rag: Ragged):
        assert rag["field0"].offsets is rag["field1"].offsets

    def test_field_returns_ragged(self, rag: Ragged):
        assert isinstance(rag["field0"], Ragged)
        assert isinstance(rag["field1"], Ragged)

    def test_dtype_structured(self, rag: Ragged):
        dt = rag.dtype
        assert isinstance(dt, np.dtype)
        assert dt.names == ("field0", "field1")
        assert dt["field0"] == np.dtype(np.int64)
        assert dt["field1"] == np.dtype(np.float64)

    def test_dtype_field_order_preserved(self):
        r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
        rag = rag_zip({"zeta": r1, "alpha": r2})
        assert list(rag.dtype.names) == ["zeta", "alpha"]

    def test_view_raises_on_record(self, rag: Ragged):
        with pytest.raises(NotImplementedError, match="record"):
            _ = rag.view(np.float32)

    def test_to_numpy_raises_on_record(self):
        # _core contract (replaces _array-era NotImplementedError): to_numpy on a
        # record returns a dict of dense per-field arrays, not a single ndarray.
        lens = np.array([2, 2, 2], dtype=np.uint32)
        f0 = Ragged.from_lengths(np.arange(6, dtype=np.int64), lens)
        f1 = Ragged.from_lengths(np.arange(6, dtype=np.float64), lens)
        rec = rag_zip({"field0": f0, "field1": f1})
        out = rec.to_numpy()
        assert isinstance(out, dict)
        assert set(out.keys()) == {"field0", "field1"}
        assert isinstance(out["field0"], np.ndarray)
        assert isinstance(out["field1"], np.ndarray)

    def test_field_setitem_roundtrip(self, rag: Ragged):
        original = rag["field0"].data.copy()
        rag["field0"] = rag["field0"].view(np.uint64)
        np.testing.assert_array_equal(rag["field0"].data.view(np.int64), original)

    def test_squeeze_record(self):
        f0 = Ragged.from_lengths(
            np.arange(6, dtype=np.int64).reshape(6, 1), np.array([2, 1, 3])
        )
        f1 = Ragged.from_lengths(
            np.arange(6, dtype=np.float64).reshape(6, 1), np.array([2, 1, 3])
        )
        rag = rag_zip({"a": f0, "b": f1})
        sq = rag.squeeze()
        assert isinstance(sq, Ragged)
        assert sq.shape == (3, None)
        np.testing.assert_array_equal(sq["a"].data, np.arange(6))
        assert sq["a"].offsets is sq.offsets

    def test_reshape_record(self):
        lengths = np.array([2, 1, 3, 1, 2, 1])
        data_a = np.arange(10, dtype=np.int64)
        data_b = np.arange(10, dtype=np.float64)
        f0 = Ragged.from_lengths(data_a, lengths)
        f1 = Ragged.from_lengths(data_b, lengths)
        rag = rag_zip({"a": f0, "b": f1})
        re = rag.reshape(2, 3, None)
        assert isinstance(re, Ragged)
        assert re.shape == (2, 3, None)
        assert re["a"].offsets is re.offsets
        np.testing.assert_array_equal(re["a"].data, data_a)

    def test_zip_produces_initialized_ragged(self):
        r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
        z = rag_zip({"a": r1, "b": r2})
        assert isinstance(z, Ragged)
        np.testing.assert_array_equal(
            z.offsets, np.array([0, 2, 3, 6], dtype=z.offsets.dtype)
        )


class TestZip:
    # seqpro.rag.zip (alias of Ragged.from_fields) is the _core-native record
    # builder, replacing the _array-era ak.zip({...}) idiom. It builds a record
    # (struct-of-arrays) Ragged from single-field Ragged inputs sharing one
    # ragged axis.
    def _mk(self, dtype):
        return Ragged.from_lengths(np.arange(6, dtype=dtype), np.array([2, 1, 3]))

    def test_zip_returns_ragged(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = rag_zip({"a": r1, "b": r2})
        assert isinstance(z, Ragged)

    def test_zip_explicit_wrap_returns_ragged(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = Ragged(rag_zip({"a": r1, "b": r2}))
        assert isinstance(z, Ragged)

    def test_zip_three_fields(self):
        r1, r2, r3 = self._mk(np.int64), self._mk(np.float64), self._mk(np.int32)
        z = rag_zip({"a": r1, "b": r2, "c": r3})
        assert list(z.dtype.names) == ["a", "b", "c"]
        np.testing.assert_array_equal(z["c"].data, np.arange(6, dtype=np.int32))

    def test_zip_field_order_preserved(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = rag_zip({"zeta": r1, "alpha": r2})
        assert list(z.dtype.names) == ["zeta", "alpha"]

    def test_zip_offsets_shared_across_fields(self):
        r1, r2 = self._mk(np.int64), self._mk(np.float64)
        z = rag_zip({"a": r1, "b": r2})
        assert z["a"].offsets is z["b"].offsets

    def test_zip_with_extra_dim(self):
        # Fields may carry a trailing fixed dim (here K=2); zip preserves it.
        data_a = np.arange(12, dtype=np.int64).reshape(6, 2)
        data_b = np.arange(12, dtype=np.float64).reshape(6, 2)
        r1 = Ragged.from_lengths(data_a, np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(data_b, np.array([2, 1, 3]))
        z = rag_zip({"a": r1, "b": r2})
        assert isinstance(z, Ragged)
        np.testing.assert_array_equal(z["a"].data, data_a)


@pytest.fixture
def rag_record():
    return Ragged(
        ak.Array(
            {
                "field0": [[1, 2], [3], [4, 5, 6]],
                "field1": [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
            }
        )
    )


def test_is_rag_dtype_record_primitive_query_returns_false(rag_record):
    assert not is_rag_dtype(rag_record, np.int64)
    assert not is_rag_dtype(rag_record, np.float64)


def test_is_rag_dtype_record_matching_struct_returns_true(rag_record):
    dt = np.dtype([("field0", np.int64), ("field1", np.float64)])
    assert is_rag_dtype(rag_record, dt)


def test_is_rag_dtype_record_wrong_struct_returns_false(rag_record):
    dt_wrong_types = np.dtype([("field0", np.float32), ("field1", np.float64)])
    dt_wrong_names = np.dtype([("x", np.int64), ("y", np.float64)])
    assert not is_rag_dtype(rag_record, dt_wrong_types)
    assert not is_rag_dtype(rag_record, dt_wrong_names)


# _core.Ragged is not an ak.Array subclass and does not register an awkward
# typestr; the structural facts the old "3 * var * Ragged[int64]" typestr encoded
# (leading dims, the single ragged axis, trailing fixed dims, and the leaf dtype)
# are exposed directly via .shape and .dtype, which these tests now assert.
def test_typestr_scalar_dtype():
    r = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
    assert r.shape == (3, None)  # "3 * var"
    assert r.dtype == np.dtype(np.int64)  # "Ragged[int64]"


def test_typestr_multidim_inner():
    data = np.zeros((6, 4), dtype=np.int32)
    r = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6]))
    assert r.shape == (2, None, 4)  # "2 * var * 4"
    assert r.dtype == np.dtype(np.int32)  # "Ragged[int32]"


def test_typestr_float():
    r = Ragged.from_lengths(np.ones(6, dtype=np.float32), np.array([2, 1, 3]))
    assert r.shape == (3, None)  # "3 * var"
    assert r.dtype == np.dtype(np.float32)  # "Ragged[float32]"
