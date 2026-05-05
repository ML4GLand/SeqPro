import awkward as ak
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases
from seqpro.rag import OFFSET_TYPE, Ragged, lengths_to_offsets
from seqpro.rag._array import RagParts


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

    def test_parts_dict(self, rag: Ragged):
        p = rag.parts
        assert isinstance(p, dict)
        assert list(p.keys()) == ["field0", "field1"]
        for v in p.values():
            assert isinstance(v, RagParts)

    def test_parts_dict_shares_offsets(self, rag: Ragged):
        p = rag.parts
        assert p["field0"].offsets is rag.offsets
        assert p["field1"].offsets is rag.offsets

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

    def test_dtype_dict(self, rag: Ragged):
        dt = rag.dtype
        assert isinstance(dt, dict)
        assert list(dt.keys()) == ["field0", "field1"]
        assert dt["field0"] == np.int64
        assert dt["field1"] == np.float64

    def test_dtype_field_order_preserved(self):
        r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
        rag = Ragged(ak.zip({"zeta": r1, "alpha": r2}))
        assert list(rag.dtype.keys()) == ["zeta", "alpha"]

    def test_zip_produces_initialized_ragged(self):
        r1 = Ragged.from_lengths(np.arange(6, dtype=np.int64), np.array([2, 1, 3]))
        r2 = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([2, 1, 3]))
        z = ak.zip({"a": r1, "b": r2})
        assert isinstance(z, Ragged)
        np.testing.assert_array_equal(
            z.offsets, np.array([0, 2, 3, 6], dtype=z.offsets.dtype)
        )
