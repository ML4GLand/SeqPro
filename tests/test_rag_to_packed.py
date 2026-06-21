import awkward as ak
import numpy as np
import pytest

from seqpro.rag import Ragged
from seqpro.rag._ops import to_packed


def _unpacked(lengths, dtype):
    """A Ragged backed by a ListArray (2-D offsets) via row-reversal.

    ``lengths`` may include zeros to exercise empty rows.
    """
    lengths = np.asarray(lengths)
    data = np.arange(int(lengths.sum()), dtype=dtype)
    rev = Ragged.from_lengths(data, lengths)[::-1]  # reorder -> ListArray
    assert rev.offsets.ndim == 2
    return rev


class TestToPackedFlat:
    def test_2d_offsets_matches_awkward(self):
        rag = _unpacked([3, 0, 2, 4], np.dtype("float64"))
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert out.is_contiguous
        assert ak.to_list(out) == ak.to_list(rag)
        # matches awkward's packing exactly
        assert ak.to_list(out) == ak.to_list(ak.to_packed(rag))

    def test_1d_offsets_unpacked_rebases(self):
        # A contiguous slice keeps a 1-D ListOffsetArray but with a non-zero
        # base (offsets[0] != 0) and untrimmed content -> exercises the
        # offsets.ndim == 1 gather path (starts = offsets[:-1]) in _pack_parts.
        full = Ragged.from_lengths(np.arange(9, dtype=np.float64), np.array([3, 2, 4]))
        rag = full[1:]
        assert rag.offsets.ndim == 1
        assert rag.offsets[0] != 0  # not zero-based -> unpacked
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert out.is_contiguous
        assert out.data.shape[0] == 6  # content trimmed to packed extent
        assert not np.shares_memory(out.data, rag.data)  # rebased into fresh buffer
        assert ak.to_list(out) == ak.to_list(rag)
        assert ak.to_list(out) == ak.to_list(ak.to_packed(rag))

    def test_bytes_dtype(self):
        seqs = ["ATG", "C", "GGGG"]
        data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").copy()
        lengths = np.array([len(s) for s in seqs])
        rag = Ragged.from_lengths(data, lengths)[::-1]
        out = to_packed(rag)
        assert out.dtype == np.dtype("S1")
        assert ak.to_list(out) == ak.to_list(rag)

    def test_trailing_fixed_dims(self):
        # OHE-like: (n, None, 4) uint8
        data = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        rag = Ragged.from_lengths(data, np.array([2, 1]))[::-1]
        out = to_packed(rag)
        assert out.shape[1:] == rag.shape[1:]
        assert ak.to_list(out) == ak.to_list(rag)

    def test_empty_array(self):
        rag = Ragged.empty((0, None), np.float64)
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert len(out) == 0

    def test_copy_true_returns_owned_buffer(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        out = to_packed(rag, copy=True)
        assert not np.shares_memory(
            out.data, rag.data
        )  # freshly allocated, no shared memory
        out.data[0] = 999.0
        assert rag.data[0] == 0.0  # input untouched

    def test_copy_false_passthrough_when_packed(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        out = to_packed(rag, copy=False)
        assert out is rag  # zero-copy passthrough

    def test_copy_false_raises_when_unpacked(self):
        rag = _unpacked([3, 2], np.dtype("float64"))
        with pytest.raises(ValueError, match="already-packed"):
            to_packed(rag, copy=False)

    def test_trailing_fixed_dims_multibyte(self):
        # float64 (itemsize 8) with trailing dims: exercises elem = prod(trailing)*8
        data = np.arange(3 * 3, dtype=np.float64).reshape(3, 3)
        rag = Ragged.from_lengths(data, np.array([2, 1]))[::-1]
        out = to_packed(rag)
        assert out.dtype == np.dtype("float64")
        assert out.shape[1:] == rag.shape[1:]
        assert ak.to_list(out) == ak.to_list(rag)


class TestToPackedRecord:
    def _record(self):
        import awkward as ak

        lengths = np.array([3, 2, 4])
        scores = np.arange(9, dtype=np.float64)
        flags = np.arange(9, dtype=np.int8)
        rec = ak.zip(
            {
                "score": Ragged.from_lengths(scores, lengths),
                "flag": Ragged.from_lengths(flags, lengths),
            }
        )
        return Ragged(rec)

    def test_record_unpacked_packs_all_fields(self):
        rec = self._record()[::-1]  # reorder -> ListArray-backed fields
        out = to_packed(rec)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert ak.to_list(out) == ak.to_list(rec)
        # fields share one offsets object (zero-copy SoA contract)
        assert out["score"].offsets is out["flag"].offsets

    def test_record_copy_false_passthrough(self):
        rec = self._record()
        out = to_packed(rec, copy=False)
        assert out is rec


class TestToPackedMethod:
    def test_method_delegates(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))[
            ::-1
        ]
        out = rag.to_packed()
        assert out.offsets.ndim == 1
        assert ak.to_list(out) == ak.to_list(rag)

    def test_method_copy_false(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        assert rag.to_packed(copy=False) is rag

    def test_exported_from_package(self):
        import seqpro.rag as rag_mod

        assert hasattr(rag_mod, "to_packed")


class TestIndexedLayouts:
    """An IndexedArray wrapper arises from indexing a *record* then pulling a
    field (e.g. ak.zip(...)[perm]["x"]). seqpro's layout walkers must traverse
    it. Plain Ragged[perm] yields a ListArray (already covered elsewhere)."""

    def test_indexed_field_unbox_and_to_packed(self):
        lengths = np.array([3, 0, 2, 4])
        perm = np.array([2, 0, 3, 1])
        data = np.arange(int(lengths.sum()), dtype=np.float64)
        r = Ragged.from_lengths(data, lengths)
        field = ak.zip({"x": r}, depth_limit=1)[perm]["x"]

        from awkward.contents import IndexedArray

        assert isinstance(field.layout, IndexedArray)

        rag = Ragged(field)  # used to raise: Expected 1 ragged dimension, got 0
        # accessors that route through unbox() must all work
        assert rag.offsets is not None
        assert rag.data is not None
        out = to_packed(rag)
        assert out.offsets.ndim == 1 and out.offsets[0] == 0
        assert out.is_contiguous
        assert ak.to_list(out) == ak.to_list(field)
        assert ak.to_list(out) == ak.to_list(ak.to_packed(field))

    def test_indexed_record_layout_offsets(self):
        # ak.zip(..., depth_limit=1) -> RecordArray; indexing it ->
        # IndexedArray(RecordArray(...)), which _extract_list_offsets must walk.
        r = Ragged.from_lengths(np.arange(9, dtype=np.float64), np.array([3, 2, 4]))
        rec = ak.zip({"a": r, "b": r}, depth_limit=1)[np.array([2, 0, 1])]

        from awkward.contents import IndexedArray

        assert isinstance(rec.layout, IndexedArray)

        rag = Ragged(rec)  # record-layout Ragged over an indexed layout
        # offsets extraction (via _extract_list_offsets) must not crash
        assert rag.offsets is not None
        assert ak.to_list(rag["a"]) == ak.to_list(rec["a"])


class TestToPackedNested:
    def test_to_packed_nested_after_outer_slice(self):
        from seqpro.rag._core import Ragged as CoreRagged

        data = np.arange(10, dtype=np.int32)
        rag = CoreRagged.from_offsets(
            data,
            (3, None, None),
            [np.array([0, 2, 3, 4]), np.array([0, 3, 5, 8, 10])],
        )
        packed = rag[1:3].to_packed()
        assert packed._layout.offsets[0].ndim == 1 and packed._layout.offsets[0][0] == 0
        assert packed._layout.offsets[1].ndim == 1 and packed._layout.offsets[1][0] == 0
        assert packed.data.flags.c_contiguous
        np.testing.assert_array_equal(packed[0, 0], np.array([5, 6, 7]))
        np.testing.assert_array_equal(packed[1, 0], np.array([8, 9]))
        np.testing.assert_array_equal(packed.data, np.array([5, 6, 7, 8, 9]))
