import awkward as ak
import numpy as np
import pytest

from seqpro.rag import Ragged
from seqpro.rag import zip as rag_zip
from seqpro.rag._ops import to_packed


def _to_list(rag):
    """Awkward nested-list view of a _core.Ragged for value comparisons.

    _core.Ragged is not an ak.Array, so ``ak.to_list(rag)`` cannot introspect it
    directly; ``rag.to_ak()`` bridges to the awkward layout that the old
    _array-backed tests compared against.
    """
    return ak.to_list(rag.to_ak()) if isinstance(rag, Ragged) else ak.to_list(rag)


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
        assert _to_list(out) == _to_list(rag)
        # matches awkward's packing exactly
        assert _to_list(out) == ak.to_list(ak.to_packed(rag.to_ak()))

    def test_unpacked_slice_rebases(self):
        # The contiguous-slice fast path now returns a zero-copy, already-contiguous
        # result for simple (R=1, non-step) slices.  to_packed must still work
        # correctly on the result: produce a 1-D, zero-based, contiguous output
        # whose content matches the original slice.
        full = Ragged.from_lengths(np.arange(9, dtype=np.float64), np.array([3, 2, 4]))
        rag = full[1:]
        # Fast path: slicing now returns a contiguous view, not an unpacked gather.
        assert rag.is_contiguous
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert out.is_contiguous
        assert out.data.shape[0] == 6  # content trimmed to packed extent
        assert _to_list(out) == _to_list(rag)
        assert _to_list(out) == ak.to_list(ak.to_packed(rag.to_ak()))

    def test_bytes_dtype(self):
        seqs = ["ATG", "C", "GGGG"]
        data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1").copy()
        lengths = np.array([len(s) for s in seqs])
        rag = Ragged.from_lengths(data, lengths)[::-1]
        out = to_packed(rag)
        # _core models a 1-D S1 from_lengths collection as an opaque variable-width
        # string (dtype descriptor 'S'); the underlying char buffer stays S1.
        assert out.dtype == np.dtype("S")
        assert out.data.dtype == np.dtype("S1")
        assert _to_list(out) == _to_list(rag)

    def test_trailing_fixed_dims(self):
        # OHE-like: (n, None, 4) uint8
        data = np.arange(3 * 4, dtype=np.uint8).reshape(3, 4)
        rag = Ragged.from_lengths(data, np.array([2, 1]))[::-1]
        out = to_packed(rag)
        assert out.shape[1:] == rag.shape[1:]
        assert _to_list(out) == _to_list(rag)

    def test_empty_array(self):
        rag = Ragged.empty((0, None), np.float64)
        out = to_packed(rag)
        assert out.offsets.ndim == 1
        assert out.shape[0] == 0  # _core.Ragged has no __len__; check leading dim

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
        assert _to_list(out) == _to_list(rag)


class TestToPackedRecord:
    def _record(self):
        lengths = np.array([3, 2, 4])
        scores = np.arange(9, dtype=np.float64)
        flags = np.arange(9, dtype=np.int8)
        return rag_zip(
            {
                "score": Ragged.from_lengths(scores, lengths),
                "flag": Ragged.from_lengths(flags, lengths),
            }
        )

    def test_record_unpacked_packs_all_fields(self):
        rec = self._record()[::-1]  # reorder -> ListArray-backed fields
        out = to_packed(rec)
        assert out.offsets.ndim == 1
        assert out.offsets[0] == 0
        assert _to_list(out) == _to_list(rec)
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
        assert _to_list(out) == _to_list(rag)

    def test_method_copy_false(self):
        rag = Ragged.from_lengths(np.arange(6, dtype=np.float64), np.array([3, 3]))
        assert rag.to_packed(copy=False) is rag

    def test_exported_from_package(self):
        import seqpro.rag as rag_mod

        assert hasattr(rag_mod, "to_packed")


class TestIndexedLayouts:
    """An IndexedArray wrapper arises from indexing a *record* then pulling a
    field (e.g. ak.zip(rag.to_ak(), ...)[perm]["x"]). seqpro's awkward-layout
    ingestion (Ragged(ak_array)) must traverse it. Plain Ragged[perm] yields a
    ListArray (already covered elsewhere). The awkward record is built from
    rag.to_ak() since _core.Ragged is not itself an ak.Array."""

    def test_indexed_field_unbox_and_to_packed(self):
        lengths = np.array([3, 0, 2, 4])
        perm = np.array([2, 0, 3, 1])
        data = np.arange(int(lengths.sum()), dtype=np.float64)
        r = Ragged.from_lengths(data, lengths)
        field = ak.zip({"x": r.to_ak()}, depth_limit=1)[perm]["x"]

        from awkward.contents import IndexedArray

        assert isinstance(field.layout, IndexedArray)

        rag = Ragged(field)  # used to raise: Expected 1 ragged dimension, got 0
        # accessors that route through unbox() must all work
        assert rag.offsets is not None
        assert rag.data is not None
        out = to_packed(rag)
        assert out.offsets.ndim == 1 and out.offsets[0] == 0
        assert out.is_contiguous
        assert _to_list(out) == ak.to_list(field)
        assert _to_list(out) == ak.to_list(ak.to_packed(field))

    def test_indexed_record_layout_offsets(self):
        # ak.zip(..., depth_limit=1) -> RecordArray; indexing it ->
        # IndexedArray(RecordArray(...)), which _extract_list_offsets must walk.
        r = Ragged.from_lengths(np.arange(9, dtype=np.float64), np.array([3, 2, 4]))
        rec = ak.zip({"a": r.to_ak(), "b": r.to_ak()}, depth_limit=1)[
            np.array([2, 0, 1])
        ]

        from awkward.contents import IndexedArray

        assert isinstance(rec.layout, IndexedArray)

        rag = Ragged(rec)  # record-layout Ragged over an indexed layout
        # offsets extraction (via _extract_list_offsets) must not crash
        assert rag.offsets is not None
        assert _to_list(rag["a"]) == ak.to_list(rec["a"])


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


def _record(var_off, char_off, chars):
    alt = Ragged.from_offsets(
        chars, (len(var_off) - 1, None, None), [var_off, char_off]
    ).to_strings()
    start = Ragged.from_offsets(
        np.arange(int(var_off[-1]), dtype=np.int32),
        (len(var_off) - 1, None),
        alt.offsets,
    )
    return Ragged.from_fields({"alt": alt, "start": start}), alt


def test_to_packed_opaque_string_under_axis():
    var_off = np.array([0, 2, 3], dtype=np.int64)
    char_off = np.array([0, 2, 3, 6], dtype=np.int64)
    rv, alt = _record(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    sl = alt[np.array([1, 0])]  # produces (2,N) gather offsets
    packed = sl.to_packed()  # must not raise
    assert packed.to_ak().to_list() == [[b"TTT"], [b"AC", b"G"]]


def test_to_packed_record_with_string_field():
    var_off = np.array([0, 2, 3], dtype=np.int64)
    char_off = np.array([0, 2, 3, 6], dtype=np.int64)
    rv, _ = _record(var_off, char_off, np.frombuffer(b"ACGTTT", "S1").copy())
    sl = rv[np.array([1, 0])]
    packed = sl.to_packed()  # must not raise
    assert packed["alt"].to_ak().to_list() == [[b"TTT"], [b"AC", b"G"]]
    assert packed["start"].to_ak().to_list() == [[2], [0, 1]]
