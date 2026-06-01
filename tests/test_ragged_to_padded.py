"""Tests for seqpro.rag.to_padded (flat-buffer ragged densify-and-pad)."""

from __future__ import annotations

import awkward as ak
import awkward.operations.str as ak_str
import numpy as np
import pytest

from seqpro.rag import Ragged, lengths_to_offsets
from seqpro.rag._ops import to_padded


# --------------------------------------------------------------------------- #
# Awkward references (mirror gvl's _ragged.to_padded)
# --------------------------------------------------------------------------- #


def _naive_pad_bytes(rag: Ragged, pad_value: bytes, length: int) -> np.ndarray:
    return Ragged(ak_str.rpad(rag, length, pad_value)).to_numpy()


def _naive_pad_numeric(rag: Ragged, pad_value, length: int) -> np.ndarray:
    orig_dtype = rag.dtype
    r = ak.pad_none(rag, length, axis=-1, clip=True)
    r = ak.fill_none(r, pad_value)
    return ak.to_numpy(r).astype(orig_dtype, copy=False)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_bytes_rag(seqs: list[str]) -> Ragged:
    lengths = np.array([len(s) for s in seqs], dtype=np.uint32)
    flat = "".join(seqs).encode()
    data = np.frombuffer(flat, dtype="S1").copy()
    return Ragged.from_lengths(data, lengths)


def _make_numeric_rag(rows: list[list], dtype) -> Ragged:
    lengths = np.array([len(r) for r in rows], dtype=np.uint32)
    flat = np.array([x for r in rows for x in r], dtype=dtype)
    return Ragged.from_lengths(flat, lengths)


# --------------------------------------------------------------------------- #
# Core: pad to batch max
# --------------------------------------------------------------------------- #


def test_pad_to_max_bytes_basic():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N")
    expected = np.array(
        [[b"A", b"T", b"C", b"G"], [b"G", b"G", b"N", b"N"]], dtype="S1"
    )
    np.testing.assert_array_equal(out, expected)


def test_pad_to_max_int32_basic():
    rag = _make_numeric_rag([[0, 1, 2, 3], [4, 5]], np.int32)
    out = to_padded(rag, -1)
    expected = np.array([[0, 1, 2, 3], [4, 5, -1, -1]], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.int32


def test_pad_to_max_float32_basic():
    rag = _make_numeric_rag([[1.5], [2.5, 3.5, 4.5]], np.float32)
    out = to_padded(rag, 0.0)
    expected = np.array([[1.5, 0.0, 0.0], [2.5, 3.5, 4.5]], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)
    assert out.dtype == np.float32


def test_pad_to_max_matches_awkward_bytes():
    rng = np.random.default_rng(1)
    n = 12
    lengths = rng.integers(0, 20, size=n)
    seqs = ["".join(rng.choice(list("ACGT"), size=int(L))) for L in lengths]
    rag = _make_bytes_rag(seqs)
    out = to_padded(rag, b"N")
    expected = _naive_pad_bytes(rag, b"N", int(lengths.max()))
    np.testing.assert_array_equal(out, expected)


def test_pad_to_max_leading_dims():
    """(2, 3, None) input densifies to (2, 3, out_len) with rows in the right cells."""
    rows = ["AT", "G", "TTTT", "CC", "A", "GGG"]
    lengths = np.array([len(s) for s in rows], dtype=np.uint32)
    data = np.frombuffer("".join(rows).encode(), dtype="S1").copy()
    offsets = lengths_to_offsets(lengths)
    rag = Ragged.from_offsets(data, (2, 3, None), offsets)
    out = to_padded(rag, b"N")
    assert out.shape == (2, 3, 4)
    np.testing.assert_array_equal(out[0, 0], np.frombuffer(b"ATNN", dtype="S1"))
    np.testing.assert_array_equal(out[1, 2], np.frombuffer(b"GGGN", dtype="S1"))


def test_length_pad_beyond_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=6)
    expected = np.array(
        [[b"A", b"T", b"C", b"G", b"N", b"N"], [b"G", b"G", b"N", b"N", b"N", b"N"]],
        dtype="S1",
    )
    np.testing.assert_array_equal(out, expected)


def test_length_truncate_below_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=3)
    expected = np.array([[b"A", b"T", b"C"], [b"G", b"G", b"N"]], dtype="S1")
    np.testing.assert_array_equal(out, expected)


def test_length_equal_to_max():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out_explicit = to_padded(rag, b"N", length=4)
    out_default = to_padded(rag, b"N")
    np.testing.assert_array_equal(out_explicit, out_default)
    assert out_explicit.dtype == out_default.dtype
    assert out_explicit.shape == out_default.shape


def test_length_truncate_numeric():
    rag = _make_numeric_rag([[0, 1, 2, 3], [4, 5]], np.int32)
    out = to_padded(rag, -1, length=2)
    expected = np.array([[0, 1], [4, 5]], dtype=np.int32)
    np.testing.assert_array_equal(out, expected)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_record_layout_raises():
    seq = _make_bytes_rag(["ATCG", "GG"])
    score = _make_numeric_rag([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0]], np.float32)
    rec = ak.zip({"seq": seq, "score": score})
    assert isinstance(rec, Ragged)
    with pytest.raises(NotImplementedError, match="record-layout"):
        to_padded(rec, b"N")


def test_trailing_fixed_dim_raises():
    data = np.zeros((6, 4), dtype="S1")
    rag = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="ragged axis to be last"):
        to_padded(rag, b"N")


def test_sliced_nonzero_offset_start_correct():
    """A sliced Ragged with a nonzero starting offset still densifies correctly."""
    rag = _make_bytes_rag(["ATCG", "GG", "TTTAA", "C"])
    sliced = rag[1:3]  # 1-D offsets with a nonzero start (e.g. [4, 6, 11])
    out = to_padded(sliced, b"N")
    expected = np.array(
        [[b"G", b"G", b"N", b"N", b"N"], [b"T", b"T", b"T", b"A", b"A"]], dtype="S1"
    )
    np.testing.assert_array_equal(out, expected)


# --------------------------------------------------------------------------- #
# Edge cases + multi-dtype awkward baseline
# --------------------------------------------------------------------------- #


def test_empty_batch():
    rag = Ragged.from_lengths(
        np.frombuffer(b"", dtype="S1").copy(), np.array([], dtype=np.uint32)
    )
    out = to_padded(rag, b"N")
    assert out.shape == (0, 0)


def test_all_empty_rows():
    rag = Ragged.from_lengths(
        np.frombuffer(b"", dtype="S1").copy(), np.array([0, 0, 0], dtype=np.uint32)
    )
    out = to_padded(rag, b"N")
    assert out.shape == (3, 0)


def test_length_zero_truncates_all():
    rag = _make_bytes_rag(["ATCG", "GG"])
    out = to_padded(rag, b"N", length=0)
    assert out.shape == (2, 0)


def test_matches_awkward_int32_iinfo_max():
    rng = np.random.default_rng(3)
    rows = [
        list(rng.integers(0, 100, size=int(L))) for L in rng.integers(0, 15, size=10)
    ]
    rag = _make_numeric_rag(rows, np.int32)
    pad = int(np.iinfo(np.int32).max)
    length = max((len(r) for r in rows), default=0)
    out = to_padded(rag, pad)
    expected = _naive_pad_numeric(rag, pad, length)
    np.testing.assert_array_equal(out, expected)


def test_matches_awkward_float32():
    rng = np.random.default_rng(4)
    rows = [
        list(rng.random(int(L)).astype(np.float32)) for L in rng.integers(1, 12, size=9)
    ]
    rag = _make_numeric_rag(rows, np.float32)
    length = max(len(r) for r in rows)
    out = to_padded(rag, 0.0)
    expected = _naive_pad_numeric(rag, 0.0, length)
    np.testing.assert_array_equal(out, expected)
