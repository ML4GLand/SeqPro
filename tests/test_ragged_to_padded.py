"""Tests for seqpro.rag.to_padded (flat-buffer ragged densify-and-pad)."""

from __future__ import annotations

import awkward as ak
import awkward.operations.str as ak_str
import numpy as np

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
