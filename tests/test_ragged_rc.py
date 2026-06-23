"""Tests for seqpro.rag.reverse_complement (flat-buffer ragged kernel)."""

from __future__ import annotations

import numpy as np
import pytest

import seqpro as sp
from seqpro.rag import Ragged, lengths_to_offsets
from seqpro.rag._ops import reverse_complement

# --------------------------------------------------------------------------- #
# Naive numpy reference
# --------------------------------------------------------------------------- #
#
# The original oracle used ak.str.reverse, which is unusable in this environment
# (the installed pyarrow lacks PyExtensionType, which awkward's string ops
# require). This independent per-row numpy reference computes the same value:
# reverse-complement each masked row over a packed copy of the flat buffer.

_COMP_DNA_U8 = np.frombuffer(bytes.maketrans(b"ACGT", b"TGCA"), np.uint8)


def _naive_rc_data(rag: Ragged, mask: np.ndarray) -> np.ndarray:
    """Packed S1 flat buffer after reverse-complementing each masked row."""
    packed = rag.to_packed()
    off = np.asarray(packed.offsets)
    data = np.ascontiguousarray(packed.data).view(np.uint8).copy()
    mask = np.asarray(mask).reshape(-1)
    for i in range(len(off) - 1):
        if mask[i]:
            seg = data[off[i] : off[i + 1]]
            data[off[i] : off[i + 1]] = _COMP_DNA_U8[seg[::-1]]
    return data.view("S1")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

COMP_LUT = sp.DNA.bytes_comp_array.view(np.uint8)


def _make_rag(seqs: list[str]) -> Ragged:
    """Build a 1-D ragged array from a list of DNA strings (writable buffer)."""
    lengths = np.array([len(s) for s in seqs], dtype=np.uint32)
    flat = "".join(seqs).encode()
    data = np.array(list(flat), dtype=np.uint8).view("S1")
    return Ragged.from_lengths(data, lengths)


def _rag_rows(rag: Ragged) -> list[bytes]:
    offsets = rag.offsets
    data = rag.data
    return [
        data[offsets[i] : offsets[i + 1]].tobytes() for i in range(len(offsets) - 1)
    ]


# --------------------------------------------------------------------------- #
# Basic correctness
# --------------------------------------------------------------------------- #


def test_rc_single_row_no_mask():
    rag = _make_rag(["ATCG"])
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"CGAT"]


def test_rc_all_rows_no_mask():
    seqs = ["ATCG", "GCTA", "TTTT"]
    rag = _make_rag(seqs)
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"CGAT", b"TAGC", b"AAAA"]


def test_rc_palindrome_unchanged():
    rag = _make_rag(["GAATTC"])  # EcoRI site
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"GAATTC"]


def test_rc_single_base():
    rag = _make_rag(["A"])
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"T"]


def test_rc_even_length():
    rag = _make_rag(["AATT"])
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"AATT"]


def test_rc_odd_length():
    rag = _make_rag(["ATG"])
    result = reverse_complement(rag, COMP_LUT)
    assert _rag_rows(result) == [b"CAT"]


# --------------------------------------------------------------------------- #
# Mask behaviour
# --------------------------------------------------------------------------- #


def test_rc_all_false_mask_unchanged():
    seqs = ["ATCG", "GCTA"]
    rag = _make_rag(seqs)
    mask = np.zeros(2, dtype=bool)
    result = reverse_complement(rag, COMP_LUT, mask=mask)
    assert _rag_rows(result) == [b"ATCG", b"GCTA"]


def test_rc_partial_mask_only_selected_rows():
    seqs = ["ATCG", "GCTA", "TTTT"]
    rag = _make_rag(seqs)
    mask = np.array([True, False, True])
    result = reverse_complement(rag, COMP_LUT, mask=mask)
    assert _rag_rows(result) == [b"CGAT", b"GCTA", b"AAAA"]


def test_rc_all_true_mask_same_as_no_mask():
    seqs = ["ATCG", "GCTA"]
    rag = _make_rag(seqs)
    mask = np.ones(2, dtype=bool)
    result_masked = reverse_complement(rag, COMP_LUT, mask=mask)
    result_no_mask = reverse_complement(rag, COMP_LUT)
    np.testing.assert_array_equal(result_masked.data, result_no_mask.data)


# --------------------------------------------------------------------------- #
# Copy semantics
# --------------------------------------------------------------------------- #


def test_rc_copy_true_does_not_mutate_input():
    rag = _make_rag(["ATCG", "GCTA"])
    original_data = rag.data.copy()
    _ = reverse_complement(rag, COMP_LUT, copy=True)
    np.testing.assert_array_equal(rag.data, original_data)


def test_rc_copy_true_returns_different_object():
    rag = _make_rag(["ATCG"])
    result = reverse_complement(rag, COMP_LUT, copy=True)
    assert result is not rag


def test_rc_copy_false_mutates_input_buffer():
    rag = _make_rag(["ATCG"])
    data_before = rag.data
    reverse_complement(rag, COMP_LUT, copy=False)
    # The flat buffer is mutated — original data view sees the change
    assert data_before.tobytes() == b"CGAT"


# --------------------------------------------------------------------------- #
# Agrees with awkward baseline
# --------------------------------------------------------------------------- #


def test_rc_matches_awkward_baseline_batch():
    rng = np.random.default_rng(42)
    n = 16
    lengths = rng.integers(10, 30, size=n).astype(np.uint32)
    total = int(lengths.sum())
    bases = np.array([65, 67, 71, 84], np.uint8)  # ACGT
    raw = bases[rng.integers(0, 4, size=total)].view("S1")
    rag = Ragged.from_lengths(raw, lengths)
    mask = rng.random(n) < 0.5

    exp_data = _naive_rc_data(rag, mask)
    got = reverse_complement(rag, COMP_LUT, mask=mask, copy=True)

    np.testing.assert_array_equal(got.data, exp_data)


def test_rc_no_mask_matches_awkward_baseline():
    rng = np.random.default_rng(7)
    lengths = rng.integers(5, 20, size=8).astype(np.uint32)
    bases = np.array([65, 67, 71, 84], np.uint8)
    raw = bases[rng.integers(0, 4, size=int(lengths.sum()))].view("S1")
    rag = Ragged.from_lengths(raw, lengths)
    mask = np.ones(8, dtype=bool)

    exp_data = _naive_rc_data(rag, mask)
    got = reverse_complement(rag, COMP_LUT)

    np.testing.assert_array_equal(got.data, exp_data)


# --------------------------------------------------------------------------- #
# Multi-dimensional leading shape
# --------------------------------------------------------------------------- #


def test_rc_2d_leading_shape():
    """Shape (2, 3, None): mask must cover 6 rows total."""
    seqs = ["AT", "GC", "TTTT", "CCGG", "AAAA", "TGCA"]
    lengths = np.array([len(s) for s in seqs], dtype=np.uint32)
    data = np.frombuffer(b"".join(s.encode() for s in seqs), dtype="S1")
    offsets = lengths_to_offsets(lengths)
    rag = Ragged.from_offsets(data, (2, 3, None), offsets)

    mask = np.array([[True, False, True], [False, True, False]])
    result = reverse_complement(rag, COMP_LUT, mask=mask)

    rows = _rag_rows(result)
    assert rows[0] == b"AT"  # RC of AT is AT
    assert rows[1] == b"GC"  # unchanged (mask=False)
    assert rows[2] == b"AAAA"  # RC of TTTT
    assert rows[3] == b"CCGG"  # unchanged
    assert rows[4] == b"TTTT"  # RC of AAAA
    assert rows[5] == b"TGCA"  # unchanged


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #


def test_rc_wrong_dtype_raises():
    rag = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([2, 1, 3]))
    with pytest.raises(ValueError, match="S1"):
        reverse_complement(rag, COMP_LUT)


def test_rc_trailing_fixed_dim_raises():
    data = np.zeros((6, 4), dtype="S1")
    rag = Ragged.from_offsets(data, (2, None, 4), np.array([0, 2, 6], dtype=np.int64))
    with pytest.raises(ValueError, match="ragged axis to be last"):
        reverse_complement(rag, COMP_LUT)


def test_rc_mask_length_mismatch_raises():
    rag = _make_rag(["ATCG", "GCTA", "TTTT"])
    mask = np.ones(5, dtype=bool)  # wrong size
    with pytest.raises(ValueError, match="mask has 5 entries"):
        reverse_complement(rag, COMP_LUT, mask=mask)


def test_rc_bad_comp_lut_shape_raises():
    rag = _make_rag(["ATCG"])
    with pytest.raises(ValueError, match="256-entry"):
        reverse_complement(rag, COMP_LUT[:128])
