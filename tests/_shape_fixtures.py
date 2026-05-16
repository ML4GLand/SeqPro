"""Helpers for shape-matrix tests.

A "shape-matrix" test verifies that a function applies independently across
arbitrary leading and trailing dimensions surrounding its key axes (length,
ohe, etc). The pattern: build a baseline input plus a "patterned" input that
broadcasts the baseline along new leading and/or trailing axes, call the
function on both, and check that every leading/trailing slice of the
patterned output equals the baseline output.
"""

from __future__ import annotations

import numpy as np

from seqpro.rag import Ragged

SHAPE_PATTERNS = ("baseline", "leading", "trailing", "combined")
B = 2  # leading dim size
M = 3  # trailing dim size
RAG_LENGTHS = np.array([4, 2, 5])  # used for ragged baselines (n=3)


# -----------------------------------------------------------------------------
# Dense helpers
# -----------------------------------------------------------------------------


def add_leading_trailing_dense(
    baseline: np.ndarray,
    pattern: str,
    length_axis: int = -1,
    b: int = B,
    m: int = M,
) -> tuple[np.ndarray, int]:
    """Build a patterned dense input from a 1D-ish baseline.

    Parameters
    ----------
    baseline
        Baseline array whose "length" axis is at ``length_axis``.
    pattern
        One of ``SHAPE_PATTERNS``.
    length_axis
        Position of the length axis in ``baseline``. Negative indices supported.

    Returns
    -------
    patterned, new_length_axis
        The patterned array and the new length-axis index.
    """
    if length_axis < 0:
        length_axis = baseline.ndim + length_axis

    if pattern == "baseline":
        return baseline, length_axis

    if pattern == "leading":
        out = np.broadcast_to(baseline[None, ...], (b, *baseline.shape)).copy()
        return out, length_axis + 1

    if pattern == "trailing":
        # tile baseline along a new trailing axis so each length-position value
        # is replicated m times.
        out = np.broadcast_to(baseline[..., None], (*baseline.shape, m)).copy()
        return out, length_axis  # unchanged (new axis appended at end)

    if pattern == "combined":
        tmp = np.broadcast_to(baseline[None, ...], (b, *baseline.shape)).copy()
        out = np.broadcast_to(tmp[..., None], (*tmp.shape, m)).copy()
        return out, length_axis + 1

    raise ValueError(f"Unknown pattern: {pattern}")


def assert_broadcast_equal_dense(
    patterned_out: np.ndarray,
    baseline_out: np.ndarray,
    pattern: str,
    b: int = B,
    m: int = M,
):
    """Assert patterned_out equals baseline_out broadcast across leading/trailing dims.

    The convention is that "leading" dims (if any) are at the front of the
    patterned-output shape and "trailing" dims (if any) are at the back.
    """
    patterned_out = np.asarray(patterned_out)
    baseline_out = np.asarray(baseline_out)
    if pattern == "baseline":
        np.testing.assert_array_equal(patterned_out, baseline_out)
        return

    if pattern == "leading":
        # Expected: (b, *baseline_out.shape)
        expected = np.broadcast_to(baseline_out[None, ...], (b, *baseline_out.shape))
        np.testing.assert_array_equal(patterned_out, expected)
        return

    if pattern == "trailing":
        expected = np.broadcast_to(baseline_out[..., None], (*baseline_out.shape, m))
        np.testing.assert_array_equal(patterned_out, expected)
        return

    if pattern == "combined":
        tmp = np.broadcast_to(baseline_out[None, ...], (b, *baseline_out.shape))
        expected = np.broadcast_to(tmp[..., None], (*tmp.shape, m))
        np.testing.assert_array_equal(patterned_out, expected)
        return

    raise ValueError(f"Unknown pattern: {pattern}")


# -----------------------------------------------------------------------------
# Ragged helpers
# -----------------------------------------------------------------------------


def make_ragged_baseline(seqs: list[str]) -> "Ragged[np.bytes_]":
    """Build a (n, ~L) Ragged[bytes] from a list of strings."""
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    return Ragged.from_lengths(data, lengths)


def make_ragged_with_trailing(rag_baseline: "Ragged", m: int = M) -> "Ragged":
    """Build (n, ~L, m) Ragged by tiling each element along a new trailing axis."""
    data = rag_baseline.data
    new_data = np.broadcast_to(data[..., None], (*data.shape, m)).copy()
    lengths = rag_baseline.lengths.ravel()
    return Ragged.from_lengths(new_data, lengths)


def make_ragged_with_leading(rag_baseline: "Ragged", b: int = B) -> "Ragged":
    """Build (b, n, ~L) Ragged by repeating the baseline b times as leading dim."""
    data = rag_baseline.data
    lengths = rag_baseline.lengths.ravel()
    new_data = np.tile(data, (b, *([1] * (data.ndim - 1))))
    new_lengths = np.tile(lengths, (b, 1))
    return Ragged.from_lengths(new_data, new_lengths)


def make_ragged_combined(rag_baseline: "Ragged", b: int = B, m: int = M) -> "Ragged":
    """Build (b, n, ~L, m)."""
    leading = make_ragged_with_leading(rag_baseline, b)
    return make_ragged_with_trailing(leading, m)


def make_ragged_patterned(
    rag_baseline: "Ragged", pattern: str, b: int = B, m: int = M
) -> "Ragged":
    if pattern == "baseline":
        return rag_baseline
    if pattern == "leading":
        return make_ragged_with_leading(rag_baseline, b)
    if pattern == "trailing":
        return make_ragged_with_trailing(rag_baseline, m)
    if pattern == "combined":
        return make_ragged_combined(rag_baseline, b, m)
    raise ValueError(f"Unknown pattern: {pattern}")


def _ragged_data_per_seq(rag: "Ragged") -> list[np.ndarray]:
    """Return a list of per-segment flat-data arrays (length n_segments).

    Works for arbitrary trailing dims on the flat data.
    """
    offsets = rag.offsets
    if offsets.ndim == 1:
        starts = offsets[:-1]
        stops = offsets[1:]
    else:
        starts = offsets[0]
        stops = offsets[1]
    return [rag.data[s:e] for s, e in zip(starts, stops)]


def assert_broadcast_equal_ragged(
    patterned_out: "Ragged",
    baseline_out: "Ragged",
    pattern: str,
    b: int = B,
    m: int = M,
):
    """Assert patterned Ragged output equals baseline output broadcast across
    leading/trailing dims.
    """
    base_segs = _ragged_data_per_seq(baseline_out)
    pat_segs = _ragged_data_per_seq(patterned_out)
    n = len(base_segs)

    if pattern == "baseline":
        assert len(pat_segs) == n
        for ps, bs in zip(pat_segs, base_segs):
            np.testing.assert_array_equal(ps, bs)
        return

    if pattern == "leading":
        # patterned has b * n segments; each block of n matches baseline
        assert len(pat_segs) == b * n, f"Expected {b * n} segments, got {len(pat_segs)}"
        for bi in range(b):
            for i in range(n):
                np.testing.assert_array_equal(pat_segs[bi * n + i], base_segs[i])
        return

    if pattern == "trailing":
        assert len(pat_segs) == n
        for ps, bs in zip(pat_segs, base_segs):
            # ps should equal bs broadcast across trailing m dim
            expected = np.broadcast_to(bs[..., None], (*bs.shape, m))
            np.testing.assert_array_equal(ps, expected)
        return

    if pattern == "combined":
        assert len(pat_segs) == b * n
        for bi in range(b):
            for i in range(n):
                bs = base_segs[i]
                expected = np.broadcast_to(bs[..., None], (*bs.shape, m))
                np.testing.assert_array_equal(pat_segs[bi * n + i], expected)
        return

    raise ValueError(f"Unknown pattern: {pattern}")
