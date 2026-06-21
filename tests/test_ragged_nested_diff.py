"""Differential property tests: native R=2 Ragged vs awkward oracle.

Each property test draws a random nested R=2 structure and checks that the
native implementation agrees with an awkward-array oracle built from the same
raw buffers.  Empty groups (outer_counts == 0) and empty inner segments
(inner_lens == 0) are exercised by the st.integers(0, …) ranges.
"""

from __future__ import annotations

import awkward as ak
import numpy as np
from hypothesis import given, settings, strategies as st

from seqpro.rag._core import Ragged


# ---------------------------------------------------------------------------
# Strategy: random nested R=2 structure
# ---------------------------------------------------------------------------


@st.composite
def nested_r2(draw):
    """Return (outer_counts, inner_lens, data) as numpy arrays."""
    n_outer = draw(st.integers(1, 4))
    outer_counts = draw(st.lists(st.integers(0, 3), min_size=n_outer, max_size=n_outer))
    n_mid = sum(outer_counts)
    inner_lens = draw(st.lists(st.integers(0, 4), min_size=n_mid, max_size=n_mid))
    total = sum(inner_lens)
    data = np.arange(total, dtype=np.int64)
    return (
        np.array(outer_counts, dtype=np.int64),
        np.array(inner_lens, dtype=np.int64),
        data,
    )


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------


def _oracle(outer: np.ndarray, inner: np.ndarray, data: np.ndarray) -> ak.Array:
    """Build a nested-list awkward array from the same buffers."""
    o0 = np.concatenate([[0], np.cumsum(outer)]).astype(np.int64)
    o1 = np.concatenate([[0], np.cumsum(inner)]).astype(np.int64)
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(o0),
            ak.contents.ListOffsetArray(
                ak.index.Index64(o1),
                ak.contents.NumpyArray(data),
            ),
        )
    )


def _oracle_nested_list(outer: np.ndarray, inner: np.ndarray, data: np.ndarray) -> list:
    """Build the expected nested Python list without going through awkward."""
    result = []
    mid_offset = 0
    data_offset = 0
    for oc in outer:
        group = []
        for j in range(oc):
            il = int(inner[mid_offset + j]) if (mid_offset + j) < len(inner) else 0
            group.append(list(data[data_offset : data_offset + il]))
            data_offset += il
        mid_offset += oc
        result.append(group)
    return result


def _rag_from(outer, inner, data):
    return Ragged.from_lengths(data, (outer, inner))


# ---------------------------------------------------------------------------
# Test 1: construct, index (int), pack
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2())
def test_diff_construct_index_pack(t):
    """Construction, .to_ak(), .lengths, integer indexing, and .to_packed() all
    match the awkward oracle."""
    outer, inner, data = t
    rag = _rag_from(outer, inner, data)
    ora = _oracle(outer, inner, data)

    # Full round-trip
    assert rag.to_ak().to_list() == ora.to_list()

    # .lengths matches ak.num(ora, axis=1) (outer middle counts)
    assert rag.lengths.tolist() == ak.num(ora, axis=1).to_list()

    # Integer indexing (outer row)
    for i in range(len(outer)):
        assert rag[i].to_ak().to_list() == ora[i].to_list()

    # to_packed
    assert rag.to_packed().to_ak().to_list() == ak.to_packed(ora).to_list()


# ---------------------------------------------------------------------------
# Test 2: inner integer indexing  rag[:, k]
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2(), st.integers(0, 2))
def test_diff_inner_int(t, k):
    """rag[:, k] (peel k-th middle from every group) matches ora[:, k]."""
    outer, inner, data = t
    # Skip if any group has fewer than k+1 middles
    if np.any(outer <= k):
        return
    rag = _rag_from(outer, inner, data)
    ora = _oracle(outer, inner, data)
    assert rag[:, k].to_ak().to_list() == ora[:, k].to_list()


# ---------------------------------------------------------------------------
# Test 3: inner slice  rag[:, a:b]
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2(), st.integers(0, 3), st.integers(0, 3))
def test_diff_inner_slice(t, a, b):
    """rag[:, a:b] (per-group slice over middles) matches ora[:, a:b]."""
    outer, inner, data = t
    rag = _rag_from(outer, inner, data)
    ora = _oracle(outer, inner, data)
    assert rag[:, a:b].to_ak().to_list() == ora[:, a:b].to_list()


# ---------------------------------------------------------------------------
# Test 4: inner boolean mask  rag[:, mask]
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2())
def test_diff_inner_mask(t):
    """rag[:, mask] (mask over the global middle axis) matches a hand-built oracle.

    We build the expected nested Python list directly (without going through
    awkward masking semantics which differ) to avoid oracle misspecification.
    """
    outer, inner, data = t
    n_mid = int(np.sum(outer))
    if n_mid == 0:
        return  # nothing to mask
    mask = np.zeros(n_mid, dtype=np.bool_)
    # Toggle every other middle to exercise both True and False
    mask[::2] = True

    rag = _rag_from(outer, inner, data)

    # Build expected nested list by filtering middles per group
    expected = []
    mid_offset = 0
    data_offset_per_mid = np.concatenate([[0], np.cumsum(inner)]).astype(np.int64)
    for oc in outer:
        group = []
        for j in range(oc):
            global_mid = mid_offset + j
            if mask[global_mid]:
                lo = int(data_offset_per_mid[global_mid])
                hi = int(data_offset_per_mid[global_mid + 1])
                group.append(list(data[lo:hi]))
        mid_offset += oc
        expected.append(group)

    assert rag[:, mask].to_ak().to_list() == expected


# ---------------------------------------------------------------------------
# Test 5: outer slice  rag[a:b]
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2(), st.integers(0, 3), st.integers(0, 4))
def test_diff_outer_slice(t, a, b):
    """rag[a:b] (outer row slice) matches ora[a:b]."""
    outer, inner, data = t
    rag = _rag_from(outer, inner, data)
    ora = _oracle(outer, inner, data)
    assert rag[a:b].to_ak().to_list() == ora[a:b].to_list()


# ---------------------------------------------------------------------------
# Test 6: outer boolean mask  rag[mask]
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2())
def test_diff_outer_mask(t):
    """rag[mask] (mask over outer rows) matches ora[mask]."""
    outer, inner, data = t
    n_outer = len(outer)
    # Draw a reproducible alternating mask (avoids another strategy parameter)
    mask = np.zeros(n_outer, dtype=np.bool_)
    mask[::2] = True

    rag = _rag_from(outer, inner, data)
    ora = _oracle(outer, inner, data)
    assert rag[mask].to_ak().to_list() == ora[mask].to_list()


# ---------------------------------------------------------------------------
# Test 7: record of two numeric fields
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(nested_r2())
def test_diff_record(t):
    """Two-field record over the same R=2 nesting matches ak.zip oracle."""
    outer, inner, data = t
    data2 = data * 2

    a = _rag_from(outer, inner, data)
    b = _rag_from(outer, inner, data2)
    rec = Ragged.from_fields({"a": a, "b": b})

    ora_a = _oracle(outer, inner, data)
    ora_b = _oracle(outer, inner, data2)
    ora_rec = ak.zip({"a": ora_a, "b": ora_b}, depth_limit=1)

    assert rec.to_ak().to_list() == ora_rec.to_list()
    assert rec.to_packed().to_ak().to_list() == ak.to_packed(ora_rec).to_list()


# ---------------------------------------------------------------------------
# Test 8: string-under-axis round-trip + to_chars / to_strings
# ---------------------------------------------------------------------------


@st.composite
def string_under_axis(draw):
    """Return (lengths, bytedata) for a flat string-under-axis Ragged.

    Each 'row' is a sequence of bytes (simulating DNA/protein strings with
    variable length).  lengths is 1-D (N,) and bytedata is S1 flat array.
    """
    n = draw(st.integers(1, 6))
    row_lens = draw(st.lists(st.integers(0, 8), min_size=n, max_size=n))
    total = sum(row_lens)
    # Use printable ASCII bytes for deterministic readable content
    raw = np.frombuffer(bytes((65 + (i % 26)) for i in range(total)), dtype="S1")
    return np.array(row_lens, dtype=np.int64), raw


def _string_oracle(row_lens: np.ndarray, bytedata: np.ndarray) -> ak.Array:
    """Build an ak.Array of bytestrings from lengths + flat S1 data."""
    offsets = np.concatenate([[0], np.cumsum(row_lens)]).astype(np.int64)
    byte_leaf = ak.contents.NumpyArray(
        bytedata.view(np.uint8),
        parameters={"__array__": "byte"},
    )
    inner = ak.contents.ListOffsetArray(
        ak.index.Index64(offsets),
        byte_leaf,
        parameters={"__array__": "bytestring"},
    )
    return ak.Array(inner)


@settings(max_examples=200)
@given(string_under_axis())
def test_diff_string_under_axis(t):
    """String-under-axis Ragged round-trips through .to_ak() and the
    to_chars() → to_strings() pair matches the oracle."""
    row_lens, bytedata = t

    # Build native string-under-axis Ragged
    rag = Ragged.from_lengths(bytedata, row_lens.astype(np.uint32))
    assert rag.is_string

    ora = _string_oracle(row_lens, bytedata)

    # Direct to_ak() round-trip
    native_list = rag.to_ak().to_list()
    oracle_list = ora.to_list()
    assert native_list == oracle_list

    # to_chars() → to_strings() round-trip
    chars = rag.to_chars()
    strings = chars.to_strings()
    assert strings.is_string
    assert strings.to_ak().to_list() == oracle_list
