"""Issue #73: fixed axes between the leading axis and two ragged axes.

Shape ``(b, p, ~v, ~w)`` is the layout genvarloader's ``_FlatWindow.to_ragged()``
returns for variant windows. Indexing and ``to_ak()`` must honor the fixed ``p``
axis. The oracle is awkward indexing on the equivalent nested Python list.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from seqpro.rag import Ragged

# (b=2, p=2, ~v, ~w)
NESTED = [
    [[[0, 1, 2], [3, 4]], [[5]]],
    [[], [[6, 7], [8], [9, 10, 11]]],
]
O0 = np.array([0, 2, 3, 3, 6])
O1 = np.array([0, 3, 5, 6, 8, 9, 12])


def _rag(scale: int = 1) -> Ragged[Any]:
    data = np.arange(12, dtype=np.int64) * scale
    return Ragged.from_offsets(data, (2, 2, None, None), [O0, O1])


def _tolist(x: Any) -> Any:
    if isinstance(x, Ragged):
        return x.to_ak().tolist()
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    return np.asarray(x).tolist()


KEYS = [
    0,
    -1,
    slice(1, None),
    slice(None, None, -1),
    [1, 0],
    np.array([False, True]),
    (0, 1),
    (1, 0),
    (-1, -1),
    (slice(None), 1),
    (slice(None), [1, 0]),
    ([1, 0], 0),
    (0, slice(None)),
    ([0, 1], [1, 0]),
    (np.array([True, False]), slice(None)),
]


def test_issue_repro():
    r = Ragged.from_offsets(
        np.arange(9, dtype=np.uint8),
        (2, 1, None, None),
        [np.array([0, 2, 3]), np.array([0, 3, 5, 9])],
    )
    assert r[0].shape == (1, None, None)
    assert r[0, 0].shape == (2, None)
    assert _tolist(r[0, 0]) == [[0, 1, 2], [3, 4]]
    assert str(r.to_ak().type) == "2 * 1 * var * var * uint8"


def test_to_ak_type():
    assert str(_rag().to_ak().type) == "2 * 2 * var * var * int64"
    assert _rag().to_ak().tolist() == NESTED


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_getitem_matches_awkward(key: Any):
    expected = ak.Array(NESTED)[key].tolist()
    assert _tolist(_rag()[key]) == expected


@pytest.mark.parametrize("key", KEYS, ids=repr)
def test_getitem_on_gathered_matches_awkward(key: Any):
    # Gathered (2, N) outer offsets must index the same as packed ones.
    gathered = _rag()[[1, 0]]
    expected = ak.Array(NESTED)[[1, 0]][key].tolist()
    assert _tolist(gathered[key]) == expected


@pytest.mark.parametrize(
    ("key", "shape"),
    [
        (0, (2, None, None)),
        ((0, 1), (1, None)),
        ((1, 0), (0, None)),
        ((slice(None), 1), (2, None, None)),
        ([1, 0], (2, 2, None, None)),
        (([0, 1], [1, 0]), (2, None, None)),
    ],
    ids=repr,
)
def test_getitem_shape(key: Any, shape: tuple[int | None, ...]):
    assert _rag()[key].shape == shape


def test_chained_equals_tuple():
    r = _rag()
    assert _tolist(r[1][1]) == _tolist(r[1, 1])


def test_record_getitem_matches_awkward():
    rec = Ragged.from_fields({"a": _rag(), "b": _rag(10)})
    oracle = {"a": ak.Array(NESTED), "b": ak.Array(NESTED) * 10}
    for key in [0, [1, 0], (slice(None), 1), (0, slice(None))]:
        got = rec[key]
        assert isinstance(got, Ragged)
        for name, arr in oracle.items():
            assert _tolist(got[name]) == arr[key].tolist(), (key, name)
    peeled = rec[1, 1]
    assert isinstance(peeled, dict)
    for name, arr in oracle.items():
        assert _tolist(peeled[name]) == arr[1, 1].tolist(), name


@pytest.mark.parametrize("shape", [(2, None), (2, None, None), (2, 2, None, None)])
def test_record_to_ak_is_leaf_level(shape: tuple[int | None, ...]):
    # One record depth for every rag_dim (#75): records sit at the leaf, so
    # indexing then converting equals converting then indexing.
    if shape == (2, None):
        a = Ragged.from_offsets(np.arange(5), shape, np.array([0, 3, 5]))
    elif shape == (2, None, None):
        a = Ragged.from_offsets(
            np.arange(9), shape, [np.array([0, 2, 3]), np.array([0, 3, 5, 9])]
        )
    else:
        a = _rag()
    rec = Ragged.from_fields({"a": a, "b": a * 10})
    out = rec.to_ak()
    assert str(out.type).endswith("{a: int64, b: int64}")
    assert out.tolist() == ak.zip({"a": a.to_ak(), "b": (a * 10).to_ak()}).tolist()
    got = rec[[1, 0]]
    assert isinstance(got, Ragged)
    assert got.to_ak().tolist() == out[[1, 0]].tolist()


def test_record_to_ak_trailing_dim_stays_in_leaf():
    off = np.array([0, 2, 3])
    a = Ragged.from_offsets(np.arange(6).reshape(3, 2), (2, None, 2), off)
    b = Ragged.from_offsets(np.arange(3), (2, None), off)
    out = Ragged.from_fields({"a": a, "b": b}).to_ak()
    assert str(out.type) == "2 * var * {a: 2 * int64, b: int64}"
    assert (
        out.tolist() == ak.zip({"a": a.to_ak(), "b": b.to_ak()}, depth_limit=2).tolist()
    )
