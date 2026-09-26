"""Issue #74: tuple keys after the leading axes index the right axis.

Each key in a tuple targets the next *output* axis, as in NumPy/awkward. Keys
after a non-int key used to be applied to axis 0 of the intermediate result.
The oracle is awkward indexing on the equivalent array.
"""

from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np
import pytest

from seqpro.rag import Ragged

S = slice(None)

A_LIST = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]  # (3, ~)
B_LIST = [[[0, 1, 2], [3, 4]], [[5, 6, 7, 8], [9, 10, 11]]]  # (2, 2, ~)
C_LIST = [
    [[0, 1], [2, 3], [4, 5]],
    [[6, 7], [8, 9]],
    [[10, 11], [12, 13], [14, 15], [16, 17]],
]  # (3, ~, 2)
N_LIST = [[[0, 1, 2], [3, 4]], [[5, 6, 7, 8]]]  # (2, ~, ~)
F_LIST = [  # (2, 2, ~, ~)
    [[[0, 1, 2], [3, 4]], [[5]]],
    [[], [[6, 7], [8], [9, 10, 11]]],
]


def _a() -> Ragged[Any]:
    return Ragged.from_offsets(np.arange(9), (3, None), np.array([0, 3, 5, 9]))


def _b() -> Ragged[Any]:
    return Ragged.from_offsets(np.arange(12), (2, 2, None), np.array([0, 3, 5, 9, 12]))


def _c() -> Ragged[Any]:
    data = np.arange(18).reshape(9, 2)
    return Ragged.from_offsets(data, (3, None, 2), np.array([0, 3, 5, 9]))


def _n() -> Ragged[Any]:
    return Ragged.from_offsets(
        np.arange(9), (2, None, None), [np.array([0, 2, 3]), np.array([0, 3, 5, 9])]
    )


def _f() -> Ragged[Any]:
    return Ragged.from_offsets(
        np.arange(12),
        (2, 2, None, None),
        [np.array([0, 2, 3, 3, 6]), np.array([0, 3, 5, 6, 8, 9, 12])],
    )


FIXTURES = {
    "a": (_a, A_LIST),
    "b": (_b, B_LIST),
    "c": (_c, C_LIST),
    "n": (_n, N_LIST),
    "f": (_f, F_LIST),
}

CASES = [
    ("a", (S, 0)),
    ("a", (S, -1)),
    ("a", (S, slice(1, None))),
    ("a", (S, slice(None, 2))),
    ("a", (S, slice(-2, None))),
    ("a", (S, slice(-5, -1))),
    ("a", ([2, 0], 0)),
    ("a", (slice(1, None), 1)),
    ("a", (np.array([True, False, True]), -1)),
    ("a", (0, 1)),
    ("a", (S, 2)),  # row 1 too short -> IndexError
    ("b", (S, S, 0)),
    ("b", (0, S, 0)),
    ("b", (S, 1, -1)),
    ("b", ([1, 0], S, slice(1, None))),
    ("b", (1, 0, 1)),
    ("b", (S, 0, slice(None, 2))),
    ("c", (S, 0)),
    ("c", (S, S, 1)),
    ("c", (0, S, 1)),
    ("c", (S, 0, 1)),
    ("c", (S, slice(1, None), 0)),
    ("c", (0, 1, 1)),
    ("c", (S, -1, S)),
    ("c", (S, S, [1, 0])),
    ("n", (S, 0)),
    ("n", (S, 0, 1)),
    ("n", (S, S, 0)),
    ("n", (S, slice(1, None), slice(1, None))),
    ("n", (0, 1, 1)),
    ("n", (S, -1, -1)),
    ("n", (S, 0, slice(None, 2))),
    ("n", (S, S, -1)),
    ("f", (S, 1, 0)),
    ("f", (S, 1, 0, -1)),
    ("f", (S, S, S, 0)),
    ("f", (0, S, S, slice(1, None))),
    ("f", (S, S, S, slice(None, 1))),
    ("f", (S, S, 0)),  # b1p0 has no variants -> IndexError
    ("f", (1, 1, S, 0)),
]


def _tolist(x: Any) -> Any:
    if isinstance(x, Ragged):
        return x.to_ak().tolist()
    if isinstance(x, dict):
        return {k: _tolist(v) for k, v in x.items()}
    if isinstance(x, bytes):
        return x
    return np.asarray(x).tolist()


@pytest.mark.parametrize(("name", "key"), CASES, ids=repr)
def test_tuple_index_matches_awkward(name: str, key: tuple[Any, ...]):
    make, nested = FIXTURES[name]
    try:
        expected = ak.Array(nested)[key]
    except (IndexError, ValueError):
        with pytest.raises(IndexError):
            make()[key]
        return
    expected = expected.tolist() if isinstance(expected, ak.Array) else expected
    assert _tolist(make()[key]) == expected


@pytest.mark.parametrize(("name", "key"), CASES, ids=repr)
def test_tuple_index_on_gathered(name: str, key: tuple[Any, ...]):
    # Gathered (2, N) offsets must index the same as packed ones.
    make, nested = FIXTURES[name]
    perm = [1, 0] if name in {"b", "n", "f"} else [2, 0, 1]
    try:
        expected = ak.Array(nested)[perm][key]
    except (IndexError, ValueError):
        with pytest.raises(IndexError):
            make()[perm][key]
        return
    expected = expected.tolist() if isinstance(expected, ak.Array) else expected
    assert _tolist(make()[perm][key]) == expected


def _strings() -> Ragged[Any]:
    # (3, ~) opaque strings: ["A", "CG"], ["T"], ["GA", "", "C"]
    data = np.frombuffer(b"ACGTGAC", "S1")
    str_off = np.array([0, 1, 3, 4, 6, 6, 7])
    return Ragged.from_offsets(
        data, (3, None), np.array([0, 2, 3, 6]), str_offsets=str_off
    )


@pytest.mark.parametrize(
    "key", [(S, 0), (S, -1), (S, slice(1, None)), ([2, 0], 0)], ids=repr
)
def test_string_under_axis(key: tuple[Any, ...]):
    rag = _strings()
    assert _tolist(rag[key]) == rag.to_ak()[key].tolist()


@pytest.mark.parametrize("name", ["a", "b", "n"])
@pytest.mark.parametrize("key", [(S, 0), (S, slice(1, None)), (S, -1)], ids=repr)
def test_record(name: str, key: tuple[Any, ...]):
    make, nested = FIXTURES[name]
    rec = Ragged.from_fields({"x": make(), "y": make() * 10})
    oracle = {"x": ak.Array(nested), "y": ak.Array(nested) * 10}
    got = rec[key]
    for field, arr in oracle.items():
        assert _tolist(got[field]) == arr[key].tolist(), field


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("a", (S, slice(None, None, 2))),  # stepped ragged slice
        ("a", (S, [0, 1])),  # fancy key on a ragged axis
        ("a", (S, np.array([True, False]))),
        ("n", (S, S, [0, 1])),
    ],
    ids=repr,
)
def test_unsupported_raises(name: str, key: tuple[Any, ...]):
    with pytest.raises(NotImplementedError):
        FIXTURES[name][0]()[key]


def test_record_trailing_key_raises():
    rec = Ragged.from_fields({"x": _c(), "y": _c()})
    with pytest.raises(NotImplementedError):
        rec[:, :, 0]


def test_newaxis_with_ragged_key():
    # np.newaxis positions are still tracked around ragged-axis keys.
    got = _a()[None, :, 0]
    assert _tolist(got) == [[0, 3, 5]]
