import numpy as np
import pytest
from seqpro.rag import Ragged
from seqpro.rag._utils import lengths_to_offsets


def to_py(x):
    """Universal, layout-agnostic materialization to nested python lists/bytes.
    Recurses by peeling the outer (always-int) axis; never calls to_packed, so it
    works on string and string-record results too."""
    if isinstance(x, dict):
        return {k: to_py(v) for k, v in x.items()}
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    if isinstance(x, Ragged):
        return [to_py(x[i]) for i in range(len(x))]
    return np.asarray(x).tolist()


def assert_slice_parity(rag, sl):
    new = rag[sl]  # fast path (gate fires)
    old = rag._getitem(sl)  # bypass gate -> original gather path
    if isinstance(new, Ragged):
        assert new.is_contiguous, "fast-path result must be contiguous"
    assert to_py(new) == to_py(old)


def _r1(lengths, dtype=np.int32, shape=None):
    lengths = np.asarray(lengths, np.int64)
    off = lengths_to_offsets(lengths)
    total = int(off[-1])
    data = np.arange(total, dtype=dtype)
    shp = shape if shape is not None else (len(lengths), None)
    return Ragged.from_offsets(data, shp, off)


@pytest.mark.parametrize(
    "sl",
    [slice(1, 4), slice(0, 5), slice(2, 2), slice(3, 1), slice(None), slice(-2, None)],
)
def test_r1_simple_parity(sl):
    assert_slice_parity(_r1([4, 2, 5, 3, 6]), sl)


def test_r1_multidim_parity():
    # shape (B=3, P=2, None): 6 segments.
    # _getitem produces 2D (non-contiguous) offsets for multidim inner shapes, so the
    # parity oracle is per-element equality (rag[i]) rather than _getitem(slice).
    rag = _r1([4, 2, 5, 3, 6, 1], shape=(3, 2, None))
    new = rag[slice(1, 3)]
    assert new.is_contiguous
    assert to_py(new) == [to_py(rag[1]), to_py(rag[2])]


def test_r1_trailing_dim_parity():
    # OHE-style trailing fixed dim: shape (N, None, 4)
    off = lengths_to_offsets(np.array([3, 1, 2], np.int64))
    data = np.arange(int(off[-1]) * 4, dtype=np.uint8).reshape(-1, 4)
    rag = Ragged.from_offsets(data, (3, None, 4), off)
    assert_slice_parity(rag, slice(0, 2))


def test_r1_result_is_narrowed_view():
    rag = _r1([4, 2, 5, 3, 6])
    out = rag[1:3]
    assert out.is_contiguous
    assert out.offsets[0] == 0
    assert np.shares_memory(out.data, rag.data)  # narrowed view, not a copy
    assert out.data.shape[0] == 2 + 5  # only rows 1,2


def _r2(group_counts, inner_lengths, dtype=np.int32):
    """R=2 array: outer groups -> middle segments -> data."""
    o0 = lengths_to_offsets(np.asarray(group_counts, np.int64))
    o1 = lengths_to_offsets(np.asarray(inner_lengths, np.int64))
    data = np.arange(int(o1[-1]), dtype=dtype)
    n_outer = len(group_counts)
    return Ragged.from_offsets(data, (n_outer, None, None), [o0, o1])


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_r2_parity(sl):
    # 3 groups with 2,1,2 middle segments; middles have these lengths
    rag = _r2([2, 1, 2], [4, 3, 5, 2, 6])
    assert_slice_parity(rag, sl)


def _str_flat(strings):
    """Flat opaque-string collection: shape (N,)."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    so = lengths_to_offsets(np.array([len(s) for s in strings], np.int64))
    return Ragged.from_offsets(data, (len(strings),), so, str_offsets=so)


def _str_under_axis(rows):
    """String-under-axis: shape (N, None); each row is a list of byte strings."""
    flat = [s for row in rows for s in row]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    so = lengths_to_offsets(np.array([len(s) for s in flat], np.int64))
    o0 = lengths_to_offsets(np.array([len(r) for r in rows], np.int64))
    return Ragged.from_offsets(data, (len(rows), None), o0, str_offsets=so)


@pytest.mark.parametrize("sl", [slice(1, 3), slice(0, 4), slice(2, 2), slice(None)])
def test_string_flat_parity(sl):
    assert_slice_parity(_str_flat([b"AC", b"GGG", b"T", b"CCGT"]), sl)


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2)])
def test_string_under_axis_parity(sl):
    rows = [[b"AC", b"G"], [b"TT"], [b"CCG", b"A", b"T"]]
    assert_slice_parity(_str_under_axis(rows), sl)


def _record_r1():
    """Record R=1 with a numeric field and a string-under-axis field sharing O0.
    shape (3, None); rows have 2,1,3 variants."""
    o0 = lengths_to_offsets(np.array([2, 1, 3], np.int64))
    n_var = int(o0[-1])
    start_field = Ragged.from_offsets(np.arange(n_var, dtype=np.int32), (3, None), o0)
    alts = [b"AC", b"G", b"T", b"CC", b"A", b"GG"]  # one per variant
    sdata = np.frombuffer(b"".join(alts), dtype="S1")
    sso = lengths_to_offsets(np.array([len(a) for a in alts], np.int64))
    alt_field = Ragged.from_offsets(sdata, (3, None), o0, str_offsets=sso)
    return Ragged.from_fields({"start": start_field, "alt": alt_field})


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_record_r1_parity(sl):
    assert_slice_parity(_record_r1(), sl)


def _record_r2():
    """Record R=2: two numeric fields sharing [O0, O1]. 3 outer groups."""
    o0 = lengths_to_offsets(np.array([2, 1, 2], np.int64))  # middles per group
    o1 = lengths_to_offsets(np.array([3, 2, 4, 1, 5], np.int64))  # data per middle
    n = int(o1[-1])
    a = Ragged.from_offsets(np.arange(n, dtype=np.int32), (3, None, None), [o0, o1])
    b = Ragged.from_offsets(np.arange(n, dtype=np.float32), (3, None, None), [o0, o1])
    return Ragged.from_fields({"a": a, "b": b})


@pytest.mark.parametrize("sl", [slice(0, 2), slice(1, 3), slice(2, 2), slice(None)])
def test_record_r2_parity(sl):
    assert_slice_parity(_record_r2(), sl)


def _record_r2_multidim_outer():
    """Record R=2 with a middle fixed dim: shape (B=2, P=2, None, None).
    _outer_n_inner() returns 2; there are B*P=4 offset entries in o0."""
    o0 = lengths_to_offsets(np.array([1, 2, 1, 1], np.int64))  # 4 entries for B*P
    o1 = lengths_to_offsets(np.array([3, 2, 4, 1, 5], np.int64))
    n = int(o1[-1])
    a = Ragged.from_offsets(np.arange(n, dtype=np.int32), (2, 2, None, None), [o0, o1])
    b = Ragged.from_offsets(
        np.arange(n, dtype=np.float32), (2, 2, None, None), [o0, o1]
    )
    return Ragged.from_fields({"a": a, "b": b})


def test_record_r2_multidim_outer_preserves_middle_dim():
    """Slicing a (B, P, None, None) record must keep P in the result shape."""
    rag = _record_r2_multidim_outer()
    sl = slice(0, 1)
    result = rag[sl]
    # shape must be (1, 2, None, None) — NOT (1, None, None)
    assert result.shape == (1, 2, None, None), (
        f"expected (1, 2, None, None), got {result.shape}"
    )
    assert result.is_contiguous
    # per-element ground-truth oracle (old gather path broken for multidim-outer)
    assert to_py(result) == [to_py(rag[i]) for i in range(0, 1)]

    # also check full-range slice
    result_all = rag[slice(None)]
    assert result_all.shape == (2, 2, None, None)
    assert to_py(result_all) == [to_py(rag[i]) for i in range(2)]


def test_to_numpy_validate_false_matches_true():
    off = np.arange(4 + 1, dtype=np.int64) * 3  # 4 uniform rows of len 3
    data = np.arange(4 * 3, dtype=np.int32)
    rag = Ragged.from_offsets(data, (4, None), off)
    a = rag.to_numpy()  # validate=True (default)
    b = rag.to_numpy(validate=False)  # trust-me
    np.testing.assert_array_equal(a, b)
    assert np.shares_memory(b, rag.data)  # zero-copy reshape


def test_to_numpy_validate_true_still_raises_on_jagged():
    rag = _r1([4, 2, 5])
    with pytest.raises(ValueError):
        rag.to_numpy()  # jagged -> raise (unchanged default)


def test_to_numpy_validate_false_multidim():
    rag = _r1([3, 3, 3, 3], shape=(2, 2, None))  # (2,2,None) uniform len 3
    out = rag.to_numpy(validate=False)
    assert out.shape == (2, 2, 3)


def test_from_offsets_validate_true_raises_on_size_mismatch():
    data = np.arange(10, dtype=np.int32)
    bad = np.array([0, 4, 20], np.int64)  # implies 20 elements, data has 10
    with pytest.raises(ValueError):
        Ragged.from_offsets(data, (2, None), bad, validate=True)


def test_from_offsets_default_skips_size_check():
    data = np.arange(10, dtype=np.int32)
    bad = np.array([0, 4, 20], np.int64)
    # default validate=False: constructs without raising (caller's contract)
    rag = Ragged.from_offsets(data, (2, None), bad)
    assert rag.shape == (2, None)


def test_from_offsets_preserves_already_canonical_offsets():
    data = np.arange(7, dtype=np.int32)
    off = np.array([0, 4, 7], np.int64)  # already C-contiguous int64
    rag = Ragged.from_offsets(data, (2, None), off)
    assert rag.offsets is off  # no ascontiguousarray copy
