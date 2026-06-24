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
    new = rag[sl]                 # fast path (gate fires)
    old = rag._getitem(sl)        # bypass gate -> original gather path
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


@pytest.mark.parametrize("sl", [slice(1, 4), slice(0, 5), slice(2, 2),
                                slice(3, 1), slice(None), slice(-2, None)])
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
    assert np.shares_memory(out.data, rag.data)   # narrowed view, not a copy
    assert out.data.shape[0] == 2 + 5             # only rows 1,2
