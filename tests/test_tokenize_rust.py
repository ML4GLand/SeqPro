import numpy as np
import pytest

import seqpro as sp
from seqpro.rag import Ragged

TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
UNK = 7


def _ref_tokenize(seqs, token_map, unk):
    """Pure-NumPy reference (LUT + np.take), independent of the impl under test."""
    keys = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    vals = np.array(list(token_map.values()), dtype=np.int32)
    lut = np.full(256, np.int32(unk), dtype=np.int32)
    lut[keys] = vals
    arr = sp.cast_seqs(seqs)
    return np.take(lut, arr.view(np.uint8))


@pytest.mark.parametrize("parallel", [None, False, True])
@pytest.mark.parametrize("n", [0, 1, 5, 100, 50_000])
def test_tokenize_matches_reference_1d(n, parallel):
    rng = np.random.default_rng(0)
    seqs = rng.choice(np.frombuffer(b"ACGTN", "S1"), size=n)
    got = sp.tokenize(seqs, TOKEN_MAP, UNK, parallel=parallel)
    assert got.dtype == np.int32
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))


def test_tokenize_multidim_preserves_shape():
    rng = np.random.default_rng(1)
    seqs = rng.choice(np.frombuffer(b"ACGT", "S1"), size=(4, 8, 3))
    got = sp.tokenize(seqs, TOKEN_MAP, UNK)
    assert got.shape == (4, 8, 3)
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))


def test_tokenize_ragged_matches_reference():
    data = np.frombuffer(b"ACGTACG", "S1")
    offsets = np.array([0, 3, 7], dtype=np.int64)
    rag = Ragged.from_offsets(data, (2, None), offsets)
    got = sp.tokenize(rag, TOKEN_MAP, UNK)
    np.testing.assert_array_equal(got.data, _ref_tokenize(data, TOKEN_MAP, UNK))
    np.testing.assert_array_equal(got.offsets, offsets)


def test_tokenize_out_dtype_typeerror():
    seqs = np.frombuffer(b"ACGT", "S1")
    bad = np.empty(4, dtype=np.int64)
    with pytest.raises(TypeError, match="int32"):
        sp.tokenize(seqs, TOKEN_MAP, UNK, out=bad)


def test_tokenize_parallel_true_strided_out_valueerror():
    seqs = np.frombuffer(b"ACGTACGT", "S1")
    out = np.empty(16, dtype=np.int32)[::2]  # non-contiguous
    assert not out.flags.c_contiguous
    with pytest.raises(ValueError, match="C-contiguous"):
        sp.tokenize(seqs, TOKEN_MAP, UNK, out=out, parallel=True)


def test_tokenize_out_strided_serial_ok():
    seqs = np.frombuffer(b"ACGTACGT", "S1")
    out = np.empty(16, dtype=np.int32)[::2]
    got = sp.tokenize(seqs, TOKEN_MAP, UNK, out=out, parallel=False)
    np.testing.assert_array_equal(got, _ref_tokenize(seqs, TOKEN_MAP, UNK))
