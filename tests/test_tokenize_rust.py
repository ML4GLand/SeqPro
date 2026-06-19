import numpy as np
import pytest

import seqpro as sp

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
