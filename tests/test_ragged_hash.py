import hashlib

import numpy as np
import pytest


def _pack(strings):
    """Return (uint8 data buffer, int64 offsets) for a list of byte strings."""
    data = np.frombuffer(b"".join(strings), dtype=np.uint8)
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in strings])]).astype(
        np.int64
    )
    return np.ascontiguousarray(data), np.ascontiguousarray(offsets)


@pytest.mark.parametrize("algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)])
def test_kernel_crypto_matches_hashlib(algo, hl):
    from seqpro.seqpro import _ragged_hash

    strings = [b"ACGT", b"", b"hello world", b"x", b"the quick brown fox"]
    data, offsets = _pack(strings)
    out = _ragged_hash(data, offsets, algo, None)
    assert out.dtype == np.uint8
    assert out.shape == (len(strings), hl().digest_size)
    expected = np.stack(
        [np.frombuffer(hl(s).digest(), dtype=np.uint8) for s in strings]
    )
    np.testing.assert_array_equal(out, expected)


def test_kernel_rapidhash_properties():
    from seqpro.seqpro import _ragged_hash

    strings = [b"abc", b"abc", b"abd", b""]
    data, offsets = _pack(strings)
    out = _ragged_hash(data, offsets, "rapidhash", None)
    assert out.dtype == np.uint64
    assert out.shape == (len(strings),)
    assert out[0] == out[1]  # identical input -> identical hash
    assert out[0] != out[2]  # different input -> different hash

    seeded = _ragged_hash(data, offsets, "rapidhash", 12345)
    assert seeded[0] != out[0]  # seed changes output
    again = _ragged_hash(data, offsets, "rapidhash", 12345)
    np.testing.assert_array_equal(seeded, again)  # deterministic


def test_kernel_unknown_algo_raises():
    from seqpro.seqpro import _ragged_hash

    data, offsets = _pack([b"AC"])
    with pytest.raises(ValueError, match="unknown algo"):
        _ragged_hash(data, offsets, "sha1", None)
