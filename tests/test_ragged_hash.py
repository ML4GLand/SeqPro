import hashlib

import numpy as np
import pytest

import seqpro.rag as rag_mod
from seqpro.rag._core import Ragged


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


# --- fixtures: one per string representation ---------------------------------


def _opaque_flat(strings):
    """Opaque-string leaf, flat: is_string True, shape (N,)."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    lengths = np.array([len(s) for s in strings], dtype=np.int64)
    return Ragged.from_lengths(data, lengths)


def _chars_r1(strings):
    """Chars / S1 leaf, R=1: shape (N, None), one string per row."""
    data = np.frombuffer(b"".join(strings), dtype="S1")
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in strings])]).astype(
        np.int64
    )
    return Ragged.from_offsets(data, (len(strings), None), offsets)


def _opaque_under_axis(groups):
    """Opaque strings grouped: shape (G, None), str_offsets + outer o0."""
    flat = [s for g in groups for s in g]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    str_off = np.concatenate([[0], np.cumsum([len(s) for s in flat])]).astype(np.int64)
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    return Ragged.from_offsets(data, (len(groups), None), o0, str_offsets=str_off)


def _chars_r2(groups):
    """Chars / S1 leaf, R=2: shape (G, None, None)."""
    flat = [s for g in groups for s in g]
    data = np.frombuffer(b"".join(flat), dtype="S1")
    o1 = np.concatenate([[0], np.cumsum([len(s) for s in flat])]).astype(np.int64)
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    return Ragged.from_offsets(data, (len(groups), None, None), [o0, o1])


def _digest_bytes(hl, s):
    return np.frombuffer(hl(s).digest(), dtype=np.uint8)


# --- regular (ungrouped) output: opaque flat + chars R=1 --------------------


@pytest.mark.parametrize("algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)])
@pytest.mark.parametrize("ctor", [_opaque_flat, _chars_r1])
def test_crypto_regular_matches_hashlib(algo, hl, ctor):
    strings = [b"ACGT", b"", b"hello world", b"x"]
    r = ctor(strings)
    out = r.hash(algo)
    assert not isinstance(out, Ragged)
    assert out.dtype == np.uint8
    assert out.shape == (len(strings), hl().digest_size)
    expected = np.stack([_digest_bytes(hl, s) for s in strings])
    np.testing.assert_array_equal(out, expected)


# --- grouped output: opaque-under-axis + chars R=2 --------------------------


@pytest.mark.parametrize("algo,hl", [("md5", hashlib.md5), ("sha256", hashlib.sha256)])
@pytest.mark.parametrize("ctor", [_opaque_under_axis, _chars_r2])
def test_crypto_grouped_returns_ragged(algo, hl, ctor):
    groups = [[b"AA", b"B"], [b"CCC", b"DDDD"]]
    flat = [s for g in groups for s in g]
    r = ctor(groups)
    out = r.hash(algo)
    assert isinstance(out, Ragged)
    # output offsets identical to the input outer offsets
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    np.testing.assert_array_equal(out.offsets, o0)
    # per-string digests correct, in packed order
    packed = out.to_packed().data.reshape(len(flat), hl().digest_size)
    expected = np.stack([_digest_bytes(hl, s) for s in flat])
    np.testing.assert_array_equal(packed, expected)


@pytest.mark.parametrize("ctor", [_chars_r2, _opaque_under_axis])
def test_rapidhash_grouped_returns_ragged_uint64(ctor):
    groups = [[b"AA", b"B"], [b"CCC"]]
    r = ctor(groups)
    out = r.hash("rapidhash")
    assert isinstance(out, Ragged)
    assert out.to_packed().data.dtype == np.uint64
    o0 = np.cumsum([0] + [len(g) for g in groups]).astype(np.int64)
    np.testing.assert_array_equal(out.offsets, o0)


def test_leading_fixed_dims_regular_output():
    """Exercises the reshape branch in hash() for inputs with leading fixed dims.

    Constructs a chars Ragged with shape (B, M, None) — one ragged dim, two
    leading fixed dims — and confirms the output is reshaped to (B, M, 16).
    """
    B, M = 2, 2
    strings = [b"ACGT", b"hello", b"foo", b"bar"]
    data = np.frombuffer(b"".join(strings), dtype="S1")
    offsets = np.concatenate([[0], np.cumsum([len(s) for s in strings])]).astype(
        np.int64
    )
    r = Ragged.from_offsets(data, (B, M, None), offsets)
    out = r.hash("md5")
    assert not isinstance(out, Ragged), "expected a regular NDArray, got Ragged"
    assert out.shape == (B, M, 16), f"expected ({B}, {M}, 16), got {out.shape}"
    flat = out.reshape(B * M, 16)
    for i, s in enumerate(strings):
        expected = np.frombuffer(hashlib.md5(s).digest(), dtype=np.uint8)
        np.testing.assert_array_equal(flat[i], expected)


# --- equivalences and edges -------------------------------------------------


def test_free_function_matches_method():
    r = _chars_r1([b"AC", b"GT"])
    np.testing.assert_array_equal(rag_mod.hash(r, "md5"), r.hash("md5"))


def test_unpacked_input_is_packed_internally():
    r = _chars_r1([b"AAA", b"B", b"CC", b"DDDD"])
    sub = r[np.array([3, 0, 2])]  # gather -> non-contiguous offsets
    out = sub.hash("md5")
    expected = np.stack(
        [_digest_bytes(hashlib.md5, s) for s in [b"DDDD", b"AAA", b"CC"]]
    )
    np.testing.assert_array_equal(out, expected)


def test_empty_container():
    r = _chars_r1([])
    out = r.hash("sha256")
    assert out.shape == (0, 32)


# --- error handling ---------------------------------------------------------


def test_numeric_ragged_raises():
    r = Ragged.from_lengths(np.arange(6, dtype=np.int32), np.array([3, 3]))
    with pytest.raises(ValueError, match="string/char"):
        r.hash("md5")


def test_record_ragged_raises():
    a = Ragged.from_lengths(np.frombuffer(b"catdog", "S1"), np.array([3, 3]))
    b = Ragged.from_lengths(np.frombuffer(b"birdho", "S1"), np.array([3, 3]))
    rec = Ragged.from_fields({"x": a, "y": b})
    with pytest.raises(NotImplementedError):
        rec.hash("md5")


def test_unknown_algo_raises_in_python():
    r = _chars_r1([b"AC"])
    with pytest.raises(ValueError, match="unknown algo"):
        r.hash("sha1")


def test_seed_on_crypto_raises():
    r = _chars_r1([b"AC"])
    with pytest.raises(ValueError, match="seed is only valid"):
        r.hash("md5", seed=1)
