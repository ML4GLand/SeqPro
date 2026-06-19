from typing import List

import numpy as np
import pytest
import seqpro as sp
from hypothesis import given, note
from seqpro.rag import Ragged
from hypothesis import strategies as st

seq = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=32, max_codepoint=126),
    min_size=1,
)
seqs = st.lists(seq, min_size=1, max_size=10)


@given(seq)
def test_roundtrip(seq: str):
    chars = set(seq)
    for _ in range(min(len(chars), 1)):
        chars.pop()
    token_map = dict(zip(chars, range(len(chars))))
    cast_seq = sp.cast_seqs(seq)
    tokens = sp.tokenize(cast_seq, token_map, unknown_token=-1)
    decoded = sp.decode_tokens(tokens, token_map)
    mod_seq = cast_seq.copy()
    mod_seq[np.isin(cast_seq, [c.encode("ascii") for c in chars], invert=True)] = b"N"

    note(f"{seq}, {chars}, {token_map}, {cast_seq}")

    np.testing.assert_equal(decoded, mod_seq)


@given(seq)
def test_tokenize(seq: str):
    chars = set(seq)
    for _ in range(min(len(chars), 1)):
        chars.pop()
    token_map = dict(zip(chars, range(len(chars))))
    cast_seq = sp.cast_seqs(seq)
    tokens = sp.tokenize(cast_seq, token_map, unknown_token=-1)

    note(f"{seq}, {chars}, {token_map}, {cast_seq}, {tokens}")

    for char, token in token_map.items():
        assert np.all(tokens[cast_seq == char.encode("ascii")] == token)

    assert (
        np.isin(cast_seq, [c.encode("ascii") for c in chars], invert=True).sum()
        == (tokens == -1).sum()
    )


@given(seqs)
def test_roundtrip_multiple(seqs: List[str]):
    chars = set("".join(seqs))
    for _ in range(min(len(chars), 1)):
        chars.pop()
    token_map = dict(zip(chars, range(len(chars))))
    cast_seq = sp.cast_seqs(seqs)
    tokens = sp.tokenize(cast_seq, token_map, unknown_token=-1)
    decoded = sp.decode_tokens(tokens, token_map)
    mod_seq = cast_seq.copy()
    mod_seq[np.isin(cast_seq, [c.encode("ascii") for c in chars], invert=True)] = b"N"

    note(f"{seq}, {chars}, {token_map}, {cast_seq}")

    np.testing.assert_equal(decoded, mod_seq)


@given(seqs)
def test_tokenize_multiple(seqs: List[str]):
    chars = set("".join(seqs))
    for _ in range(min(len(chars), 1)):
        chars.pop()
    token_map = dict(zip(chars, range(len(chars))))
    cast_seq = sp.cast_seqs(seqs)
    tokens = sp.tokenize(cast_seq, token_map, unknown_token=-1)

    note(f"{seq}, {chars}, {token_map}, {cast_seq}, {tokens}")

    for char, token in token_map.items():
        assert np.all(tokens[cast_seq == char.encode("ascii")] == token)

    assert (
        np.isin(cast_seq, [c.encode("ascii") for c in chars], invert=True).sum()
        == (tokens == -1).sum()
    )


def test_tokenize_ragged():
    seqs = ["ACGT", "AC", "GTTTT"]
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    result = sp.DNA.tokenize(rag, token_map, unknown_token=-1)

    assert isinstance(result, Ragged)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result.lengths.ravel(), lengths)
    np.testing.assert_array_equal(result.data[:4], [0, 1, 2, 3])  # "ACGT"
    np.testing.assert_array_equal(result.data[4:6], [0, 1])  # "AC"
    np.testing.assert_array_equal(result.data[6:], [2, 3, 3, 3, 3])  # "GTTTT"


def test_tokenize_ragged_unknown():
    seqs = ["ACGN", "NT"]
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    result = sp.tokenize(rag, token_map, unknown_token=-1)

    assert isinstance(result, Ragged)
    np.testing.assert_array_equal(result.data, [0, 1, 2, -1, -1, 3])


def test_decode_tokens_ragged():
    seqs = ["ACGT", "AC", "GTTTT"]
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    tokens = sp.DNA.tokenize(rag, token_map, unknown_token=-1)
    result = sp.DNA.decode_tokens(tokens, token_map)

    assert isinstance(result, Ragged)
    assert result.dtype == np.dtype("S1")
    np.testing.assert_array_equal(result.lengths.ravel(), lengths)
    np.testing.assert_array_equal(result.data, rag.data)


def test_decode_tokens_ragged_module_level():
    seqs = ["ACG", "T"]
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    tokens = sp.tokenize(rag, token_map, unknown_token=-1)
    result = sp.decode_tokens(tokens, token_map)

    assert isinstance(result, Ragged)
    np.testing.assert_array_equal(result.data, rag.data)


def test_tokenize_matches_gufunc_reference():
    """LUT output must be byte-for-byte identical to the linear-scan gufunc."""
    from seqpro._numba import gufunc_tokenize

    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    unknown_token = 4

    def reference(cast_seq):
        source = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
        target = np.array(list(token_map.values()), dtype=np.int32)
        return gufunc_tokenize(
            cast_seq.view(np.uint8), source, target, np.int32(unknown_token)
        )

    # Dense 2-D, with known + unknown ("N", "x") characters.
    seqs = ["ACGTN", "TTxAC", "GGGGG"]
    cast = sp.cast_seqs(seqs)  # (3, 5) S1
    expected = reference(cast)
    result = sp.tokenize(cast, token_map, unknown_token=unknown_token)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int32

    # out= path: result written in place, equals expected, returns same buffer.
    out = np.empty(cast.shape, dtype=np.int32)
    returned = sp.tokenize(cast, token_map, unknown_token=unknown_token, out=out)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(returned, expected)
    assert returned is out  # out= must return the same buffer

    # Ragged path.
    rag_seqs = ["ACGTN", "TTxAC", "GGGGG"]
    data = np.frombuffer("".join(rag_seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in rag_seqs])
    rag = Ragged.from_lengths(data, lengths)
    rag_result = sp.tokenize(rag, token_map, unknown_token=unknown_token)
    flat_expected = reference(
        np.frombuffer(b"".join(s.encode("ascii") for s in rag_seqs), dtype="S1")
    )
    np.testing.assert_array_equal(rag_result.data, flat_expected)
    np.testing.assert_array_equal(rag_result.lengths.ravel(), lengths)


def test_tokenize_out_requires_int32():
    """out= must be dtype int32; any other dtype raises TypeError."""
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    cast = sp.cast_seqs(["ACGT"])

    # int32 out: should write into the buffer and return the same object.
    out_i32 = np.empty(cast.shape, dtype=np.int32)
    returned = sp.tokenize(cast, token_map, unknown_token=-1, out=out_i32)
    np.testing.assert_array_equal(returned, [[0, 1, 2, 3]])
    assert returned is out_i32

    # non-int32 out: np.take enforces safe casting, so this must raise TypeError.
    out_i64 = np.empty(cast.shape, dtype=np.int64)
    with pytest.raises(TypeError):
        sp.tokenize(cast, token_map, unknown_token=-1, out=out_i64)


def test_tokenize_parallel_branch_matches_reference():
    """Inputs above the parallel threshold must match the gufunc reference,
    across writable/readonly, dense/ragged, and out= variants (regression guard
    for the parallel lut_gather path)."""
    from seqpro._numba import gufunc_tokenize
    from seqpro._encoders import _TOKENIZE_PARALLEL_THRESHOLD as THRESH

    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    src = np.array([c.encode("ascii") for c in token_map]).view(np.uint8)
    tgt = np.array(list(token_map.values()), dtype=np.int32)

    def ref(u8):
        return gufunc_tokenize(u8, src, tgt, np.int32(4))

    n = THRESH + 5000  # comfortably above the parallel threshold
    rng = np.random.default_rng(0)
    bases = np.frombuffer(b"ACGTN", dtype="S1")
    flat = bases[rng.integers(0, 5, n)]  # (n,) S1, writable, known + unknown

    # writable dense -> parallel kernel
    np.testing.assert_array_equal(
        sp.tokenize(flat, token_map, unknown_token=4), ref(flat.view(np.uint8))
    )

    # readonly dense -> regression guard (np.frombuffer yields a readonly buffer)
    ro = np.frombuffer(flat.tobytes(), dtype="S1")
    assert not ro.flags.writeable
    np.testing.assert_array_equal(
        sp.tokenize(ro, token_map, unknown_token=4), ref(ro.view(np.uint8))
    )

    # contiguous int32 out= -> writes through, returns same buffer
    out = np.empty(n, dtype=np.int32)
    ret = sp.tokenize(flat, token_map, unknown_token=4, out=out)
    assert ret is out
    np.testing.assert_array_equal(out, ref(flat.view(np.uint8)))

    # strided int32 out= -> forced down the np.take fallback, still correct
    strided = np.empty((n, 2), dtype=np.int32)[:, 0]
    assert not strided.flags.c_contiguous
    sp.tokenize(flat, token_map, unknown_token=4, out=strided)
    np.testing.assert_array_equal(strided, ref(flat.view(np.uint8)))

    # non-int32 out= -> TypeError on the large path too (contract is uniform)
    with pytest.raises(TypeError):
        sp.tokenize(flat, token_map, unknown_token=4, out=np.empty(n, dtype=np.int64))

    # ragged above threshold -> parallel kernel on the packed flat data
    lengths = np.full(2000, 35)  # 70k total elements
    data = bases[rng.integers(0, 5, int(lengths.sum()))]
    rag = Ragged.from_lengths(data, lengths)
    rag_got = sp.tokenize(rag, token_map, unknown_token=4)
    np.testing.assert_array_equal(rag_got.data, ref(data.view(np.uint8)))


def test_tokenize_parallel_override_forces_path(monkeypatch):
    """parallel= overrides the size heuristic in both directions, while staying
    correct. Spying on the parallel kernel reveals which branch ran."""
    import seqpro._encoders as enc
    from seqpro._encoders import _TOKENIZE_PARALLEL_THRESHOLD as THRESH

    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    bases = np.frombuffer(b"ACGT", dtype="S1")

    calls = {"n": 0}
    real_lut = enc.lut_gather

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real_lut(*args, **kwargs)

    monkeypatch.setattr(enc, "lut_gather", spy)

    # small input, parallel=True -> parallel kernel used despite being below thresh
    small = bases[np.zeros(8, dtype=int)]
    assert small.size < THRESH
    res = sp.tokenize(small, token_map, unknown_token=-1, parallel=True)
    assert calls["n"] == 1
    np.testing.assert_array_equal(res, np.zeros(8, dtype=np.int32))

    # large input, parallel=False -> single-threaded np.take, kernel not used
    calls["n"] = 0
    large = bases[np.zeros(THRESH + 1000, dtype=int)]
    res = sp.tokenize(large, token_map, unknown_token=-1, parallel=False)
    assert calls["n"] == 0
    np.testing.assert_array_equal(res, np.zeros(THRESH + 1000, dtype=np.int32))

    # ragged honors the override too: small ragged forced parallel
    calls["n"] = 0
    data = bases[np.zeros(8, dtype=int)]
    rag = Ragged.from_lengths(data, np.array([4, 4]))
    sp.tokenize(rag, token_map, unknown_token=-1, parallel=True)
    assert calls["n"] == 1


def test_tokenize_parallel_true_rejects_noncontiguous_out():
    """parallel=True cannot honor a non-C-contiguous out= buffer, so it raises."""
    token_map = {"A": 0, "C": 1, "G": 2, "T": 3}
    flat = np.frombuffer(b"ACGT" * 4, dtype="S1")
    strided = np.empty((flat.size, 2), dtype=np.int32)[:, 0]
    assert not strided.flags.c_contiguous
    with pytest.raises(ValueError):
        sp.tokenize(flat, token_map, unknown_token=-1, out=strided, parallel=True)
