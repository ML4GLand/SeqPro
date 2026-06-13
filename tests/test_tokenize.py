from typing import List

import numpy as np
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

    # Ragged path.
    rag_seqs = ["ACGTN", "TTxAC", "GGGGG"]
    data = np.frombuffer("".join(rag_seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in rag_seqs])
    rag = Ragged.from_lengths(data, lengths)
    rag_result = sp.tokenize(rag, token_map, unknown_token=unknown_token)
    flat_expected = reference(
        np.frombuffer(b"".join(s.encode() for s in rag_seqs), dtype="S1")
    )
    np.testing.assert_array_equal(rag_result.data, flat_expected)
    np.testing.assert_array_equal(rag_result.lengths.ravel(), lengths)
