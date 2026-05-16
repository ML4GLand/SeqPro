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
