from typing import List

import numpy as np
import seqpro as sp
from hypothesis import given, note
from hypothesis import strategies as st
from seqpro.rag import Ragged

seq = st.text(
    alphabet=st.characters(codec="ascii", min_codepoint=32, max_codepoint=126),
    min_size=1,
)
seqs = st.lists(seq, min_size=1, max_size=10)


@given(seq)
def test_roundtrip(seq: str):
    cast_seq = sp.cast_seqs(seq)
    ohe = sp.DNA.ohe(cast_seq)
    decoded = sp.DNA.decode_ohe(ohe, -1)
    mod_seq = cast_seq.copy()
    mod_seq[
        np.isin(cast_seq, [c.encode("ascii") for c in sp.DNA.alphabet], invert=True)
    ] = b"N"

    note(f"{seq}, {cast_seq}")

    np.testing.assert_equal(decoded, mod_seq)


@given(seq)
def test_ohe(seq: str):
    cast_seq = sp.cast_seqs(seq)
    ohe = sp.DNA.ohe(cast_seq)

    note(f"{seq}, {cast_seq}, {ohe}")

    for i in range(len(seq)):
        for char in sp.DNA.alphabet:
            if seq[i] == char:
                assert ohe[i, sp.DNA.alphabet.index(char)] == 1
            else:
                assert ohe[i, sp.DNA.alphabet.index(char)] == 0


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
def test_ohe_multiple(seqs: List[str]):
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


def test_ohe_ragged():
    seqs = ["ACGT", "AC", "GTTTT"]
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    result = sp.DNA.ohe(rag)

    assert isinstance(result, Ragged)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result.lengths.ravel(), lengths)
    A = len(sp.DNA.alphabet)
    # "A" → [1,0,0,0], "C" → [0,1,0,0], "G" → [0,0,1,0], "T" → [0,0,0,1]
    np.testing.assert_array_equal(result.data[0], [1, 0, 0, 0])  # A
    np.testing.assert_array_equal(result.data[1], [0, 1, 0, 0])  # C
    np.testing.assert_array_equal(result.data[2], [0, 0, 1, 0])  # G
    np.testing.assert_array_equal(result.data[3], [0, 0, 0, 1])  # T
    assert result.data.shape == (sum(lengths), A)


def test_ohe_ragged_unknown():
    seqs = ["ACGN", "AT"]
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    result = sp.DNA.ohe(rag)

    assert isinstance(result, Ragged)
    # "N" is not in DNA alphabet — should encode as all zeros
    np.testing.assert_array_equal(result.data[3], [0, 0, 0, 0])


def test_decode_ohe_ragged():
    seqs = ["ACGT", "AC", "GTTTT"]
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    encoded = sp.DNA.ohe(rag)
    result = sp.DNA.decode_ohe(encoded, -1)

    assert isinstance(result, Ragged)
    assert result.dtype == np.dtype("S1")
    np.testing.assert_array_equal(result.lengths.ravel(), lengths)
    np.testing.assert_array_equal(result.data, rag.data)


def test_decode_ohe_ragged_module_level():
    seqs = ["ACGT", "GC"]
    data = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    lengths = np.array([len(s) for s in seqs])
    rag = Ragged.from_lengths(data, lengths)

    encoded = sp.ohe(rag, sp.DNA)
    result = sp.decode_ohe(encoded, -1, sp.DNA)

    assert isinstance(result, Ragged)
    np.testing.assert_array_equal(result.data, rag.data)


def _make_ragged_with_trailing(seqs: list[str], m: int) -> "Ragged[np.bytes_]":
    """Build a (n, ~L, m) Ragged by tiling each sequence m times along a new trailing axis."""
    flat_chars = np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
    # shape (total, m) — repeat each character m times across the trailing axis
    data = np.tile(flat_chars[:, None], (1, m))
    lengths = np.array([len(s) for s in seqs])
    return Ragged.from_lengths(data, lengths)


def test_ohe_ragged_trailing_dim():
    """(n, ~L, m) input -> (n, ~L, A, m) output."""
    seqs = ["ACGT", "AC", "GTTTT"]
    m = 3
    rag = _make_ragged_with_trailing(seqs, m)
    assert rag.shape == (3, None, m)

    result = sp.DNA.ohe(rag)

    A = len(sp.DNA.alphabet)
    assert isinstance(result, Ragged)
    assert result.shape == (3, None, A, m)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result.lengths.ravel(), [len(s) for s in seqs])
    # Each character position should produce identical OHE vectors across the m columns
    # e.g. first position is 'A' -> one-hot [1,0,0,0] repeated across m
    np.testing.assert_array_equal(result.data[0], np.tile([1, 0, 0, 0], (m, 1)).T)


def test_decode_ohe_ragged_trailing_dim():
    """(n, ~L, A, m) input -> (n, ~L, m) output (roundtrip)."""
    seqs = ["ACGT", "AC", "GTTTT"]
    m = 3
    rag = _make_ragged_with_trailing(seqs, m)

    encoded = sp.DNA.ohe(rag)
    result = sp.DNA.decode_ohe(encoded, -1)

    assert isinstance(result, Ragged)
    assert result.shape == (3, None, m)
    assert result.dtype == np.dtype("S1")
    np.testing.assert_array_equal(result.lengths.ravel(), rag.lengths.ravel())
    np.testing.assert_array_equal(result.data, rag.data)
