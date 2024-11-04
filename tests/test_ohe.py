from typing import List

import numpy as np
import seqpro as sp
from hypothesis import given, note
from hypothesis import strategies as st

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
