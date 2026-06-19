import numpy as np
import pytest

import seqpro as sp
from seqpro import AA  # standard AminoAlphabet


def _bio_like_translate(seq_str: str) -> str:
    """Reference using AA.codon_to_aa directly (independent of the impl)."""
    out = []
    for i in range(0, len(seq_str), 3):
        out.append(AA.codon_to_aa.get(seq_str[i : i + 3], "X"))
    return "".join(out)


@pytest.mark.parametrize(
    "seq",
    ["ATGAAATAA", "atgaaataa", "ATGNNNTAA", ""],
)
def test_translate_dense_pad_matches_reference(seq):
    arr = np.frombuffer(seq.encode("ascii"), "S1")
    got = sp.AA.translate(arr, length_axis=0)
    exp = _bio_like_translate(seq.upper())
    assert got.tobytes().decode() == exp
