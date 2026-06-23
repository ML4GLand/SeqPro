import numpy as np
from seqpro.rag import Ragged


def test_to_ak_multi_leading_axis_record():
    # (b=2, p=2, ~v) record with an opaque-string and a numeric field
    var_off = np.array([0, 2, 3, 3, 4], dtype=np.int64)  # 4 groups
    char_off = np.array([0, 2, 3, 6, 7], dtype=np.int64)
    chars = np.frombuffer(b"ACGTTTX", dtype="S1").copy()
    alt = Ragged.from_offsets(
        chars, (2, 2, None, None), [var_off, char_off]
    ).to_strings()
    start = Ragged.from_offsets(np.arange(4, dtype=np.int32), (2, 2, None), alt.offsets)
    rv = Ragged.from_fields({"alt": alt, "start": start})
    got = rv.to_ak()  # must not raise
    # Both fields preserve the (b=2, p=2, ~) shape: 2 outer groups of 2 inner ragged lists.
    # alt: each inner list holds opaque bytestrings; start: each inner list holds ints.
    assert got["alt"].to_list() == [[[b"AC", b"G"], [b"TTT"]], [[], [b"X"]]]
    assert got["start"].to_list() == [[[0, 1], [2]], [[], [3]]]
