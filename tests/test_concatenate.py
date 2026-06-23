import numpy as np
import seqpro.rag as r
from seqpro.rag import Ragged


def test_concatenate_ragged_axis_prepend_regular():
    # prepend a size-1 pad per group (the prepend_pad_itv use case)
    base = Ragged.from_offsets(
        np.array([10, 11, 12], np.int32), (2, None), np.array([0, 2, 3], np.int64)
    )  # [[10,11],[12]]
    pad = Ragged.from_offsets(
        np.array([-1, -1], np.int32), (2, None), np.array([0, 1, 2], np.int64)
    )  # [[-1],[-1]]
    out = r.concatenate([pad, base], axis=-1)
    assert out.to_ak().to_list() == [[-1, 10, 11], [-1, 12]]


def test_concatenate_matches_awkward_oracle():
    import awkward as ak

    a = Ragged.from_offsets(
        np.arange(5, dtype=np.float32), (2, None), np.array([0, 3, 5], np.int64)
    )
    b = Ragged.from_offsets(
        np.arange(5, 9, dtype=np.float32), (2, None), np.array([0, 1, 4], np.int64)
    )
    got = r.concatenate([a, b], axis=-1).to_ak().to_list()
    exp = ak.concatenate([a.to_ak(), b.to_ak()], axis=-1).to_list()
    assert got == exp
