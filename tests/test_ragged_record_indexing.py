import numpy as np
import pytest

from seqpro.rag import Ragged


def _multidim_record():
    """A (3, 2, None) numeric record: 6 ragged segments over a (batch=3, ploidy=2)
    leading grid."""
    lengths = np.array([[1, 2], [3, 1], [2, 2]], np.uint32)  # (3, 2)
    data = np.arange(int(lengths.sum()), dtype=np.int32)
    start = Ragged.from_lengths(data, lengths.reshape(-1)).reshape(3, 2, None)
    return Ragged.from_fields({"start": start})


@pytest.mark.parametrize(
    "key",
    [
        0,
        slice(1, 3),
        slice(None),
        np.array([True, False, True]),
        np.array([0, 2]),
    ],
    ids=["int", "slice", "full_slice", "mask", "int_array"],
)
def test_nontuple_equals_one_tuple_multidim_record(key):
    """numpy contract: rec[k] must equal rec[(k,)] for a multi-leading-axis record."""
    rec = _multidim_record()
    got = rec[key]
    want = rec[(key,)]
    assert type(got) is type(want), (type(got), type(want))
    if isinstance(want, Ragged):
        assert got.shape == want.shape, (got.shape, want.shape)
        np.testing.assert_array_equal(
            np.asarray(got["start"].data), np.asarray(want["start"].data)
        )
        np.testing.assert_array_equal(
            np.asarray(got["start"].offsets[0]), np.asarray(want["start"].offsets[0])
        )


def test_slice_preserves_ploidy_axis():
    """Regression for the specific bug: rec[1:3] kept the ploidy axis."""
    rec = _multidim_record()
    assert rec[1:3].shape == (2, 2, None)
    assert rec[0].shape == (2, None)
