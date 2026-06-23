import numpy as np

from seqpro.rag import Ragged


class _Sub(Ragged):
    __slots__ = ()


def _ragged():
    return Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32)
    )


def test_with_layout_preserves_subclass():
    sub = _Sub(_ragged()._layout)
    out = sub._with_layout(_ragged()._layout)
    assert type(out) is _Sub


def test_with_layout_base_returns_base():
    base = _ragged()
    out = base._with_layout(_ragged()._layout)
    assert type(out) is Ragged
