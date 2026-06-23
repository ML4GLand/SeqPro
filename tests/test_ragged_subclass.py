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


def _record():
    a = Ragged.from_lengths(
        np.arange(6, dtype=np.int32), np.array([2, 1, 3], np.uint32)
    )
    b = Ragged.from_lengths(
        np.arange(6, dtype=np.int32) * 10, np.array([2, 1, 3], np.uint32)
    )
    return Ragged.from_fields({"a": a, "b": b})


def test_getitem_positional_preserves_subclass():
    sub = _Sub(_record()._layout)
    assert type(sub[0:2]) is _Sub  # positional row slice -> subclass
    assert type(sub[np.array([0, 2])]) is _Sub


def test_getitem_field_extraction_stays_base():
    sub = _Sub(_record()._layout)
    assert type(sub["a"]) is Ragged  # string key -> bare field, base Ragged
