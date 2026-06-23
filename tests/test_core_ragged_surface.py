"""Smoke check: the Rust _core.Ragged exposes the API surface consumers use.

Surface derived from grepping genoray/GenVarLoader/genvarformer source.
"""

import numpy as np
import pytest
from seqpro.rag._core import Ragged

CONSTRUCTORS = ["from_offsets", "from_lengths", "from_fields", "empty"]
METHODS = ["to_packed", "to_padded", "to_numpy", "to_ak", "view", "squeeze", "reshape"]
# note: ndim is absent — no consumer calls .ndim on _core.Ragged (grepped
#   genoray/, GenVarLoader/, genvarformer/; all .ndim hits are on numpy arrays
#   or ak.Array, not _core.Ragged objects).
# note: parts is absent — no consumer calls .parts on _core.Ragged (zero
#   matches across all three consumer repos); _array.Ragged has .parts but
#   _core.Ragged uses a different internal layout and no consumer depends on it.
ATTRS = ["data", "offsets", "lengths", "shape", "dtype"]
DUNDERS = ["__getitem__", "__setitem__", "__array__"]


@pytest.mark.parametrize("name", CONSTRUCTORS + METHODS)
def test_callable_present(name):
    assert callable(getattr(Ragged, name, None)), f"Ragged.{name} missing"


def test_instance_attrs_present():
    # 2 rows of lengths [3, 2] over data 0..4
    r = Ragged.from_lengths(np.arange(5), np.array([3, 2]))
    missing = [a for a in ATTRS if not hasattr(r, a)]
    assert not missing, f"missing instance attrs: {missing}"


@pytest.mark.parametrize("name", DUNDERS)
def test_dunder_present(name):
    assert hasattr(Ragged, name), f"Ragged.{name} missing"
