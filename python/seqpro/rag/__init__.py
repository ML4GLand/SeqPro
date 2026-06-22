from typing import Any

from ._array import DTYPE_co, RDTYPE_co
from ._core import Ragged, Ragged as _CoreRagged, is_rag_dtype
from ._ops import concatenate, reverse_complement, to_packed, to_padded
from ._utils import OFFSET_TYPE, lengths_to_offsets


def zip(fields: "dict[str, _CoreRagged[Any]]") -> "_CoreRagged[Any]":  # noqa: A001  (intentional ak.zip-compatible name)
    """Build a record Ragged from a dict of single-field Ragged inputs.

    Alias for ``Ragged.from_fields``; operates on the Rust-native core path.
    """
    return _CoreRagged.from_fields(fields)


__all__ = [
    "OFFSET_TYPE",
    "DTYPE_co",
    "RDTYPE_co",
    "Ragged",
    "concatenate",
    "is_rag_dtype",
    "lengths_to_offsets",
    "reverse_complement",
    "to_packed",
    "to_padded",
    "zip",
]
