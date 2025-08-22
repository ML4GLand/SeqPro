from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ._types import ak_dtypes

DTYPE = TypeVar("DTYPE", bound=ak_dtypes)
LENGTH_TYPE = np.uint32
OFFSET_TYPE = np.int64


def lengths_to_offsets(
    lengths: NDArray[np.integer], dtype: type[DTYPE] = OFFSET_TYPE
) -> NDArray[DTYPE]:
    """Convert lengths to offsets.

    Parameters
    ----------
    lengths
        Lengths of the segments.

    Returns
    -------
    offsets
        Offsets of the segments.
    """
    offsets = np.empty(lengths.size + 1, dtype=dtype)
    offsets[0] = 0
    offsets[1:] = lengths.cumsum()
    return offsets
