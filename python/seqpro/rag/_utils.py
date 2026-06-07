from __future__ import annotations

from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

DTYPE = TypeVar("DTYPE", bound=np.integer)
LENGTH_TYPE = np.uint32
OFFSET_TYPE = np.int64


def lengths_to_offsets(
    lengths: NDArray[np.integer],
    dtype: type[DTYPE] | DTYPE = OFFSET_TYPE,  # pyrefly: ignore[bad-function-definition] -- np.int64 satisfies bound np.integer but pyrefly can't verify TypeVar default
) -> NDArray[DTYPE]:
    """Convert lengths to offsets.

    Parameters
    ----------
    lengths
        Lengths of the segments.

    Returns
    -------
    NDArray[DTYPE]
        Offsets of the segments; length is len(lengths) + 1, starting with 0.
    """
    offsets = np.empty(lengths.size + 1, dtype=dtype)
    offsets[0] = 0
    offsets[1:] = lengths.cumsum()
    return offsets
