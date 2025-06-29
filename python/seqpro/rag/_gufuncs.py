from typing import TypeVar

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._types import ak_dtypes

DTYPE = TypeVar("DTYPE", bound=ak_dtypes)


@nb.guvectorize("(n)->(n)")
def reverse_complement(data: NDArray[DTYPE], mask: NDArray[np.bool_]):
    pass
