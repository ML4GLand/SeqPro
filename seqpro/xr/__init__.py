from typing import Optional, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

import seqpro as sp
from seqpro._numba import gufunc_ohe

__all__ = ["ohe"]


def ohe(
    seqs: Union[xr.DataArray, xr.Dataset],
    alphabet: sp.NucleotideAlphabet,
    ohe_dim: Optional[str] = None,
) -> xr.DataArray:
    if ohe_dim is None:
        ohe_dim = "_ohe"

    alpha = xr.DataArray(alphabet.array, dims=ohe_dim)

    def _ohe(seqs: NDArray[np.bytes_], alphabet: NDArray[np.bytes_]):
        return gufunc_ohe(seqs.view(np.uint8), alphabet.view(np.uint8))

    out = xr.apply_ufunc(
        _ohe,
        seqs,
        alpha,
        input_core_dims=[[], [ohe_dim]],
        output_core_dims=[[ohe_dim]],
        dask="parallelized",
    )

    return out
