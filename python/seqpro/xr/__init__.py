from typing import Union

import numpy as np
from numpy.typing import NDArray

from seqpro._numba import gufunc_ohe, gufunc_translate
from seqpro.alphabets import AminoAlphabet, NucleotideAlphabet

try:
    import xarray as xr
except ImportError:
    raise ImportError("Need to install Xarray to use seqpro.xr")

__all__ = ["ohe", "bin_coverage"]


def ohe(
    seqs: Union[xr.DataArray, xr.Dataset],
    alphabet: NucleotideAlphabet,
    ohe_dim: str = "_ohe",
) -> Union[xr.DataArray, xr.Dataset]:
    """Ohe hot encode sequences in an xr.Dataset.

    Parameters
    ----------
    seqs : Union[xr.DataArray, xr.Dataset]
    alphabet : NucleotideAlphabet
    ohe_dim : Optional[str], optional
        Name to give the new one hot encoding dimension, by default "_ohe"

    Returns
    -------
    xr.DataArray, xr.Dataset
        One hot encoded sequences.
    """
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


def bin_coverage(
    coverage: Union[xr.DataArray, xr.Dataset],
    bin_width: int,
    length_dim: str,
    binned_dim="_bin",
    normalize=False,
) -> Union[xr.DataArray, xr.Dataset]:
    """Bin coverage to a lower resolution by summing across non-overlapping windows.

    Parameters
    ----------
    coverage : Union[xr.DataArray, xr.Dataset]
        Array of coverage.
    bin_width : int
        Size of the bins (aka windows)
    length_dim : str
        Name of the length dimension.
    binned_dim : str, optional
        Name of the binned dimension, by default '_bin'
    normalize : bool, optional
        Whether to divide by bin width, by default False

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        DataArray or Dataset of binned coverage.

    Raises
    ------
    ValueError
        If the length is not evenly divisible by the bin width.
    """
    length = coverage.sizes[length_dim]
    if length % bin_width != 0:
        raise ValueError("Bin width must evenly divide length.")

    def _bin(x):
        return np.add.reduceat(x, np.arange(0, length, bin_width), axis=-1)

    binned_coverage = xr.apply_ufunc(
        _bin,
        coverage,
        input_core_dims=[[length_dim]],
        output_core_dims=[[binned_dim]],
        dask="parallelized",
        dask_gufunc_kwargs={},
    )

    if normalize:
        binned_coverage = binned_coverage / bin_width
    return binned_coverage


def translate(
    seqs: Union[xr.DataArray, xr.Dataset],
    alphabet: AminoAlphabet,
    length_dim: str,
    aa_length_dim="_aa_length",
) -> Union[xr.DataArray, xr.Dataset]:
    """Translate DNA sequences to amino acid sequences.

    Parameters
    ----------
    seqs : Union[xr.DataArray, xr.Dataset]
    alphabet : AminoAlphabet
    length_dim : str
    aa_length_dim : str, optional
        Amino acid length dimension, by default "_aa_length"
    """
    k = alphabet.codon_array.shape[-1]

    if seqs.sizes[length_dim] % k != 0:
        raise ValueError(
            f"Sequence length is not evenly divisible by codon length: {k}."
        )

    def _translate(seqs):
        n = seqs.shape[-1] // k
        shape = *seqs.shape[:-1], n, k
        strides = (
            *seqs.strides[:-1],
            k,
            seqs.strides[-1],
        )
        trimers = np.lib.stride_tricks.as_strided(seqs, shape=shape, strides=strides)
        return gufunc_translate(
            trimers.view(np.uint8),
            alphabet.codon_array.view(np.uint8),
            alphabet.aa_array.view(np.uint8),
            axes=[(-1), (-2, -1), (-1), ()],  # type: ignore
        ).view("S1")

    aa_seqs = xr.apply_ufunc(
        _translate,
        seqs,
        input_core_dims=[[length_dim]],
        output_core_dims=[[aa_length_dim]],
        dask="parallelized",
    )

    return aa_seqs


# * This function doesn't seem possible to implement with xarray.apply_ufunc using the
# * current implementation of sp.jitter. This is because tuples of DataArrays don't get
# * converted to tuples of NumPy arrays before getting passed to `func` in apply_ufunc.
# * Passing in unpacked DataArrays or Datasets causes `func` to be applied to each
# * separately.
# def jitter(
#     data: Union[xr.DataArray, xr.Dataset],
#     max_jitter: int,
#     length_dim: str,
#     jitter_dims: Union[str, List[str]],
#     jitter_length_dim: Optional[str] = None,
#     seed: Optional[int] = None,
# ) -> Union[xr.DataArray, xr.Dataset]:
#     if isinstance(jitter_dims, str):
#         jitter_dims = [jitter_dims]

#     if jitter_length_dim is None:
#         jitter_length_dim = f'_jittered_{length_dim.strip("_")}'

#     if isinstance(data, xr.Dataset):
#         _data = list(data.data_vars.values())
#     else:
#         _data = [data]

#     for a in _data:
#         a = a.transpose(..., *jitter_dims, length_dim)

#     func = partial(
#         sp_jitter,
#         max_jitter=max_jitter,
#         length_axis=-1,
#         jitter_axes=tuple(range(-len(jitter_dims) - 1, -1)),
#         seed=seed,
#     )

#     jittered = xr.apply_ufunc(
#         func,
#         *_data,
#         input_core_dims=[*(([length_dim],) * len(_data))],
#         output_core_dims=[*(([jitter_length_dim],) * len(_data))],
#         dask="parallelized",
#         output_dtypes=[a.dtype for a in _data],
#         dask_gufunc_kwargs={
#             'output_sizes': {
#                 jitter_length_dim: _data[0].sizes[length_dim] - 2*max_jitter
#             },
#         },
#     )

#     if isinstance(data, xr.Dataset):
#         jittered = []
#         for a, b in zip(jittered, data.data_vars.values()):
#             dims = []
#             for d in b.dims:
#                 if d == length_dim:
#                     dims.append(jitter_length_dim)
#                 else:
#                     dims.append(d)
#             arr = a.rename(b.name).transpose(*dims)
#             jittered.append(arr)
#         return xr.merge(jittered)
#     else:
#         dims = []
#         for d in data.dims:
#             if d == length_dim:
#                 dims.append(jitter_length_dim)
#             else:
#                 dims.append(d)
#         jittered.rename(data.name).transpose(dims)

#     return jittered
