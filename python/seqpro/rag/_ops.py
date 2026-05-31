"""Flat-buffer operations on :class:`Ragged` arrays.

These operate directly on the ``(data, offsets)`` representation with Numba
kernels instead of going through awkward-array ops, which is the hot path for
per-batch transforms (e.g. reverse-complementing negative-strand entries) in
downstream loaders.
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray

from ._array import Ragged, is_rag_dtype

__all__ = ["reverse_complement"]


@nb.njit(parallel=True, nogil=True, cache=True)
def _reverse_complement_ragged(
    data: NDArray[np.uint8],
    offsets: NDArray[np.int64],
    comp_lut: NDArray[np.uint8],
    mask: NDArray[np.bool_],
) -> None:  # pragma: no cover - exercised via reverse_complement
    """In-place reverse-complement of selected ragged rows over a flat buffer.

    Each row ``data[offsets[i]:offsets[i + 1]]`` with ``mask[i]`` true is
    reverse-complemented in place: positions are swapped end-to-end and each
    byte is mapped through ``comp_lut`` (a 256-entry byte->byte table). Rows with
    ``mask[i]`` false are left untouched. Offsets are unchanged because
    reverse-complement preserves length. The pass is single-touch (each byte is
    read and written exactly once) and parallel across rows.
    """
    n = mask.shape[0]
    for i in nb.prange(n):
        if not mask[i]:
            continue
        lo = offsets[i]
        hi = offsets[i + 1] - 1
        while lo < hi:
            a = data[lo]
            b = data[hi]
            data[lo] = comp_lut[b]
            data[hi] = comp_lut[a]
            lo += 1
            hi -= 1
        if lo == hi:
            data[lo] = comp_lut[data[lo]]


def reverse_complement(
    rag: Ragged[np.bytes_],
    comp_lut: NDArray[np.uint8],
    *,
    mask: NDArray[np.bool_] | None = None,
    copy: bool = True,
) -> Ragged[np.bytes_]:
    """Reverse-complement each row of an S1 (bytes) ragged array on its flat buffer.

    This is a flat-buffer alternative to the awkward-array idiom
    ``Ragged(ak.to_packed(ak.where(mask, rc(rag), rag)))``: it touches only the
    rows selected by ``mask``, runs a single in-place pass per row, and reuses
    the input ``offsets`` (reverse-complement is length-preserving).

    Parameters
    ----------
    rag
        S1 (bytes) ragged array with exactly one ragged dimension and no fixed
        trailing dimensions (i.e. the ragged axis is last).
    comp_lut
        256-entry ``uint8`` byte->byte complement table. For DNA this is
        ``sp.DNA.bytes_comp_array.view(np.uint8)``; non-alphabet bytes (e.g.
        ``N``) map to themselves.
    mask
        Boolean array, one entry per ragged row (any shape broadcastable to the
        leading non-ragged dimensions, flattened in C order). Rows where the
        mask is true are reverse-complemented; the rest are copied unchanged.
        ``None`` reverse-complements every row.
    copy
        When true (default), operate on a copy of the data buffer and return a
        new ragged array, leaving ``rag`` unmodified. When false, mutate the
        input buffer in place — only safe when the caller owns ``rag`` (e.g. a
        freshly reconstructed batch).

    Returns
    -------
    Ragged[np.bytes_]
        Reverse-complemented ragged array sharing ``rag``'s offsets.
    """
    if not is_rag_dtype(rag, np.bytes_):
        raise ValueError("reverse_complement requires an S1 (bytes) Ragged array.")

    rag_dim = rag.rag_dim
    if any(d is not None for d in rag.shape[rag_dim + 1 :]):
        raise ValueError(
            "reverse_complement requires the ragged axis to be last "
            f"(no fixed trailing dims), got shape {rag.shape}."
        )

    comp_lut = np.ascontiguousarray(comp_lut, dtype=np.uint8)
    if comp_lut.shape != (256,):
        raise ValueError(
            f"comp_lut must be a 256-entry uint8 table, got shape {comp_lut.shape}."
        )

    # The kernel assumes contiguous (N+1,) offsets and a contiguous flat buffer.
    if not rag.is_contiguous:
        import awkward as ak

        rag = Ragged(ak.to_packed(rag))

    offsets = np.ascontiguousarray(rag.offsets, dtype=np.int64)
    n_rows = offsets.shape[0] - 1

    if mask is None:
        mask_flat = np.ones(n_rows, dtype=np.bool_)
    else:
        mask_flat = np.ascontiguousarray(mask, dtype=np.bool_).reshape(-1)
        if mask_flat.shape[0] != n_rows:
            raise ValueError(
                f"mask has {mask_flat.shape[0]} entries but ragged array has "
                f"{n_rows} rows."
            )

    u1 = np.ascontiguousarray(rag.data).view(np.uint8)
    if copy:
        u1 = u1.copy()
    _reverse_complement_ragged(u1, offsets, comp_lut, mask_flat)
    return Ragged.from_offsets(u1.view("S1"), rag.shape, offsets)
