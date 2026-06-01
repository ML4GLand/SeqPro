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
from ._utils import OFFSET_TYPE

__all__ = ["reverse_complement", "to_packed"]


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


@nb.njit(parallel=True, nogil=True, cache=True)
def _pack(
    src_bytes: NDArray[np.uint8],
    in_starts: NDArray[np.int64],
    in_stops: NDArray[np.int64],
    out_bytes: NDArray[np.uint8],
    out_starts: NDArray[np.int64],
) -> None:  # pragma: no cover - exercised via to_packed
    """Gather each row's contiguous byte span into the packed output buffer.

    All offsets are in *byte* units. Row ``i`` copies
    ``src_bytes[in_starts[i]:in_stops[i]]`` to
    ``out_bytes[out_starts[i]:out_starts[i] + (in_stops[i] - in_starts[i])]``.
    One contiguous read + write per row, parallel across rows.
    """
    n = in_starts.shape[0]
    for i in nb.prange(n):
        length = in_stops[i] - in_starts[i]
        out_bytes[out_starts[i] : out_starts[i] + length] = src_bytes[
            in_starts[i] : in_stops[i]
        ]


def _pack_parts(
    data: NDArray, shape: tuple, offsets: NDArray, copy: bool
) -> tuple[NDArray, NDArray]:
    """Pack one flat (data, offsets) pair. Returns (packed_data, packed_offsets_1d).

    Raises ValueError if ``copy=False`` and the input is not already packed.
    """
    rag_dim = shape.index(None)
    trailing = shape[rag_dim + 1 :]
    elem = int(np.prod(trailing, dtype=np.int64)) * data.dtype.itemsize

    if offsets.ndim == 1:
        starts = offsets[:-1]
        stops = offsets[1:]
        zero_based = offsets.size > 0 and offsets[0] == 0
    else:
        starts = offsets[0]
        stops = offsets[1]
        zero_based = False  # ListArray -> treat as unpacked

    n_elems = data.shape[0]
    is_packed = (
        offsets.ndim == 1
        and zero_based
        and data.flags.c_contiguous
        and int(offsets[-1]) == n_elems
    )
    if is_packed and not copy:
        return data, offsets
    if not copy:
        raise ValueError(
            "to_packed(copy=False) requires already-packed input "
            "(contiguous, zero-based, 1-D offsets); got an unpacked array."
        )

    lengths = (stops - starts).astype(np.int64)
    out_offsets = np.empty(lengths.size + 1, dtype=OFFSET_TYPE)
    out_offsets[0] = 0
    np.cumsum(lengths, out=out_offsets[1:])

    if is_packed:  # copy=True on already-packed input
        return data.copy(), out_offsets

    if not data.flags.c_contiguous:
        data = np.ascontiguousarray(data)
    src_bytes = data.view(np.uint8).reshape(-1)
    out_bytes = np.empty(int(out_offsets[-1]) * elem, dtype=np.uint8)
    _pack(
        src_bytes,
        (starts.astype(np.int64) * elem),
        (stops.astype(np.int64) * elem),
        out_bytes,
        (out_offsets[:-1] * elem),
    )
    out_data = out_bytes.view(data.dtype)
    if trailing:
        out_data = out_data.reshape(-1, *trailing)
    return out_data, out_offsets


def to_packed(rag: Ragged, *, copy: bool = True) -> Ragged:
    """Pack a Ragged array's data into a fresh contiguous, zero-based buffer.

    A Numba-parallelized replacement for ``Ragged(ak.to_packed(rag))``: it
    gathers each row's contiguous byte span into a new buffer with one
    parallel read+write per row, which is fast even when the source data is a
    ``np.memmap``. The result always has 1-D offsets starting at zero.

    Parameters
    ----------
    rag
        Ragged array (flat or record layout) with one ragged dimension.
    copy
        When ``True`` (default), always return a freshly allocated, owned
        packed array (safe to mutate in place afterwards). When ``False``,
        return the input zero-copy if it is already packed, and raise
        ``ValueError`` otherwise.

    Returns
    -------
    Ragged
        Contiguous, zero-based Ragged equal in value to ``rag``.
    """
    rag._ensure_parts()
    if isinstance(rag._parts, dict):
        import awkward as ak

        offsets = rag.offsets
        if not copy:
            # passthrough only if 1-D zero-based and every field already packed
            is_packed = (
                offsets.ndim == 1
                and (offsets.size == 0 or offsets[0] == 0)
                and all(
                    p.data.flags.c_contiguous and int(offsets[-1]) == p.data.shape[0]
                    for p in rag._parts.values()
                )
            )
            if is_packed:
                return rag
            raise ValueError(
                "to_packed(copy=False) requires already-packed input; "
                "got an unpacked record array."
            )
        fields = {}
        for name, p in rag._parts.items():
            packed_data, packed_offsets = _pack_parts(
                p.data, p.shape, offsets, copy=True
            )
            fields[name] = Ragged.from_offsets(packed_data, p.shape, packed_offsets)
        return Ragged(ak.zip(fields, depth_limit=1))

    parts = rag._parts
    packed_data, packed_offsets = _pack_parts(
        parts.data, parts.shape, parts.offsets, copy
    )
    if packed_data is parts.data and packed_offsets is parts.offsets:
        return rag  # copy=False passthrough
    return Ragged.from_offsets(packed_data, parts.shape, packed_offsets)
