"""Flat-buffer operations on :class:`Ragged` arrays.

These operate directly on the ``(data, offsets)`` representation using Rust
kernels (via the compiled ``seqpro.seqpro`` extension) instead of going
through awkward-array ops, which is the hot path for per-batch transforms
(e.g. reverse-complementing negative-strand entries) in downstream loaders.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._core import Ragged, is_rag_dtype
from ._utils import OFFSET_TYPE

__all__ = ["concatenate", "reverse_complement", "to_packed", "to_padded"]


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
        rag = to_packed(rag)

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
    from seqpro.seqpro import _ragged_reverse_complement  # type: ignore[missing-import]  # rust

    _ragged_reverse_complement(u1, offsets, comp_lut, mask_flat)
    return Ragged.from_offsets(u1.view("S1"), rag.shape, offsets)


def _pack_parts(
    data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any], copy: bool
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Pack one flat (data, offsets) pair. Returns (packed_data, packed_offsets_1d).

    Raises ValueError if ``copy=False`` and the input is not already packed.
    """
    rag_dim = shape.index(None)
    trailing = shape[rag_dim + 1 :]
    # trailing dims are all int (only the ragged dim is None); cast away int|None for np.prod
    elem = (
        int(np.prod([d for d in trailing if d is not None], dtype=np.int64))
        * data.dtype.itemsize
    )

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
    from seqpro.seqpro import _ragged_pack  # type: ignore[missing-import]  # rust

    # Pre-allocate a plain numpy-owned buffer and let rust write into it.
    # This avoids: (a) a zero-init pass inside Rust, (b) a PySliceContainer-
    # backed return value (which breaks `is_base`), and (c) an extra .copy().
    out_bytes = np.empty(int(out_offsets[-1]) * elem, dtype=np.uint8)
    _ragged_pack(
        np.ascontiguousarray(starts, np.int64),
        np.ascontiguousarray(stops, np.int64),
        src_bytes,
        elem,
        out_bytes,
    )
    out_data = out_bytes.view(data.dtype)
    if trailing:
        out_data = out_data.reshape(-1, *trailing)
    return out_data, out_offsets


def _nested_pack_parts(
    data: NDArray[Any],
    shape: tuple[int | None, ...],
    offsets_list: list[NDArray[Any]],
    copy: bool,
) -> tuple[NDArray[Any], list[NDArray[Any]]]:
    """Pack a nested R=2 ragged array into canonical zero-based contiguous layout.

    Returns ``(packed_data, [o0_out, o1_out])`` where both offset arrays are 1-D
    and zero-based, and ``packed_data`` is contiguous.

    When ``copy=False`` and the input is already packed, returns the original
    objects unchanged (identity-preserving); raises ``ValueError`` otherwise.
    """
    from ._layout import _level_bounds

    if len(offsets_list) != 2:
        raise ValueError("_nested_pack_parts expects exactly 2 offset levels")
    o0, o1 = offsets_list
    o0_starts, o0_stops = _level_bounds(o0)
    o1_starts, o1_stops = _level_bounds(o1)
    rag_dim = shape.index(None)
    trailing = shape[rag_dim + 2 :]
    elem = (
        int(np.prod([d for d in trailing if d is not None], dtype=np.int64))
        * data.dtype.itemsize
    )
    already = (
        o0.ndim == 1
        and o1.ndim == 1
        and (o0.size == 0 or o0[0] == 0)
        and (o1.size == 0 or o1[0] == 0)
        and data.flags.c_contiguous
        and int(o1[-1]) == data.shape[0]
    )
    if already and not copy:
        return data, [o0, o1]
    if not copy:
        raise ValueError(
            "to_packed(copy=False) requires already-packed input; "
            "got an unpacked nested array."
        )
    from seqpro.seqpro import _ragged_nested_pack  # type: ignore[missing-import]

    src = np.ascontiguousarray(data).reshape(-1).view(np.uint8)
    out_o0, out_o1, out_bytes = _ragged_nested_pack(
        np.ascontiguousarray(o0_starts, np.int64),
        np.ascontiguousarray(o0_stops, np.int64),
        np.ascontiguousarray(o1_starts, np.int64),
        np.ascontiguousarray(o1_stops, np.int64),
        src,
        elem,
    )
    out_data = out_bytes.view(data.dtype)
    if trailing:
        out_data = out_data.reshape(-1, *trailing)
    return out_data, [out_o0.astype(OFFSET_TYPE), out_o1.astype(OFFSET_TYPE)]


def to_packed(rag: Any, *, copy: bool = True) -> Any:
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
    # _core.Ragged has a native to_packed() that handles all layout cases:
    # record, R=2 nested, opaque-string, and flat.  Delegate to it directly.
    if isinstance(rag, Ragged):
        return rag.to_packed(copy=copy)

    raise TypeError(f"Unsupported Ragged type: {type(rag)}")


def to_padded(
    rag: Ragged[Any],
    pad_value: Any,
    *,
    length: int | None = None,
) -> NDArray[Any]:
    """Densify a Ragged into a right-padded rectilinear array via a flat-buffer kernel.

    Flat-buffer alternative to the awkward idiom
    ``Ragged(ak_str.rpad(rag, L, v)).to_numpy()`` (bytes) /
    ``ak.to_numpy(ak.fill_none(ak.pad_none(rag, L, axis=-1, clip=True), v))`` (numeric):
    each row is copied once into a pre-filled output buffer in a single parallel pass.
    Pads the last axis to ``length`` if given, otherwise to the batch maximum ``rag.lengths.max()``.

    Parameters
    ----------
    rag
        Ragged array with exactly one ragged dimension and no fixed trailing
        dimensions (the ragged axis is last). Any fixed-itemsize dtype.
    pad_value
        Fill value for positions past each row's length; must be castable to
        ``rag.data.dtype`` (e.g. ``b"N"`` for S1, ``-1`` for int32).
    length
        Target length of the last axis. ``None`` (default) uses the batch maximum
        ``rag.lengths.max()``. An explicit ``length`` right-pads shorter rows and
        truncates longer rows to exactly ``length``.

    Returns
    -------
    NDArray
        Dense array of dtype ``rag.data.dtype`` and shape
        ``(*rag.shape[:rag_dim], out_len)``.
    """
    _is_rec = rag._is_record
    if _is_rec:
        raise NotImplementedError(
            "to_padded is not defined on record-layout Ragged arrays; "
            "convert fields individually."
        )

    rag_dim = rag.rag_dim
    if any(d is not None for d in rag.shape[rag_dim + 1 :]):
        raise ValueError(
            "to_padded requires the ragged axis to be last "
            f"(no fixed trailing dims), got shape {rag.shape}."
        )

    if not rag.is_contiguous:
        rag = to_packed(rag)

    offsets = np.ascontiguousarray(rag.offsets, dtype=np.int64)
    n_rows = offsets.shape[0] - 1

    if length is not None:
        out_len = int(length)
    elif n_rows:
        out_len = int(rag.lengths.max())
    else:
        out_len = 0

    # rag.data is NDArray here (record layout already rejected above)
    rag_data: NDArray[Any] = rag.data  # type: ignore[assignment]
    dtype = rag_data.dtype
    itemsize = dtype.itemsize

    out = np.full((n_rows, out_len), pad_value, dtype=dtype)
    if n_rows and out_len:
        data_u1 = np.ascontiguousarray(rag_data).reshape(-1).view(np.uint8)
        out_u1 = out.reshape(-1).view(np.uint8)
        from seqpro.seqpro import _ragged_to_padded  # type: ignore[missing-import]  # rust

        _ragged_to_padded(data_u1, offsets, out_u1, itemsize, out_len)

    leading = rag.shape[:rag_dim]
    if leading:
        out = out.reshape((*leading, out_len))  # pyrefly: ignore[no-matching-overload] -- leading contains int|None but dims before rag_dim are always int; numpy stub can't verify this
    return out


def concatenate(rags: Any, axis: int) -> "Ragged[Any]":
    """Concatenate Ragged arrays along the ragged axis.

    Given N ``Ragged`` arrays that share the same number of groups and the same
    leading fixed dimensions, concatenate their per-group element sequences so
    that each output group contains the elements of ``rags[0]`` followed by
    ``rags[1]``, …, ``rags[-1]``.

    Parameters
    ----------
    rags
        Sequence of :class:`Ragged` arrays.  All must have the same ``shape``
        except for the total element count (which varies per group).
    axis
        Axis to concatenate along.  Must be the ragged axis (``None`` position
        in ``shape``); negative values are normalised.  ``axis=-1`` is the
        typical call for a 1-D ragged array ``(G, None)``.

    Returns
    -------
    Ragged
        A new packed :class:`Ragged` whose per-group data is the concatenation
        of the inputs.  dtype is inherited from the first input.

    Notes
    -----
    No Python loops over elements.  Offset arithmetic and buffered copy are
    performed by a Rust/rayon kernel (``_ragged_concat``).  Supports any
    fixed-itemsize numeric dtype (e.g. ``int32``, ``float32``).
    """
    from ._core import Ragged

    if not rags:
        raise ValueError("concatenate requires at least one Ragged")
    rags = [r if isinstance(r, Ragged) else Ragged(r) for r in rags]
    ref = rags[0]
    ax = axis % len(ref.shape)
    if ax != ref.rag_dim:
        raise ValueError(
            f"concatenate only supports the ragged axis ({ref.rag_dim}), got {axis}"
        )

    # Pack each input so offsets are 1-D (G+1,) and data is contiguous.
    packed = [x.to_packed() for x in rags]

    # Compute byte-element size: dtype.itemsize * product of trailing fixed dims.
    trailing = ref.shape[ref.rag_dim + 1 :]
    trailing_ints = [d for d in trailing if d is not None]
    elem = (
        int(np.prod(trailing_ints, dtype=np.int64)) * ref.data.dtype.itemsize
        if trailing_ints
        else ref.data.dtype.itemsize
    )  # type: ignore[union-attr]

    from seqpro.seqpro import _ragged_concat  # type: ignore[missing-import]  # rust

    data_list = [
        np.ascontiguousarray(p.data).reshape(-1).view(np.uint8)  # type: ignore[union-attr]
        for p in packed
    ]
    offsets_list = [np.ascontiguousarray(p.offsets, dtype=np.int64) for p in packed]

    out_bytes, out_offsets = _ragged_concat(data_list, offsets_list, elem)
    out_data = out_bytes.view(ref.data.dtype)  # type: ignore[union-attr]
    if trailing_ints:
        out_data = out_data.reshape(-1, *trailing_ints)
    return Ragged.from_offsets(out_data, ref.shape, out_offsets)
