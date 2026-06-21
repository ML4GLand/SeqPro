from __future__ import annotations

from typing import Any, cast

import awkward as ak
import numpy as np
from awkward.contents import (
    EmptyArray,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
)
from awkward.index import Index

from ._array import unbox  # proven awkward extractor (RagParts: data, shape, offsets)
from ._layout import RaggedLayout, RecordLayout
from ._utils import OFFSET_TYPE


def _walk_ragged(node: Any) -> "tuple[Any, list[Any], Any | None]":
    """Walk a nested awkward list layout → (data, offsets_levels, str_offsets).

    Non-bytestring list axes contribute one entry to ``offsets_levels``.
    A bytestring list axis becomes the opaque-string leaf: its offsets go
    into ``str_offsets`` and the walk stops there.
    """
    offsets_levels: list[Any] = []
    str_offsets = None
    while True:
        if isinstance(node, (IndexedArray, IndexedOptionArray)):
            node = node.project()
            continue
        if isinstance(node, (ListOffsetArray, ListArray)):
            if isinstance(node, ListOffsetArray):
                off = np.ascontiguousarray(node.offsets.data, dtype=OFFSET_TYPE)
            else:
                off = np.stack(
                    [np.asarray(node.starts.data), np.asarray(node.stops.data)], 0
                ).astype(OFFSET_TYPE)
            if node.parameter("__array__") == "bytestring":
                str_offsets = off  # opaque string leaf
                node = node.content
                break
            offsets_levels.append(off)
            node = node.content
            continue
        break
    if isinstance(node, EmptyArray):
        node = node.to_NumpyArray(dtype=np.float64)
    numpy_node = cast(NumpyArray, node)
    data = np.asarray(numpy_node.data)
    if numpy_node.parameter("__array__") == "byte" or str_offsets is not None:
        data = data.view("S1")
    return data, offsets_levels, str_offsets


def _nested_layout_from_ak(node: Any) -> "RaggedLayout[Any]":
    """Build a RaggedLayout from an awkward node with 2 ragged axes OR
    a list-of-bytestrings (string-under-axis)."""
    data, offsets_levels, str_offsets = _walk_ragged(node)
    n_ragged = len(offsets_levels)
    trailing = tuple(data.shape[1:])
    if n_ragged == 0:
        # string-under-axis: only str_offsets set, no ragged axes collected
        # (shouldn't happen in practice via this path, but be defensive)
        assert str_offsets is not None
        n = int(len(str_offsets) - 1)
        return RaggedLayout(data=data, offsets=[], shape=(n,), str_offsets=str_offsets)
    outer = offsets_levels[0]
    l0 = int(len(outer) - 1) if outer.ndim == 1 else int(outer.shape[1])
    shape: tuple[int | None, ...] = (l0, *([None] * n_ragged), *trailing)
    return RaggedLayout(
        data=data, offsets=offsets_levels, shape=shape, str_offsets=str_offsets
    )


def _field_layout_from_ak(field_arr: Any) -> "RaggedLayout[Any]":
    """Build a per-field RaggedLayout (single-level or nested) for a record field."""
    try:
        parts = unbox(field_arr)
        # Single-level field (numeric or chars): build a minimal RaggedLayout.
        off = np.ascontiguousarray(parts.offsets, dtype=OFFSET_TYPE)
        return RaggedLayout(data=parts.data, offsets=[off], shape=parts.shape)
    except ValueError as e:
        if "ragged dimension" not in str(e):
            raise
        return _nested_layout_from_ak(ak.to_layout(field_arr, allow_record=False))


def layout_from_ak(arr: Any) -> "RaggedLayout[Any] | RecordLayout":
    if ak.fields(arr):
        names = ak.fields(arr)
        shared: list[Any] | None = None
        rec_shape: tuple[int | None, ...] | None = None
        fields: dict[str, RaggedLayout[Any]] = {}
        for f in names:
            fl = _field_layout_from_ak(arr[f])
            if shared is None:
                shared, rec_shape = fl.offsets, fl.shape
            fields[f] = RaggedLayout(
                data=fl.data, offsets=shared, shape=fl.shape, str_offsets=fl.str_offsets
            )
        assert shared is not None
        assert rec_shape is not None
        return RecordLayout(offsets=shared, shape=rec_shape, fields=fields)

    try:
        parts = unbox(arr)
    except ValueError as e:
        if "ragged dimension" not in str(e):
            raise
        return _nested_layout_from_ak(ak.to_layout(arr, allow_record=False))

    # --- existing single-level paths (unchanged) ---
    is_bytes = parts.data.dtype.kind == "S"
    if (
        is_bytes
        and parts.shape.count(None) == 1
        and parts.shape.index(None) == len(parts.shape) - 1
    ):
        leading = parts.shape[: parts.shape.index(None)]
        return RaggedLayout(
            data=parts.data, offsets=[], shape=leading, str_offsets=parts.offsets
        )
    return RaggedLayout(data=parts.data, offsets=[parts.offsets], shape=parts.shape)


def _wrap_list(off: Any, content: Any) -> Any:
    """Wrap ``content`` in a list layout using the given offsets array.

    Does NOT add the ``__list__: "Ragged"`` behavior parameter — the registered
    behavior dispatches ``to_list()`` into the single-level ``_array.Ragged``
    which can't handle R=2 content.  Plain list layouts let awkward's default
    ``to_list()`` walk work correctly.
    """
    off = np.ascontiguousarray(off, dtype=OFFSET_TYPE)
    if off.ndim == 1:
        return ListOffsetArray(Index(off), content)
    return ListArray(
        Index(np.ascontiguousarray(off[0], dtype=OFFSET_TYPE)),
        Index(np.ascontiguousarray(off[1], dtype=OFFSET_TYPE)),
        content,
    )


def to_ak(rag: Any) -> ak.Array:
    from ._array import _parts_to_content, RagParts
    from ._layout import RecordLayout

    if isinstance(rag._layout, RecordLayout):
        return ak.zip({f: to_ak(rag[f]) for f in rag.fields}, depth_limit=1)

    # --- R=2 nested numeric path ---
    if rag._layout.n_ragged == 2:
        o0, o1 = rag._layout.offsets
        data = rag._rl.data
        if data.dtype.kind == "S":
            leaf = NumpyArray(
                np.ascontiguousarray(data).view(np.uint8),  # type: ignore[arg-type]
                parameters={"__array__": "byte"},
            )
        else:
            leaf = NumpyArray(np.ascontiguousarray(data))  # type: ignore[arg-type]
        return ak.Array(_wrap_list(o0, _wrap_list(o1, leaf)))

    # --- string-under-axis path (shape (L0, None), is_string=True) ---
    if rag._rl.str_offsets is not None and rag._layout.offsets:
        o0 = rag._layout.offsets[0]
        str_off = np.ascontiguousarray(rag._rl.str_offsets, dtype=OFFSET_TYPE)
        byte_leaf = NumpyArray(
            np.ascontiguousarray(rag._rl.data).view(np.uint8),  # type: ignore[arg-type]
            parameters={"__array__": "byte"},
        )
        inner = ak.with_parameter(
            ListOffsetArray(Index(str_off), byte_leaf),
            "__array__",
            "bytestring",
            highlevel=False,
        )
        return ak.Array(_wrap_list(o0, inner))

    # --- existing single-level path (unchanged) ---
    content = _parts_to_content(
        RagParts(
            rag.data,
            rag.shape if None in rag.shape else (*rag.shape, None),
            rag.offsets,
        )
    )
    return ak.Array(content)
