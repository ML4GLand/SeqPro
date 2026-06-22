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
    RecordArray,
    RegularArray,
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
        rec = rag._layout
        # Fast path: shape has only one leading fixed dim (e.g. (L0, None) or
        # (L0, None, None)). In this case per-field to_ak() + ak.zip works because all
        # per-field arrays have the same outermost length and no extra RegularArray wrappers.
        rag_dim = rec.shape.index(None)
        if rag_dim <= 1:
            return ak.zip({f: to_ak(rag[f]) for f in rag.fields}, depth_limit=1)

        # Multi-leading-axis path: shape like (b, p, ~v) where rag_dim >= 2 and there is
        # exactly one ragged axis.  Per-field to_ak() produces incompatible outer shapes:
        # - numeric fields get RegularArray wrappers from _parts_to_content
        # - string-under-axis fields do NOT (string path only wraps one level)
        # Fix: build leaf contents from layout buffers and re-wrap uniformly.
        field_leaves: list[Any] = []
        field_names: list[str] = []
        for name, fl in rec.fields.items():
            if fl.str_offsets is not None:
                # string-under-axis field: leaf is the bytestring ListOffsetArray
                str_off = np.ascontiguousarray(fl.str_offsets, dtype=OFFSET_TYPE)
                byte_leaf = NumpyArray(
                    np.ascontiguousarray(fl.data).view(np.uint8),  # type: ignore[arg-type]
                    parameters={"__array__": "byte"},
                )
                leaf: Any = ak.with_parameter(
                    ListOffsetArray(Index(str_off), byte_leaf),
                    "__array__",
                    "bytestring",
                    highlevel=False,
                )
            else:
                # Numeric or S1 char leaf; the shared offsets loop below handles wrapping.
                data = fl.data
                if data.dtype.kind == "S":
                    leaf = NumpyArray(
                        np.ascontiguousarray(data).view(np.uint8),  # type: ignore[arg-type]
                        parameters={"__array__": "byte"},
                    )
                else:
                    leaf = NumpyArray(np.ascontiguousarray(data))  # type: ignore[arg-type]
            field_leaves.append(leaf)
            field_names.append(name)

        content: Any = RecordArray(field_leaves, field_names)

        # Re-wrap by iterating reversed(shape[1:]), applying ListOffsetArray for None dims
        # (innermost ragged first) and RegularArray for fixed dims.  Shared offsets are
        # consumed innermost-to-outermost so that the outer list wraps the inner list.
        # The outermost dim (shape[0]) is implicit in the final array length.
        off_iter = iter(reversed(rec.offsets))
        for size in reversed(rec.shape[1:]):
            if size is None:
                off = next(off_iter)
                content = _wrap_list(off, content)
            else:
                content = RegularArray(content, size)

        return ak.Array(content)

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
