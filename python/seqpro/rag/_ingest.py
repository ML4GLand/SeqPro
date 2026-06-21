from __future__ import annotations

from typing import Any

import awkward as ak
import numpy as np

from ._array import unbox  # proven awkward extractor (RagParts: data, shape, offsets)
from ._layout import RaggedLayout
from ._utils import OFFSET_TYPE


def layout_from_ak(arr: Any) -> "RaggedLayout[Any] | Any":
    if ak.fields(arr):
        from ._array import _extract_list_offsets
        from ._layout import RecordLayout

        shared = np.ascontiguousarray(
            _extract_list_offsets(ak.to_layout(arr)), dtype=OFFSET_TYPE
        )
        fields: dict[str, RaggedLayout[Any]] = {}
        rec_shape: tuple[int | None, ...] | None = None
        for f in ak.fields(arr):
            parts = unbox(
                arr[f]
            )  # data, shape (with None), offsets (ignored; use shared)
            # records hold chars, not opaque strings: keep the None axis as-is
            fields[f] = RaggedLayout(
                data=parts.data, offsets=[shared], shape=parts.shape
            )
            if rec_shape is None:
                rec_shape = parts.shape
        assert rec_shape is not None
        return RecordLayout(offsets=[shared], shape=rec_shape, fields=fields)

    parts = unbox(arr)
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


def to_ak(rag: Any) -> ak.Array:
    from ._array import _parts_to_content, RagParts
    from ._layout import RecordLayout

    if isinstance(rag._layout, RecordLayout):
        return ak.zip({f: to_ak(rag[f]) for f in rag.fields}, depth_limit=1)

    content = _parts_to_content(
        RagParts(
            rag.data,
            rag.shape if None in rag.shape else (*rag.shape, None),
            rag.offsets,
        )
    )
    return ak.Array(content)
