from __future__ import annotations

from typing import Any

import awkward as ak

from ._array import unbox  # proven awkward extractor (RagParts: data, shape, offsets)
from ._layout import RaggedLayout


def layout_from_ak(arr: Any) -> RaggedLayout[Any]:
    if ak.fields(arr):
        raise NotImplementedError("record-layout Ragged lands in Spec B")
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

    content = _parts_to_content(
        RagParts(
            rag.data,
            rag.shape if None in rag.shape else (*rag.shape, None),
            rag.offsets,
        )
    )
    return ak.Array(content)
