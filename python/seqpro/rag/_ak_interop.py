"""Interop helpers between seqpro _core.Ragged buffers and awkward-array layouts.

These functions are used by _ingest.py's ``layout_from_ak`` / ``to_ak`` paths to
convert between awkward's Content tree and the flat (data, offsets, shape) buffer
representation used by _core.Ragged.  Awkward remains a seqpro interop/test
dependency; these helpers centralise the boundary code so that the main _core
module has no awkward imports.
"""

from __future__ import annotations

from typing import Any, cast

import awkward as ak
import numpy as np
from awkward.contents import (
    Content,
    EmptyArray,
    IndexedArray,
    IndexedOptionArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
    RecordArray,
)
from awkward.index import Index
from attrs import define
from numpy.typing import NDArray
from typing_extensions import Generic, Self, TypeVar

from ._types import ak_dtypes
from ._utils import OFFSET_TYPE, lengths_to_offsets

DTYPE_co = TypeVar("DTYPE_co", bound=ak_dtypes | np.void, covariant=True)
RDTYPE_co = TypeVar("RDTYPE_co", bound=ak_dtypes | np.void, covariant=True)

# The string used as the __list__ behavior parameter for the legacy awkward Ragged backend.
# Kept here so that _parts_to_content can still produce awkward arrays that carry it (for
# round-trip fidelity in to_ak / tests that compare against old ak output).
_AK_RAGGED_NAME = "Ragged"


@define
class RagParts(Generic[DTYPE_co]):
    data: NDArray[DTYPE_co]
    shape: tuple[int | None, ...]
    offsets: NDArray[OFFSET_TYPE]
    """(n_ragged + 1) or (2, n_ragged)"""

    @property
    def contiguous(self) -> bool:
        """Whether offsets are stored as a contiguous (N+1,) array rather than (2, N) starts/stops.

        Returns
        -------
        bool
        """
        return self.offsets.ndim == 1

    @classmethod
    def from_lengths(
        cls, data: NDArray[DTYPE_co], lengths: NDArray[np.integer]
    ) -> Self:
        """Create a RagParts from data and segment lengths.

        Parameters
        ----------
        data
            Flat data array.
        lengths
            Lengths of the segments.

        Returns
        -------
        Self
        """
        offsets = lengths_to_offsets(lengths)
        shape = (*lengths.shape, None, *data.shape[1:])
        return cls(data, shape, offsets)


def unbox(arr: ak.Array) -> "RagParts[Any]":
    """Unbox an awkward array with a single ragged dimension into data, offsets, and shape.

    Always a view: the returned data references the original flat buffer. Indexed
    layouts (e.g. from fancy-indexing a record then extracting a field) are collapsed
    via ``project()``, which only reorders the list pointers into ``starts``/``stops``
    (the flat data is still shared), so callers needing contiguous, row-ordered data
    must pack afterwards (e.g. ``to_packed``).

    Parameters
    ----------
    arr
        The awkward array to unbox.

    Returns
    -------
    RagParts
        Data, shape, and offsets extracted from the awkward array.
    """
    node = cast(Content, ak.to_layout(arr, allow_record=False))
    shape: list[int | None] = [len(node)]
    n_ragged = 0
    offsets = None

    while isinstance(
        node,
        (
            ListArray,
            ListOffsetArray,
            RegularArray,
            RecordArray,
            IndexedArray,
            IndexedOptionArray,
        ),
    ):
        if isinstance(node, (IndexedArray, IndexedOptionArray)):
            # Ragged carries no missing values; project() collapses the index.
            # (An option array with real None would drop rows and desync shape[0].)
            node = node.project()
            continue
        if isinstance(node, RecordArray):
            raise ValueError(  # noqa: TRY004
                "Must extract a single field before unboxing a Ragged array of records."
            )
        elif isinstance(node, RegularArray):
            shape.append(node.size)
        else:
            shape.append(None)
            n_ragged += 1
            if isinstance(node, ListOffsetArray):
                offsets = node.offsets.data
            else:
                offsets = np.stack(  # pyrefly: ignore[no-matching-overload]
                    [node.starts.data, node.stops.data],  # type: ignore
                    0,
                )

        node = node.content

    if n_ragged != 1:
        raise ValueError(f"Expected 1 ragged dimension, got {n_ragged}")

    if isinstance(node, EmptyArray):
        node = node.to_NumpyArray(dtype=np.float64)

    if isinstance(node, NumpyArray):
        data = cast(NDArray[Any], node.data)  # type: ignore

        if node.parameter("__array__") == "byte":
            # view uint8 as bytes
            data = data.view("S1")

        shape.extend(data.shape[1:])

        if offsets is None:
            raise ValueError("Did not find offsets.")
        offsets = cast(NDArray[Any], offsets)

        rag_dim = shape.index(None)
        reshape = cast(tuple[int, ...], (-1, *shape[rag_dim + 1 :]))
        return RagParts(data.reshape(reshape), tuple(shape), offsets)

    msg = f"Awkward Array type must have regular and irregular lists only, not:\n{arr.layout}"
    raise TypeError(msg)


def _parts_to_content(parts: "RagParts[Any]") -> Content:
    if parts.data.ndim > 1:
        parts.data = parts.data.ravel()

    if parts.data.dtype.str == "|S1":
        layout: Content = NumpyArray(
            parts.data.view(np.uint8),  # type: ignore
            parameters={"__array__": "byte"},
        )
    else:
        layout = NumpyArray(parts.data)  # type: ignore

    for i, size in enumerate(reversed(parts.shape[1:])):
        if size is None:
            if parts.contiguous:
                layout = ListOffsetArray(Index(parts.offsets), layout)
            else:
                layout = ListArray(
                    Index(parts.offsets[0, :]), Index(parts.offsets[1, :]), layout
                )
            layout = ak.with_parameter(
                layout, "__list__", _AK_RAGGED_NAME, highlevel=False
            )
        else:
            layout = RegularArray(layout, size)

        if i == 0 and parts.data.dtype.str == "|S1":
            layout = ak.with_parameter(
                layout, "__array__", "bytestring", highlevel=False
            )

    if isinstance(layout, NumpyArray):
        raise ValueError("Data is effectively a 1D array, and thus not ragged.")  # noqa: TRY004

    if len(layout) != parts.shape[0]:
        raise ValueError(
            f"Length of layout {len(layout)} does not match size of first dimension {parts.shape[0]}"
        )

    return layout
