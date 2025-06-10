from __future__ import annotations

from copy import deepcopy
from typing import Callable, Generic, TypeVar, Union, cast

import awkward as ak
import numpy as np
from attrs import define
from awkward.contents import (
    Content,
    EmptyArray,
    ListArray,
    ListOffsetArray,
    NumpyArray,
    RegularArray,
)
from awkward.index import Index
from numpy.typing import NDArray
from typing_extensions import Self

__all__ = ["Ragged"]

ak_dtypes = Union[np.number, np.bytes_, np.datetime64, np.timedelta64]
DTYPE = TypeVar("DTYPE", bound=ak_dtypes)
RDTYPE = TypeVar("RDTYPE", bound=ak_dtypes)
LENGTH_TYPE = np.uint32
OFFSET_TYPE = np.int64


class Ragged(ak.Array, Generic[RDTYPE]):
    """An awkward array with exactly 1 ragged dimension. The ragged dimension is :code:`None` in its shape tuple.

    .. important::
        Ragged arrays only support a subset of Awkward array features.

        - Strings are not supported since ASCII is sufficient for the bioinformatics domain.
        - Byte strings will have their length encoded as an explicit dimension, which allows variable sized groups
        of fixed length strings to adhere to the Ragged invariant of exactly 1 ragged dimension. For example:

            .. code-block:: python

                seqs = sp.random_seqs((5,2), sp.DNA)
                lengths = np.array([2, 3])
                rag = Ragged.from_lengths(seqs, lengths)

                [[b'AT', b'AC'],
                [b'CG', b'CA', b'CC']]
                ---------------------------------
                type: 2 * var * 2 * bytes[ragged]

        - Ragged arrays are not tested with support for Awkward fields or union types. Any functionality that appears
        to work with fields is experimental.

    """

    _parts: RagParts[RDTYPE]

    def __init__(self, data: Content | ak.Array | Ragged[RDTYPE] | RagParts[RDTYPE]):
        if isinstance(data, RagParts):
            data = _parts_to_content(data)
        else:
            data = ak.with_parameter(data, "__list__", "ragged", highlevel=False)
        super().__init__(data, behavior=deepcopy(behavior))
        self._parts = unbox(self)
        type_parts = []
        name = self.parts.data.dtype.name
        if name == "bytes8":
            name = "bytes"
        else:
            type_parts.append("var")
        type_parts.extend([str(s) for s in self.shape[self.rag_dim + 1 :]])
        type_parts.append(f"ragged[{name}]")
        self.behavior["__typestr__", "ragged"] = " * ".join(type_parts)  # type: ignore

    @staticmethod
    def from_offsets(
        data: NDArray[DTYPE],
        offsets: NDArray[OFFSET_TYPE],
        shape: tuple[int | None, ...],
    ) -> Ragged[DTYPE]:
        """Create a Ragged array from data, offsets, and shape.

        Parameters
        ----------
        data
            The data to create the Ragged array from.
        offsets
            The offsets to create the Ragged array from.
        shape
            The shape of the Ragged array.
        """
        parts = RagParts[DTYPE](data, offsets, shape)
        return Ragged(parts)

    @staticmethod
    def from_lengths(
        data: NDArray[DTYPE], lengths: NDArray[np.integer]
    ) -> Ragged[DTYPE]:
        """Create a Ragged array from data and lengths.

        Parameters
        ----------
        data
            The data to create the Ragged array from.
        lengths
            The lengths of the segments.
        """
        parts = RagParts[DTYPE].from_lengths(data, lengths)
        return Ragged(parts)

    @property
    def parts(self) -> RagParts[RDTYPE]:
        """The parts of the Ragged array."""
        return self._parts

    @property
    def data(self) -> NDArray[RDTYPE]:
        """The data of the Ragged array."""
        return self._parts.data

    @property
    def offsets(self) -> NDArray[OFFSET_TYPE]:
        """The offsets of the Ragged array."""
        return self._parts.offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        """The shape of the Ragged array. The ragged dimension is :code:`None`."""
        return self._parts.shape

    @property
    def dtype(self) -> np.dtype[RDTYPE]:
        """The dtype of the Ragged array."""
        return self.data.dtype

    @property
    def rag_dim(self) -> int:
        """The index of the ragged dimension."""
        return self.shape.index(None)

    @property
    def lengths(self) -> NDArray[np.integer]:
        """The lengths of the segments."""
        if self.offsets.ndim == 1:
            lengths = np.diff(self.offsets)
        else:
            lengths = self.offsets[:, 1] - self.offsets[:, 0]

        return lengths.reshape(self.shape[: self.rag_dim])  # type: ignore

    def view(self, dtype: type[DTYPE] | str) -> Ragged[DTYPE]:
        """Return a view of the data with the given dtype."""
        # get a new layout, same data
        view = ak.without_parameters(self)

        # change view of the data
        parts = unbox(view)
        parts.data = parts.data.view(dtype)

        # init a new array with same base data
        view = Ragged(parts)
        return view

    @classmethod
    def empty(
        cls, shape: int | tuple[int | None, ...], dtype: type[RDTYPE]
    ) -> Ragged[RDTYPE]:
        """Create an empty Ragged array with the given shape and dtype."""
        data = np.empty(0, dtype=dtype)
        if isinstance(shape, int):
            shape = (shape,)
        rag_dim = shape.index(None)
        offsets = np.zeros(
            np.prod(shape[:rag_dim]) + 1,  # type: ignore
            dtype=OFFSET_TYPE,
        )
        parts = RagParts(data, offsets, shape)
        content = _parts_to_content(parts)
        return cls(content)

    @property
    def is_empty(self) -> bool:
        """Whether the Ragged array is empty."""
        return self.data.size == 0

    @property
    def is_contiguous(self) -> bool:
        """Whether the Ragged array is contiguous."""
        return self.offsets.ndim == 1 and self.data.flags.contiguous

    @property
    def is_base(self) -> bool:
        """Whether the Ragged array is a base array."""
        return (
            self.data.base is None
            and self.is_contiguous
            and self.offsets[0] == 0
            and self.offsets[-1] == self.data.size
        )

    def to_numpy(self, allow_missing: bool = False) -> NDArray[RDTYPE]:
        """Note: not zero-copy if offsets or data are non-contiguous."""
        arr = super().to_numpy(allow_missing=allow_missing)
        if self.dtype.type == np.bytes_:
            arr = arr.view("S1")
        return arr

    def __getitem__(self, where):
        arr = super().__getitem__(where)
        if isinstance(arr, ak.Array):
            if _n_var(arr) == 1:
                return type(self)(arr)
            else:
                return _without_ragged(arr)
        else:
            return arr

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> Self:
        """Squeeze the ragged array along the given non-ragged axis."""
        parts = RagParts[RDTYPE].from_lengths(
            self._parts.data, self.lengths.squeeze(axis)
        )
        content = _parts_to_content(parts)
        return type(self)(content)

    def reshape(self, shape: int | tuple[int | None, ...]) -> Self:
        """Reshape non-ragged axes."""
        # this is correct because all reshaping operations preserve the layout i.e. raveled ordered
        if isinstance(shape, int):
            shape = (shape,)
        rag_dim = shape.index(None)
        parts = RagParts[
            RDTYPE
        ].from_lengths(
            self._parts.data.reshape(-1, *shape[rag_dim + 1 :]),  # type: ignore
            self.lengths.reshape(shape[:rag_dim]),  # type: ignore
        )
        content = _parts_to_content(parts)
        return type(self)(content)

    def to_ak(self):
        """Convert to an Awkward array."""
        arr = _without_ragged(self)
        arr.behavior = None
        return arr

    def apply(self, gufunc: Callable[[NDArray[RDTYPE]], DTYPE]) -> Ragged[DTYPE]:
        """Apply a gufunc to the data of the Ragged array."""
        ...


behavior = deepcopy(ak.behavior)
behavior["*", "ragged"] = Ragged


def _n_var(arr: ak.Array) -> int:
    node = cast(Content, arr.layout)
    n_var = 0
    while not isinstance(node, (EmptyArray, NumpyArray)):
        if isinstance(node, (ListArray, ListOffsetArray)):
            n_var += 1
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    return n_var


def _without_ragged(arr: ak.Array | Ragged[DTYPE]) -> ak.Array:
    def fn(layout, **kwargs):
        if isinstance(layout, (ListArray, ListOffsetArray)):
            return ak.with_parameter(layout, "__list__", None, highlevel=False)

    return ak.transform(fn, arr)


def lengths_to_offsets(
    lengths: NDArray[np.integer], dtype: type[DTYPE] = OFFSET_TYPE
) -> NDArray[DTYPE]:
    """Convert lengths to offsets.

    Parameters
    ----------
    lengths
        Lengths of the segments.

    Returns
    -------
    offsets
        Offsets of the segments.
    """
    offsets = np.empty(lengths.size + 1, dtype=dtype)
    offsets[0] = 0
    offsets[1:] = lengths.cumsum()
    return offsets


@define
class RagParts(Generic[DTYPE]):
    data: NDArray[DTYPE]
    offsets: NDArray[OFFSET_TYPE]
    shape: tuple[int | None, ...]

    @property
    def contiguous(self) -> bool:
        return self.offsets.ndim == 1

    @classmethod
    def from_lengths(cls, data: NDArray[DTYPE], lengths: NDArray[np.integer]) -> Self:
        offsets = lengths_to_offsets(lengths)
        shape = (*lengths.shape, None, *data.shape[1:])
        return cls(data, offsets, shape)


def unbox(
    arr: ak.Array | Ragged[DTYPE], as_contiguous: bool = False
) -> RagParts[DTYPE]:
    """Unbox an awkward array with a single ragged dimension into data, offsets, and shape.
    Is guaranteed to be zero-copy if as_contiguous is False, in which case the data is a view
    of the original array.

    Parameters
    ----------
    arr
        The awkward array to unbox.
    as_contiguous
        If True, the data will be returned as a contiguous array. May force a copy into memory.
        If the underlying data is memory-mapped, this could cause an out-of-memory error.

    Returns
    -------
        Parts of the ragged array.
    """
    if as_contiguous:
        arr = ak.to_packed(arr)

    node = cast(Content, arr.layout)
    shape: list[int | None] = [len(node)]
    offsets = None

    while isinstance(node, (ListArray, ListOffsetArray, RegularArray)):
        if isinstance(node, RegularArray):
            shape.append(node.size)
        else:
            shape.append(None)
            if isinstance(node, ListOffsetArray):
                offsets = node.offsets.data
            else:
                offsets = np.stack(
                    [node.starts.data, node.stops.data],  # type: ignore
                    1,
                )

        node = node.content

    if isinstance(node, EmptyArray):
        node = node.to_NumpyArray(dtype=np.float64)

    if isinstance(node, NumpyArray):
        data = cast(NDArray, node.data)

        if node.parameter("__array__") == "byte":
            # view uint8 as bytes
            data = data.view("S1")

        shape.extend(data.shape[1:])

        if offsets is None:
            raise ValueError("Did not find offsets.")
        offsets = cast(NDArray, offsets)

        rag_dim = shape.index(None)
        reshape = cast(tuple[int, ...], (-1, *shape[rag_dim + 1 :]))
        return RagParts(data.reshape(reshape), offsets, tuple(shape))

    msg = f"Awkward Array type must have regular and irregular lists only, not:\n{arr.layout}"
    raise TypeError(msg)


def _parts_to_content(parts: RagParts[DTYPE]) -> Content:
    list_params = {"__list__": "ragged"}
    if parts.data.dtype.str == "|S1":
        list_params["__array__"] = "bytestring"
        layout = NumpyArray(
            parts.data.view(np.uint8).ravel(),  # type: ignore
            parameters={"__array__": "byte"},
        )
    else:
        layout = NumpyArray(parts.data.ravel())  # type: ignore

    for size in reversed(parts.shape[1:]):
        if size is None:
            if parts.contiguous:
                layout = ListOffsetArray(
                    Index(parts.offsets),
                    layout,
                    parameters=list_params,
                )
            else:
                layout = ListArray(
                    Index(parts.offsets[:, 0]),
                    Index(parts.offsets[:, 1]),
                    layout,
                    parameters=list_params,
                )
        else:
            layout = RegularArray(layout, size)

    if len(layout) != parts.shape[0]:
        raise ValueError(
            f"Length of layout {len(layout)} does not match size of first dimension {parts.shape[0]}"
        )

    return layout
