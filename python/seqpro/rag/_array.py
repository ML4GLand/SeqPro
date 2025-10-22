from __future__ import annotations

from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)

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
from typing_extensions import Concatenate, ParamSpec, Self, TypeGuard

from ._types import ak_dtypes
from ._utils import OFFSET_TYPE, lengths_to_offsets

DTYPE = TypeVar("DTYPE", bound=ak_dtypes, covariant=True)
RDTYPE = TypeVar("RDTYPE", bound=ak_dtypes, covariant=True)
P = ParamSpec("P")


def is_rag_dtype(rag: Any, dtype: type[DTYPE]) -> TypeGuard[Ragged[DTYPE]]:
    return isinstance(rag, Ragged) and np.issubdtype(rag.dtype, dtype)


class Ragged(ak.Array, Generic[RDTYPE]):
    """An awkward array with exactly 1 ragged dimension. The ragged dimension is :code:`None` in its shape tuple.

    .. important::
        Ragged arrays only support a subset of Awkward array features.

        - Strings are not supported since ASCII is sufficient for the bioinformatics domain.
        - Bytestrings count as a ragged dimension, and we break from the Awkward convention to not include a "var" in the type string.
        - Ragged arrays are not tested with support for Awkward fields or union types. Any functionality that appears
        to work with fields is experimental.

    """

    _parts: RagParts[RDTYPE]

    def __init__(
        self,
        data: Content | ak.Array | Ragged[RDTYPE] | RagParts[RDTYPE],
    ):
        if isinstance(data, RagParts):
            content = _parts_to_content(data)
        else:
            content = _with_ragged(data, highlevel=False)
        super().__init__(content, behavior=deepcopy(ak.behavior))
        self._parts = unbox(self)
        type_parts: list[str] = []
        type_parts.append("var")
        type_parts.extend([str(s) for s in self.shape[self.rag_dim + 1 :]])
        type_parts.append(Ragged.__name__)
        self.behavior["__typestr__", Ragged.__name__] = " * ".join(type_parts)  # type: ignore

    @staticmethod
    def from_offsets(
        data: NDArray[DTYPE],
        shape: tuple[int | None, ...],
        offsets: NDArray[OFFSET_TYPE],
    ) -> Ragged[DTYPE]:
        """Create a Ragged array from data, offsets, and shape.

        Parameters
        ----------
        data
            The data to create the Ragged array from.
        shape
            The shape of the Ragged array.
        offsets
            The offsets to create the Ragged array from.
        """
        try:
            rag_dim = shape.index(None)
        except ValueError:
            raise ValueError("Shape must have exactly one None dimension.")

        if offsets.ndim == 1:
            n_rag = len(offsets) - 1
        else:
            n_rag = offsets.shape[1]
        if n_rag != np.prod(shape[:rag_dim], dtype=int):  # type: ignore
            raise ValueError(
                f"Number of ragged segments {n_rag} does not match product of ragged components of shape {shape[:rag_dim]}"
            )

        if offsets.ndim == 1:
            size = offsets[-1] * np.prod(shape[rag_dim + 1 :], dtype=int)  # type: ignore
            if data.size != size:
                raise ValueError(
                    f"Data size {data.size} does not match size implied by shape and contiguous offsets: {size}"
                )

        parts = RagParts[DTYPE](data, shape, offsets)
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
        if not hasattr(self, "_parts"):
            self._parts = unbox(self)
        return self._parts

    @property
    def data(self) -> NDArray[RDTYPE]:
        """The data of the Ragged array."""
        return self.parts.data

    @property
    def offsets(self) -> NDArray[OFFSET_TYPE]:
        """The offsets of the Ragged array. May be 1- or 2-dimensional."""
        return self.parts.offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        """The shape of the Ragged array. The ragged dimension is :code:`None`."""
        return self.parts.shape

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
            lengths = np.diff(self.offsets, axis=0)

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
        cls, shape: int | tuple[int | None, ...], dtype: type[DTYPE]
    ) -> Ragged[DTYPE]:
        """Create an empty Ragged array with the given shape and dtype."""
        data = np.empty(0, dtype=dtype)
        if isinstance(shape, int):
            shape = (shape,)
        rag_dim = shape.index(None)
        offsets = np.zeros(
            np.prod(shape[:rag_dim]) + 1,  # type: ignore
            dtype=OFFSET_TYPE,
        )
        parts = RagParts(data, shape, offsets)
        content = _parts_to_content(parts)
        return cast(Ragged[DTYPE], cls(content))

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
            arr = arr[:, None].view("S1")
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

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> Self | NDArray[RDTYPE]:
        """Squeeze the ragged array along the given non-ragged axis.
        If squeezing would result in a 1D array, return the data as a numpy array."""
        if axis is None:
            data = self._parts.data.squeeze()
            shape = tuple(s for s in self.shape if s != 1)
            parts = RagParts[RDTYPE](data, shape, self.offsets)
            return type(self)(parts)

        if isinstance(axis, int):
            axis = (axis,)
        axis = tuple(a if a >= 0 else self.ndim + a + 1 for a in axis)
        for a in axis:
            if (size := self.shape[a]) != 1:
                raise ValueError(f"Cannot squeeze axis {a} of size {size}.")

        shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)
        data_shape = tuple(
            s for i, s in enumerate(self.shape) if i not in axis and i > self.rag_dim
        )
        data = self._parts.data.reshape(len(self._parts.data), *data_shape)

        if shape == (None,):
            return data

        parts = RagParts[RDTYPE](data, shape, self.offsets)
        return type(self)(parts)

    def reshape(self, *shape: int | None | tuple[int | None, ...]) -> Self:
        """Reshape non-ragged axes."""
        # this is correct because all reshaping operations preserve the layout i.e. raveled ordered
        if isinstance(shape[0], tuple):
            if len(shape) > 1:
                raise ValueError("Cannot mix tuple and non-tuple shapes.")
            shape = cast(tuple[tuple[int | None, ...]], shape)
            shape = shape[0]

        if TYPE_CHECKING:
            shape = cast(tuple[int | None, ...], shape)

        rag_dim = shape.index(None)
        rag_shape = cast(tuple[int, ...], self.shape[: self.rag_dim])
        n_rag = np.prod(rag_shape)
        new_rag_shape = cast(tuple[int, ...], shape[:rag_dim])
        n_new_rag = abs(np.prod(new_rag_shape))
        new_rag_shape = tuple(
            s if s >= 0 else int(n_rag // n_new_rag) for s in new_rag_shape
        )
        data = self._parts.data.reshape(len(self._parts.data), *shape[rag_dim + 1 :])
        new_shape = (*new_rag_shape, None, *data.shape[1:])
        parts = RagParts[RDTYPE](data, new_shape, self.offsets)
        return type(self)(parts)

    def to_ak(self):
        """Convert to an Awkward array."""
        arr = _without_ragged(self)
        arr.behavior = None
        return arr

    def apply(
        self,
        gufunc: Callable[Concatenate[NDArray[RDTYPE], P], NDArray[DTYPE]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Ragged[DTYPE]:
        """Apply a gufunc to the data of the Ragged array that does not alter the shape of the data."""
        data = gufunc(self.data, *args, **kwargs)
        parts = RagParts(data, self.shape, self.offsets)
        return Ragged(parts)


def apply_ufunc(
    ufunc: np.ufunc, method: str, args: tuple[Any, ...], kwargs: dict[str, Any]
):
    args = tuple(a.to_ak() if isinstance(a, Ragged) else a for a in args)
    return Ragged(getattr(ufunc, method)(*args, **kwargs))


ak.behavior["*", Ragged.__name__] = Ragged
ak.behavior[np.ufunc, Ragged.__name__] = apply_ufunc


def _n_var(arr: ak.Array) -> int:
    node = cast(Content, arr.layout)
    n_var = 0
    while not isinstance(node, (EmptyArray, NumpyArray)):
        if isinstance(node, (ListArray, ListOffsetArray)):
            n_var += 1
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    return n_var


@overload
def _with_ragged(
    arr: ak.Array | Content, highlevel: Literal[True] = True
) -> ak.Array: ...
@overload
def _with_ragged(arr: ak.Array | Content, highlevel: Literal[False]) -> Content: ...
def _with_ragged(arr: ak.Array | Content, highlevel: bool = True) -> ak.Array | Content:
    def fn(layout: Content, **kwargs):
        if isinstance(layout, (ListArray, ListOffsetArray)):
            return ak.with_parameter(
                layout, "__list__", Ragged.__name__, highlevel=False
            )
        else:
            if layout._parameters is not None:
                layout._parameters = None

    return ak.transform(fn, arr, highlevel=highlevel)  # type: ignore


@overload
def _without_ragged(
    arr: ak.Array | Ragged[DTYPE], highlevel: Literal[True] = True
) -> ak.Array: ...
@overload
def _without_ragged(
    arr: ak.Array | Ragged[DTYPE], highlevel: Literal[False]
) -> Content: ...
def _without_ragged(
    arr: ak.Array | Ragged[DTYPE], highlevel: bool = True
) -> ak.Array | Content:
    def fn(layout, **kwargs):
        if isinstance(layout, (ListArray, ListOffsetArray)):
            return ak.with_parameter(layout, "__list__", None, highlevel=False)

    return ak.transform(fn, arr, highlevel=highlevel)  # type: ignore


@define
class RagParts(Generic[DTYPE]):
    data: NDArray[DTYPE]
    shape: tuple[int | None, ...]
    offsets: NDArray[OFFSET_TYPE]
    """(n_ragged + 1) or (2, n_ragged)"""

    @property
    def contiguous(self) -> bool:
        return self.offsets.ndim == 1

    @classmethod
    def from_lengths(cls, data: NDArray[DTYPE], lengths: NDArray[np.integer]) -> Self:
        offsets = lengths_to_offsets(lengths)
        shape = (*lengths.shape, None, *data.shape[1:])
        return cls(data, shape, offsets)


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
    n_ragged = 0
    offsets = None

    while isinstance(node, (ListArray, ListOffsetArray, RegularArray)):
        if isinstance(node, RegularArray):
            shape.append(node.size)
        else:
            shape.append(None)
            n_ragged += 1
            if isinstance(node, ListOffsetArray):
                offsets = node.offsets.data
            else:
                offsets = np.stack(
                    [node.starts.data, node.stops.data],  # type: ignore
                    0,
                )

        node = node.content

    if n_ragged != 1:
        raise ValueError(f"Expected 1 ragged dimension, got {n_ragged}")

    if isinstance(node, EmptyArray):
        node = node.to_NumpyArray(dtype=np.float64)

    if isinstance(node, NumpyArray):
        data = cast(NDArray, node.data)  # type: ignore

        if node.parameter("__array__") == "byte":
            # view uint8 as bytes
            data = data.view("S1")

        shape.extend(data.shape[1:])

        if offsets is None:
            raise ValueError("Did not find offsets.")
        offsets = cast(NDArray, offsets)

        rag_dim = shape.index(None)
        reshape = cast(tuple[int, ...], (-1, *shape[rag_dim + 1 :]))
        return RagParts(data.reshape(reshape), tuple(shape), offsets)

    msg = f"Awkward Array type must have regular and irregular lists only, not:\n{arr.layout}"
    raise TypeError(msg)


def _parts_to_content(parts: RagParts[DTYPE]) -> Content:
    if parts.data.ndim > 1:
        parts.data = parts.data.ravel()

    if parts.data.dtype.str == "|S1":
        layout = NumpyArray(
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
                layout, "__list__", Ragged.__name__, highlevel=False
            )
        else:
            layout = RegularArray(layout, size)

        if i == 0 and parts.data.dtype.str == "|S1":
            layout = ak.with_parameter(
                layout, "__array__", "bytestring", highlevel=False
            )

    if isinstance(layout, NumpyArray):
        raise ValueError("Data is effectively a 1D array, and thus not ragged.")

    if len(layout) != parts.shape[0]:
        raise ValueError(
            f"Length of layout {len(layout)} does not match size of first dimension {parts.shape[0]}"
        )

    return layout
