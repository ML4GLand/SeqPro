from __future__ import annotations

from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeGuard,
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
    RecordArray,
    RegularArray,
)
from awkward.index import Index
from numpy.typing import NDArray
from typing_extensions import ParamSpec, Self

from ._types import ak_dtypes
from ._utils import OFFSET_TYPE, lengths_to_offsets

DTYPE_co = TypeVar("DTYPE_co", bound=ak_dtypes, covariant=True)
RDTYPE_co = TypeVar("RDTYPE_co", bound=ak_dtypes, covariant=True)
P = ParamSpec("P")


def is_rag_dtype(
    rag: Any, dtype: DTYPE_co | type[DTYPE_co]
) -> TypeGuard[Ragged[DTYPE_co]]:
    """Check if an object is a `Ragged` array with the given dtype (fails for record-layout Ragged arrays).

    Parameters
    ----------
    rag
        Object to check.
    dtype
        Expected dtype.

    Returns
    -------
    TypeGuard[Ragged[DTYPE_co]]
        True if `rag` is a `Ragged` array whose dtype is a subtype of `dtype`.
    """
    if not isinstance(rag, Ragged):
        return False
    if isinstance(rag.dtype, dict):
        return False
    return np.issubdtype(rag.dtype, dtype)


def _is_record_layout(layout: Content) -> bool:
    """Return True if a list layer wraps a RecordArray (past any regular wrappers)."""
    node = layout
    has_list = False
    while isinstance(node, (ListOffsetArray, ListArray, RegularArray)):
        if isinstance(node, (ListOffsetArray, ListArray)):
            has_list = True
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    return has_list and isinstance(node, RecordArray)


def _extract_list_offsets(layout: Content) -> NDArray[OFFSET_TYPE]:
    """Extract offsets from the (single) list layer in a record-layout Ragged.

    The list layer can sit outside the RecordArray (e.g., `ak.zip` output:
    `ListOffsetArray(RecordArray(...))`) or inside it (e.g., dict-of-lists
    `ak.Array({"f0": [[...]], "f1": [[...]]})`: `RecordArray({"f0": ListOffsetArray, ...})`).
    Walks past any `RegularArray` / `RecordArray` (diving into field 0,
    since all fields share the same list layer for a Ragged record).

    Returns a 1-D `(N+1,)` array for `ListOffsetArray` or a 2-D `(2, N)`
    starts/stops array for `ListArray` — same convention as `unbox()`.
    """
    node = layout
    while True:
        if isinstance(node, ListOffsetArray):
            return cast(NDArray, node.offsets.data)
        if isinstance(node, ListArray):
            return np.stack([node.starts.data, node.stops.data], 0)  # type: ignore
        if isinstance(node, RegularArray):
            node = node.content
        elif isinstance(node, RecordArray):
            node = node.content(0)
        else:
            raise ValueError(  # noqa: TRY004
                f"No list layer found while extracting offsets from layout:\n{layout.form}"
            )


class Ragged(ak.Array, Generic[RDTYPE_co]):
    """An awkward array with exactly 1 ragged dimension. The ragged dimension is `None` in its shape tuple.

    !!! warning
        Ragged arrays only support a subset of Awkward array features.

        - Strings are not supported since ASCII is sufficient for the bioinformatics domain.
        - Bytestrings count as a ragged dimension, and we break from the Awkward convention to not include a "var" in the type string.
        - Record-layout Ragged arrays (produced by `ak.zip` of Ragged inputs or by passing a record-layout
          `ak.Array`) return field-keyed dicts from `dtype`, `data`, and `parts`. Use `rag["field"]`
          for zero-copy single-field access. `view`, `apply`, and `to_numpy` are not defined on record
          layouts; access individual fields. Union types remain unsupported.

    """

    _parts: RagParts[RDTYPE_co] | None

    def __init__(
        self,
        data: Content | ak.Array | Ragged[RDTYPE_co] | RagParts[RDTYPE_co],
    ):
        if isinstance(data, RagParts):
            content = _parts_to_content(data)
        else:
            content = _as_ragged(data, highlevel=False)
        super().__init__(content, behavior=deepcopy(ak.behavior))
        if isinstance(content, RecordArray) or _is_record_layout(content):
            # ak._update_class() demotes RecordArray layouts to plain ak.Array
            # because there is no "__list__" parameter at the record level.
            # Restore the Ragged subclass and leave _parts unset for records.
            self.__class__ = Ragged  # type: ignore[assignment]
            self._parts = None
        else:
            self._parts = unbox(self)

    def _ensure_parts(self) -> None:
        """Idempotent lazy init for `_parts`. Handles Ragged instances created
        via awkward behavior dispatch (e.g. `ak.zip`) that bypass `__init__`."""
        if hasattr(self, "_parts"):
            return
        layout = cast(Content, ak.to_layout(self))
        if isinstance(layout, RecordArray) or _is_record_layout(layout):
            object.__setattr__(self, "_parts", None)
        else:
            object.__setattr__(self, "_parts", unbox(self))

    @staticmethod
    def from_offsets(
        data: NDArray[DTYPE_co],
        shape: tuple[int | None, ...],
        offsets: NDArray[OFFSET_TYPE],
    ) -> Ragged[DTYPE_co]:
        """Create a Ragged array from data, offsets, and shape.

        Parameters
        ----------
        data
            The data to create the Ragged array from.
        shape
            The shape of the Ragged array.
        offsets
            The offsets to create the Ragged array from.

        Returns
        -------
        Ragged[DTYPE_co]
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

        parts = RagParts[DTYPE_co](data, shape, offsets)
        return Ragged(parts)

    @staticmethod
    def from_lengths(
        data: NDArray[DTYPE_co], lengths: NDArray[np.integer]
    ) -> Ragged[DTYPE_co]:
        """Create a Ragged array from data and lengths.

        Parameters
        ----------
        data
            The data to create the Ragged array from.
        lengths
            The lengths of the segments.

        Returns
        -------
        Ragged[DTYPE_co]
        """
        parts = RagParts[DTYPE_co].from_lengths(data, lengths)
        return Ragged(parts)

    @property
    def parts(self) -> RagParts[RDTYPE_co] | dict[str, RagParts]:
        """The parts of the Ragged array. For record layouts, a dict of
        field name -> RagParts; all share the same offsets ndarray.

        Returns
        -------
        RagParts[RDTYPE_co] | dict[str, RagParts]
        """
        self._ensure_parts()
        if self._parts is None:
            return {f: self[f].parts for f in ak.fields(self)}  # type: ignore[reportUnknownReturnType]
        return self._parts

    @property
    def data(self) -> NDArray[RDTYPE_co] | dict[str, NDArray]:
        """The data of the Ragged array. For record layouts, a dict of
        field name -> zero-copy ndarray view, in awkward field order.

        Returns
        -------
        NDArray[RDTYPE_co] | dict[str, NDArray]
        """
        self._ensure_parts()
        if self._parts is None:
            return {f: self[f].data for f in ak.fields(self)}  # type: ignore[reportUnknownReturnType]
        return self._parts.data

    @property
    def offsets(self) -> NDArray[OFFSET_TYPE]:
        """The offsets of the Ragged array. May have shape (n_ragged + 1) or (2, n_ragged).

        Returns
        -------
        NDArray[np.int64]
        """
        self._ensure_parts()
        if self._parts is None:
            # Record layout — extract offsets via the unified helper, cache for sharing.
            # object.__setattr__ used in case ak.Array intercepts __setattr__.
            if not hasattr(self, "_offsets_cache"):
                layout = cast(Content, ak.to_layout(self))
                offsets = _extract_list_offsets(layout)
                object.__setattr__(self, "_offsets_cache", offsets)
            return self._offsets_cache  # type: ignore[return-value]
        return self._parts.offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        """The shape of the Ragged array. The ragged dimension is `None`.

        Returns
        -------
        tuple[int | None, ...]
        """
        self._ensure_parts()
        if self._parts is None:
            # All fields share the ragged structure; derive from any.
            return self[ak.fields(self)[0]].shape
        return self._parts.shape

    @property
    def dtype(self) -> np.dtype[RDTYPE_co] | dict[str, np.dtype]:
        """The dtype of the Ragged array. For record layouts, a dict of
        field name -> dtype, in awkward field order.

        Returns
        -------
        np.dtype[RDTYPE_co] | dict[str, np.dtype]
        """
        self._ensure_parts()
        if self._parts is None:
            return {f: self[f].dtype for f in ak.fields(self)}  # type: ignore[reportUnknownReturnType]
        return self._parts.data.dtype

    @property
    def rag_dim(self) -> int:
        """The index of the ragged dimension.

        Returns
        -------
        int
        """
        return self.shape.index(None)

    @property
    def lengths(self) -> NDArray[np.integer]:
        """The lengths of the segments.

        Returns
        -------
        NDArray[np.integer]
        """
        if self.offsets.ndim == 1:
            lengths = np.diff(self.offsets)
        else:
            lengths = np.diff(self.offsets, axis=0)

        return lengths.reshape(self.shape[: self.rag_dim])  # type: ignore

    def view(self, dtype: type[DTYPE_co] | str) -> Ragged[DTYPE_co]:
        """Return a view of the data with the given dtype.

        Parameters
        ----------
        dtype
            Target dtype.

        Returns
        -------
        Ragged[DTYPE_co]
            Zero-copy view with reinterpreted dtype.
        """
        self._ensure_parts()
        if self._parts is None:
            raise NotImplementedError(
                "view is not defined on record-layout Ragged arrays; "
                "update fields individually, e.g. rag['f'] = rag['f'].view(dtype)."
            )
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
        cls, shape: int | tuple[int | None, ...], dtype: type[DTYPE_co]
    ) -> Ragged[DTYPE_co]:
        """Create an empty Ragged array with the given shape and dtype.

        Parameters
        ----------
        shape
            Shape of the array. Must include exactly one `None` for the ragged dimension.
        dtype
            Element dtype.

        Returns
        -------
        Ragged[DTYPE_co]
        """
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
        return cast(Ragged[DTYPE_co], cls(content))

    @property
    def is_empty(self) -> bool:
        """Whether the Ragged array is empty.

        Returns
        -------
        bool
        """
        if self.offsets.ndim == 1:
            return self.offsets[-1] == 0
        else:
            return np.all(self.offsets[0] == self.offsets[1]).item()

    @property
    def is_contiguous(self) -> bool:
        """Whether the Ragged array is contiguous.

        Returns
        -------
        bool
        """
        contiguous_offsets = self.offsets.ndim == 1
        if isinstance(self.data, dict):
            contiguous_data = all(d.flags.contiguous for d in self.data.values())
        else:
            contiguous_data = self.data.flags.contiguous
        return contiguous_offsets and contiguous_data

    @property
    def is_base(self) -> bool:
        """Whether the Ragged array is a base array (owns its data, contiguous, no offset).

        Returns
        -------
        bool
        """
        if isinstance(self.data, dict):
            base_data = all(d.base is None for d in self.data.values())
            data_size = next(iter(self.data.values())).size
        else:
            base_data = self.data.base is None
            data_size = self.data.size
        return (
            base_data
            and self.is_contiguous
            and self.offsets[0] == 0
            and self.offsets[-1] == data_size
        )

    def to_numpy(self, allow_missing: bool = False) -> NDArray[RDTYPE_co]:
        """Convert to a dense NumPy array. Not zero-copy if offsets or data are non-contiguous.

        Parameters
        ----------
        allow_missing
            Passed through to `ak.Array.to_numpy`.

        Returns
        -------
        NDArray[RDTYPE_co]
        """
        self._ensure_parts()
        if self._parts is None:
            raise NotImplementedError(
                "to_numpy is not defined on record-layout Ragged arrays; "
                "convert fields individually."
            )
        arr = super().to_numpy(allow_missing=allow_missing)
        if self.dtype.type == np.bytes_:  # type: ignore[attr-defined] guaranteed by record check
            arr = arr[..., None].view("S1")
        return arr

    def __getitem__(self, where):
        arr = super().__getitem__(where)
        if isinstance(arr, ak.Array):
            if _n_var(arr) == 1:
                result = type(self)(arr)
                # For record field access, share the parent's offsets object (zero-copy).
                self._ensure_parts()
                if isinstance(where, str) and self._parts is None:
                    result._parts = RagParts(
                        result._parts.data,  # type: ignore[reportUnknownAttribute]
                        result._parts.shape,  # type: ignore[reportUnknownAttribute]
                        self.offsets,
                    )
                return result
            else:
                return _as_ak(arr)
        else:
            return arr

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> Self | NDArray[RDTYPE_co] | dict[str, NDArray[RDTYPE_co]]:
        """Squeeze the ragged array along the given non-ragged axis.
        If squeezing would result in a 1D array, return the data as a numpy array.
        For record layouts, dispatches per-field; if fields collapse to 1D ndarrays,
        returns a dict of ndarrays, otherwise returns a record Ragged.

        Parameters
        ----------
        axis
            Axis or axes to squeeze. Must have size 1. If `None`, squeeze all size-1 axes.

        Returns
        -------
        Self | NDArray[RDTYPE_co] | dict[str, NDArray[RDTYPE_co]]
        """
        self._ensure_parts()
        if self._parts is None:
            squeezed = {f: self[f].squeeze(axis) for f in ak.fields(self)}
            first = next(iter(squeezed.values()))
            if isinstance(first, np.ndarray):
                return squeezed  # type: ignore[reportUnknownReturnType]
            return type(self)(ak.zip(squeezed, depth_limit=1))  # type: ignore[reportUnknownReturnType]
        if axis is None:
            data = self._parts.data.squeeze()
            shape = tuple(s for s in self.shape if s != 1)
            parts = RagParts[RDTYPE_co](data, shape, self.offsets)
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

        parts = RagParts[RDTYPE_co](data, shape, self.offsets)
        return type(self)(parts)

    def reshape(self, *shape: int | None | tuple[int | None, ...]) -> Self:
        """Reshape non-ragged axes.

        Parameters
        ----------
        *shape
            New shape including exactly one `None` for the ragged dimension.

        Returns
        -------
        Self
        """
        self._ensure_parts()
        if self._parts is None:
            reshaped = {f: self[f].reshape(*shape) for f in ak.fields(self)}
            return type(self)(ak.zip(reshaped, depth_limit=1))
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
        parts = RagParts[RDTYPE_co](data, new_shape, self.offsets)
        return type(self)(parts)

    def to_ak(self) -> ak.Array:
        """Convert to a plain Awkward array, stripping the Ragged behavior.

        Returns
        -------
        ak.Array
        """
        arr = _as_ak(self)
        arr.behavior = None
        return arr


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
    while not isinstance(node, (EmptyArray, NumpyArray, RecordArray)):
        if isinstance(node, (ListArray, ListOffsetArray)):
            n_var += 1
        node = node.content  # type: ignore[reportAttributeAccessIssue]
    return n_var


@overload
def _as_ragged(
    arr: ak.Array | Content, highlevel: Literal[True] = True
) -> ak.Array: ...
@overload
def _as_ragged(arr: ak.Array | Content, highlevel: Literal[False]) -> Content: ...
def _as_ragged(arr: ak.Array | Content, highlevel: bool = True) -> ak.Array | Content:
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
def _as_ak(
    arr: ak.Array | Ragged[DTYPE_co], highlevel: Literal[True] = True
) -> ak.Array: ...
@overload
def _as_ak(arr: ak.Array | Ragged[DTYPE_co], highlevel: Literal[False]) -> Content: ...
def _as_ak(
    arr: ak.Array | Ragged[DTYPE_co], highlevel: bool = True
) -> ak.Array | Content:
    def fn(layout, **kwargs):
        if isinstance(layout, (ListArray, ListOffsetArray)):
            return ak.with_parameter(layout, "__list__", None, highlevel=False)

    return ak.transform(fn, arr, highlevel=highlevel)  # type: ignore


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


def unbox(
    arr: ak.Array | Ragged[DTYPE_co], as_contiguous: bool = False
) -> RagParts[DTYPE_co]:
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
        result
    """
    if as_contiguous:
        arr = ak.to_packed(arr)

    node = cast(Content, ak.to_layout(arr, allow_record=False))
    shape: list[int | None] = [len(node)]
    n_ragged = 0
    offsets = None

    while isinstance(node, (ListArray, ListOffsetArray, RegularArray, RecordArray)):
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


def _parts_to_content(parts: RagParts[DTYPE_co]) -> Content:
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
        raise ValueError("Data is effectively a 1D array, and thus not ragged.")  # noqa: TRY004

    if len(layout) != parts.shape[0]:
        raise ValueError(
            f"Length of layout {len(layout)} does not match size of first dimension {parts.shape[0]}"
        )

    return layout
