from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

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
from awkward.types.listtype import ListType as _ListType
from awkward.types.regulartype import RegularType as _RegularType
from numpy.typing import NDArray
from typing_extensions import ParamSpec, Self, TypeIs

from ._types import ak_dtypes
from ._utils import OFFSET_TYPE, lengths_to_offsets

# Patch ListType._get_typestr to support callable __typestr__ values so that
# the Ragged type string can be computed dynamically from the content type.
_orig_list_get_typestr = _ListType._get_typestr


def _callable_list_get_typestr(self, behavior):
    typestr = _orig_list_get_typestr(self, behavior)
    if callable(typestr):
        return typestr(self._content, behavior)
    return typestr


_ListType._get_typestr = _callable_list_get_typestr  # type: ignore[method-assign]

DTYPE_co = TypeVar("DTYPE_co", bound=ak_dtypes | np.void, covariant=True)
RDTYPE_co = TypeVar("RDTYPE_co", bound=ak_dtypes | np.void, covariant=True)
P = ParamSpec("P")


def is_rag_dtype(
    rag: Any, dtype: DTYPE_co | type[DTYPE_co]
) -> TypeIs[Ragged[DTYPE_co]]:
    """Check if an object is a `Ragged` array with the given dtype (fails for record-layout Ragged arrays).

    Parameters
    ----------
    rag
        Object to check.
    dtype
        Expected dtype.

    Returns
    -------
    TypeIs[Ragged[DTYPE_co]]
        True if `rag` is a `Ragged` array whose dtype is a subtype of `dtype`.
    """
    if not isinstance(rag, Ragged):
        return False
    if np.issubdtype(rag.dtype, np.void):  # structured dtype → record layout
        if not np.issubdtype(dtype, np.void):
            return False  # can't match structured Ragged with primitive dtype
        return rag.dtype == np.dtype(dtype)
    return np.issubdtype(rag.dtype, dtype)


def _is_record_layout(layout: Content) -> bool:
    """Return True if a list layer wraps a RecordArray (past any regular wrappers)."""
    node = layout
    has_list = False
    while isinstance(node, (ListOffsetArray, ListArray, RegularArray)):
        if isinstance(node, (ListOffsetArray, ListArray)):
            has_list = True
        node = node.content
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
            return np.asarray(node.offsets.data)
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


class _PartsDescriptor:
    """Descriptor for `Ragged.parts` with self-typed overloads."""

    @overload
    def __get__(self, obj: Ragged[np.void], objtype: Any) -> dict[str, RagParts]: ...
    @overload
    def __get__(self, obj: Ragged[RDTYPE_co], objtype: Any) -> RagParts[RDTYPE_co]: ...
    @overload
    def __get__(self, obj: None, objtype: Any) -> Self: ...
    def __get__(self, obj: Ragged | None, objtype: Any = None):
        if obj is None:
            return self
        obj._ensure_parts()
        return obj._parts


class _DataDescriptor:
    """Descriptor for `Ragged.data` with self-typed overloads."""

    @overload
    def __get__(self, obj: Ragged[np.void], objtype: Any) -> dict[str, NDArray]: ...
    @overload
    def __get__(self, obj: Ragged[RDTYPE_co], objtype: Any) -> NDArray[RDTYPE_co]: ...
    @overload
    def __get__(self, obj: None, objtype: Any) -> Self: ...
    def __get__(self, obj: Ragged | None, objtype: Any = None):
        if obj is None:
            return self
        obj._ensure_parts()
        if isinstance(obj._parts, dict):
            return {f: p.data for f, p in obj._parts.items()}
        return obj._parts.data


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

    _parts: RagParts[RDTYPE_co] | dict[str, RagParts]

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
            # Restore the Ragged subclass and cache per-field RagParts.
            self.__class__ = Ragged  # type: ignore[assignment]
            # Set sentinel first: self[f] -> __getitem__ -> _ensure_parts checks hasattr
            object.__setattr__(self, "_parts", {})
            shared_offsets = _extract_list_offsets(cast(Content, ak.to_layout(self)))
            self._parts = {
                f: RagParts(p.data, p.shape, shared_offsets)
                for f in ak.fields(self)
                for p in (unbox(self[f]),)
            }
        else:
            self._parts = unbox(self)

    def _ensure_parts(self) -> None:
        """Idempotent lazy init for `_parts`. Handles Ragged instances created
        via awkward behavior dispatch (e.g. `ak.zip`) that bypass `__init__`."""
        if hasattr(self, "_parts"):
            return
        layout = cast(Content, ak.to_layout(self))
        if isinstance(layout, RecordArray) or _is_record_layout(layout):
            # Set sentinel first to break the self[f] -> _ensure_parts cycle.
            object.__setattr__(self, "_parts", {})
            shared_offsets = _extract_list_offsets(layout)
            object.__setattr__(
                self,
                "_parts",
                {
                    f: RagParts(p.data, p.shape, shared_offsets)
                    for f in ak.fields(self)
                    for p in (unbox(self[f]),)
                },
            )
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

    parts = _PartsDescriptor()
    """The parts of the Ragged array. For record layouts, a dict of
    field name -> RagParts; all share the same offsets ndarray."""

    data = _DataDescriptor()
    """The data of the Ragged array. For record layouts, a dict of
    field name -> zero-copy ndarray view, in awkward field order."""

    @property
    def offsets(self) -> NDArray[OFFSET_TYPE]:
        """The offsets of the Ragged array. May have shape (n_ragged + 1) or (2, n_ragged).

        Returns
        -------
        NDArray[np.int64]
        """
        self._ensure_parts()
        if isinstance(self._parts, dict):
            return next(iter(self._parts.values())).offsets
        return self._parts.offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        """The shape of the Ragged array. The ragged dimension is `None`.

        Returns
        -------
        tuple[int | None, ...]
        """
        self._ensure_parts()
        if isinstance(self._parts, dict):
            return next(iter(self._parts.values())).shape
        return self._parts.shape

    @property
    def dtype(self) -> np.dtype[RDTYPE_co]:
        """The dtype of the Ragged array.

        For non-record layouts, returns the numpy dtype of the flat data buffer
        (e.g. ``np.dtype('int32')``).

        For record layouts, returns a numpy *structured* dtype whose field names
        and per-field dtypes match the Ragged record fields — for example::

            np.dtype([("seq", "S1"), ("score", "f4")])

        .. note::
            **Memory layout is SoA, not AoS.**  A numpy structured dtype normally
            implies Array-of-Structs packing, but here each field lives in its own
            contiguous buffer (Structure of Arrays).  The structured dtype is used
            purely as a convenient, numpy-compatible descriptor: it carries all
            field/dtype information in a single object without inventing a new type.

        Returns
        -------
        np.dtype[RDTYPE_co]
        """
        self._ensure_parts()
        if isinstance(self._parts, dict):
            return np.dtype([(f, p.data.dtype) for f, p in self._parts.items()])  # type: ignore[return-value]
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
        if isinstance(self._parts, dict):
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
        self._ensure_parts()
        if isinstance(self._parts, dict):
            contiguous_data = all(p.data.flags.contiguous for p in self._parts.values())
        else:
            contiguous_data = self._parts.data.flags.contiguous
        return contiguous_offsets and contiguous_data

    @property
    def is_base(self) -> bool:
        """Whether the Ragged array is a base array (owns its data, contiguous, no offset).

        Returns
        -------
        bool
        """
        self._ensure_parts()
        if isinstance(self._parts, dict):
            parts_list = list(self._parts.values())
            base_data = all(p.data.base is None for p in parts_list)
            data_size = parts_list[0].data.size
        else:
            base_data = self._parts.data.base is None
            data_size = self._parts.data.size
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
        if isinstance(self._parts, dict):
            raise NotImplementedError(
                "to_numpy is not defined on record-layout Ragged arrays; "
                "convert fields individually."
            )
        arr = super().to_numpy(allow_missing=allow_missing)
        if self.dtype.type == np.bytes_:  # type: ignore[attr-defined] guaranteed by record check
            arr = arr[..., None].view("S1")
        return arr

    def to_packed(self, *, copy: bool = True) -> Ragged[RDTYPE_co]:
        """Pack into a fresh contiguous, zero-based Ragged (1-D offsets).

        Numba-parallelized replacement for ``Ragged(ak.to_packed(self))``.
        See :func:`seqpro.rag.to_packed` for the ``copy`` semantics.

        Parameters
        ----------
        copy
            When ``True`` (default), return a freshly allocated owned array.
            When ``False``, return zero-copy if already packed, else raise.

        Returns
        -------
        Ragged[RDTYPE_co]
        """
        from ._ops import to_packed as _to_packed

        return _to_packed(self, copy=copy)

    def __getitem__(self, where):
        arr = super().__getitem__(where)
        if isinstance(arr, ak.Array):
            if _n_var(arr) == 1:
                result = type(self)(arr)
                # For record field access, share the parent's offsets object (zero-copy).
                self._ensure_parts()
                if (
                    isinstance(where, str)
                    and isinstance(self._parts, dict)
                    and where in self._parts
                ):
                    result._ensure_parts()
                    assert isinstance(result._parts, RagParts)
                    result._parts = RagParts(
                        result._parts.data,
                        result._parts.shape,
                        self._parts[where].offsets,
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
        if isinstance(self._parts, dict):
            squeezed = {f: self[f].squeeze(axis) for f in self._parts}
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
        if isinstance(self._parts, dict):
            reshaped = {f: self[f].reshape(*shape) for f in self._parts}
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


def _ragged_typestr(content_type, behavior):
    # Walk RegularType wrappers to collect fixed dims, then wrap the innermost
    # scalar type: e.g. RegularType(4, NumpyType("int32")) → "var * 4 * Ragged[int32]"
    dims = []
    t = content_type
    while isinstance(t, _RegularType):
        dims.append(str(t.size))
        t = t.content
    inner = "".join(t._str("", True, behavior))
    prefix = "".join(f"{d} * " for d in dims)
    return f"var * {prefix}Ragged[{inner}]"


ak.behavior["__typestr__", Ragged.__name__] = _ragged_typestr


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


def unbox(arr: ak.Array | Ragged[DTYPE_co]) -> RagParts[DTYPE_co]:
    """Unbox an awkward array with a single ragged dimension into data, offsets, and shape.
    Always zero-copy: the returned data is a view of the original array.

    Parameters
    ----------
    arr
        The awkward array to unbox.

    Returns
    -------
    RagParts[DTYPE_co]
        Data, shape, and offsets extracted from the awkward array.
    """
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
