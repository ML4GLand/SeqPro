from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Optional, TypeVar, Union, cast, overload

import awkward as ak
import numpy as np
from attrs import define
from awkward.contents import ListArray, ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index64
from numpy.typing import ArrayLike, NDArray
from typing_extensions import Self

from ._utils import cast_seqs

__all__ = ["Ragged"]

DTYPE = TypeVar("DTYPE", bound=np.generic)
RDTYPE = TypeVar("RDTYPE", bound=np.generic)
LENGTH_TYPE = np.uint32
OFFSET_TYPE = np.int64


@define(eq=False, order=False)
class Ragged(Generic[RDTYPE]):
    """Ragged array i.e. a rectilinear array where the final axis is ragged. Should not be initialized
    directly, use :meth:`from_offsets()` or :meth:`from_lengths()` instead.

    Examples
    --------

    .. code-block:: python

        r = Ragged.from_lengths(np.arange(10), np.array([3, 2, 5]))
        assert r.offsets ==  np.array([0, 3, 5, 10])
        assert r.data[r.offsets[0]:r.offsets[1]] == np.array([0, 1, 2])
        assert r.data[r.offsets[1]:r.offsets[2]] == np.array([3, 4])
        assert r.data[r.offsets[2]:r.offsets[3]] == np.array([5, 6, 7, 8, 9])

    """

    data: NDArray[RDTYPE]
    """A 1D array of the data."""
    shape: tuple[int, ...]
    """Shape of the ragged array, excluding the length dimension. For example, if
        the shape is (2, 3), then the j, k-th element can be mapped to an index for
        offsets with :code:`i = np.ravel_multi_index((j, k), shape)`. The number of ragged
        elements corresponds to the product of the shape."""
    maybe_offsets: Optional[NDArray[OFFSET_TYPE]] = None
    maybe_lengths: Optional[NDArray[LENGTH_TYPE]] = None

    def __attrs_post_init__(self):
        if self.maybe_offsets is None and self.maybe_lengths is None:
            raise ValueError("Either offsets or lengths must be provided.")

    def __len__(self):
        return self.shape[0]

    def item(self):
        a = self.squeeze()
        if a.shape != ():
            raise ValueError("Array has more than 1 ragged element.")
        return a.data

    def view(self, dtype: type[DTYPE]) -> Ragged[DTYPE]:
        """Return a view of the data with the given dtype."""
        return Ragged(
            self.data.view(dtype), self.shape, self.maybe_offsets, self.maybe_lengths
        )

    @property
    def ndim(self) -> int:
        """Number of dimensions of the ragged array."""
        return len(self.shape)

    @property
    def dtype(self) -> np.dtype[RDTYPE]:
        """Data type of the data array."""
        return self.data.dtype

    @property
    def offsets(self) -> NDArray[OFFSET_TYPE]:
        """Offsets into the data array to get corresponding elements. The i-th element
        is accessible as :code:`data[offsets[i]:offsets[i+1]]`."""
        if self.maybe_offsets is None:
            self.maybe_offsets = lengths_to_offsets(self.lengths, dtype=OFFSET_TYPE)
        return self.maybe_offsets

    @property
    def lengths(self) -> NDArray[LENGTH_TYPE]:
        """Array with appropriate shape containing lengths of each element in the ragged array."""
        if self.maybe_lengths is None:
            if self.offsets.ndim == 1:
                self.maybe_lengths = np.diff(self.offsets).reshape(self.shape)
            else:
                self.maybe_lengths = np.diff(self.offsets, axis=0).reshape(self.shape)
        return self.maybe_lengths

    @classmethod
    def from_offsets(
        cls,
        data: NDArray[RDTYPE],
        shape: Union[int, tuple[int, ...]],
        offsets: NDArray[OFFSET_TYPE],
    ) -> Self:
        """Create a Ragged array from data and offsets.

        Parameters
        ----------
        data
            1D data array.
        shape
            Shape of the ragged array, excluding the length dimension.
        offsets
            Offsets into the data array to get corresponding elements.
        """
        if isinstance(shape, int):
            shape = (shape,)
        return cls(data, shape, maybe_offsets=offsets)

    @classmethod
    def from_lengths(cls, data: NDArray[RDTYPE], lengths: NDArray[LENGTH_TYPE]) -> Self:
        """Create a Ragged array from data and lengths. The lengths array should have
        the intended shape of the Ragged array.

        Parameters
        ----------
        data
            1D data array.
        lengths
            Lengths of each element in the ragged array.
        """
        return cls(data, lengths.shape, maybe_lengths=lengths)

    @classmethod
    def empty(cls, shape: int | tuple[int, ...], dtype: type[RDTYPE]) -> Self:
        """Create an empty Ragged array with the given shape and dtype."""
        data = np.empty(0, dtype=dtype)
        offsets = np.zeros(np.prod(shape) + 1, dtype=OFFSET_TYPE)
        return cls.from_offsets(data, shape, offsets)

    @property
    def is_empty(self) -> bool:
        return self.data.size == 0

    def to_numpy(self) -> NDArray[RDTYPE]:
        """Note: potentially not zero-copy if offsets are ListArray."""
        arr = self.to_awkward().to_numpy(allow_missing=False)
        if self.dtype.type == np.bytes_:
            arr = arr.view("S1")
        return arr

    def squeeze(self, axis: Optional[Union[int, tuple[int, ...]]] = None) -> Self:
        """Squeeze the ragged array along the given non-ragged axis."""
        return type(self).from_lengths(self.data, self.lengths.squeeze(axis))

    def reshape(self, shape: int | tuple[int, ...]) -> Self:
        """Reshape non-ragged axes."""
        # this is correct because all reshaping operations preserve the layout i.e. raveled ordered
        return type(self).from_lengths(self.data, self.lengths.reshape(shape))

    def __str__(self):
        return (
            f"Ragged<shape={self.shape} dtype={self.data.dtype} size={self.data.size}>"
        )

    @overload
    def __getitem__(self, idx: int | tuple[int, ...]) -> NDArray[RDTYPE]: ...
    @overload
    def __getitem__(
        self, idx: slice | Sequence[int] | tuple[slice | Sequence[int], ...]
    ) -> Self: ...
    @overload
    def __getitem__(
        self, idx: slice | ArrayLike | tuple[slice | ArrayLike, ...]
    ) -> Self | NDArray[RDTYPE]: ...
    def __getitem__(
        self, idx: slice | ArrayLike | tuple[slice | ArrayLike, ...]
    ) -> Self | NDArray[RDTYPE]:
        item = self.to_awkward()[idx]
        if isinstance(item, ak.Array):
            # check if 1D, meaning the array is not ragged
            if len(item.typestr.split(" * ")) == 2:
                return item.to_numpy()
            return type(self).from_awkward(item)
        elif isinstance(item, (str, bytes)):
            return cast_seqs(item)  # type: ignore
        else:
            return item  # type: ignore

    def __setitem__(self, idx, value):
        awk = self.to_awkward()
        if isinstance(value, Ragged):
            value = value.to_awkward()
        awk[idx] = value
        return type(self).from_awkward(awk)

    def to_awkward(self) -> ak.Array:
        """Convert to an `Awkward <https://awkward-array.org/doc/main/>`_ array without copying. Note that this effectively
        returns a view of the data, so modifying the data will modify the original array."""
        if self.dtype.type == np.void:
            raise NotImplementedError(
                "Converting Ragged arrays with structured dtype to/from Awkward is not supported."
            )

        if self.dtype.type == np.bytes_ and self.dtype.itemsize == 1:
            data = NumpyArray(
                self.data.view(np.uint8),  # type: ignore
                parameters={"__array__": "char"},
            )
        else:
            data = NumpyArray(self.data)  # type: ignore

        if self.offsets.ndim == 1:
            layout = ListOffsetArray(Index64(self.offsets), data)
        else:
            layout = ListArray(Index64(self.offsets[0]), Index64(self.offsets[1]), data)

        for size in reversed(self.shape[1:]):
            layout = RegularArray(layout, size)

        return ak.Array(layout)

    @classmethod
    def from_awkward(cls, awk: "ak.Array") -> Self:
        """Convert from an `Awkward <https://awkward-array.org/doc/main/>`_ array without copying. Note that this effectively
        returns a view of the data, so modifying the data will modify the original array."""
        # parse shape
        shape_str = awk.typestr.split(" * ")
        try:
            shape = tuple(map(int, shape_str[:-2]))
        except ValueError as err:
            raise ValueError(
                f"Only the final axis of an awkward array may be variable to convert to ragged, but got {awk.type}."
            ) from err

        # extract data and offsets
        layout = awk.layout
        offsets = None
        while hasattr(layout, "content"):
            if isinstance(layout, ListOffsetArray):
                offsets = np.asarray(layout.offsets.data, dtype=OFFSET_TYPE)
                content = layout.content
                break
            elif isinstance(layout, ListArray):
                starts = layout.starts.data
                stops = layout.stops.data
                offsets = cast(
                    NDArray[OFFSET_TYPE],
                    np.stack(
                        [starts, stops],  # type: ignore
                        axis=0,
                        dtype=OFFSET_TYPE,
                    ),
                )
                content = layout.content
                break
            else:
                layout = layout.content
        else:
            raise ValueError

        if offsets is None:
            raise ValueError

        data = np.asarray(content.data)  # type: ignore
        if content.parameter("__array__") == "char":
            data = data.view("S1")

        rag = cls.from_offsets(
            data,  # type: ignore
            shape,
            offsets,
        )

        return rag

    def __repr__(self):
        msg = ""
        if self.dtype.type != np.void:
            msg += f"{self.to_awkward().show(5, stream=None)}\n"
        msg += str(self)
        return msg

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            inputs = tuple(i.data if isinstance(i, Ragged) else i for i in inputs)
            data = ufunc(*inputs, **kwargs)
            return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)
        else:
            return NotImplemented

    def __add__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data + other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __sub__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data - other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rsub__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other - self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __mul__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data * other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __truediv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data / other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rtruediv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other / self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __floordiv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data // other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rfloordiv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other // self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __mod__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data % other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rmod__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other % self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __pow__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data**other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rpow__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other**self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __eq__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data == other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __ne__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data != other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __lt__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data < other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __le__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data <= other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __gt__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data > other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __ge__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data >= other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __or__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data | other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __and__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data & other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __xor__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data ^ other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __lshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data << other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rlshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other << self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = self.data >> other
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __rrshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        data = other >> self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __neg__(self: Ragged):
        data = -self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __pos__(self: Ragged):
        data = +self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __invert__(self: Ragged):
        data = ~self.data
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __abs__(self: Ragged):
        data = np.abs(self.data)
        return Ragged(data, self.shape, self.maybe_offsets, self.maybe_lengths)

    def __iadd__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data += other
        return self

    def __isub__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data -= other
        return self

    def __imul__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data *= other
        return self

    def __itruediv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data /= other
        return self

    def __ifloordiv__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data //= other
        return self

    def __imod__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data %= other
        return self

    def __ipow__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data **= other
        return self

    def __iand__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data &= other
        return self

    def __ior__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data |= other
        return self

    def __ixor__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data ^= other
        return self

    def __ilshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data <<= other
        return self

    def __irshift__(self: Ragged, other: int | float | Ragged | ak.Array | NDArray):
        if isinstance(other, ak.Array):
            other = Ragged.from_awkward(other)
        if isinstance(other, Ragged):
            other = other.data
        self.data >>= other
        return self

    __radd__ = __add__
    __rmul__ = __mul__
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__


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
