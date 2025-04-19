from __future__ import annotations

from typing import Generic, Optional, Tuple, TypeVar, Union, cast

import awkward as ak
import numpy as np
from attrs import define
from awkward.contents import ListArray, ListOffsetArray, NumpyArray, RegularArray
from awkward.index import Index64
from numpy.typing import NDArray
from typing_extensions import Self

__all__ = ["Ragged"]

DTYPE = TypeVar("DTYPE", bound=np.generic)
RDTYPE = TypeVar("RDTYPE", bound=np.generic)
LENGTH_TYPE = np.uint32
OFFSET_TYPE = np.int64


@define
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
    shape: Tuple[int, ...]
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
            self.maybe_lengths = np.diff(self.offsets).reshape(self.shape)
        return self.maybe_lengths

    @classmethod
    def from_offsets(
        cls,
        data: NDArray[RDTYPE],
        shape: Union[int, Tuple[int, ...]],
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

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Self:
        """Squeeze the ragged array along the given non-ragged axis."""
        return type(self).from_lengths(self.data, self.lengths.squeeze(axis))

    def reshape(self, shape: Tuple[int, ...]) -> Self:
        """Reshape non-ragged axes."""
        # this is correct because all reshaping operations preserve the layout i.e. raveled ordered
        return type(self).from_lengths(self.data, self.lengths.reshape(shape))

    def __str__(self):
        return (
            f"Ragged<shape={self.shape} dtype={self.data.dtype} size={self.data.size}>"
        )

    def to_awkward(self) -> ak.Array:
        """Convert to an `Awkward <https://awkward-array.org/doc/main/>`_ array without copying. Note that this effectively
        returns a view of the data, so modifying the data will modify the original array."""
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
            layout = ListArray(
                Index64(self.offsets[:, 0]), Index64(self.offsets[:, 1]), data
            )

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
                        axis=-1,
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
        return cast(
            str,
            self.to_awkward().show(
                5,
                stream=None,  # type: ignore
            ),
        )


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
