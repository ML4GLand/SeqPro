from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray
from typing_extensions import override

from ._layout import RaggedLayout, validate_layout
from ._utils import OFFSET_TYPE, lengths_to_offsets

RDTYPE_co = TypeVar("RDTYPE_co", covariant=True)
DTYPE_co = TypeVar("DTYPE_co", covariant=True)


def _build_layout(
    data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any]
) -> RaggedLayout[Any]:
    """Build a single-level layout, applying the string-leaf rule.

    For bytes (S1) data with no trailing fixed dims, the single ragged axis is
    the string itself -> collapse to a string leaf: drop the None, store the
    supplied offsets as str_offsets.
    """
    is_bytes = data.dtype.kind == "S"
    n_none = shape.count(None)
    if is_bytes and n_none == 1:
        rag_dim = shape.index(None)
        trailing = shape[rag_dim + 1 :]
        if all(d is not None for d in trailing) and len(trailing) == 0:
            leaf_shape = shape[:rag_dim]
            return RaggedLayout(
                data=data, offsets=[], shape=leaf_shape, str_offsets=offsets
            )
    return RaggedLayout(data=data, offsets=[offsets], shape=shape)


class Ragged(NDArrayOperatorsMixin, Generic[RDTYPE_co]):
    """A non-branching ragged array with a single ragged axis (Spec A)."""

    __slots__ = ("_layout",)

    def __init__(self, data: RaggedLayout[Any]):
        if not isinstance(data, RaggedLayout):
            raise TypeError(
                "Ragged(...) currently accepts a RaggedLayout; "
                "awkward/Ragged ingestion is added in a later task."
            )
        validate_layout(data)
        self._layout = data

    @staticmethod
    def from_offsets(
        data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any]
    ) -> "Ragged[Any]":
        if shape.count(None) > 1:
            raise NotImplementedError(
                "nested raggedness (>1 ragged level) lands in Spec C"
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have exactly one None ragged dimension")
        offsets = np.ascontiguousarray(offsets, dtype=OFFSET_TYPE)
        return Ragged(_build_layout(data, shape, offsets))

    @staticmethod
    def from_lengths(data: NDArray[Any], lengths: NDArray[Any]) -> "Ragged[Any]":
        offsets = lengths_to_offsets(lengths)
        trailing = data.shape[1:]
        shape: tuple[int | None, ...] = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets)

    @classmethod
    def empty(cls, shape: int | tuple[int | None, ...], dtype: Any) -> "Ragged[Any]":
        if isinstance(shape, int):
            shape = (shape,)
        rag_dim = shape.index(None)
        trailing = shape[rag_dim + 1 :]  # all int (only the ragged dim is None)
        trailing_ints: list[int] = [d for d in trailing if d is not None]
        empty_shape: Sequence[int] = [0, *trailing_ints] if trailing_ints else [0]
        data: NDArray[Any] = (
            np.empty(empty_shape, dtype=dtype) if trailing else np.empty(0, dtype=dtype)
        )
        leading = [d for d in shape[:rag_dim] if d is not None]
        n_seg = int(np.prod(np.array(leading, dtype=np.int64))) if leading else 1
        offsets: NDArray[Any] = np.zeros(n_seg + 1, dtype=OFFSET_TYPE)
        return Ragged.from_offsets(data, shape, offsets)

    @property
    def data(self) -> NDArray[Any]:
        return self._layout.data

    @property
    def offsets(self) -> NDArray[Any]:
        if self._layout.offsets:
            return self._layout.offsets[0]
        assert self._layout.str_offsets is not None
        return self._layout.str_offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self._layout.shape

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._layout.data.dtype

    @property
    def rag_dim(self) -> int:
        return self._layout.shape.index(None)

    @property
    def is_empty(self) -> bool:
        offsets = self.offsets
        if offsets.ndim == 1:
            return bool(offsets.size == 0 or offsets[-1] == 0)
        return bool(np.all(offsets[0] == offsets[1]))

    @property
    def is_contiguous(self) -> bool:
        return self.offsets.ndim == 1 and self._layout.data.flags.c_contiguous

    @property
    def is_base(self) -> bool:
        offsets = self.offsets
        data = self._layout.data
        owns_memory = data.base is None or (
            data.base is not None and data.base.base is None
        )
        return bool(
            owns_memory
            and self.is_contiguous
            and offsets[0] == 0
            and offsets[-1] == data.shape[0]
        )

    def view(self, dtype: Any) -> "Ragged[Any]":
        new_layout = RaggedLayout(
            data=self._layout.data.view(dtype),
            offsets=self._layout.offsets,
            shape=self._layout.shape,
            str_offsets=self._layout.str_offsets,
        )
        return Ragged(new_layout)

    def _starts_stops(self) -> tuple[NDArray[Any], NDArray[Any]]:
        offsets = self.offsets
        if offsets.ndim == 1:
            return offsets[:-1], offsets[1:]
        return offsets[0], offsets[1]

    def __getitem__(self, where: Any) -> "NDArray[Any] | bytes | Ragged[Any]":
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            row = self._layout.data[lo:hi]
            if self._layout.is_string:
                return row.tobytes()
            return row
        # slice / mask / int-array on the leading axis -> gather to (2, M)
        sel_starts = np.ascontiguousarray(starts[where], dtype=OFFSET_TYPE)
        sel_stops = np.ascontiguousarray(stops[where], dtype=OFFSET_TYPE)
        new_offsets = np.stack([sel_starts, sel_stops], 0)
        if None not in self._layout.shape:  # string-leaf flat collection
            new_layout = RaggedLayout(
                data=self._layout.data,
                offsets=[],
                shape=(len(sel_starts),),
                str_offsets=new_offsets,
            )
        else:
            new_layout = RaggedLayout(
                data=self._layout.data,
                offsets=[new_offsets],
                shape=(len(sel_starts), *self._layout.shape[self.rag_dim :]),
                str_offsets=self._layout.str_offsets,
            )
        return Ragged(new_layout)

    def _with_data(self, new_data: NDArray[Any]) -> "Ragged[Any]":
        return Ragged(
            RaggedLayout(
                data=new_data,
                offsets=self._layout.offsets,
                shape=self._layout.shape,
                str_offsets=self._layout.str_offsets,
            )
        )

    @override
    def __array_ufunc__(
        self, ufunc: Any, method: str, *inputs: Any, **kwargs: Any
    ) -> "Ragged[Any]":
        if method != "__call__":
            raise NotImplementedError(
                f"Ragged supports only element-wise ufuncs, not method={method!r}"
            )
        ref_offsets = self.offsets
        raw_inputs = []
        for x in inputs:
            if isinstance(x, Ragged):
                if x.offsets is not ref_offsets and not np.array_equal(
                    x.offsets, ref_offsets
                ):
                    raise ValueError("ufunc operands must share offsets")
                raw_inputs.append(x.data)
            else:
                raw_inputs.append(x)
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        return self._with_data(result)

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> "Ragged[Any] | NDArray[Any]":
        if axis is None:
            data = self._layout.data.squeeze()
            shape = tuple(s for s in self._layout.shape if s != 1)
            return Ragged(
                RaggedLayout(
                    data=data,
                    offsets=self._layout.offsets,
                    shape=shape,
                    str_offsets=self._layout.str_offsets,
                )
            )
        if isinstance(axis, int):
            axis = (axis,)
        ndim = len(self._layout.shape)
        axis = tuple(a % ndim for a in axis)
        for a in axis:
            if self._layout.shape[a] != 1:
                raise ValueError(
                    f"cannot squeeze axis {a} of size {self._layout.shape[a]}"
                )
        shape = tuple(s for i, s in enumerate(self._layout.shape) if i not in axis)
        data_trailing = tuple(
            s
            for i, s in enumerate(self._layout.shape)
            if i not in axis and i > self.rag_dim
        )
        data = self._layout.data.reshape(len(self._layout.data), *data_trailing)
        return Ragged(
            RaggedLayout(
                data=data,
                offsets=self._layout.offsets,
                shape=shape,
                str_offsets=self._layout.str_offsets,
            )
        )

    def reshape(self, *shape: int | None) -> "Ragged[Any]":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]  # type: ignore[assignment]
        rag_dim = shape.index(None)
        new_rag_shape = shape[:rag_dim]
        leading_ints = [s for s in self._layout.shape[: self.rag_dim] if s is not None]
        n_rag = int(np.prod(np.array(leading_ints, dtype=np.int64)))
        n_new = (
            abs(
                int(
                    np.prod([s for s in new_rag_shape if s is not None], dtype=np.int64)
                )
            )
            or 1
        )
        new_rag_shape = tuple(
            s if s is not None and s >= 0 else n_rag // n_new for s in new_rag_shape
        )
        data = self._layout.data.reshape(len(self._layout.data), *shape[rag_dim + 1 :])
        new_shape: tuple[int | None, ...] = (*new_rag_shape, None, *data.shape[1:])
        return Ragged(
            RaggedLayout(
                data=data,
                offsets=self._layout.offsets,
                shape=new_shape,
                str_offsets=self._layout.str_offsets,
            )
        )

    @property
    def lengths(self) -> NDArray[Any]:
        offsets = self.offsets
        raw = np.diff(offsets) if offsets.ndim == 1 else np.diff(offsets, axis=0)
        rag_dim = (
            self._layout.shape.index(None)
            if None in self._layout.shape
            else len(self._layout.shape)
        )
        leading = self._layout.shape[:rag_dim]
        reshape_arg: Sequence[int] = (
            [d for d in leading if d is not None] if leading else [-1]
        )
        return raw.reshape(reshape_arg)

    def to_packed(self, *, copy: bool = True) -> "Ragged[Any]":
        from ._ops import _pack_parts

        packed_data, packed_offsets = _pack_parts(
            self._layout.data, self._layout.shape, self.offsets, copy
        )
        if packed_data is self._layout.data and packed_offsets is self.offsets:
            return self
        return Ragged.from_offsets(packed_data, self._layout.shape, packed_offsets)

    def to_padded(self, pad_value: Any, *, length: int | None = None) -> NDArray[Any]:
        from ._ops import _to_padded_copy

        rag = self if self.is_contiguous else self.to_packed()
        offsets = np.ascontiguousarray(rag.offsets, dtype=OFFSET_TYPE)
        n_rows = offsets.shape[0] - 1
        out_len = (
            int(length)
            if length is not None
            else (int(rag.lengths.max()) if n_rows else 0)
        )
        dtype = rag.data.dtype
        out = np.full((n_rows, out_len), pad_value, dtype=dtype)
        if n_rows and out_len:
            data_u1 = np.ascontiguousarray(rag.data).reshape(-1).view(np.uint8)
            out_u1 = out.reshape(-1).view(np.uint8)
            _to_padded_copy(data_u1, offsets, out_u1, dtype.itemsize, out_len)
        leading = rag.shape[: rag.rag_dim]
        return out.reshape((*leading, out_len)) if leading else out  # pyrefly: ignore[no-matching-overload] -- leading dims before rag_dim are always int; numpy stub can't verify this

    def to_numpy(self, allow_missing: bool = False) -> NDArray[Any]:
        lengths = self.lengths
        if lengths.size and not np.all(lengths == lengths.flat[0]):
            raise ValueError("cannot convert a jagged Ragged to a dense array")
        packed = self if self.is_base else self.to_packed()
        row_len = int(lengths.flat[0]) if lengths.size else 0
        leading = packed.shape[: packed.rag_dim]
        return packed.data.reshape(*(leading or (-1,)), row_len, *packed.data.shape[1:])  # pyrefly: ignore[no-matching-overload] -- leading dims before rag_dim are always int; numpy stub can't verify this

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr
