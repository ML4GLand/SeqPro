from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar, cast

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray
from typing_extensions import override

from ._layout import RaggedLayout, RecordLayout, validate_layout
from ._utils import OFFSET_TYPE, lengths_to_offsets

RDTYPE_co = TypeVar("RDTYPE_co", covariant=True)
DTYPE_co = TypeVar("DTYPE_co", covariant=True)


def _where_is_bool(where: Any) -> bool:
    return isinstance(where, np.ndarray) and where.dtype == np.bool_


def _build_layout(
    data: NDArray[Any], shape: tuple[int | None, ...], offsets: NDArray[Any]
) -> RaggedLayout[Any]:
    """Build a single-level layout under the string/char rule.

    - ``None`` present in ``shape``  -> counted ragged axis: numeric, or S1 *chars*
      (length is an axis). ``offsets`` is the ragged axis; ``str_offsets`` is None.
    - no ``None`` + S1 data          -> opaque *string* leaf: ``offsets`` is stored
      as ``str_offsets``; the byte-length is not an axis.
    """
    if None in shape:
        return RaggedLayout(data=data, offsets=[offsets], shape=shape)
    if data.dtype.kind == "S":
        return RaggedLayout(data=data, offsets=[], shape=shape, str_offsets=offsets)
    raise ValueError(
        "shape must have exactly one None ragged dimension for numeric data"
    )


class Ragged(NDArrayOperatorsMixin, Generic[RDTYPE_co]):
    """A non-branching ragged array with a single ragged axis (Spec A)."""

    __slots__ = ("_layout",)

    def __init__(self, data: Any):
        if isinstance(data, Ragged):
            data = data._layout
        if not isinstance(data, (RaggedLayout, RecordLayout)):
            from ._ingest import layout_from_ak

            data = layout_from_ak(data)
        validate_layout(data)
        self._layout = data

    @property
    def _is_record(self) -> bool:
        return isinstance(self._layout, RecordLayout)

    @property
    def _rl(self) -> "RaggedLayout[Any]":
        """Narrow ``_layout`` to ``RaggedLayout`` for single-layout code paths.

        All methods that access ``data``/``offsets``/``str_offsets``/``is_string``
        are only valid on non-record Rageds; this assert enforces that contract at
        runtime while satisfying the type-checker.
        """
        assert isinstance(self._layout, RaggedLayout)
        return self._layout

    def to_ak(self):
        from ._ingest import to_ak as _to_ak

        return _to_ak(self)

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
        if data.dtype.kind == "S" and data.ndim == 1:
            # opaque string collection: (N,), byte-length is not an axis
            shape: tuple[int | None, ...] = tuple(lengths.shape)
            return Ragged.from_offsets(data, shape, offsets)
        trailing = data.shape[1:]
        shape = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets)

    @staticmethod
    def from_fields(fields: "dict[str, Ragged[Any]]") -> "Ragged[Any]":
        """Build a record (struct-of-arrays) from named single-field Ragged inputs
        that share one ragged axis. Sequence fields must be chars (see to_chars)."""
        if not fields:
            raise ValueError("from_fields requires at least one field (got empty)")
        items = list(fields.items())
        for name, f in items:
            if f._is_record:
                raise NotImplementedError(
                    f"record-of-record field {name!r} lands in Spec C"
                )
            if f.is_string:
                raise NotImplementedError(
                    f"opaque-string field {name!r} is Spec C; pass chars via .to_chars()"
                )
        shared = items[0][1].offsets
        for name, f in items[1:]:
            if not np.array_equal(f.offsets, shared):
                raise ValueError(
                    f"field {name!r} offsets are not equal to the first field's"
                )
        rec_shape = items[0][1].shape
        rebound: dict[str, RaggedLayout[Any]] = {}
        for name, f in items:
            rebound[name] = RaggedLayout(
                data=f._rl.data, offsets=[shared], shape=f._layout.shape
            )
        return Ragged(RecordLayout(offsets=[shared], shape=rec_shape, fields=rebound))

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
    def data(self) -> "NDArray[Any] | dict[str, NDArray[Any]]":
        if isinstance(self._layout, RecordLayout):
            return {f: fl.data for f, fl in self._layout.fields.items()}
        return self._rl.data

    @property
    def offsets(self) -> NDArray[Any]:
        if isinstance(self._layout, RecordLayout):
            return self._layout.offsets[0]
        if self._layout.offsets:
            return self._layout.offsets[0]
        assert self._rl.str_offsets is not None
        return self._rl.str_offsets

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self._layout.shape

    @property
    def dtype(self) -> "np.dtype[Any]":
        if isinstance(self._layout, RecordLayout):
            return np.dtype(
                [(f, fl.data.dtype) for f, fl in self._layout.fields.items()]
            )
        if self._rl.is_string:
            return np.dtype(
                "S"
            )  # opaque variable-width string: descriptor, not S1 storage
        return self._rl.data.dtype

    @property
    def fields(self) -> list[str]:
        """Field names for a record Ragged. Raises TypeError on non-record arrays."""
        if isinstance(self._layout, RecordLayout):
            return list(self._layout.fields)
        raise TypeError("fields is only defined on record Ragged arrays")

    @property
    def is_string(self) -> bool:
        """True for an opaque variable-width string Ragged (dtype 'S', shape (N,))."""
        return self._rl.is_string

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
        if isinstance(self._layout, RecordLayout):
            return self.offsets.ndim == 1 and all(
                fl.data.flags.c_contiguous for fl in self._layout.fields.values()
            )
        return self.offsets.ndim == 1 and self._rl.data.flags.c_contiguous

    @property
    def is_base(self) -> bool:
        offsets = self.offsets
        if isinstance(self._layout, RecordLayout):
            fields = self._layout.fields.values()
            owns = all(fl.data.base is None for fl in fields)
            size0 = next(iter(self._layout.fields.values())).data.shape[0]
            return bool(
                owns and self.is_contiguous and offsets[0] == 0 and offsets[-1] == size0
            )
        data = self._rl.data
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
            data=self._rl.data.view(dtype),
            offsets=self._layout.offsets,
            shape=self._layout.shape,
            str_offsets=self._rl.str_offsets,
        )
        return Ragged(new_layout)

    def to_chars(self) -> "Ragged[Any]":
        """Zero-copy view of an opaque string ('S', (N,)) as ascii chars
        ('S1', (N, None)); the byte-length becomes a counted ragged axis."""
        if not self._rl.is_string:
            raise ValueError("to_chars() requires an opaque string Ragged (dtype 'S')")
        assert self._rl.str_offsets is not None
        new_shape = (*self._layout.shape, None)
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=[self._rl.str_offsets],
                shape=new_shape,
            )
        )

    def to_strings(self) -> "Ragged[Any]":
        """Zero-copy view of a 1-D ascii-char leaf ('S1', (N, None)) as an opaque
        string ('S', (N,)); the length axis becomes an uncounted byte leaf."""
        if self._rl.is_string:
            return self
        if self._rl.data.dtype.kind != "S":
            raise ValueError("to_strings() requires an S1 char Ragged")
        if self._rl.data.ndim != 1 or self._layout.shape[self.rag_dim + 1 :]:
            raise ValueError(
                "to_strings() requires a 1-D S1 char leaf (no trailing dims)"
            )
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=[],
                shape=self._layout.shape[: self.rag_dim],
                str_offsets=self.offsets,
            )
        )

    def _starts_stops(self) -> tuple[NDArray[Any], NDArray[Any]]:
        offsets = self.offsets
        if offsets.ndim == 1:
            return offsets[:-1], offsets[1:]
        return offsets[0], offsets[1]

    def __getitem__(
        self, where: Any
    ) -> "NDArray[Any] | bytes | dict[str, Any] | Ragged[Any]":
        if isinstance(self._layout, RecordLayout):
            return self._getitem_record(where)
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            row = self._rl.data[lo:hi]
            if self._rl.is_string:
                return row.tobytes()
            return row
        # slice / mask / int-array on the leading axis -> gather to (2, M)
        sel_starts, sel_stops = self._row_gather(where)
        new_offsets = np.stack([sel_starts, sel_stops], 0)
        if None not in self._layout.shape:  # string-leaf flat collection
            new_layout = RaggedLayout(
                data=self._rl.data,
                offsets=[],
                shape=(len(sel_starts),),
                str_offsets=new_offsets,
            )
        else:
            new_layout = RaggedLayout(
                data=self._rl.data,
                offsets=[new_offsets],
                shape=(len(sel_starts), *self._layout.shape[self.rag_dim :]),
                str_offsets=self._rl.str_offsets,
            )
        return Ragged(new_layout)

    def _getitem_record(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if isinstance(where, str):
            try:
                field = rec.fields[where]
            except KeyError:
                raise KeyError(where)
            return Ragged(field)  # field.offsets[0] is the shared object (zero-copy)
        return self._getitem_record_rows(where)  # Task 8

    def _row_gather(self, where: Any) -> "tuple[NDArray[Any], NDArray[Any]]":
        """Given a slice/mask/int-array ``where``, return (sel_starts, sel_stops)
        as contiguous OFFSET_TYPE arrays for the shared ragged axis."""
        starts, stops = self._starts_stops()
        n = len(starts)
        if _where_is_bool(where):
            if where.shape[0] != n:
                raise IndexError(
                    f"boolean index did not match indexed array along axis 0; "
                    f"size of axis is {n} but size of corresponding boolean axis is {where.shape[0]}"
                )
            idx = np.flatnonzero(where).astype(np.int64)
        else:
            idx = np.atleast_1d(np.asarray(np.arange(n)[where], dtype=np.int64))
            idx = np.where(idx < 0, idx + n, idx)
        try:
            from seqpro.seqpro import _ragged_select  # type: ignore[missing-import]

            sel_starts, sel_stops = _ragged_select(
                np.ascontiguousarray(starts, np.int64),
                np.ascontiguousarray(stops, np.int64),
                idx,
            )
        except ImportError:  # pragma: no cover
            sel_starts, sel_stops = starts[idx], stops[idx]
        return (
            np.ascontiguousarray(sel_starts, dtype=OFFSET_TYPE),
            np.ascontiguousarray(sel_stops, dtype=OFFSET_TYPE),
        )

    def _getitem_record_rows(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            out: dict[str, Any] = {}
            for name, fl in rec.fields.items():
                row = fl.data[lo:hi]
                out[name] = row
            return out
        sel_starts, sel_stops = self._row_gather(where)
        new_offsets = np.stack([sel_starts, sel_stops], 0)
        new_shape = (len(sel_starts), *rec.shape[rec.shape.index(None) :])
        new_fields = {
            name: RaggedLayout(
                data=fl.data,
                offsets=[new_offsets],
                shape=(len(sel_starts), *fl.shape[fl.shape.index(None) :]),
            )
            for name, fl in rec.fields.items()
        }
        return Ragged(
            RecordLayout(offsets=[new_offsets], shape=new_shape, fields=new_fields)
        )

    def __getattr__(self, name: str) -> "Ragged[Any]":
        # Only reached when `name` is not a real attribute/slot.
        # Must avoid recursion: use object.__getattribute__ to fetch _layout
        # without going through __getattr__ again (which would happen if _layout
        # is not set yet, e.g. during unpickling/copy before __init__).
        try:
            layout = object.__getattribute__(self, "_layout")
        except AttributeError:
            raise AttributeError(name)
        if isinstance(layout, RecordLayout) and name in layout.fields:
            return Ragged(layout.fields[name])
        raise AttributeError(name)

    def __setitem__(self, key: str, value: "Ragged[Any]") -> None:
        if not isinstance(self._layout, RecordLayout):
            raise TypeError("item assignment is only supported on record Ragged arrays")
        if not isinstance(key, str):
            raise TypeError("record field assignment requires a string field name")
        if value._is_record or value.is_string:
            raise NotImplementedError(
                "record fields must be numeric/char single fields"
            )
        shared = self._layout.offsets[0]
        if not np.array_equal(value.offsets, shared):
            raise ValueError("assigned field offsets must equal the record's offsets")
        new_field = RaggedLayout(
            data=value._rl.data, offsets=[shared], shape=value._layout.shape
        )
        new_fields = dict(self._layout.fields)
        new_fields[key] = new_field
        self._layout = RecordLayout(
            offsets=[shared], shape=self._layout.shape, fields=new_fields
        )

    def _with_data(self, new_data: NDArray[Any]) -> "Ragged[Any]":
        return Ragged(
            RaggedLayout(
                data=new_data,
                offsets=self._layout.offsets,
                shape=self._layout.shape,
                str_offsets=self._rl.str_offsets,
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
                raw_inputs.append(x._rl.data)
            else:
                raw_inputs.append(x)
        result = getattr(ufunc, method)(*raw_inputs, **kwargs)
        return self._with_data(result)

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> "Ragged[Any] | NDArray[Any]":
        if isinstance(self._layout, RecordLayout):
            return Ragged.from_fields(
                cast(
                    "dict[str, Ragged[Any]]",
                    {
                        f: Ragged(fl).squeeze(axis)
                        for f, fl in self._layout.fields.items()
                    },
                )
            )
        if axis is None:
            data = self._rl.data.squeeze()
            shape = tuple(s for s in self._layout.shape if s != 1)
            return Ragged(
                RaggedLayout(
                    data=data,
                    offsets=self._layout.offsets,
                    shape=shape,
                    str_offsets=self._rl.str_offsets,
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
        data = self._rl.data.reshape(len(self._rl.data), *data_trailing)
        return Ragged(
            RaggedLayout(
                data=data,
                offsets=self._layout.offsets,
                shape=shape,
                str_offsets=self._rl.str_offsets,
            )
        )

    def reshape(self, *shape: int | None) -> "Ragged[Any]":
        if isinstance(self._layout, RecordLayout):
            return Ragged.from_fields(
                {f: Ragged(fl).reshape(*shape) for f, fl in self._layout.fields.items()}
            )
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
        data = self._rl.data.reshape(len(self._rl.data), *shape[rag_dim + 1 :])
        new_shape: tuple[int | None, ...] = (*new_rag_shape, None, *data.shape[1:])
        return Ragged(
            RaggedLayout(
                data=data,
                offsets=self._layout.offsets,
                shape=new_shape,
                str_offsets=self._rl.str_offsets,
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

        if isinstance(self._layout, RecordLayout):
            rec = self._layout
            shared = self.offsets
            if not copy:
                already = (
                    shared.ndim == 1
                    and (shared.size == 0 or shared[0] == 0)
                    and all(
                        fl.data.flags.c_contiguous
                        and int(shared[-1]) == fl.data.shape[0]
                        for fl in rec.fields.values()
                    )
                )
                if already:
                    return self
                raise ValueError(
                    "to_packed(copy=False) requires already-packed input; got an unpacked record."
                )
            packed_offsets: NDArray[Any] | None = None
            new_fields: dict[str, RaggedLayout[Any]] = {}
            for name, fl in rec.fields.items():
                pdata, poff = _pack_parts(fl.data, fl.shape, shared, copy=True)
                if packed_offsets is None:
                    packed_offsets = poff
                # Ensure pdata owns its memory (base is None) so is_base holds.
                # _pack_parts may return a view of an internal uint8 buffer.
                if pdata.base is not None:
                    pdata = pdata.copy()
                new_fields[name] = RaggedLayout(
                    data=pdata, offsets=[packed_offsets], shape=fl.shape
                )
            assert packed_offsets is not None
            return Ragged(
                RecordLayout(
                    offsets=[packed_offsets], shape=rec.shape, fields=new_fields
                )
            )

        packed_data, packed_offsets = _pack_parts(
            self._rl.data, self._layout.shape, self.offsets, copy
        )
        if packed_data is self._rl.data and packed_offsets is self.offsets:
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
        dtype = rag._rl.data.dtype
        out = np.full((n_rows, out_len), pad_value, dtype=dtype)
        if n_rows and out_len:
            data_u1 = np.ascontiguousarray(rag._rl.data).reshape(-1).view(np.uint8)
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
        return packed._rl.data.reshape(  # pyrefly: ignore[no-matching-overload] -- leading dims before rag_dim are always int; numpy stub can't verify this
            *(leading or (-1,)), row_len, *packed._rl.data.shape[1:]
        )

    def __array__(self, dtype: Any = None) -> NDArray[Any]:
        arr = self.to_numpy()
        return arr.astype(dtype) if dtype is not None else arr
