from __future__ import annotations

from typing import Any, Generic, Sequence, TypeVar, cast

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.typing import NDArray
from typing_extensions import override

from ._layout import RaggedLayout, RecordLayout, _level_bounds, validate_layout
from ._utils import OFFSET_TYPE, lengths_to_offsets

RDTYPE_co = TypeVar("RDTYPE_co", covariant=True)
DTYPE_co = TypeVar("DTYPE_co", covariant=True)


def _where_is_bool(where: Any) -> bool:
    return isinstance(where, np.ndarray) and where.dtype == np.bool_


def _build_layout(
    data: NDArray[Any],
    shape: tuple[int | None, ...],
    offsets: "NDArray[Any] | list[NDArray[Any]]",
) -> RaggedLayout[Any]:
    """Build a ragged layout (R=0, R=1, or R=2) under the string/char rule.

    - ``None`` present in ``shape``  -> counted ragged axis(es): numeric, or S1 *chars*
      (length is an axis). ``offsets`` is a list of offset arrays; ``str_offsets`` is None.
    - no ``None`` + S1 data          -> opaque *string* leaf: first offset array is stored
      as ``str_offsets``; the byte-length is not an axis.
    """
    n_none = shape.count(None)
    off_list = offsets if isinstance(offsets, list) else [offsets]
    if n_none == 0:
        if data.dtype.kind == "S":
            # S1-only by convention; multi-byte S (S4/S100) is unsupported and not tightened here (see Spec D)
            (off,) = off_list
            return RaggedLayout(data=data, offsets=[], shape=shape, str_offsets=off)
        raise ValueError("shape must have a None ragged dimension for numeric data")
    if len(off_list) != n_none:
        raise ValueError(f"expected {n_none} offsets arrays, got {len(off_list)}")
    return RaggedLayout(data=data, offsets=off_list, shape=shape)


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
        data: NDArray[Any],
        shape: tuple[int | None, ...],
        offsets: "NDArray[Any] | list[NDArray[Any]]",
        *,
        str_offsets: "NDArray[Any] | None" = None,
    ) -> "Ragged[Any]":
        if shape.count(None) >= 3:
            raise NotImplementedError("nested raggedness with R >= 3 is unsupported")
        off_list = offsets if isinstance(offsets, list) else [offsets]
        off_list = [np.ascontiguousarray(o, dtype=OFFSET_TYPE) for o in off_list]
        if str_offsets is not None:
            str_offsets = np.ascontiguousarray(str_offsets, dtype=OFFSET_TYPE)
            return Ragged(
                RaggedLayout(
                    data=data, offsets=off_list, shape=shape, str_offsets=str_offsets
                )
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have a None ragged dimension")
        return Ragged(_build_layout(data, shape, off_list))

    @staticmethod
    def from_lengths(
        data: NDArray[Any], lengths: "NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]"
    ) -> "Ragged[Any]":
        if isinstance(lengths, tuple):
            outer_counts, inner_lengths = lengths
            o0 = lengths_to_offsets(np.asarray(outer_counts).reshape(-1))
            o1 = lengths_to_offsets(np.asarray(inner_lengths).reshape(-1))
            trailing = data.shape[1:]
            shape: tuple[int | None, ...] = (
                *np.asarray(outer_counts).shape,
                None,
                None,
                *trailing,
            )
            return Ragged.from_offsets(data, shape, [o0, o1])
        offsets = lengths_to_offsets(lengths)
        if data.dtype.kind == "S" and data.ndim == 1:
            # opaque string collection: (N,), byte-length is not an axis
            shape = tuple(lengths.shape)
            return Ragged.from_offsets(data, shape, offsets)
        trailing = data.shape[1:]
        shape = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets)

    @staticmethod
    def from_fields(fields: "dict[str, Ragged[Any]]") -> "Ragged[Any]":
        """Build a record (struct-of-arrays) from named single-field Ragged inputs
        that share one ragged axis. Supports numeric, char, string-under-axis, and
        R=2 fields; record-of-record and R>=3 fields are not supported."""
        if not fields:
            raise ValueError("from_fields requires at least one field (got empty)")
        items = list(fields.items())
        for name, f in items:
            if f._is_record:
                raise NotImplementedError(
                    f"record-of-record field {name!r} is unsupported"
                )
            if f._rl.n_ragged >= 3:
                raise NotImplementedError(f"R>=3 field {name!r} is unsupported")
        shared = items[0][1]._layout.offsets  # the FULL list (not public .offsets)
        for name, f in items[1:]:
            fo = f._layout.offsets
            if len(fo) != len(shared) or any(
                not np.array_equal(a, b) for a, b in zip(fo, shared)
            ):
                raise ValueError(
                    f"field {name!r} offsets are not equal to the first field's"
                )
        rec_shape = items[0][1].shape
        rebound: dict[str, RaggedLayout[Any]] = {
            name: RaggedLayout(
                data=f._rl.data,
                offsets=shared,
                shape=f._layout.shape,
                str_offsets=f._rl.str_offsets,
            )
            for name, f in items
        }
        return Ragged(RecordLayout(offsets=shared, shape=rec_shape, fields=rebound))

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
        if isinstance(self._layout, RecordLayout):
            return False
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
            owns = all(
                fl.data.base is None or fl.data.base.base is None for fl in fields
            )
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
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError(
                "view is not defined on record Ragged arrays; view a field, "
                "e.g. rag['f'] = rag['f'].view(dtype)."
            )
        new_layout = RaggedLayout(
            data=self._rl.data.view(dtype),
            offsets=self._layout.offsets,
            shape=self._layout.shape,
            str_offsets=self._rl.str_offsets,
        )
        return Ragged(new_layout)

    def to_chars(self) -> "Ragged[Any]":
        """Zero-copy view of an opaque string ('S', (..., None?)) as ascii chars
        ('S1', (..., None?, None)); str_offsets becomes the innermost real axis."""
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError(
                "to_chars() is not defined on record Ragged arrays; "
                "convert individual fields instead."
            )
        if not self._rl.is_string:
            raise ValueError("to_chars() requires an opaque string Ragged (dtype 'S')")
        assert self._rl.str_offsets is not None
        new_offsets = [
            *self._layout.offsets,
            self._rl.str_offsets,
        ]  # str_offsets -> innermost real level
        new_shape = (*self._layout.shape, None)
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=new_offsets,
                shape=new_shape,
            )
        )

    def to_strings(self) -> "Ragged[Any]":
        """Zero-copy view of a 1-D ascii-char leaf ('S1', (..., None)) as an opaque
        string ('S', (...)); the innermost length axis becomes an uncounted byte leaf."""
        if isinstance(self._layout, RecordLayout):
            raise NotImplementedError(
                "to_strings() is not defined on record Ragged arrays; "
                "convert individual fields instead."
            )
        if self._rl.is_string:
            return self
        if self._rl.data.dtype.kind != "S":
            raise ValueError("to_strings() requires an S1 char Ragged")
        inner_none = max(i for i, d in enumerate(self._layout.shape) if d is None)
        if self._rl.data.ndim != 1 or self._layout.shape[inner_none + 1 :]:
            raise ValueError(
                "to_strings() requires a 1-D S1 char leaf (no trailing dims)"
            )
        *outer_offsets, inner = (
            self._layout.offsets
        )  # innermost real level -> str_offsets
        new_shape = self._layout.shape[:-1]  # drop the inner None
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=outer_offsets,
                shape=new_shape,
                str_offsets=inner,
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
        if (
            isinstance(where, tuple)
            and len(where) == 2
            and not self._is_record
            and self._rl.n_ragged == 2
            and self._is_full_slice(where[0])
        ):
            return self._getitem_inner(where[1])
        if isinstance(where, tuple):
            result: Any = self
            for k in where:
                result = result[k]
            return result
        if isinstance(self._layout, RecordLayout):
            return self._getitem_record(where)
        if self._layout.n_ragged == 2:
            return self._getitem_r2(where)
        starts, stops = self._starts_stops()
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            if self._rl.str_offsets is not None and self._layout.offsets:
                # string-under-axis: outer offsets index variants -> map to bytes via str_offsets
                so = self._rl.str_offsets
                return self._rl.data[int(so[lo]) : int(so[hi])].tobytes()
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

    def _gather_indices(
        self, where: Any, starts: "NDArray[Any]", stops: "NDArray[Any]"
    ) -> "tuple[NDArray[Any], NDArray[Any]]":
        """Resolve ``where`` (slice/mask/int-array) against ``starts``/``stops`` and
        return ``(sel_starts, sel_stops)`` as contiguous OFFSET_TYPE arrays."""
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

    def _row_gather(self, where: Any) -> "tuple[NDArray[Any], NDArray[Any]]":
        """Given a slice/mask/int-array ``where``, return (sel_starts, sel_stops)
        as contiguous OFFSET_TYPE arrays for the shared ragged axis."""
        return self._gather_indices(where, *self._starts_stops())

    @staticmethod
    def _is_full_slice(k: Any) -> bool:
        return isinstance(k, slice) and k == slice(None)

    def _getitem_inner(self, sel: Any) -> "Ragged[Any]":
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        o1_starts, o1_stops = _level_bounds(o1)
        if isinstance(sel, (int, np.integer)):  # k-th middle of each group -> R=1
            counts = o0_stops - o0_starts
            if np.any(sel >= counts) or np.any(-sel > counts):
                raise IndexError(f"inner index {sel} out of range for some group")
            mid_idx = (o0_starts + (sel if sel >= 0 else counts + sel)).astype(np.int64)
            ds = np.ascontiguousarray(o1_starts[mid_idx], dtype=OFFSET_TYPE)
            de = np.ascontiguousarray(o1_stops[mid_idx], dtype=OFFSET_TYPE)
            trailing = self._layout.shape[self.rag_dim + 2 :]
            return Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[np.stack([ds, de], 0)],
                    shape=(len(mid_idx), None, *trailing),
                )
            )
        if isinstance(sel, slice):  # local per-group slice -> R=2
            start, stop, step = sel.indices(1 << 62)
            if step != 1:
                raise NotImplementedError("step != 1 inner slices are unsupported")
            if (sel.start is not None and sel.start < 0) or (
                sel.stop is not None and sel.stop < 0
            ):
                raise NotImplementedError(
                    "negative inner-slice bounds (rag[:, -k:]) are unsupported"
                )
            new_starts = np.minimum(o0_starts + start, o0_stops)
            new_stops = np.minimum(o0_starts + stop, o0_stops)
            new_o0 = np.stack(
                [new_starts.astype(OFFSET_TYPE), new_stops.astype(OFFSET_TYPE)], 0
            )
            trailing = self._layout.shape[self.rag_dim + 2 :]
            return Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[new_o0, o1],
                    shape=(len(new_starts), None, None, *trailing),
                )
            )
        return self._getitem_inner_gather(sel)  # mask / int-array -> Task 8

    def _getitem_inner_gather(self, sel: Any) -> "Ragged[Any]":
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        o1_starts, o1_stops = _level_bounds(o1)
        trailing = self._layout.shape[self.rag_dim + 2 :]
        if _where_is_bool(sel):  # mask over the global middle axis -> R=2
            from seqpro.seqpro import _ragged_nested_gather  # type: ignore[missing-import]

            counts, sel_idx = _ragged_nested_gather(
                np.ascontiguousarray(o0_starts, np.int64),
                np.ascontiguousarray(o0_stops, np.int64),
                np.ascontiguousarray(sel, np.bool_),
            )
            new_o0 = lengths_to_offsets(counts.astype(np.uint32))
            ds = np.ascontiguousarray(o1_starts[sel_idx], dtype=OFFSET_TYPE)
            de = np.ascontiguousarray(o1_stops[sel_idx], dtype=OFFSET_TYPE)
            return Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[new_o0, np.stack([ds, de], 0)],
                    shape=(len(o0_starts), None, None, *trailing),
                )
            )
        idx = np.atleast_1d(
            np.asarray(sel, dtype=np.int64)
        )  # uniform per-group int array
        counts = o0_stops - o0_starts
        if np.any(idx.max() >= counts) or np.any(idx.min() < -counts.min()):
            raise IndexError("uniform inner index out of range for some group")
        cols = [self._getitem_inner(int(k)) for k in idx]  # each is (L0, ~K) R=1
        ds = np.stack([c._layout.offsets[0][0] for c in cols], 1).reshape(-1)
        de = np.stack([c._layout.offsets[0][1] for c in cols], 1).reshape(-1)
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=[np.stack([ds, de], 0)],
                shape=(len(counts), len(idx), None, *trailing),
            )
        )

    def _getitem_r2(self, where: Any) -> "Ragged[Any]":
        """Index an R=2 array on the outer axis.

        - ``int`` → peel one outer row to a 1-level ``Ragged`` (zero-copy inner slice).
        - ``slice`` / bool mask / int-array → lazy gather: build a ``(2, L0')`` outer
          offset that references ranges in the global O1; no data or O1 movement.
        """
        o0, o1 = self._layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        if isinstance(where, (int, np.integer)):
            # peel one outer row -> 1-level Ragged
            a, b = int(o0_starts[where]), int(o0_stops[where])
            if o1.ndim == 1:
                inner = o1[a : b + 1]  # contiguous slice, zero-copy
            else:
                inner = np.stack([o1[0][a:b], o1[1][a:b]], 0)
            trailing = self._layout.shape[self.rag_dim + 2 :]
            return Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[inner],
                    shape=(b - a, None, *trailing),
                )
            )
        # slice / mask / int-array on the outer axis: gather O0 ranges, keep O1 global
        sel_starts, sel_stops = self._gather_indices(where, o0_starts, o0_stops)
        new_o0 = np.stack([sel_starts, sel_stops], 0)
        trailing = self._layout.shape[self.rag_dim + 2 :]
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=[new_o0, o1],
                shape=(len(sel_starts), None, None, *trailing),
            )
        )

    def _getitem_record_rows(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if len(rec.offsets) == 2:
            return self._getitem_record_rows_r2(where)
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

    def _getitem_record_rows_r2(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        o0, o1 = rec.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        if isinstance(where, (int, np.integer)):
            # peel one outer row -> dict of per-field 1-level Rageds
            return {name: Ragged(fl)[where] for name, fl in rec.fields.items()}
        sel_starts, sel_stops = self._gather_indices(where, o0_starts, o0_stops)
        new_o0 = np.stack([sel_starts, sel_stops], 0)
        new_shared = [new_o0, o1]  # keep O1 global; share across fields
        rag_dim = rec.shape.index(None)
        new_shape = (len(sel_starts), None, None, *rec.shape[rag_dim + 2 :])
        new_fields = {
            name: RaggedLayout(
                data=fl.data,
                offsets=new_shared,
                shape=(
                    len(sel_starts),
                    None,
                    None,
                    *fl.shape[fl.shape.index(None) + 2 :],
                ),
                str_offsets=fl.str_offsets,
            )
            for name, fl in rec.fields.items()
        }
        return Ragged(
            RecordLayout(offsets=new_shared, shape=new_shape, fields=new_fields)
        )

    def _to_packed_record_r2(self, copy: bool) -> "Ragged[Any]":
        from ._ops import _nested_pack_parts

        rec = self._layout
        assert isinstance(rec, RecordLayout)
        o0, o1 = rec.offsets
        if not copy:
            already = (
                o0.ndim == 1
                and o1.ndim == 1
                and (o0.size == 0 or o0[0] == 0)
                and (o1.size == 0 or o1[0] == 0)
                and all(
                    fl.data.flags.c_contiguous and int(o1[-1]) == fl.data.shape[0]
                    for fl in rec.fields.values()
                )
            )
            if already:
                return self
            raise ValueError(
                "to_packed(copy=False) requires already-packed input; got an unpacked nested record."
            )
        shared_packed: list[NDArray[Any]] | None = None
        new_fields: dict[str, RaggedLayout[Any]] = {}
        for name, fl in rec.fields.items():
            pdata, poff = _nested_pack_parts(fl.data, fl.shape, rec.offsets, copy=True)
            if shared_packed is None:
                shared_packed = (
                    poff  # all fields produce identical [o0,o1]; share the first
                )
            new_fields[name] = RaggedLayout(
                data=pdata,
                offsets=shared_packed,
                shape=fl.shape,
                str_offsets=fl.str_offsets,
            )
        assert shared_packed is not None
        return Ragged(
            RecordLayout(offsets=shared_packed, shape=rec.shape, fields=new_fields)
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
        if any(isinstance(x, Ragged) and x._is_record for x in inputs):
            raise NotImplementedError(
                "element-wise ufuncs are not defined on record Ragged arrays; "
                "operate on individual fields."
            )
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
        inner_axis = self.rag_dim + (1 if self._layout.n_ragged == 2 else 0)
        data_trailing = tuple(
            s
            for i, s in enumerate(self._layout.shape)
            if i not in axis and i > inner_axis
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
        if self._layout.n_ragged == 2:
            data = self._rl.data.reshape(len(self._rl.data), *shape[rag_dim + 2 :])
            new_shape: tuple[int | None, ...] = (
                *new_rag_shape,
                None,
                None,
                *data.shape[1:],
            )
        else:
            data = self._rl.data.reshape(len(self._rl.data), *shape[rag_dim + 1 :])
            new_shape = (*new_rag_shape, None, *data.shape[1:])
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
        o0: NDArray[Any] = (
            self._layout.offsets[0] if self._layout.offsets else self._rl.str_offsets  # type: ignore[assignment]
        )
        starts, stops = _level_bounds(o0)
        raw = stops - starts
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
            if len(rec.offsets) == 2:
                return self._to_packed_record_r2(copy)
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
                # copy is always True here; the copy=False path returns/raises above
                pdata, poff = _pack_parts(fl.data, fl.shape, shared, copy=True)
                if packed_offsets is None:
                    packed_offsets = poff
                new_fields[name] = RaggedLayout(
                    data=pdata, offsets=[packed_offsets], shape=fl.shape
                )
            assert packed_offsets is not None
            return Ragged(
                RecordLayout(
                    offsets=[packed_offsets], shape=rec.shape, fields=new_fields
                )
            )

        if self._layout.n_ragged == 2:
            from ._ops import _nested_pack_parts

            n2_data, n2_offsets = _nested_pack_parts(
                self._rl.data, self._layout.shape, self._layout.offsets, copy
            )
            if n2_data is self._rl.data and all(
                po is so for po, so in zip(n2_offsets, self._layout.offsets)
            ):
                return self
            return Ragged.from_offsets(n2_data, self._layout.shape, n2_offsets)

        packed_data, packed_offsets = _pack_parts(
            self._rl.data, self._layout.shape, self.offsets, copy
        )
        if packed_data is self._rl.data and packed_offsets is self.offsets:
            return self
        return Ragged.from_offsets(packed_data, self._layout.shape, packed_offsets)

    def to_padded(
        self,
        pad_value: Any,
        *,
        length: "int | tuple[int | None, int | None] | None" = None,
        axis: "int | None" = None,
    ) -> "NDArray[Any] | Ragged[Any] | dict[str, NDArray[Any]]":
        if isinstance(self._layout, RecordLayout):
            return {  # pyrefly: ignore[bad-return] -- fields are never records; inner calls return NDArray
                f: cast(
                    "NDArray[Any]",
                    cast("Ragged[Any]", self[f]).to_padded(
                        pad_value, length=length, axis=axis
                    ),
                )
                for f in self._layout.fields
            }
        if self._layout.n_ragged == 2:
            return self._to_padded_nested(pad_value, length=length, axis=axis)
        from ._ops import _to_padded_copy

        # Part A: support trailing regular dims (e.g. (N, None, K) -> (N, out_len, K))
        rag = self if self.is_contiguous else self.to_packed()
        offsets = np.ascontiguousarray(rag.offsets, dtype=OFFSET_TYPE)
        n_rows = offsets.shape[0] - 1
        out_len = (
            int(length)  # type: ignore[arg-type]
            if length is not None
            else (int(rag.lengths.max()) if n_rows else 0)
        )
        dtype = rag._rl.data.dtype
        trailing = tuple(rag._rl.data.shape[1:])  # regular trailing dims (() for plain)
        out = np.full((n_rows, out_len, *trailing), pad_value, dtype=dtype)
        if n_rows and out_len:
            data_u1 = np.ascontiguousarray(rag._rl.data).reshape(-1).view(np.uint8)
            out_u1 = out.reshape(-1).view(np.uint8)
            elem_bytes = (
                dtype.itemsize * int(np.prod(trailing, dtype=np.int64))
                if trailing
                else dtype.itemsize
            )
            _to_padded_copy(data_u1, offsets, out_u1, elem_bytes, out_len)
        leading = rag.shape[: rag.rag_dim]
        if leading or trailing:
            return out.reshape((*leading, out_len, *trailing))  # pyrefly: ignore[no-matching-overload] -- leading/trailing dims are always int; numpy stub can't verify this
        return out

    def _to_padded_nested(
        self,
        pad_value: Any,
        *,
        length: "int | tuple[int | None, int | None] | None",
        axis: "int | None",
    ) -> "NDArray[Any] | Ragged[Any]":
        """Pad an R=2 Ragged array along one or both ragged axes.

        axis=-1  -> pad inner only; returns R=1 Ragged with shape (*leading, ~M, K)
        axis=-2  -> deferred (NotImplementedError, Spec C)
        axis=None -> pad both axes; returns dense ndarray with shape (*leading, M, K)
        """
        rag = self if self.is_contiguous else self.to_packed()
        o0 = rag._layout.offsets[0]
        o1 = rag._layout.offsets[1]
        len_m, len_k = length if isinstance(length, tuple) else (length, length)
        rag_dim = rag.rag_dim
        trailing = rag._layout.shape[rag_dim + 2 :]
        # middle-as-rows single-level view over O1, then pad each middle's data to K
        inner_view = Ragged(
            RaggedLayout(
                data=rag._rl.data, offsets=[o1], shape=(len(o1) - 1, None, *trailing)
            )
        )
        padded_raw = inner_view.to_padded(
            pad_value, length=len_k
        )  # (M_total, K, *trailing)
        assert isinstance(
            padded_raw, np.ndarray
        )  # inner_view is R=1, never dict or Ragged
        padded: NDArray[Any] = padded_raw
        if axis == -2:
            raise NotImplementedError(
                "to_padded(axis=-2) (outer-only with ragged inner) is not supported in Spec C"
            )
        result_shape: tuple[int | None, ...] = (
            *rag.shape[:rag_dim],
            None,
            *padded.shape[1:],
        )
        if axis == -1:  # pad inner only -> Ragged (*leading, ~M, K)
            return Ragged(RaggedLayout(data=padded, offsets=[o0], shape=result_shape))
        # axis is None -> pad BOTH -> dense (*leading, M, K)
        outer_view = Ragged(RaggedLayout(data=padded, offsets=[o0], shape=result_shape))
        result = outer_view.to_padded(pad_value, length=len_m)
        assert isinstance(
            result, np.ndarray
        )  # outer_view is R=1 with trailing dim, always dense
        return result

    def to_numpy(
        self, allow_missing: bool = False
    ) -> "NDArray[Any] | dict[str, NDArray[Any]]":
        if isinstance(self._layout, RecordLayout):
            return {  # pyrefly: ignore[bad-return] -- fields are never records; inner calls return NDArray
                f: cast(
                    "NDArray[Any]", cast("Ragged[Any]", self[f]).to_numpy(allow_missing)
                )
                for f in self._layout.fields
            }
        if self._layout.n_ragged == 2:
            packed = self.to_packed()
            o0, o1 = packed._layout.offsets  # canonical 1-D after pack
            grp_lens = np.diff(o0)
            mid_lens = np.diff(o1)
            if grp_lens.size and not np.all(grp_lens == grp_lens[0]):
                raise ValueError("cannot convert a jagged outer axis to a dense array")
            if mid_lens.size and not np.all(mid_lens == mid_lens[0]):
                raise ValueError("cannot convert a jagged inner axis to a dense array")
            result = self.to_padded(
                np.zeros((), self.dtype)[()]
            )  # rectangular -> pad is identity (both dense)
            assert isinstance(
                result, np.ndarray
            )  # axis=None on R=2 always returns dense
            return result
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
        if isinstance(self._layout, RecordLayout):
            raise TypeError(
                "record Ragged arrays have no single dense array form; "
                "use to_numpy() per field."
            )
        arr = self.to_numpy()
        assert isinstance(
            arr, np.ndarray
        )  # keep for pyrefly narrowing on single-level path
        return arr.astype(dtype) if dtype is not None else arr
