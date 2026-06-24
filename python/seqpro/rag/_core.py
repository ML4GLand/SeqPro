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


def is_rag_dtype(rag: Any, dtype: Any) -> bool:
    """Dtype check for _core.Ragged arrays.

    Returns True if *rag* is a Ragged with a dtype that is a numpy subtype of *dtype*.
    Always returns False for record-layout Rageds when a primitive dtype is queried.
    """
    if not isinstance(rag, Ragged):
        return False
    rag_dtype = rag.dtype
    if np.issubdtype(rag_dtype, np.void):  # structured dtype → record layout
        if not np.issubdtype(dtype, np.void):
            return False
        return rag_dtype == np.dtype(dtype)
    return np.issubdtype(rag_dtype, dtype)


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

    def __init__(self, data: Any, *, validate: bool = False):
        if isinstance(data, Ragged):
            data = data._layout
        if not isinstance(data, (RaggedLayout, RecordLayout)):
            from ._ingest import layout_from_ak

            data = layout_from_ak(data)
        if validate:
            validate_layout(data)
        self._layout = data

    def _with_layout(self, layout: Any) -> "Ragged[Any]":
        """Reconstruct a same-kind container around ``layout``, preserving the
        concrete subclass. Bypasses ``__init__`` (subclasses carry no state beyond
        ``_layout``; see the subclassing contract). Used by structural transforms
        so a ``Ragged`` subclass survives slicing/reshape/squeeze/to_packed."""
        obj = object.__new__(type(self))
        obj._layout = layout
        return obj

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
        validate: bool = False,
    ) -> "Ragged[Any]":
        if shape.count(None) >= 3:
            raise NotImplementedError("nested raggedness with R >= 3 is unsupported")
        off_list = offsets if isinstance(offsets, list) else [offsets]
        off_list = [
            o
            if (
                isinstance(o, np.ndarray)
                and o.dtype == OFFSET_TYPE
                and o.flags.c_contiguous
            )
            else np.ascontiguousarray(o, dtype=OFFSET_TYPE)
            for o in off_list
        ]
        if str_offsets is not None:
            if not (
                isinstance(str_offsets, np.ndarray)
                and str_offsets.dtype == OFFSET_TYPE
                and str_offsets.flags.c_contiguous
            ):
                str_offsets = np.ascontiguousarray(str_offsets, dtype=OFFSET_TYPE)
            return Ragged(
                RaggedLayout(
                    data=data, offsets=off_list, shape=shape, str_offsets=str_offsets
                ),
                validate=validate,
            )
        if shape.count(None) == 0 and data.dtype.kind != "S":
            raise ValueError("shape must have a None ragged dimension")
        if validate:
            # eager data-size check (only when validating)
            n_none = shape.count(None)
            if n_none == 1 and len(off_list) == 1 and off_list[0].ndim == 1:
                rag_dim = shape.index(None)
                trailing = shape[rag_dim + 1 :]
                trailing_ints: list[int] = [d for d in trailing if d is not None]
                trailing_size = int(np.prod(trailing_ints)) if trailing_ints else 1
                if off_list[0].size > 0:
                    expected_size = int(off_list[0][-1]) * trailing_size
                    if data.size != expected_size:
                        raise ValueError(
                            f"Data size {data.size} does not match size implied by shape and contiguous offsets: {expected_size}"
                        )
        return Ragged(_build_layout(data, shape, off_list), validate=validate)

    @staticmethod
    def from_lengths(
        data: NDArray[Any],
        lengths: "NDArray[Any] | tuple[NDArray[Any], NDArray[Any]]",
        *,
        validate: bool = False,
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
            return Ragged.from_offsets(data, shape, [o0, o1], validate=validate)
        offsets = lengths_to_offsets(lengths)
        if data.dtype.kind == "S" and data.ndim == 1:
            # opaque string collection: (N,), byte-length is not an axis
            shape = tuple(lengths.shape)
            return Ragged.from_offsets(data, shape, offsets, validate=validate)
        trailing = data.shape[1:]
        shape = (*lengths.shape, None, *trailing)
        return Ragged.from_offsets(data, shape, offsets, validate=validate)

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

    def __len__(self) -> int:
        """Return the size of the outermost dimension (shape[0])."""
        s = self._layout.shape[0]
        if s is None:
            raise TypeError("len() of unsized Ragged (shape[0] is the ragged axis)")
        return int(s)

    @property
    def data(self) -> "NDArray[Any]":  # type: ignore[override]
        """Return the underlying data array. For record Rageds, returns the dict of fields."""
        if isinstance(self._layout, RecordLayout):
            return {f: fl.data for f, fl in self._layout.fields.items()}  # type: ignore[return-value]
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
        if None in self._layout.shape:
            return self._layout.shape.index(None)
        # Opaque-string layout: the byte-length axis is implicit, conceptually
        # immediately after all explicit leading dims.
        if not isinstance(self._layout, RecordLayout) and self._rl.is_string:
            return len(self._layout.shape)
        raise ValueError(
            f"Ragged has no ragged dimension (shape={self._layout.shape!r})"
        )

    @property
    def is_empty(self) -> bool:
        offsets = self.offsets
        if offsets.ndim == 1:
            return bool(offsets.size == 0 or offsets[-1] == 0)
        return bool(np.all(offsets[0] == offsets[1]))

    @property
    def is_contiguous(self) -> bool:
        all_1d = all(o.ndim == 1 for o in self._layout.offsets)
        if isinstance(self._layout, RecordLayout):
            return all_1d and all(
                fl.data.flags.c_contiguous for fl in self._layout.fields.values()
            )
        if self._rl.is_string:
            # Opaque-string: str_offsets must also be 1-D and zero-based
            str_off = self._rl.str_offsets
            if (
                str_off is None
                or str_off.ndim != 1
                or (str_off.size > 0 and str_off[0] != 0)
            ):
                return False
        return all_1d and self._rl.data.flags.c_contiguous

    @staticmethod
    def _owns_memory(arr: "NDArray[Any]") -> bool:
        """Return True if ``arr`` owns its memory (not backed by mmap or a non-owned ndarray).

        The contract matches ``_array.Ragged.is_base`` for the mmap case: a
        memory-mapped array whose ``base`` is a ``mmap.mmap`` object is *not*
        considered owned (returns ``False``).  A normal view of a freshly
        allocated ndarray (``base`` is an ndarray whose own ``base is None``)
        *is* considered owned (returns ``True``), consistent with the pre-Task-5
        behaviour and the ``to_packed`` contract.

        The previous crash-fix returned ``True`` for any non-ndarray base
        (including ``mmap.mmap``), diverging from the oracle.
        """
        base = arr.base
        if base is None:
            return True
        # mmap.mmap (or any non-ndarray base) = memory-mapped / external buffer
        # -> not owned.  This also avoids AttributeError from accessing .base on
        # types that don't have it (e.g. mmap.mmap).
        if not isinstance(base, np.ndarray):
            return False
        return base.base is None

    @property
    def is_base(self) -> bool:
        offsets = self.offsets
        if isinstance(self._layout, RecordLayout):
            fields = self._layout.fields.values()
            owns = all(self._owns_memory(fl.data) for fl in fields)
            size0 = next(iter(self._layout.fields.values())).data.shape[0]
            return bool(
                owns and self.is_contiguous and offsets[0] == 0 and offsets[-1] == size0
            )
        data = self._rl.data
        owns_memory = self._owns_memory(data)
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
        # Fast path: a plain step-1 slice of a contiguous array becomes the
        # already-packed slice (narrowed buffer + rebased (N+1,) offsets) with no
        # gather, no (2,N) drift, no copy. Any other index uses the path below.
        if (
            isinstance(where, slice)
            and self._layout.shape
            and self._layout.shape[0] is not None
            and self.is_contiguous
        ):
            start, stop, step = where.indices(self._layout.shape[0])
            if step == 1:
                if stop < start:
                    stop = start  # numpy empty-slice semantics (e.g. a[5:2])
                fast = self._slice_contiguous(start, stop)
                if fast is not None:
                    return (
                        self._with_layout(fast._layout)
                        if type(self) is not Ragged
                        else fast
                    )
        result = self._getitem(where)
        if (
            type(self) is not Ragged
            and not isinstance(where, str)
            and isinstance(result, Ragged)
        ):
            return self._with_layout(result._layout)
        return result

    def _outer_n_inner(self) -> int:
        """Product of the fixed dims between the outer axis and the first ragged
        axis (1 when the outer axis is immediately followed by the ragged axis)."""
        shape = self._layout.shape
        rag_dim = shape.index(None)
        inner = [d for d in shape[1:rag_dim] if d is not None]
        return int(np.prod(np.array(inner, dtype=np.int64))) if inner else 1

    def _slice_contiguous(self, start: int, stop: int) -> "Ragged[Any] | None":
        """Build the already-packed result of a contiguous step-1 outer slice, or
        None to fall back. Caller guarantees self.is_contiguous and shape[0] int."""
        layout = self._layout
        if isinstance(layout, RecordLayout):
            return self._slice_contig_record(start, stop)
        rl = self._rl
        if rl.is_string:
            return self._slice_contig_string(start, stop)
        if rl.n_ragged == 2:
            return self._slice_contig_r2(start, stop)
        if rl.n_ragged == 1:
            return self._slice_contig_r1(start, stop)
        return None

    def _slice_contig_r1(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        n_inner = self._outer_n_inner()
        o0 = rl.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        base = int(o0[g0])
        new_off = o0[g0 : g1 + 1] - base  # contiguous (M+1,) int64
        new_data = rl.data[base : int(o0[g1])]  # narrowed view
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(RaggedLayout(data=new_data, offsets=[new_off], shape=new_shape))

    def _slice_contig_string(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        so = rl.str_offsets
        assert so is not None  # _slice_contig_string is only called for string layouts
        if rl.n_ragged == 0:
            # flat string collection: shape (N,), no axis offsets; slice str_offsets
            b0, b1 = int(so[start]), int(so[stop])
            new_so = so[start : stop + 1] - b0
            new_data = rl.data[b0:b1]
            return Ragged(
                RaggedLayout(
                    data=new_data, offsets=[], shape=(stop - start,), str_offsets=new_so
                )
            )
        # string-under-axis: O0 (outer -> variant) then str_offsets (variant -> byte)
        n_inner = self._outer_n_inner()
        o0 = rl.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        v0, v1 = int(o0[g0]), int(o0[g1])
        new_o0 = o0[g0 : g1 + 1] - v0
        b0, b1 = int(so[v0]), int(so[v1])
        new_so = so[v0 : v1 + 1] - b0
        new_data = rl.data[b0:b1]
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(
            RaggedLayout(
                data=new_data, offsets=[new_o0], shape=new_shape, str_offsets=new_so
            )
        )

    def _slice_contig_r2(self, start: int, stop: int) -> "Ragged[Any]":
        rl = self._rl
        n_inner = self._outer_n_inner()
        o0, o1 = rl.offsets
        g0, g1 = start * n_inner, stop * n_inner
        m0, m1 = int(o0[g0]), int(o0[g1])  # middle-segment range
        new_o0 = o0[g0 : g1 + 1] - m0
        d0, d1 = int(o1[m0]), int(o1[m1])  # data range
        new_o1 = o1[m0 : m1 + 1] - d0
        new_data = rl.data[d0:d1]
        new_shape = (stop - start, *rl.shape[1:])
        return Ragged(
            RaggedLayout(data=new_data, offsets=[new_o0, new_o1], shape=new_shape)
        )

    def _slice_contig_record(self, start: int, stop: int) -> "Ragged[Any] | None":
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if len(rec.offsets) == 2:
            return self._slice_contig_record_r2(start, stop)
        n_inner = self._outer_n_inner()
        o0 = rec.offsets[0]
        g0, g1 = start * n_inner, stop * n_inner
        v0, v1 = int(o0[g0]), int(o0[g1])
        shared = [o0[g0 : g1 + 1] - v0]  # one shared (M+1,) object for all
        new_fields: dict[str, RaggedLayout[Any]] = {}
        for name, fl in rec.fields.items():
            fld_tail = fl.shape[1:]
            if fl.str_offsets is not None:
                so = fl.str_offsets
                b0, b1 = int(so[v0]), int(so[v1])
                new_fields[name] = RaggedLayout(
                    data=fl.data[b0:b1],
                    offsets=shared,
                    shape=(stop - start, *fld_tail),
                    str_offsets=so[v0 : v1 + 1] - b0,
                )
            else:
                new_fields[name] = RaggedLayout(
                    data=fl.data[v0:v1],
                    offsets=shared,
                    shape=(stop - start, *fld_tail),
                )
        return Ragged(
            RecordLayout(
                offsets=shared, shape=(stop - start, *rec.shape[1:]), fields=new_fields
            )
        )

    def _slice_contig_record_r2(self, start: int, stop: int) -> "Ragged[Any] | None":
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if any(fl.str_offsets is not None for fl in rec.fields.values()):
            return None  # string-under-axis R=2 record: fall back to gather path
        n_inner = self._outer_n_inner()
        o0, o1 = rec.offsets
        g0, g1 = start * n_inner, stop * n_inner
        m0, m1 = int(o0[g0]), int(o0[g1])
        d0, d1 = int(o1[m0]), int(o1[m1])
        shared = [o0[g0 : g1 + 1] - m0, o1[m0 : m1 + 1] - d0]
        out_tail = rec.shape[rec.shape.index(None) :]
        new_fields = {
            name: RaggedLayout(
                data=fl.data[d0:d1],
                offsets=shared,
                shape=(stop - start, *fl.shape[fl.shape.index(None) :]),
            )
            for name, fl in rec.fields.items()
        }
        return Ragged(
            RecordLayout(
                offsets=shared, shape=(stop - start, *out_tail), fields=new_fields
            )
        )

    def _getitem(self, where: Any) -> Any:
        # np.newaxis / None: insert a size-1 leading axis.
        # e.g. shape (batch, None)[None] -> (1, batch, None)
        if where is None:
            new_shape = (1, *self._layout.shape)
            new_layout = RaggedLayout(
                data=self._rl.data,
                offsets=list(self._layout.offsets),
                shape=new_shape,
                str_offsets=self._rl.str_offsets,
            )
            return Ragged(new_layout)
        if (
            isinstance(where, tuple)
            and len(where) == 2
            and not self._is_record
            and self._rl.n_ragged == 2
            and self._is_full_slice(where[0])
        ):
            return self._getitem_inner(where[1])
        if isinstance(where, tuple):
            # Multi-dim tuple: each key in the tuple targets successive leading axes.
            # We collapse all the integer-dim keys into one combined index so that
            # e.g. rag[:, [0]] correctly selects sample axis 1, not range axis 0.
            # Only meaningful when there are *multiple* leading int axes to combine
            # (rag_dim > 1). When rag_dim == 1 there is a single leading axis and a
            # tuple just indexes it then reaches into the ragged/trailing dims; the
            # combining path would mis-handle the trailing fixed dim (e.g. a
            # (d0, None, K) array padded by to_padded), so fall through to the
            # sequential per-key path below.
            if self._layout.shape.count(None) == 1 and self.rag_dim > 1:
                return self._getitem_tuple_multidim(where)
            # Sequential per-key path.  np.newaxis (None) keys are handled here:
            # track their positions, strip them from the tuple, apply the remaining
            # keys, then insert size-1 axes at the correct positions in the result.
            has_none = any(k is None for k in where)
            if has_none:
                # Separate None positions from real indexing keys.
                real_keys: list[Any] = []
                none_positions: list[int] = []
                pos = 0  # tracks the output axis number as we accumulate
                for k in where:
                    if k is None:
                        none_positions.append(pos)
                        pos += 1  # new axis consumes one output position
                    else:
                        real_keys.append(k)
                        # An integer key collapses one axis (no output position);
                        # a slice/array key keeps one axis.
                        if not isinstance(k, (int, np.integer)):
                            pos += 1
                result: Any = self
                for k in real_keys:
                    result = result[k]
                # Insert size-1 leading axes at recorded positions.
                if isinstance(result, Ragged):
                    for p in reversed(none_positions):
                        cur_shape = result._layout.shape
                        new_shape = cur_shape[:p] + (1,) + cur_shape[p:]
                        result = Ragged(
                            RaggedLayout(
                                data=result._rl.data,
                                offsets=list(result._layout.offsets),
                                shape=new_shape,
                                str_offsets=result._rl.str_offsets,
                            )
                        )
                else:
                    # numpy array or scalar: use np.expand_dims
                    for p in reversed(none_positions):
                        result = np.expand_dims(result, p)
                return result
            result = self
            for k in where:
                result = result[k]
            return result
        if isinstance(self._layout, RecordLayout):
            return self._getitem_record(where)
        if self._layout.n_ragged == 2:
            return self._getitem_r2(where)
        # Multi-dim leading shape: when rag_dim > 1 (e.g. shape (d0, d1, ..., None))
        # index the first axis treating each "row" as a block of n_inner contiguous
        # segments.  Two offset encodings are used:
        #   - 1-D offsets: canonical / packed layout produced by from_offsets or
        #     to_packed(); always uses _getitem_multidim.
        #   - 2-D offsets (shape (2, n_segs)): two sub-cases:
        #       * rag_dim == 2: lazy-gather result from _getitem_inner_gather.
        #         The shape (d0, n_inner, None) is a virtual label; flat integer
        #         indexing is correct (returns a raw ndarray).  Use the flat path.
        #       * rag_dim >= 3: canonical multi-dim layout from genoray's
        #         from_offsets(data, (n_ranges, n_samples, ploidy, None), flat_2d).
        #         Outer dim must be peeled to return a Ragged.  Use _getitem_multidim.
        # NOTE: opaque string leaves have no None in shape (rag_dim would crash)
        # and are handled by the flat path below.
        if (
            None in self._layout.shape
            and self.rag_dim > 1
            and self._layout.offsets
            and (self._layout.offsets[0].ndim == 1 or self.rag_dim >= 3)
        ):
            return self._getitem_multidim(where)
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

    def _getitem_multidim(self, where: Any) -> "NDArray[Any] | Ragged[Any]":
        """Index the first leading dim of a multi-dim Ragged (rag_dim > 1).

        Shape is ``(d0, d1, ..., dk, None, ...)``.  Each "row" on the first axis
        corresponds to ``n_inner = d1 * ... * dk`` contiguous ragged segments.
        Supports int (peel first dim), slice, bool-mask, and int-array.
        """
        rag_dim = self.rag_dim
        leading = self._layout.shape[:rag_dim]  # (d0, d1, ..., dk)
        d0: int = cast(int, leading[0])  # always int before rag_dim
        inner_leading = leading[1:]  # (d1, ..., dk)
        n_inner = (
            int(
                np.prod(
                    np.array(
                        [int(d) for d in inner_leading if d is not None], dtype=np.int64
                    )
                )
            )
            if inner_leading
            else 1
        )
        trailing = self._layout.shape[rag_dim:]  # (None, *trailing_int)

        starts, stops = self._starts_stops()
        # starts/stops are flat over all d0*n_inner segments

        if isinstance(where, (int, np.integer)):
            i = int(where) % d0
            seg_lo = i * n_inner
            seg_hi = (i + 1) * n_inner
            new_starts = np.ascontiguousarray(starts[seg_lo:seg_hi], dtype=OFFSET_TYPE)
            new_stops = np.ascontiguousarray(stops[seg_lo:seg_hi], dtype=OFFSET_TYPE)
            new_offsets = np.stack([new_starts, new_stops], 0)
            new_shape: tuple[int | None, ...] = (*inner_leading, *trailing)
            return Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[new_offsets],
                    shape=new_shape,
                    str_offsets=self._rl.str_offsets,
                )
            )

        # Build the "outer" index (which d0-rows to select)
        if _where_is_bool(where):
            if where.shape[0] != d0:
                raise IndexError(
                    f"boolean index did not match axis 0 size {d0} but has size {where.shape[0]}"
                )
            d0_idx = np.flatnonzero(where).astype(np.int64)
        elif isinstance(where, slice):
            d0_idx = np.arange(*where.indices(d0), dtype=np.int64)
        else:
            arr = np.atleast_1d(np.asarray(np.arange(d0)[where], dtype=np.int64))
            d0_idx = np.where(arr < 0, arr + d0, arr)

        # For each selected d0-row, gather the n_inner segment offsets
        n_sel = len(d0_idx)
        flat_idx = (
            d0_idx[:, None] * n_inner + np.arange(n_inner, dtype=np.int64)
        ).reshape(-1)
        new_starts = np.ascontiguousarray(starts[flat_idx], dtype=OFFSET_TYPE)
        new_stops = np.ascontiguousarray(stops[flat_idx], dtype=OFFSET_TYPE)
        new_offsets = np.stack([new_starts, new_stops], 0)
        new_shape = (n_sel, *inner_leading, *trailing)
        return Ragged(
            RaggedLayout(
                data=self._rl.data,
                offsets=[new_offsets],
                shape=new_shape,
                str_offsets=self._rl.str_offsets,
            )
        )

    def _getitem_tuple_multidim(self, where: tuple[Any, ...]) -> Any:
        """Handle a tuple index on a single-ragged Ragged (record or non-record) with rag_dim > 1.

        Each key in ``where`` targets successive leading integer axes (like NumPy
        multi-dim indexing).  All leading-dim keys are resolved simultaneously into
        a single flat segment selection so that e.g. ``rag[:, [0]]`` correctly
        selects from axis 1 (samples), not axis 0 (ranges).

        Keys that reach the ragged ``None`` axis or beyond are not handled here
        (fall through to normal single-key dispatch after leading dims are consumed).
        """
        rag_dim = self.rag_dim
        leading = self._layout.shape[:rag_dim]  # e.g. (d0, d1, ..., dk)
        trailing = self._layout.shape[rag_dim:]  # (None, *trailing_int)

        # Split tuple: leading-dim keys vs. keys that reach or exceed the ragged axis.
        n_leading_keys = min(len(where), rag_dim)
        leading_keys = where[:n_leading_keys]
        remainder_keys = where[n_leading_keys:]

        # For each leading dim, resolve the key to an index array (or scalar).
        # We track whether each dim produces an int (scalar), a "slice" range, or a
        # "fancy" array.  Fancy arrays follow NumPy broadcast semantics: when multiple
        # fancy arrays index successive leading dimensions they are broadcast element-
        # wise, NOT outer-producted.  Slices/ints use the outer-product (meshgrid) path.
        idx_per_dim: list[NDArray[Any]] = []
        out_leading: list[int | None] = []
        is_fancy: list[bool] = []  # True = fancy array; False = int/slice
        for j, k in enumerate(leading_keys):
            dj: int = cast(
                int, leading[j]
            )  # leading dims before rag_dim are always int
            if isinstance(k, (int, np.integer)):
                ij = int(k) % dj
                idx_per_dim.append(np.array([ij], dtype=np.int64))
                is_fancy.append(False)
                # integer key: scalar — do NOT append to out_leading (dim is squeezed)
            elif isinstance(k, slice):
                ij_arr = np.arange(*k.indices(dj), dtype=np.int64)
                idx_per_dim.append(ij_arr)
                is_fancy.append(False)
                out_leading.append(len(ij_arr))
            elif _where_is_bool(k):
                if k.shape[0] != dj:
                    raise IndexError(
                        f"boolean index size {k.shape[0]} != axis {j} size {dj}"
                    )
                ij_arr = np.flatnonzero(k).astype(np.int64)
                idx_per_dim.append(ij_arr)
                is_fancy.append(True)
                out_leading.append(len(ij_arr))
            else:
                ij_arr = np.atleast_1d(np.asarray(np.arange(dj)[k], dtype=np.int64))
                idx_per_dim.append(ij_arr)
                is_fancy.append(True)
                out_leading.append(len(ij_arr))

        # Append any un-keyed leading dims (if len(where) < rag_dim)
        for j in range(n_leading_keys, rag_dim):
            dj = cast(int, leading[j])  # always int before rag_dim
            idx_per_dim.append(np.arange(dj, dtype=np.int64))
            is_fancy.append(False)
            out_leading.append(dj)

        # Compute strides for the flat segment layout: segment(i0,i1,...,ik) =
        # i0 * stride[0] + i1 * stride[1] + ... + ik * stride[k]
        strides = np.ones(rag_dim, dtype=np.int64)
        for j in range(rag_dim - 2, -1, -1):
            dj1 = leading[j + 1]
            strides[j] = strides[j + 1] * int(dj1 if dj1 is not None else 1)

        # Fancy-index semantics: when more than one index is a fancy array, broadcast
        # them together element-wise (NumPy / awkward convention) rather than forming
        # an outer product.  Non-fancy indices (ints squeezed, slices kept, or un-keyed)
        # form an outer product with the broadcast fancy result — matching NumPy semantics
        # for mixed fancy/slice indexing.
        n_fancy = sum(is_fancy)
        if n_fancy >= 2:
            # Step 1: compute the broadcast (element-wise) contribution of fancy dims.
            fancy_arrays = [idx_per_dim[j] for j in range(rag_dim) if is_fancy[j]]
            bc = np.broadcast(*fancy_arrays)
            bc_shape: tuple[int, ...] = bc.shape

            fancy_iter = iter(fancy_arrays)
            fancy_combined: NDArray[Any] = np.zeros(bc_shape, dtype=np.int64)
            for j in range(rag_dim):
                if is_fancy[j]:
                    fancy_combined = fancy_combined + next(fancy_iter) * strides[j]

            # Step 2: outer-product non-fancy (slice/unkeyed) dims with the fancy result.
            non_fancy_parts: list[NDArray[Any]] = [
                idx_per_dim[j] for j in range(rag_dim) if not is_fancy[j]
            ]
            non_fancy_strides: list[int] = [
                int(strides[j]) for j in range(rag_dim) if not is_fancy[j]
            ]
            non_fancy_sizes: list[int] = [len(a) for a in non_fancy_parts]

            if non_fancy_parts:
                # Build outer-product of non-fancy contributions, then broadcast with fancy.
                nf_grids = np.meshgrid(*non_fancy_parts, indexing="ij")
                nf_combined: NDArray[Any] = np.zeros(
                    [len(a) for a in non_fancy_parts], dtype=np.int64
                )
                for a, sf in zip(nf_grids, non_fancy_strides):
                    nf_combined = nf_combined + a * sf
                # Outer product: fancy shape + nf shape
                combined: NDArray[Any] = fancy_combined.reshape(
                    bc_shape + (1,) * len(non_fancy_sizes)
                ) + nf_combined.reshape((1,) * len(bc_shape) + tuple(non_fancy_sizes))
                flat_seg_idx = np.asarray(combined.reshape(-1), dtype=np.int64)
                # out_leading: fancy broadcast shape PLUS non-fancy (slice/unkeyed) sizes.
                # Un-keyed dims contribute new output dims (like NumPy's behaviour for
                # unindexed axes appended after the fancy block).
                out_leading = list(bc_shape) + non_fancy_sizes  # type: ignore[assignment]
            else:
                flat_seg_idx = np.asarray(fancy_combined.reshape(-1), dtype=np.int64)
                out_leading = list(bc_shape)  # type: ignore[assignment]
        else:
            # Outer-product via meshgrid: original behaviour for slices and single fancy.
            grids = np.meshgrid(*idx_per_dim, indexing="ij")
            grid_combined: NDArray[Any] = np.zeros(grids[0].shape, dtype=np.int64)
            for j in range(rag_dim):
                grid_combined = grid_combined + grids[j] * strides[j]
            flat_seg_idx = np.asarray(grid_combined.reshape(-1), dtype=np.int64)

        # Gather segment starts/stops
        starts, stops = self._starts_stops()
        new_starts = np.ascontiguousarray(starts[flat_seg_idx], dtype=OFFSET_TYPE)
        new_stops = np.ascontiguousarray(stops[flat_seg_idx], dtype=OFFSET_TYPE)
        new_offsets = np.stack([new_starts, new_stops], 0)

        # Build new shape: scalar axes squeezed, array-selected axes kept
        new_leading: tuple[int | None, ...] = tuple(out_leading)
        new_shape: tuple[int | None, ...] = (*new_leading, *trailing)

        # Build result: record Ragged or non-record Ragged depending on layout type
        if isinstance(self._layout, RecordLayout):
            rec = self._layout
            new_fields = {
                name: RaggedLayout(
                    data=fl.data,
                    offsets=[new_offsets],
                    shape=(
                        *new_leading,
                        *fl.shape[fl.shape.index(None) :],
                    ),
                    str_offsets=fl.str_offsets,
                )
                for name, fl in rec.fields.items()
            }
            result: Any = Ragged(
                RecordLayout(
                    offsets=[new_offsets],
                    shape=new_shape,
                    fields=new_fields,
                )
            )
        else:
            result = Ragged(
                RaggedLayout(
                    data=self._rl.data,
                    offsets=[new_offsets],
                    shape=new_shape,
                    str_offsets=self._rl.str_offsets,
                )
            )

        # Apply any remainder keys (reaching into or past the ragged axis)
        for k in remainder_keys:
            result = result[k]

        return result

    def _getitem_record(self, where: Any) -> Any:
        rec = self._layout
        assert isinstance(rec, RecordLayout)
        if isinstance(where, str):
            try:
                field = rec.fields[where]
            except KeyError:
                raise KeyError(where)
            return Ragged(field)  # field.offsets[0] is the shared object (zero-copy)
        # numpy contract: a non-tuple key is treated as a 1-tuple (A[x] == A[(x,)]).
        # For a single-ragged-axis record with >1 leading fixed axis, route through
        # the multidim peel path directly (under the SAME guard it requires, so we
        # never re-dispatch through __getitem__ and risk recursion). Records that are
        # not single-None fall through to the flat record-rows path unchanged.
        if (
            not isinstance(where, tuple)
            and self.rag_dim > 1
            and rec.shape.count(None) == 1
        ):
            return self._getitem_tuple_multidim((where,))
        return self._getitem_record_rows(where)

    def _gather_indices(
        self, where: Any, starts: "NDArray[Any]", stops: "NDArray[Any]"
    ) -> "tuple[NDArray[Any], NDArray[Any]]":
        """Resolve ``where`` (slice/mask/int-array) against ``starts``/``stops`` and
        return ``(sel_starts, sel_stops)`` as contiguous OFFSET_TYPE arrays."""
        n = len(starts)
        # Fast-path: plain slice — numpy slicing is an O(1) view; skip arange/where/rust
        if isinstance(where, slice):
            sel_starts = np.ascontiguousarray(starts[where], dtype=OFFSET_TYPE)
            sel_stops = np.ascontiguousarray(stops[where], dtype=OFFSET_TYPE)
            return sel_starts, sel_stops
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
            new_stops = np.maximum(np.minimum(o0_starts + stop, o0_stops), new_starts)
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
                if fl.str_offsets is not None:
                    so = fl.str_offsets
                    row = fl.data[int(so[lo]) : int(so[hi])]
                else:
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
                str_offsets=fl.str_offsets,
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
        # guard: string-under-axis fields cannot be packed in Spec C
        for fname, fl in rec.fields.items():
            if fl.str_offsets is not None:
                raise NotImplementedError(
                    f"to_packed() on a record with a string-under-axis field {fname!r} "
                    "is not supported in Spec C; convert via .to_chars() first, or pack in Spec D."
                )
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
        result = self._squeeze_impl(axis)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result

    def _squeeze_impl(
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
        result = self._reshape_impl(*shape)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result

    def _reshape_impl(self, *shape: int | None) -> "Ragged[Any]":
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
        result = self._to_packed_impl(copy=copy)
        if type(self) is not Ragged and isinstance(result, Ragged):
            return self._with_layout(result._layout)
        return result

    def _to_packed_impl(self, *, copy: bool = True) -> "Ragged[Any]":
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
                        and (
                            # numeric/char field: data length matches outer offsets
                            (
                                fl.str_offsets is None
                                and int(shared[-1]) == fl.data.shape[0]
                            )
                            # string-under-axis: str_offsets must also be 1-D and zero-based
                            or (
                                fl.str_offsets is not None
                                and fl.str_offsets.ndim == 1
                                and (fl.str_offsets.size == 0 or fl.str_offsets[0] == 0)
                            )
                        )
                        for fl in rec.fields.values()
                    )
                )
                if already:
                    return self
                raise ValueError(
                    "to_packed(copy=False) requires already-packed input; got an unpacked record."
                )
            from ._ops import _nested_pack_parts

            packed_offsets: NDArray[Any] | None = None
            new_fields: dict[str, RaggedLayout[Any]] = {}
            for name, fl in rec.fields.items():
                # copy is always True here; the copy=False path returns/raises above
                if fl.str_offsets is not None and fl.offsets:
                    # String-under-axis field: treat as R=2 (outer offsets + char offsets)
                    r2_shape = (*fl.shape, None)
                    pdata, poff2 = _nested_pack_parts(
                        fl.data, r2_shape, [shared, fl.str_offsets], copy=True
                    )
                    new_o0 = poff2[0]
                    new_str_off = poff2[1]
                    if packed_offsets is None:
                        packed_offsets = new_o0
                    new_fields[name] = RaggedLayout(
                        data=pdata,
                        offsets=[packed_offsets],
                        shape=fl.shape,
                        str_offsets=new_str_off,
                    )
                else:
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

        if self._rl.str_offsets is not None and self._layout.offsets:
            # Opaque-string-under-axis: shape (…, None), offsets=[o0], str_offsets=char_off.
            # Structurally R=2 — pack both levels via _nested_pack_parts.
            from ._ops import _nested_pack_parts

            o0 = self._layout.offsets[0]
            str_off = self._rl.str_offsets
            # Build a temporary R=2 shape for _nested_pack_parts: append a second None
            r2_shape = (*self._layout.shape, None)
            str_packed_data, str_packed_offs = _nested_pack_parts(
                self._rl.data, r2_shape, [o0, str_off], copy
            )
            return Ragged(
                RaggedLayout(
                    data=str_packed_data,
                    offsets=[str_packed_offs[0]],
                    shape=self._layout.shape,
                    str_offsets=str_packed_offs[1],
                )
            )
        if self._rl.is_string:
            # Opaque-string layout: str_offsets IS the packed dimension; there is
            # no None in shape, so _pack_parts (which calls shape.index(None)) cannot
            # be used.  Pack the string buffer directly using str_offsets as the row
            # delimiters, then rebuild with new zero-based str_offsets.
            str_off = self._rl.str_offsets
            assert str_off is not None
            packed_data, packed_str_offsets = _pack_parts(
                self._rl.data, (*self._layout.shape, None), str_off, copy
            )
            if packed_data is self._rl.data and packed_str_offsets is str_off:
                return self
            new_layout = RaggedLayout(
                data=packed_data,
                offsets=[],
                shape=self._layout.shape,
                str_offsets=packed_str_offsets,
            )
            return Ragged(new_layout)
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
        from seqpro.seqpro import _ragged_to_padded  # type: ignore[missing-import]  # rust

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
            _ragged_to_padded(data_u1, offsets, out_u1, elem_bytes, out_len)
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
        self, allow_missing: bool = False, *, validate: bool = True
    ) -> "NDArray[Any] | dict[str, NDArray[Any]]":
        if isinstance(self._layout, RecordLayout):
            # _core contract: return a dict of dense per-field arrays.
            return {
                field: cast(
                    "NDArray[Any]",
                    cast("Ragged[Any]", self[field]).to_numpy(
                        allow_missing=allow_missing, validate=validate
                    ),
                )
                for field in self._layout.fields
            }
        if self._layout.n_ragged == 2:
            if self._rl.str_offsets is not None:
                raise NotImplementedError(
                    "to_numpy() on a string-under-axis Ragged is not supported in Spec C; "
                    "convert via .to_chars() first."
                )
            packed = self.to_packed()
            o0, o1 = packed._layout.offsets  # canonical 1-D after pack
            grp_lens = np.diff(o0)
            mid_lens = np.diff(o1)
            if validate:
                if grp_lens.size and not np.all(grp_lens == grp_lens[0]):
                    raise ValueError(
                        "cannot convert a jagged outer axis to a dense array"
                    )
                if mid_lens.size and not np.all(mid_lens == mid_lens[0]):
                    raise ValueError(
                        "cannot convert a jagged inner axis to a dense array"
                    )
            result = packed.to_padded(
                np.zeros((), self.dtype)[()]
            )  # rectangular -> pad is identity (both dense)
            assert isinstance(
                result, np.ndarray
            )  # axis=None on R=2 always returns dense
            return result
        if self._rl.str_offsets is not None and self._layout.offsets:
            raise NotImplementedError(
                "to_numpy() on a string-under-axis Ragged is not supported in Spec C; "
                "convert via .to_chars() first."
            )
        if validate:
            lengths = self.lengths
            if lengths.size and not np.all(lengths == lengths.flat[0]):
                raise ValueError("cannot convert a jagged Ragged to a dense array")
            packed = self if self.is_base else self.to_packed()
            row_len = int(lengths.flat[0]) if lengths.size else 0
        else:
            # trust the caller: infer row_len from total // n_rows, no uniformity
            # scan. numpy's reshape still rejects a total-size mismatch for free.
            packed = self if self.is_base else self.to_packed()
            leading_dims = [d for d in packed.shape[: packed.rag_dim]]
            n_rows = (
                int(np.prod(np.array(leading_dims, dtype=np.int64)))
                if leading_dims
                else 1
            )
            total = packed._rl.data.shape[0]
            row_len = total // n_rows if n_rows else 0
        leading = packed.shape[: packed.rag_dim]
        return packed._rl.data.reshape(  # pyrefly: ignore[no-matching-overload]
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
