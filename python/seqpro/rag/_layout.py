from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

from ._utils import OFFSET_TYPE  # noqa: F401  (re-exported convenience)

DTYPE_co = TypeVar("DTYPE_co", covariant=True)


@define
class RaggedLayout(Generic[DTYPE_co]):
    """Buffers backing a single-level Ragged array.

    data
        Flat 1-D numeric buffer, or an S1 buffer for a string leaf; 2-D
        ``(total, *trailing)`` when the leaf has trailing regular dims.
    offsets
        One ``(N+1,)`` or ``(2, N)`` array per ragged *axis*, outermost-first.
        Empty for a flat string collection (string leaf, no axis).
    shape
        ``(*leading_int, None x R, *trailing_int)``.
    str_offsets
        Per-element byte boundaries for a string leaf; ``None`` for numeric.
        Never counted in ``shape``/``offsets``.
    """

    data: NDArray[Any]
    offsets: list[NDArray[Any]]
    shape: tuple[int | None, ...]
    str_offsets: NDArray[Any] | None = field(default=None)

    @property
    def is_string(self) -> bool:
        return self.str_offsets is not None

    @property
    def n_ragged(self) -> int:
        return self.shape.count(None)


def _is_monotonic(offsets: NDArray[Any]) -> bool:
    if offsets.ndim == 2:
        # (2, M) gather layout: each column is [start, stop]; stop >= start required
        return bool(np.all(offsets[1] >= offsets[0])) if offsets.size else True
    return bool(np.all(np.diff(offsets) >= 0)) if offsets.size else True


def _level_bounds(entry: NDArray[Any]) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return (starts, stops) for one offsets entry (1-D canonical or (2, n) gather)."""
    if entry.ndim == 2:
        return entry[0], entry[1]
    return entry[:-1], entry[1:]


@define
class RecordLayout:
    """Struct-of-arrays: named numeric/char fields sharing one ragged offsets object.

    offsets
        The single shared ragged offsets object (Spec B: ``len == 1``); identical
        object to every field's ``offsets[0]``.
    shape
        Canonical ragged shape ``(*leading, None, *trailing?)`` (the first field's).
    fields
        Insertion-ordered field name -> single-level ``RaggedLayout`` (numeric or
        S1 chars). Opaque-string fields are out of scope (Spec C).
    """

    offsets: list[NDArray[Any]]
    shape: tuple[int | None, ...]
    fields: dict[str, RaggedLayout[Any]]


def _validate_record_layout(layout: RecordLayout) -> None:
    if not layout.fields:
        raise ValueError("record layout must have at least one field (got empty)")
    if not layout.offsets:
        raise ValueError("record layout must have a shared offsets array")
    shared = layout.offsets  # the full shared list
    rag_dim = layout.shape.index(None)
    ragged_shape = layout.shape[: rag_dim + 1]
    for name, fld in layout.fields.items():
        if len(fld.offsets) != len(shared) or any(
            fo is not so for fo, so in zip(fld.offsets, shared)
        ):
            raise ValueError(
                f"field {name!r} must use the shared offsets list (zero-copy SoA)"
            )
        if fld.shape[: fld.shape.index(None) + 1] != ragged_shape:
            raise ValueError(
                f"field {name!r} ragged shape {fld.shape} disagrees with record {layout.shape}"
            )
        validate_layout(fld)


def validate_layout(layout: RaggedLayout[Any] | RecordLayout) -> None:
    if isinstance(layout, RecordLayout):
        _validate_record_layout(layout)
        return
    if layout.n_ragged > 2:
        raise NotImplementedError(
            "nested raggedness with 3 or more levels (R >= 3) is unsupported"
        )

    for off in layout.offsets:
        if not _is_monotonic(off):
            raise ValueError("offsets must be monotonic non-decreasing")

    if layout.n_ragged == 2:
        if len(layout.offsets) != 2:
            raise ValueError(
                f"expected 2 offsets arrays for 2 ragged axes, got {len(layout.offsets)}"
            )
        o0, o1 = layout.offsets
        o0_starts, o0_stops = _level_bounds(o0)
        rag_dim = layout.shape.index(None)
        leading = [d for d in layout.shape[:rag_dim] if d is not None]
        expected_l0 = int(np.prod(np.array(leading, dtype=np.int64))) if leading else 1
        if len(o0_starts) != expected_l0:
            raise ValueError(
                f"outer segment count {len(o0_starts)} != product of leading dims {expected_l0}"
            )
        n_middle = len(o1) - 1 if o1.ndim == 1 else o1.shape[1]
        max_mid = int(o0_stops.max()) if len(o0_stops) else 0
        if o0.ndim == 1 and int(o0[-1]) != n_middle:
            raise ValueError(
                f"O0 references {int(o0[-1])} middle segments but O1 has {n_middle}"
            )
        if max_mid > n_middle:
            raise ValueError(
                f"O0 middle index {max_mid} exceeds O1 segment count {n_middle}"
            )
        return

    if layout.n_ragged == 1:
        if len(layout.offsets) != 1:
            raise ValueError(
                f"expected 1 offsets array for 1 ragged axis, got {len(layout.offsets)}"
            )
        offsets = layout.offsets[0]
        n_seg = len(offsets) - 1 if offsets.ndim == 1 else offsets.shape[1]
        rag_dim = layout.shape.index(None)
        leading: list[int] = [d for d in layout.shape[:rag_dim] if d is not None]
        expected = int(np.prod(np.array(leading, dtype=np.int64)))
        if n_seg != expected:
            raise ValueError(
                f"segment count {n_seg} != product of leading dims {expected}"
            )
        if layout.str_offsets is not None:
            if not _is_monotonic(layout.str_offsets):
                raise ValueError("str_offsets must be monotonic non-decreasing")
            if layout.str_offsets.ndim == 1 and int(layout.str_offsets[-1]) != int(
                layout.data.shape[0]
            ):
                raise ValueError("str_offsets must end at the data length")

        try:
            from seqpro.seqpro import _ragged_validate  # type: ignore[missing-import]  # compiled Rust extension

            off = layout.offsets[0]
            if off.ndim == 1:
                _ragged_validate(
                    np.ascontiguousarray(off, np.int64),
                    int(layout.data.shape[0]),
                    len(off) - 1,
                )
        except ImportError:  # pragma: no cover - fallback to pure-Python checks above
            pass
