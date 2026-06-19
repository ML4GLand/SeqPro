from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np
from attrs import define, field
from numpy.typing import NDArray

from ._utils import OFFSET_TYPE  # noqa: F401  (re-exported convenience)

DTYPE_co = TypeVar("DTYPE_co", covariant=True)

_SPEC_C_MSG = "nested raggedness (>1 ragged level) lands in Spec C"


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


def validate_layout(layout: RaggedLayout[Any]) -> None:
    if layout.n_ragged > 1:
        raise NotImplementedError(_SPEC_C_MSG)

    for off in layout.offsets:
        if not _is_monotonic(off):
            raise ValueError("offsets must be monotonic non-decreasing")

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
