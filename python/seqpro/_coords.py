from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class CoordSchema:
    chrom: str
    start: str
    end: str
    zero_based: bool
    strand: str | None = None


_SCHEMAS: dict[str, CoordSchema] = {
    "bed": CoordSchema("chrom", "chromStart", "chromEnd", zero_based=True,  strand="strand"),
    "pb":  CoordSchema("chrom", "start",      "end",      zero_based=True,  strand="strand"),
    "pr":  CoordSchema("Chromosome", "Start", "End",      zero_based=True,  strand="Strand"),
    "gtf": CoordSchema("seqname",    "start", "end",      zero_based=False, strand="strand"),
    "gff": CoordSchema("seqname",    "start", "end",      zero_based=False, strand="strand"),
}

SchemaLike = Union[str, tuple, "CoordSchema"]


def _resolve_schema(s: SchemaLike) -> CoordSchema:
    if isinstance(s, CoordSchema):
        return s
    if isinstance(s, str):
        if s not in _SCHEMAS:
            raise ValueError(
                f"Unknown schema shorthand {s!r}. Valid shorthands: {list(_SCHEMAS)}."
            )
        return _SCHEMAS[s]
    if isinstance(s, tuple):
        if len(s) == 3:
            chrom, start, end = s
            return CoordSchema(chrom, start, end, zero_based=False)
        if len(s) == 4:
            chrom, start, end, strand = s
            return CoordSchema(chrom, start, end, zero_based=False, strand=strand)
        raise ValueError(
            f"Schema tuple must have 3 or 4 elements (chrom, start, end[, strand]), got {len(s)}."
        )
    raise TypeError(f"Cannot resolve schema from {type(s).__name__!r}.")
