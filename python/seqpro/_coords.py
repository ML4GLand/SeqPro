from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import narwhals as nw


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


_HINT_MSG = (
    'Specify explicitly with hint="bed", hint="pb", hint="pr", hint="gtf", '
    'or hint=("chrom_col", "start_col", "end_col").'
)


def detect_schema(df, hint: SchemaLike | None = None) -> CoordSchema:
    cols = set(nw.from_native(df).columns)

    if hint is not None:
        schema = _resolve_schema(hint)
        missing = {schema.chrom, schema.start, schema.end} - cols
        if missing:
            raise ValueError(
                f"Schema {hint!r} requires columns {missing!r} which are missing from the DataFrame. "
                + _HINT_MSG
            )
        return schema

    # Deduplicate by (chrom, start, end) key so gtf/gff count as one
    seen: dict[tuple[str, str, str], tuple[str, CoordSchema]] = {}
    for name, schema in _SCHEMAS.items():
        key = (schema.chrom, schema.start, schema.end)
        if key not in seen and {schema.chrom, schema.start, schema.end} <= cols:
            seen[key] = (name, schema)

    unique_matches = list(seen.values())

    if len(unique_matches) == 1:
        return unique_matches[0][1]

    if len(unique_matches) == 0:
        raise ValueError(
            f"Could not detect coordinate schema: no known schema columns found. "
            f"DataFrame columns: {sorted(cols)!r}. " + _HINT_MSG
        )

    names = [name for name, _ in unique_matches]
    raise ValueError(
        f"Could not detect coordinate schema: columns match {names!r}. " + _HINT_MSG
    )
