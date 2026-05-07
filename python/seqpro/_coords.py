from __future__ import annotations

import narwhals as nw
import polars as pl
import polars_config_meta  # noqa: F401
from attrs import define
from narwhals.typing import Frame, FrameT


@define(frozen=True)
class CoordSchema:
    chrom: str
    start: str
    end: str
    zero_based: bool
    strand: str | None = None


_SCHEMAS: dict[str, CoordSchema] = {
    "bed": CoordSchema(
        "chrom", "chromStart", "chromEnd", zero_based=True, strand="strand"
    ),
    "pb": CoordSchema("chrom", "start", "end", zero_based=True, strand="strand"),
    "pr": CoordSchema("Chromosome", "Start", "End", zero_based=True, strand="Strand"),
    "gtf": CoordSchema("seqname", "start", "end", zero_based=False, strand="strand"),
    "gff": CoordSchema("seqname", "start", "end", zero_based=False, strand="strand"),
}

SchemaLike = str | tuple[str, str, str] | tuple[str, str, str, str] | CoordSchema


def _resolve_schema(s: SchemaLike) -> CoordSchema:
    if isinstance(s, CoordSchema):
        return s
    elif isinstance(s, str):
        if s not in _SCHEMAS:
            raise ValueError(
                f"Unknown schema shorthand {s!r}. Valid shorthands: {list(_SCHEMAS)}."
            )
        return _SCHEMAS[s]
    elif len(s) == 3:
        chrom, start, end = s
        return CoordSchema(chrom, start, end, zero_based=False)
    elif len(s) == 4:
        chrom, start, end, strand = s
        return CoordSchema(chrom, start, end, zero_based=False, strand=strand)
    elif isinstance(s, tuple) and len(s) not in {3, 4}:
        raise ValueError(
            f"Schema tuple must have 3 or 4 elements (chrom, start, end[, strand]), got {len(s)}."
        )
    else:
        raise ValueError(f"Invalid schema: {s!r}")


_HINT_MSG = (
    'Specify explicitly with hint="bed", hint="pb", hint="pr", hint="gtf", '
    'or hint=("chrom_col", "start_col", "end_col").'
)


def detect_schema(bed: Frame, hint: SchemaLike | None = None) -> CoordSchema:
    bed = nw.from_native(bed)

    cols = set(bed.columns)

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


def set_schema(bed: FrameT, to: SchemaLike, from_: SchemaLike | None = None) -> FrameT:
    """Rename coordinate columns to match a target schema.

    Parameters
    ----------
    df
        A polars or pandas DataFrame with genomic coordinate columns.
    to
        Target schema: a shorthand string ("bed", "pb", "pr", "gtf") or
        a tuple of column names (chrom, start, end[, strand]).
    from_
        Source schema hint. Auto-detected if not provided.
    """
    bed = nw.from_native(bed)

    src = detect_schema(bed, hint=from_)
    tgt = _resolve_schema(to)

    cols = bed.columns
    rename_map: dict[str, str] = {}
    for src_col, tgt_col in [
        (src.chrom, tgt.chrom),
        (src.start, tgt.start),
        (src.end, tgt.end),
    ]:
        if src_col != tgt_col and src_col in cols:
            rename_map[src_col] = tgt_col

    if (
        src.strand is not None
        and tgt.strand is not None
        and src.strand != tgt.strand
        and src.strand in cols
    ):
        rename_map[src.strand] = tgt.strand

    result = nw.to_native(bed.rename(rename_map))

    if isinstance(result, pl.DataFrame):
        result.config_meta.set(coordinate_system_zero_based=tgt.zero_based)  # type: ignore[attr-defined]

    return result
