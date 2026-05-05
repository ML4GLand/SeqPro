# Design: Narwhals Integration & Coordinate Schema System

**Date:** 2026-05-05  
**Status:** Approved

## Overview

Two related changes to `bed.py` and `gtf.py`:

1. Make DataFrame-transforming functions generic over input/output frame types via narwhals (`@nw.narwhalify`), so callers using polars or pandas get the same type back.
2. Add a coordinate schema system: a registry of named column layouts (`"bed"`, `"pb"`, `"pr"`, `"gtf"`), auto-detection from a DataFrame's columns, and a `set_schema` function that renames columns to match a target schema.

File-reading functions (`bed.read`, `gtf.scan`) stay polars-concrete. Schema-aware features (polars-bio `config_meta`) are applied only when the native frame is `pl.DataFrame`.

---

## New File: `python/seqpro/_coords.py`

Central home for the schema system. Both `bed.py` and `gtf.py` import from here.

### `CoordSchema`

```python
@dataclass(frozen=True)
class CoordSchema:
    chrom: str
    start: str
    end: str
    zero_based: bool
```

Strand is not part of the schema definition — it is passed through unchanged by all operations.

### Built-in registry

```python
_SCHEMAS: dict[str, CoordSchema] = {
    "bed": CoordSchema("chrom", "chromStart", "chromEnd", zero_based=True),
    "pb":  CoordSchema("chrom", "start",      "end",      zero_based=True),
    "pr":  CoordSchema("Chromosome", "Start", "End",      zero_based=True),
    "gtf": CoordSchema("seqname",    "start", "end",      zero_based=False),
    "gff": CoordSchema("seqname",    "start", "end",      zero_based=False),
}
```

### `_resolve_schema(s) → CoordSchema`

Private helper. Accepts:
- A shorthand string (`"bed"`, `"pb"`, `"pr"`, `"gtf"`, `"gff"`) → looks up `_SCHEMAS`.
- A 3-tuple `(chrom_col, start_col, end_col)` → constructs `CoordSchema(..., zero_based=False)`.
- A 4-tuple `(chrom_col, start_col, end_col, strand_col)` → constructs `CoordSchema` from the first three columns; `strand_col` is accepted for convenience but ignored (strand is not part of the schema definition).
- A `CoordSchema` instance → returned as-is.

Raises `ValueError` for unrecognized strings or wrong-length tuples.

### `detect_schema(df, hint=None) → CoordSchema`

Public, `@nw.narwhalify`. Resolves the coordinate schema of `df`.

- If `hint` is given: calls `_resolve_schema(hint)`, then checks that the schema's three columns exist in `df`. Raises `ValueError` if any are missing.
- If `hint` is `None`: checks each built-in schema's three required columns against `df.columns`. If exactly one matches, returns it. If zero or more than one match, raises `ValueError`:
  - Zero matches: lists the available shorthands and tuple syntax.
  - Multiple matches: names the ambiguous schemas and tells the user to pass `hint=`.

### `set_schema(df, to, from_=None) → FrameT`

Public, `@nw.narwhalify`.

1. Resolve source: `src = detect_schema(df, hint=from_)`.
2. Resolve target: `tgt = _resolve_schema(to)`.
3. Build rename map for the three coord columns: `{src.chrom: tgt.chrom, src.start: tgt.start, src.end: tgt.end}`. Skip any pair where source == target.
4. Call `nw.to_native(df.rename(rename_map))` to get the native result before returning.
5. If `to` resolves to `"pb"` and the native result is a `pl.DataFrame`, call `result.config_meta.set(coordinate_system_zero_based=True)` guarded by `hasattr(result, "config_meta")`.
6. Return the native result directly (do not use `@nw.narwhalify` for this function — manage wrapping manually via `nw.from_native` / `nw.to_native` so the config_meta call happens before the return).

---

## Changes to `bed.py`

### Dependencies

- Add `import narwhals as nw` and `from narwhals.typing import FrameT`.
- Import `detect_schema`, `set_schema`, `_resolve_schema` from `._coords`.
- Add `narwhals` to `pixi.toml` as a hard dependency.

### `sort(bed) → FrameT`

Add `@nw.narwhalify`. Replace `natsorted(bed["chrom"].unique())` with:

```python
natsorted(bed.select(nw.col("chrom").unique()).collect()["chrom"].to_list())
```

Replace `pl.col("chrom").cast(pl.Enum(order))` with `nw.col("chrom").cast(nw.Enum(order))`.

### `with_len(bed, length) → FrameT`

Add `@nw.narwhalify`. Replace `pl.col(...)` / `pl.when(...)` with `nw.col(...)` / `nw.when(...)`. The `"peak" in bed` membership check becomes `"peak" in bed.columns`.

### `to_pyr(bedlike) → pr.PyRanges`

Not `@nw.narwhalify` (output is PyRanges, not a frame). Accept any narwhals-supported frame by wrapping input: `nw.from_native(bedlike, eager_or_interchange_only=True).to_pandas()`. Then apply existing rename + `pr.PyRanges(...)`.

### `from_pyr(pyr) → pl.DataFrame`

No change — always returns polars.

### `read(path) → pl.DataFrame`

No signature change. After the existing read+validate pipeline, add:

```python
if hasattr(result, "config_meta"):
    result.config_meta.set(coordinate_system_zero_based=True)
return result
```

### `__all__`

Add `"detect_schema"` and `"set_schema"`.

---

## Changes to `gtf.py`

No functional changes. The `"gtf"`/`"gff"` schemas are available in `_SCHEMAS` so `detect_schema` and `set_schema` work on GTF-originated frames.

`gtf.attr()` returns a `pl.Expr` (expression builder), not a frame — narwhals does not apply.

---

## Error message design

When `detect_schema` raises on ambiguity or missing columns, the message includes:

- What was found (which columns were present / which schemas matched).
- A reminder that schemas can be specified explicitly: `hint="bed"`, `hint="pb"`, `hint="pr"`, `hint="gtf"`, or `hint=("chrom_col", "start_col", "end_col")`.

Example:

```
Could not detect coordinate schema: columns match both 'pb' and 'gtf'.
Specify explicitly with hint="pb", hint="gtf", or hint=("chrom", "start", "end").
```

---

## Testing

- `test_coords.py`: unit tests for `_resolve_schema`, `detect_schema` (happy path, zero matches, multiple matches, hint override), and `set_schema` (all schema pairs, config_meta guard).
- `test_bed.py`: extend existing tests to verify `sort` and `with_len` return the same type as input (polars in → polars out, pandas in → pandas out).
- `test_bed.py`: verify `read()` result has `config_meta` set when polars-bio is present (or gracefully skips when absent).
