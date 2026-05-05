# Narwhals Integration & Coordinate Schema System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add narwhals generics to BED/GTF transform functions and a coordinate schema registry with auto-detection and column renaming.

**Architecture:** A new `_coords.py` module owns `CoordSchema`, the built-in schema registry, `_resolve_schema`, `detect_schema`, and `set_schema`. `bed.py` imports these and re-exports the public ones; transform functions gain `@nw.narwhalify`. `read()` stays polars-concrete but gains a `config_meta` call guarded by `hasattr`.

**Tech Stack:** narwhals (new hard dep), polars, pandas, pyranges, pandera, natsort, pixi/maturin

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `python/seqpro/_coords.py` | CoordSchema, _SCHEMAS, _resolve_schema, detect_schema, set_schema |
| Create | `tests/test_coords.py` | All schema system tests |
| Modify | `python/seqpro/bed.py` | Narwhalify sort/with_len/to_pyr, config_meta in read(), import/export _coords symbols |
| Modify | `tests/bed/test_sort.py` | Add pandas passthrough test |
| Modify | `tests/bed/test_with_length.py` | Add pandas passthrough test |
| Modify | `pyproject.toml` | Add narwhals>=1.0 to runtime dependencies |
| Modify | `pixi.toml` | Add narwhals to [feature.dev.dependencies] |

---

## Task 1: Add narwhals dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`

- [ ] **Step 1: Add narwhals to pyproject.toml runtime dependencies**

In `pyproject.toml`, add to `dependencies`:

```toml
dependencies = [
    "numba>=0.58.1",
    "numpy>=1.26.0",
    "polars>=1.10.0,<2",
    "pyranges>=0.1.3,<0.2",
    "pandera>=0.22.1",
    "pandas",
    "pyarrow",
    "natsort",
    "narwhals>=1.0",
    "setuptools>=70",
    "awkward>=2.5.0",
]
```

- [ ] **Step 2: Add narwhals to pixi.toml dev feature**

In `pixi.toml`, add to `[feature.dev.dependencies]`:

```toml
[feature.dev.dependencies]
numba = "==0.58.1"
numpy = "==1.26.0"
polars = "==1.10.0"
pandera = "==0.22.1"
awkward = "==2.5.0"
pandas = "*"
natsort = "*"
pyarrow = "*"
ipykernel = "*"
prek = "*"
narwhals = ">=1.0"
```

- [ ] **Step 3: Reinstall dev environment**

```bash
pixi install -e dev
```

Expected: resolves without error, narwhals present in environment.

- [ ] **Step 4: Verify narwhals is importable**

```bash
pixi run -e dev python -c "import narwhals as nw; print(nw.__version__)"
```

Expected: prints a version string like `1.x.x`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml pixi.toml pixi.lock
git commit -m "feat: add narwhals as hard dependency"
```

---

## Task 2: `_coords.py` — CoordSchema, registry, and _resolve_schema

**Files:**
- Create: `python/seqpro/_coords.py`
- Create: `tests/test_coords.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_coords.py`:

```python
import pytest
from seqpro._coords import CoordSchema, _SCHEMAS, _resolve_schema


def test_coordschema_fields():
    s = CoordSchema("chrom", "chromStart", "chromEnd", zero_based=True, strand="strand")
    assert s.chrom == "chrom"
    assert s.start == "chromStart"
    assert s.end == "chromEnd"
    assert s.zero_based is True
    assert s.strand == "strand"


def test_coordschema_strand_optional():
    s = CoordSchema("chrom", "start", "end", zero_based=False)
    assert s.strand is None


def test_coordschema_frozen():
    s = CoordSchema("chrom", "start", "end", zero_based=True)
    with pytest.raises((AttributeError, TypeError)):
        s.chrom = "other"  # type: ignore


def test_schemas_keys():
    assert set(_SCHEMAS) >= {"bed", "pb", "pr", "gtf", "gff"}


def test_schemas_bed():
    s = _SCHEMAS["bed"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "chrom", "chromStart", "chromEnd", "strand", True
    )


def test_schemas_pb():
    s = _SCHEMAS["pb"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "chrom", "start", "end", "strand", True
    )


def test_schemas_pr():
    s = _SCHEMAS["pr"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "Chromosome", "Start", "End", "Strand", True
    )


def test_schemas_gtf():
    s = _SCHEMAS["gtf"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "seqname", "start", "end", "strand", False
    )
    assert _SCHEMAS["gff"] == _SCHEMAS["gtf"]


def test_resolve_schema_string():
    assert _resolve_schema("bed") is _SCHEMAS["bed"]
    assert _resolve_schema("pb") is _SCHEMAS["pb"]
    assert _resolve_schema("pr") is _SCHEMAS["pr"]
    assert _resolve_schema("gtf") is _SCHEMAS["gtf"]


def test_resolve_schema_unknown_string():
    with pytest.raises(ValueError, match="Unknown schema"):
        _resolve_schema("xyz")


def test_resolve_schema_3tuple():
    s = _resolve_schema(("mychr", "mystart", "myend"))
    assert s.chrom == "mychr"
    assert s.start == "mystart"
    assert s.end == "myend"
    assert s.strand is None
    assert s.zero_based is False


def test_resolve_schema_4tuple():
    s = _resolve_schema(("mychr", "mystart", "myend", "mystrand"))
    assert s.chrom == "mychr"
    assert s.strand == "mystrand"


def test_resolve_schema_bad_tuple():
    with pytest.raises(ValueError, match="3 or 4"):
        _resolve_schema(("a", "b"))


def test_resolve_schema_coordschema_passthrough():
    original = CoordSchema("a", "b", "c", zero_based=False)
    assert _resolve_schema(original) is original


def test_resolve_schema_bad_type():
    with pytest.raises(TypeError):
        _resolve_schema(42)  # type: ignore
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_coords.py -v
```

Expected: `ImportError` — `_coords` module does not exist yet.

- [ ] **Step 3: Create `python/seqpro/_coords.py` with CoordSchema, _SCHEMAS, _resolve_schema**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run -e dev pytest tests/test_coords.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_coords.py tests/test_coords.py
git commit -m "feat: add CoordSchema, _SCHEMAS, and _resolve_schema"
```

---

## Task 3: `_coords.py` — `detect_schema`

**Files:**
- Modify: `python/seqpro/_coords.py`
- Modify: `tests/test_coords.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coords.py`:

```python
import polars as pl
import pandas as pd
from seqpro._coords import detect_schema


def _bed_df():
    return pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100], "strand": ["+"]})


def _pb_df():
    return pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100], "strand": ["+"]})


def _pr_df():
    return pl.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [100], "Strand": ["+"]})


def _gtf_df():
    return pl.DataFrame({"seqname": ["chr1"], "start": [0], "end": [100], "strand": ["+"]})


def test_detect_schema_bed():
    assert detect_schema(_bed_df()) == _SCHEMAS["bed"]


def test_detect_schema_pb():
    assert detect_schema(_pb_df()) == _SCHEMAS["pb"]


def test_detect_schema_pr():
    assert detect_schema(_pr_df()) == _SCHEMAS["pr"]


def test_detect_schema_gtf():
    assert detect_schema(_gtf_df()) == _SCHEMAS["gtf"]


def test_detect_schema_hint_string():
    assert detect_schema(_bed_df(), hint="bed") == _SCHEMAS["bed"]


def test_detect_schema_hint_tuple():
    assert detect_schema(_bed_df(), hint=("chrom", "chromStart", "chromEnd")) == CoordSchema(
        "chrom", "chromStart", "chromEnd", zero_based=False
    )


def test_detect_schema_hint_missing_col():
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0]})  # missing chromEnd
    with pytest.raises(ValueError, match="missing"):
        detect_schema(df, hint="bed")


def test_detect_schema_no_match():
    df = pl.DataFrame({"foo": [1], "bar": [2]})
    with pytest.raises(ValueError, match="no known schema"):
        detect_schema(df)


def test_detect_schema_ambiguous():
    # Has both bed and pb coord columns
    df = pl.DataFrame({
        "chrom": ["chr1"], "chromStart": [0], "chromEnd": [100],
        "start": [0], "end": [100],
    })
    with pytest.raises(ValueError, match="match"):
        detect_schema(df)


def test_detect_schema_pandas():
    df = pd.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    assert detect_schema(df) == _SCHEMAS["bed"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_coords.py::test_detect_schema_bed -v
```

Expected: `ImportError` — `detect_schema` not defined yet.

- [ ] **Step 3: Add `detect_schema` to `_coords.py`**

Add the following imports at the top of `_coords.py`:

```python
import narwhals as nw
```

Add the function after `_resolve_schema`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run -e dev pytest tests/test_coords.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_coords.py tests/test_coords.py
git commit -m "feat: add detect_schema"
```

---

## Task 4: `_coords.py` — `set_schema`

**Files:**
- Modify: `python/seqpro/_coords.py`
- Modify: `tests/test_coords.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_coords.py`:

```python
import polars as pl
import pandas as pd
from polars.testing import assert_frame_equal
from seqpro._coords import set_schema


def test_set_schema_bed_to_pb():
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100], "strand": ["+"], "score": [1.0]})
    result = set_schema(df, to="pb")
    assert result.columns == ["chrom", "start", "end", "strand", "score"]
    assert isinstance(result, pl.DataFrame)


def test_set_schema_bed_to_pr():
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100], "strand": ["+"]})
    result = set_schema(df, to="pr")
    assert set(result.columns) >= {"Chromosome", "Start", "End", "Strand"}


def test_set_schema_pr_to_bed():
    df = pl.DataFrame({"Chromosome": ["chr1"], "Start": [0], "End": [100], "Strand": ["+"]})
    result = set_schema(df, to="bed")
    assert set(result.columns) >= {"chrom", "chromStart", "chromEnd", "strand"}


def test_set_schema_no_strand_col():
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    result = set_schema(df, to="pb")
    assert "strand" not in result.columns
    assert result.columns == ["chrom", "start", "end"]


def test_set_schema_explicit_from():
    df = pl.DataFrame({"chrom": ["chr1"], "start": [0], "end": [100], "strand": ["+"]})
    result = set_schema(df, to="pr", from_="pb")
    assert set(result.columns) >= {"Chromosome", "Start", "End", "Strand"}


def test_set_schema_pandas_in_pandas_out():
    df = pd.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    result = set_schema(df, to="pb")
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["chrom", "start", "end"]


def test_set_schema_preserves_values():
    df = pl.DataFrame({"chrom": ["chr1", "chr2"], "chromStart": [0, 10], "chromEnd": [100, 200]})
    result = set_schema(df, to="pb")
    assert result["start"].to_list() == [0, 10]
    assert result["end"].to_list() == [100, 200]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_coords.py::test_set_schema_bed_to_pb -v
```

Expected: `ImportError` — `set_schema` not defined yet.

- [ ] **Step 3: Add `set_schema` to `_coords.py`**

Add after `detect_schema`:

```python
def set_schema(df, to: SchemaLike, from_: SchemaLike | None = None):
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
    import polars as pl

    nw_df = nw.from_native(df, eager_or_interchange_only=True)
    src = detect_schema(nw_df, hint=from_)
    tgt = _resolve_schema(to)

    cols = nw_df.columns
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

    result = nw.to_native(nw_df.rename(rename_map))

    tgt_schema = _resolve_schema(to)
    if tgt_schema == _SCHEMAS["pb"] and isinstance(result, pl.DataFrame):
        if hasattr(result, "config_meta"):
            result.config_meta.set(coordinate_system_zero_based=True)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run -e dev pytest tests/test_coords.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/_coords.py tests/test_coords.py
git commit -m "feat: add set_schema"
```

---

## Task 5: Narwhalify `bed.sort()`

**Files:**
- Modify: `python/seqpro/bed.py`
- Modify: `tests/bed/test_sort.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/bed/test_sort.py`:

```python
import pandas as pd


def test_sort_pandas_in_pandas_out():
    bed = pd.DataFrame(
        {
            "chrom": ["10", "1", "2", "2"],
            "chromStart": [1, 2, 3, 3],
            "chromEnd": [2, 3, 5, 4],
        }
    )
    result = sp.bed.sort(bed)
    assert isinstance(result, pd.DataFrame)
    assert list(result["chrom"]) == ["1", "2", "2", "10"]
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev pytest tests/bed/test_sort.py::test_sort_pandas_in_pandas_out -v
```

Expected: FAIL — `sort` currently only accepts polars and returns polars.

- [ ] **Step 3: Update `bed.sort()` in `bed.py`**

Add to `bed.py` imports (after `import polars as pl`):

```python
import narwhals as nw
from narwhals.typing import FrameT
```

Replace the existing `sort` function:

```python
@nw.narwhalify
def sort(bed: FrameT) -> FrameT:
    """Sort a BED-like DataFrame by chromosome, start, and end position, using the natural
    order of chromosome names e.g. 1, 2, ..., 10, ..."""
    order = natsorted(
        bed.select(nw.col("chrom").unique()).collect()["chrom"].to_list()
    )
    return bed.sort(
        nw.col("chrom").cast(nw.Enum(order)),
        "chromStart",
        "chromEnd",
        maintain_order=True,
    )
```

- [ ] **Step 4: Run all sort tests to verify they pass**

```bash
pixi run -e dev pytest tests/bed/test_sort.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/bed.py tests/bed/test_sort.py
git commit -m "feat: narwhalify bed.sort()"
```

---

## Task 6: Narwhalify `bed.with_len()`

**Files:**
- Modify: `python/seqpro/bed.py`
- Modify: `tests/bed/test_with_length.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/bed/test_with_length.py`:

```python
import pandas as pd


def test_with_len_pandas_in_pandas_out():
    bed = pd.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10]})
    result = sp.bed.with_len(bed, 4)
    assert isinstance(result, pd.DataFrame)
    assert result["chromStart"].iloc[0] == 3
    assert result["chromEnd"].iloc[0] == 7
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
pixi run -e dev pytest tests/bed/test_with_length.py::test_with_len_pandas_in_pandas_out -v
```

Expected: FAIL — `with_len` currently uses `pl.col`/`pl.when`.

- [ ] **Step 3: Update `bed.with_len()` in `bed.py`**

Replace the existing `with_len` function (narwhals imports were added in Task 5):

```python
@nw.narwhalify
def with_len(bed: FrameT, length: int) -> FrameT:
    """Set the length of regions in a BED-like DataFrame to a fixed length by expanding or shrinking
    relative to the center (or peak) of the window. If the original region size + length is odd, the
    center will be 1 position closer the right end.

    Parameters
    ----------
    bed
        BED-like DataFrame with at least the columns "chromStart" and "chromEnd".
    length
        Desired length of the windows. Must be non-negative.
    """
    if length < 0:
        raise ValueError("Length must be non-negative.")

    if "peak" in bed.columns:
        double_center = (
            nw.when(nw.col("peak").is_null())
            .then(nw.col("chromStart") + nw.col("chromEnd"))
            .otherwise(2 * (nw.col("chromStart") + nw.col("peak")))
        )
    else:
        double_center = nw.col("chromStart") + nw.col("chromEnd")

    return bed.with_columns(
        chromStart=(double_center - length) // 2,
        chromEnd=(double_center + length) // 2,
    )
```

- [ ] **Step 4: Run all with_len tests to verify they pass**

```bash
pixi run -e dev pytest tests/bed/test_with_length.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/bed.py tests/bed/test_with_length.py
git commit -m "feat: narwhalify bed.with_len()"
```

---

## Task 7: Update `bed.to_pyr()` for narwhals input

**Files:**
- Modify: `python/seqpro/bed.py`
- Modify: `tests/bed/test_pyranges.py`

- [ ] **Step 1: Read the existing test file**

```bash
cat tests/bed/test_pyranges.py
```

- [ ] **Step 2: Write the failing test**

Append to `tests/bed/test_pyranges.py`:

```python
import pandas as pd


def test_to_pyr_accepts_pandas():
    bed = pd.DataFrame({
        "chrom": ["chr1", "chr2"],
        "chromStart": [0, 10],
        "chromEnd": [100, 200],
        "strand": ["+", "-"],
    })
    pyr = sp.bed.to_pyr(bed)
    assert hasattr(pyr, "df")
    assert "Chromosome" in pyr.df.columns
    assert "Start" in pyr.df.columns
```

- [ ] **Step 3: Run the test to verify it fails**

```bash
pixi run -e dev pytest tests/bed/test_pyranges.py::test_to_pyr_accepts_pandas -v
```

Expected: FAIL or TypeError — `to_pyr` calls `.to_pandas()` on a polars frame, which fails for a pandas input.

- [ ] **Step 4: Update `bed.to_pyr()` in `bed.py`**

Replace the existing `to_pyr` function (narwhals imports already added in Task 5):

```python
def to_pyr(bedlike) -> pr.PyRanges:
    """Convert a BED-like DataFrame to a PyRanges object.

    .. important::
        PyRanges automatically sorts the DataFrame by chromosome and start position, so the order of
        the regions may change after conversion. You can keep track of the original
        order by `adding an index column <https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.with_row_index.html>`_
        before converting to a PyRanges object. After converting back to a DataFrame, you can sort the DataFrame by the index to
        get the original order.

    Parameters
    ----------
    bedlike
        BED-like DataFrame (polars or pandas) with at least the columns "chrom", "chromStart", and "chromEnd".
    """
    pdf = nw.from_native(bedlike, eager_or_interchange_only=True).to_pandas()
    return pr.PyRanges(
        pdf.rename(
            columns={
                "chrom": "Chromosome",
                "chromStart": "Start",
                "chromEnd": "End",
                "strand": "Strand",
            }
        )
    )
```

- [ ] **Step 5: Run all pyranges tests to verify they pass**

```bash
pixi run -e dev pytest tests/bed/test_pyranges.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/bed.py tests/bed/test_pyranges.py
git commit -m "feat: bed.to_pyr() accepts any narwhals-supported frame"
```

---

## Task 8: Add `config_meta` to `bed.read()` and wire up `_coords` imports

**Files:**
- Modify: `python/seqpro/bed.py`
- Modify: `tests/bed/test_read.py`

- [ ] **Step 1: Read the existing read test to understand its structure**

```bash
cat tests/bed/test_read.py
```

- [ ] **Step 2: Write the failing test**

Append to `tests/bed/test_read.py` (pick any small .bed fixture that already exists in the test file, or use `tmp_path`):

```python
def test_read_returns_polars_dataframe(tmp_path):
    bed_content = "chr1\t0\t100\tname\t0\t+\n"
    bed_file = tmp_path / "test.bed"
    bed_file.write_text(bed_content)
    result = sp.bed.read(bed_file)
    assert isinstance(result, pl.DataFrame)
```

- [ ] **Step 3: Add `config_meta` call and `_coords` imports to `bed.py`**

At the top of `bed.py`, add after the existing imports:

```python
from ._coords import CoordSchema, detect_schema, set_schema
```

Replace the existing `read` function body with:

```python
def read(path: PathLike) -> pl.DataFrame:
    """Reads a bed-like (BED3+) file as a pandas DataFrame. The file type is inferred
    from the file extension and supports .bed, .narrowPeak, and .broadPeak.

    Parameters
    ----------
    path
        Path to the bed-like file.

    Returns
    -------
    polars.DataFrame
    """
    path = Path(path)
    if ".bed" in path.suffixes:
        result = _read_bed(path)
    elif ".narrowPeak" in path.suffixes:
        result = _read_narrowpeak(path)
    elif ".broadPeak" in path.suffixes:
        result = _read_broadpeak(path)
    else:
        raise ValueError(
            f"""Unrecognized file extension: {"".join(path.suffixes)}. Expected one of 
            .bed, .narrowPeak, or .broadPeak (potentially gzipped)."""
        )
    if hasattr(result, "config_meta"):
        result.config_meta.set(coordinate_system_zero_based=True)
    return result
```

Update `__all__` in `bed.py`:

```python
__all__ = [
    "sort",
    "with_len",
    "to_pyr",
    "from_pyr",
    "read",
    "BEDSchema",
    "NarrowPeakSchema",
    "BroadPeakSchema",
    "CoordSchema",
    "detect_schema",
    "set_schema",
]
```

- [ ] **Step 4: Run the full test suite**

```bash
pixi run -e dev pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/seqpro/bed.py python/seqpro/_coords.py tests/bed/test_read.py
git commit -m "feat: add config_meta to bed.read(), export detect_schema and set_schema from bed"
```

---

## Self-Review

### Spec coverage

| Spec requirement | Task |
|---|---|
| narwhals hard dep | Task 1 |
| CoordSchema with chrom/start/end/zero_based/strand | Task 2 |
| _SCHEMAS registry (bed/pb/pr/gtf/gff) | Task 2 |
| _resolve_schema (string, 3-tuple, 4-tuple, CoordSchema passthrough) | Task 2 |
| detect_schema (auto, hint, error messages) | Task 3 |
| set_schema (rename, strand, config_meta for pb, pandas passthrough) | Task 4 |
| @nw.narwhalify bed.sort() with nw.col/nw.Enum | Task 5 |
| @nw.narwhalify bed.with_len() with nw.col/nw.when | Task 6 |
| to_pyr() accepts narwhals input | Task 7 |
| read() adds config_meta with hasattr guard | Task 8 |
| detect_schema/set_schema exported from bed.__all__ | Task 8 |
| detect_schema/set_schema work on GTF frames (gtf/gff in registry) | Task 2/3 |

All spec requirements covered. ✓

### Placeholder scan

No TBDs, TODOs, or vague steps — every code block is complete. ✓

### Type consistency

- `CoordSchema` defined in Task 2 with fields `chrom, start, end, zero_based, strand` — used identically in Tasks 3/4.
- `_resolve_schema` returns `CoordSchema` — consumed by `detect_schema` (Task 3) and `set_schema` (Task 4) correctly.
- `detect_schema` called from `set_schema` with a narwhals frame (`nw_df`) — `detect_schema` uses `nw.from_native(df)` which is a no-op for already-native frames (polars/pandas). ✓
- `set_schema` compares `tgt_schema == _SCHEMAS["pb"]` — both are frozen dataclasses, equality is field-based. ✓
