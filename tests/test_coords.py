import pytest
import polars as pl
import pandas as pd
from seqpro._coords import CoordSchema, _SCHEMAS, _resolve_schema, detect_schema


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
