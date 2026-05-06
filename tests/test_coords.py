import pandas as pd
import polars as pl
import pytest
from seqpro._coords import (
    _SCHEMAS,
    CoordSchema,
    _resolve_schema,
    detect_schema,
    set_schema,
)


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
        "chrom",
        "chromStart",
        "chromEnd",
        "strand",
        True,
    )


def test_schemas_pb():
    s = _SCHEMAS["pb"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "chrom",
        "start",
        "end",
        "strand",
        True,
    )


def test_schemas_pr():
    s = _SCHEMAS["pr"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "Chromosome",
        "Start",
        "End",
        "Strand",
        True,
    )


def test_schemas_gtf():
    s = _SCHEMAS["gtf"]
    assert (s.chrom, s.start, s.end, s.strand, s.zero_based) == (
        "seqname",
        "start",
        "end",
        "strand",
        False,
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
    return pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100], "strand": ["+"]}
    )


def _pb_df():
    return pl.DataFrame(
        {"chrom": ["chr1"], "start": [0], "end": [100], "strand": ["+"]}
    )


def _pr_df():
    return pl.DataFrame(
        {"Chromosome": ["chr1"], "Start": [0], "End": [100], "Strand": ["+"]}
    )


def _gtf_df():
    return pl.DataFrame(
        {"seqname": ["chr1"], "start": [0], "end": [100], "strand": ["+"]}
    )


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
    assert detect_schema(
        _bed_df(), hint=("chrom", "chromStart", "chromEnd")
    ) == CoordSchema("chrom", "chromStart", "chromEnd", zero_based=False)


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
    df = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [100],
            "start": [0],
            "end": [100],
        }
    )
    with pytest.raises(ValueError, match="match"):
        detect_schema(df)


def test_detect_schema_pandas():
    df = pd.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    assert detect_schema(df) == _SCHEMAS["bed"]


def test_set_schema_bed_to_pb():
    df = pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [100],
            "strand": ["+"],
            "score": [1.0],
        }
    )
    result = set_schema(df, to="pb")
    assert result.columns == ["chrom", "start", "end", "strand", "score"]
    assert isinstance(result, pl.DataFrame)


def test_set_schema_bed_to_pr():
    df = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100], "strand": ["+"]}
    )
    result = set_schema(df, to="pr")
    assert set(result.columns) >= {"Chromosome", "Start", "End", "Strand"}


def test_set_schema_pr_to_bed():
    df = pl.DataFrame(
        {"Chromosome": ["chr1"], "Start": [0], "End": [100], "Strand": ["+"]}
    )
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
    df = pl.DataFrame(
        {"chrom": ["chr1", "chr2"], "chromStart": [0, 10], "chromEnd": [100, 200]}
    )
    result = set_schema(df, to="pb")
    assert result["start"].to_list() == [0, 10]
    assert result["end"].to_list() == [100, 200]


@pytest.mark.parametrize("schema", ["bed", "pb", "pr"])
def test_set_schema_zero_based_sets_meta_true(schema):
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    result = set_schema(df, to=schema)
    assert result.config_meta.get_metadata().get("coordinate_system_zero_based") is True


@pytest.mark.parametrize("schema", ["gtf", "gff"])
def test_set_schema_one_based_sets_meta_false(schema):
    df = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    result = set_schema(df, to=schema)
    assert (
        result.config_meta.get_metadata().get("coordinate_system_zero_based") is False
    )


def test_set_schema_pandas_no_config_meta():
    import pandas as pd

    df = pd.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    result = set_schema(df, to="pb")
    assert isinstance(result, pd.DataFrame)
    assert not hasattr(result, "config_meta")
