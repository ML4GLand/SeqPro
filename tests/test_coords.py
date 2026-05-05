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
