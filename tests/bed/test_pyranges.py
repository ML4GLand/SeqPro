import polars as pl
import polars.testing.parametric as plst
import pytest
import seqpro as sp
from hypothesis import given, settings  # noqa: F401
from polars.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases

settings(deadline=500)


def bedlike_strategy():
    return plst.dataframes(
        [
            plst.column("chrom", dtype=pl.Utf8, allow_null=False),
            plst.column("chromStart", dtype=pl.Int64, allow_null=False),
            plst.column("length", dtype=pl.UInt16, allow_null=False),
            plst.column("other"),
        ],
        min_size=1,
    ).map(
        lambda df: df.with_columns(
            chromEnd=pl.col("chromStart") + pl.col("length")
        ).drop("length")
    )


def case_bed():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
        },
    )


def case_narrowpeak():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
            "name": ["peak"],
            "score": [0.0],
            "strand": ["+"],
            "signalValue": [0.0],
            "pValue": [0.0],
            "qValue": [0.0],
            "peak": [0],
        }
    )


def case_broadpeak():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
            "name": ["peak"],
            "score": [0.0],
            "strand": ["+"],
            "signalValue": [0.0],
            "pValue": [0.0],
            "qValue": [0.0],
        }
    )


@parametrize_with_cases("bed", cases=".")
def test_roundtrip_bed(bed: pl.DataFrame):
    pr = sp.bed.to_pyr(bed)
    new_bed = sp.bed.from_pyr(pr)
    assert_frame_equal(bed, new_bed)


@pytest.mark.skip
@given(bedlike_strategy())
def test_roundtrip_bedlike(bed: pl.DataFrame):
    pr = sp.bed.to_pyr(bed.with_row_index())
    new_bed = sp.bed.from_pyr(pr).sort("index").drop("index")
    #! exclude Enum, Decimal, and Struct for now
    with pl.StringCache():
        if isinstance(bed.schema["other"], (pl.Decimal, pl.Enum, pl.Struct)):
            assert_frame_equal(bed.drop("other"), new_bed.drop("other"))
        else:
            assert_frame_equal(bed, new_bed)
