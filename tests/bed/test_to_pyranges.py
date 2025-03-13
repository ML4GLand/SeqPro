import polars as pl
import polars.testing.parametric as plst
import seqpro as sp
from hypothesis import given, settings
from pytest_cases import parametrize_with_cases

settings(deadline=500)


def bedlike_strategy():
    return plst.dataframes(
        [
            plst.column("chrom", dtype=pl.Utf8, allow_null=False),
            plst.column("chromStart", dtype=pl.Int64, allow_null=False),
            plst.column("length", dtype=pl.UInt16, allow_null=False),
            plst.column("other"),
        ]
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
def test_bed(bed: pl.DataFrame):
    sp.bed.to_pyranges(bed)


@given(bedlike_strategy())
def test_bedlike(bedlike: pl.DataFrame):
    sp.bed.to_pyranges(bedlike)
