from tempfile import NamedTemporaryFile

import polars as pl
import pytest
import seqpro as sp
from polars.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases


def bed_valid():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
        },
    )


def bed_negative():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [-1],
            "chromEnd": [2],
        }
    )


def bed_null():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 1],
            "chromEnd": [2, 3],
            "name": [None, "david"],
            "score": [None, 0.0],
            "strand": [None, "+"],
        }
    )


@pytest.mark.xfail(reason="Region length must be non-negative.")
def bed_invalid_length():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [-1],
        }
    )


def narrowpeak_valid():
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


def narrowpeak_null():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 1],
            "chromEnd": [2, 3],
            "name": ["peak1", "peak2"],
            "score": [0.0, 0.0],
            "strand": ["+", "-"],
            "signalValue": [0.0, 0.0],
            "pValue": [None, 0.2],
            "qValue": [None, 0.3],
            "peak": [None, 0],
        },
    )


@pytest.mark.xfail(reason="Peak < 0.")
def narrowpeak_invalid():
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
            "peak": [-2],
        }
    )


def broadpeak_valid():
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


@parametrize_with_cases("bed", cases=".", prefix="bed_")
def test_read_bed(bed: pl.DataFrame):
    with NamedTemporaryFile("w+", suffix=".bed") as f:
        bed.with_columns(pl.col(r"^(name|score|strand)$").fill_null(".")).write_csv(
            f.name, include_header=False, separator="\t"
        )
        assert_frame_equal(sp.bed.read_bedlike(f.name), bed)


@parametrize_with_cases("bed", cases=".", prefix="narrowpeak_")
def test_read_narrowpeak(bed: pl.DataFrame):
    with NamedTemporaryFile("w+", suffix=".narrowPeak") as f:
        bed.with_columns(
            pl.col("name", "score", "strand").fill_null("."),
            pl.col("pValue", "qValue", "peak").fill_null(-1),
        ).write_csv(f.name, include_header=False, separator="\t")
        assert_frame_equal(sp.bed.read_bedlike(f.name), bed)


@parametrize_with_cases("bed", cases=".", prefix="broadpeak_")
def test_read_broadpeak(bed: pl.DataFrame):
    with NamedTemporaryFile("w+", suffix=".broadPeak") as f:
        bed.with_columns(
            pl.col("name", "score", "strand").fill_null("."),
            pl.col("pValue", "qValue").fill_null(-1),
        ).write_csv(f.name, include_header=False, separator="\t")
        assert_frame_equal(sp.bed.read_bedlike(f.name), bed)
