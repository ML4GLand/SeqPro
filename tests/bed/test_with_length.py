import math
from fractions import Fraction

import hypothesis.strategies as st
import polars as pl
import pytest
import seqpro as sp
from hypothesis import given
from pytest_cases import parametrize_with_cases


def case_bed_even():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
        }
    )


def case_bed_odd():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [1],
        }
    )


def case_peak_even():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
            "peak": [0],
        }
    )


def case_peak_odd():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [1],
            "peak": [0],
        }
    )


def case_peak_null():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
            "peak": [None],
        }
    )


@pytest.mark.xfail(reason="Region length must be non-negative.")
def case_invalid_length():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [-1],
        }
    )


@pytest.mark.xfail(reason="Peak must be non-negative.")
def case_invalid_peak():
    return pl.DataFrame(
        {
            "chrom": ["chr1"],
            "chromStart": [0],
            "chromEnd": [2],
            "peak": [-1],
        }
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@parametrize_with_cases("bed", cases=".")
@given(length=st.one_of(st.just(0), st.just(1), st.integers(-(2**31), 2**31)))
def test_bed(bed: pl.DataFrame, length: int):
    if length < 0:
        with pytest.raises(ValueError):
            sp.bed.with_length(bed, length)
        return

    len_adj = sp.bed.with_length(bed, length)

    if "peak" in bed:
        _, start, end, peak = bed.row(0)
        _, adj_start, adj_end, peak = len_adj.row(0)
        if peak is None:
            center = Fraction(start + end, 2)
        else:
            center = int(start + peak)
    else:
        _, start, end = bed.row(0)
        _, adj_start, adj_end = len_adj.row(0)
        center = Fraction(start + end, 2)

    desired_start = math.floor(center - Fraction(length, 2))

    assert adj_end - adj_start == length
    assert adj_start == desired_start
