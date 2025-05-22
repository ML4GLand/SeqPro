import polars as pl
import seqpro as sp
from polars.testing import assert_frame_equal


def test_sort():
    bed = pl.DataFrame(
        {
            "chrom": ["10", "1", "2", "2"],
            "chromStart": [1, 2, 3, 3],
            "chromEnd": [2, 3, 5, 4],
        }
    )

    actual_sort = sp.bed.sort(bed)
    desired_sort = sp.bed.from_pyr(sp.bed.to_pyr(bed).sort())
    assert_frame_equal(actual_sort, desired_sort)
