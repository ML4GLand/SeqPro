from pathlib import Path

import pandera.dtypes as pat
import pandera.polars as pa
import polars as pl
import pyranges as pr

from ._types import PathLike


def with_length(bed: pl.DataFrame, length: int) -> pl.DataFrame:
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

    # * Avoid any floating point math for consistent results
    if "peak" in bed:
        double_center = (
            pl.when(pl.col("peak").is_null())
            .then(pl.col("chromStart") + pl.col("chromEnd"))
            .otherwise(2 * (pl.col("chromStart") + pl.col("peak")))
        )
    else:
        double_center = pl.col("chromStart") + pl.col("chromEnd")

    return bed.with_columns(
        chromStart=(double_center - length) // 2,
        chromEnd=(double_center + length) // 2,
    )


def to_pyranges(bedlike: pl.DataFrame) -> pr.PyRanges:
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
        BED-like DataFrame with at least the columns "chrom", "chromStart", and "chromEnd".
    """
    return pr.PyRanges(
        bedlike.rename(
            {
                "chrom": "Chromosome",
                "chromStart": "Start",
                "chromEnd": "End",
                "strand": "Strand",
            },
            strict=False,
        ).to_pandas(use_pyarrow_extension_array=True)
    )


def from_pyranges(pyr: pr.PyRanges) -> pl.DataFrame:
    """Convert a PyRanges object to a BED-like DataFrame.

    Parameters
    ----------
    pyr
        PyRanges object with at least the columns "Chromosome", "Start", and "End".
    """
    return (
        pl.from_pandas(pyr.df)
        .rename(
            {
                "Chromosome": "chrom",
                "Start": "chromStart",
                "End": "chromEnd",
                "Strand": "strand",
            },
            strict=False,
        )
        .with_columns(
            # pyranges casts these to categorical, but we want them back as strings
            pl.col(r"^(chrom|strand)$").cast(pl.Utf8),
        )
    )


def read_bedlike(path: PathLike) -> pl.DataFrame:
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
        return _read_bed(path)
    elif ".narrowPeak" in path.suffixes:
        return _read_narrowpeak(path)
    elif ".broadPeak" in path.suffixes:
        return _read_broadpeak(path)
    else:
        raise ValueError(
            f"""Unrecognized file extension: {"".join(path.suffixes)}. Expected one of 
            .bed, .narrowPeak, or .broadPeak"""
        )


BEDSchema = pa.DataFrameSchema(
    {
        "chrom": pa.Column(str),
        "chromStart": pa.Column(int),
        "chromEnd": pa.Column(int),
        "name": pa.Column(str, nullable=True, required=False),
        "score": pa.Column(float, nullable=True, required=False),
        "strand": pa.Column(
            str, nullable=True, checks=pa.Check.isin(["+", "-", "."]), required=False
        ),
        "thickStart": pa.Column(int, nullable=True, required=False),
        "thickEnd": pa.Column(int, nullable=True, required=False),
        "itemRgb": pa.Column(str, nullable=True, required=False),
        "blockCount": pa.Column(pat.UInt64, nullable=True, required=False),
        "blockSizes": pa.Column(str, nullable=True, required=False),
        "blockStarts": pa.Column(str, nullable=True, required=False),
    },
    checks=pa.Check(
        lambda df: df.lazyframe.select(pl.col("chromEnd") >= pl.col("chromStart"))
    ),
    coerce=True,
)


def _read_bed(bed_path: PathLike) -> pl.DataFrame:
    with open(bed_path) as f:
        skip_rows = 0
        while (line := f.readline()).startswith(("track", "browser")):
            skip_rows += 1
    n_cols = line.count("\t") + 1
    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        skip_rows=skip_rows,
        new_columns=list(BEDSchema.columns)[:n_cols],
        schema_overrides={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
        null_values=".",
    ).pipe(BEDSchema.validate)
    return bed


NarrowPeakSchema = pa.DataFrameSchema(
    {
        "chrom": pa.Column(str),
        "chromStart": pa.Column(int),
        "chromEnd": pa.Column(int),
        "name": pa.Column(str, nullable=True, required=False),
        "score": pa.Column(float, nullable=True, required=False),
        "strand": pa.Column(
            str, nullable=True, checks=pa.Check.isin(["+", "-", "."]), required=False
        ),
        "signalValue": pa.Column(float, nullable=True, required=False),
        "pValue": pa.Column(float, nullable=True, required=False),
        "qValue": pa.Column(float, nullable=True, required=False),
        "peak": pa.Column(
            int,
            nullable=True,
            required=False,
            checks=pa.Check.greater_than_or_equal_to(0),
        ),
    },
    checks=pa.Check(
        lambda df: df.lazyframe.select(pl.col("chromEnd") >= pl.col("chromStart"))
    ),
    coerce=True,
)


def _read_narrowpeak(narrowpeak_path: PathLike) -> pl.DataFrame:
    with open(narrowpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    narrowpeaks = (
        pl.read_csv(
            narrowpeak_path,
            separator="\t",
            has_header=False,
            skip_rows=skip_rows,
            new_columns=[
                "chrom",
                "chromStart",
                "chromEnd",
                "name",
                "score",
                "strand",
                "signalValue",
                "pValue",
                "qValue",
                "peak",
            ],
            schema_overrides={
                "chrom": pl.Utf8,
                "name": pl.Utf8,
                "strand": pl.Utf8,
                "pValue": pl.Float64,
                "qValue": pl.Float64,
                "peak": pl.Int64,
            },
            null_values=".",
        )
        .with_columns(pl.col("pValue", "qValue", "peak").replace(-1, None))
        .pipe(NarrowPeakSchema.validate)
    )
    return narrowpeaks


BroadPeakSchema = pa.DataFrameSchema(
    {
        "chrom": pa.Column(str),
        "chromStart": pa.Column(int),
        "chromEnd": pa.Column(int),
        "name": pa.Column(str, nullable=True, required=False),
        "score": pa.Column(float, nullable=True, required=False),
        "strand": pa.Column(
            str, nullable=True, checks=pa.Check.isin(["+", "-", "."]), required=False
        ),
        "signalValue": pa.Column(float, nullable=True, required=False),
        "pValue": pa.Column(float, nullable=True, required=False),
        "qValue": pa.Column(float, nullable=True, required=False),
    },
    checks=pa.Check(
        lambda df: df.lazyframe.select(pl.col("chromEnd") >= pl.col("chromStart"))
    ),
    coerce=True,
)


def _read_broadpeak(broadpeak_path: PathLike) -> pl.DataFrame:
    with open(broadpeak_path) as f:
        skip_rows = 0
        while f.readline().startswith(("track", "browser")):
            skip_rows += 1
    broadpeaks = (
        pl.read_csv(
            broadpeak_path,
            separator="\t",
            has_header=False,
            skip_rows=skip_rows,
            new_columns=[
                "chrom",
                "chromStart",
                "chromEnd",
                "name",
                "score",
                "strand",
                "signalValue",
                "pValue",
                "qValue",
            ],
            schema_overrides={"chrom": pl.Utf8, "name": pl.Utf8, "strand": pl.Utf8},
            null_values=".",
        )
        .with_columns(pl.col("pValue", "qValue").replace(-1, None))
        .pipe(BroadPeakSchema.validate)
    )
    return broadpeaks
