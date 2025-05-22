from __future__ import annotations

from pathlib import Path

import polars as pl


def scan(path: str | Path):
    """Scan a GFF or GTF file.

    Parameters
    ----------
    path
        The path to the GTF file.
    """
    REQUIRED_COLUMNS = [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]

    DEFAULT_COLUMN_DTYPES = {
        "seqname": pl.Categorical,
        "source": pl.Categorical,
        "start": pl.Int64,
        "end": pl.Int64,
        "score": pl.Float32,
        "feature": pl.Categorical,
        "strand": pl.Categorical,
        "frame": pl.UInt32,
    }

    return pl.scan_csv(
        path,
        has_header=False,
        separator="\t",
        comment_prefix="#",
        null_values=".",
        new_columns=REQUIRED_COLUMNS,
        schema_overrides=DEFAULT_COLUMN_DTYPES,
    ).with_columns(
        pl.col("frame").fill_null(0),
        pl.col("attribute").str.replace_all('"', "'"),
    )


def attr(attr: str):
    """Extract a column from the attribute field.

    Parameters
    ----------
    attr
        The attribute to extract.
    """
    return pl.col("attribute").str.extract(rf"{attr} '(.*?)';?").alias(attr)
