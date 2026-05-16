"""Shape-matrix tests for pad_seqs (dense, bytes and OHE)."""

from __future__ import annotations

import numpy as np
import pytest
import seqpro as sp

from _shape_fixtures import (
    SHAPE_PATTERNS,
    add_leading_trailing_dense,
    assert_broadcast_equal_dense,
)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_pad_seqs_bytes_shape_matrix(pattern):
    # baseline (L,) bytes seq of length 4, pad to length 6
    baseline = np.frombuffer(b"ACGT", dtype="S1")
    baseline_out = sp.pad_seqs(
        baseline, pad="right", pad_value="N", length=6, length_axis=-1
    )

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.pad_seqs(
        patterned, pad="right", pad_value="N", length=6, length_axis=new_la
    )

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_pad_seqs_ohe_shape_matrix(pattern):
    # baseline OHE (L, A); key axis is "length axis" = axis 0 (or -2)
    baseline_bytes = np.frombuffer(b"ACGT", dtype="S1")
    baseline_ohe = sp.DNA.ohe(baseline_bytes)  # (4, A)
    baseline_out = sp.pad_seqs(baseline_ohe, pad="right", length=6, length_axis=0)

    # Use length_axis=0 of baseline as the "length axis". Build patterned by
    # inserting leading/trailing dims relative to that.
    patterned, new_la = add_leading_trailing_dense(baseline_ohe, pattern, length_axis=0)
    patterned_out = sp.pad_seqs(patterned, pad="right", length=6, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)
