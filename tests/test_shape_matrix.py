"""Shape-matrix tests for analyzer/modifier/transform functions.

These tests verify that each in-scope function applies independently across
arbitrary leading and trailing dimensions around its key axes. Some tests are
expected to fail in Phase A — those failures form the Phase B backlog.
"""

from __future__ import annotations

import numpy as np
import pytest
import seqpro as sp
import seqpro.transforms as spt

from _shape_fixtures import (
    B,
    M,
    SHAPE_PATTERNS,
    add_leading_trailing_dense,
    assert_broadcast_equal_dense,
    assert_broadcast_equal_ragged,
    make_ragged_baseline,
    make_ragged_patterned,
)

# -----------------------------------------------------------------------------
# ohe (dense + ragged)
# -----------------------------------------------------------------------------


def _broadcast_ohe_expected(baseline_out: np.ndarray, patterned_in_shape, pattern):
    """ohe always appends alphabet axis last. So patterned input (..., M)
    gives output (..., M, A); leading gives (B, L, A)."""
    A = baseline_out.shape[-1]
    if pattern == "baseline":
        return baseline_out
    if pattern == "leading":
        return np.broadcast_to(baseline_out[None, ...], (B, *baseline_out.shape))
    if pattern == "trailing":
        # baseline_out is (L, A). Patterned output is (L, M, A). Want every
        # patterned_out[:, m, :] == baseline_out.
        L = baseline_out.shape[0]
        return np.broadcast_to(baseline_out[:, None, :], (L, M, A))
    if pattern == "combined":
        L = baseline_out.shape[0]
        return np.broadcast_to(baseline_out[None, :, None, :], (B, L, M, A))
    raise ValueError(pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_ohe_dense_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.DNA.ohe(baseline)  # (L, A)

    patterned, _ = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.DNA.ohe(patterned)

    expected = _broadcast_ohe_expected(baseline_out, patterned.shape, pattern)
    np.testing.assert_array_equal(patterned_out, expected)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_ohe_ragged_shape_matrix(pattern):
    seqs_list = ["ACGT", "AC", "GTTTT"]
    rag_baseline = make_ragged_baseline(seqs_list)
    baseline_out = sp.DNA.ohe(rag_baseline)

    rag_patterned = make_ragged_patterned(rag_baseline, pattern)
    patterned_out = sp.DNA.ohe(rag_patterned)

    assert_broadcast_equal_ragged(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# decode_ohe (dense + ragged)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_decode_ohe_dense_shape_matrix(pattern):
    # Baseline OHE is (L, A); ohe_axis is the key axis = -1 (alphabet)
    # We treat the LENGTH axis as the patterning axis (axis 0 of baseline).
    baseline_ohe = sp.DNA.ohe(np.frombuffer(b"ACGTAC", dtype="S1"))
    baseline_out = sp.DNA.decode_ohe(baseline_ohe, ohe_axis=-1)  # (L,)

    # Pattern the baseline OHE, treating its axis 0 (length) as the key axis
    patterned, new_la = add_leading_trailing_dense(baseline_ohe, pattern, length_axis=0)
    # The ohe_axis position is unchanged at -1 because we insert leading
    # dims at the front and trailing dims at the END (after the alphabet
    # axis). That means ohe_axis is at -1 for baseline/leading patterns and
    # at -2 for trailing/combined patterns.
    if pattern in ("trailing", "combined"):
        ohe_axis = -2
    else:
        ohe_axis = -1
    patterned_out = sp.DNA.decode_ohe(patterned, ohe_axis=ohe_axis)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_decode_ohe_ragged_shape_matrix(pattern):
    seqs_list = ["ACGT", "AC", "GTTTT"]
    rag_bytes = make_ragged_baseline(seqs_list)
    rag_baseline_ohe = sp.DNA.ohe(rag_bytes)
    baseline_out = sp.DNA.decode_ohe(rag_baseline_ohe, ohe_axis=-1)

    rag_patterned = make_ragged_patterned(rag_baseline_ohe, pattern)
    patterned_out = sp.DNA.decode_ohe(rag_patterned, ohe_axis=-1)

    assert_broadcast_equal_ragged(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# tokenize / decode_tokens (dense + ragged) — element-wise, no key axis
# -----------------------------------------------------------------------------

TOKEN_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_tokenize_dense_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.tokenize(baseline, TOKEN_MAP, unknown_token=-1)

    patterned, _ = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.tokenize(patterned, TOKEN_MAP, unknown_token=-1)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_tokenize_ragged_shape_matrix(pattern):
    seqs_list = ["ACGT", "AC", "GTTTT"]
    rag_baseline = make_ragged_baseline(seqs_list)
    baseline_out = sp.tokenize(rag_baseline, TOKEN_MAP, unknown_token=-1)

    rag_patterned = make_ragged_patterned(rag_baseline, pattern)
    patterned_out = sp.tokenize(rag_patterned, TOKEN_MAP, unknown_token=-1)

    assert_broadcast_equal_ragged(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_decode_tokens_dense_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_tokens = sp.tokenize(baseline, TOKEN_MAP, unknown_token=-1)
    baseline_out = sp.decode_tokens(baseline_tokens, TOKEN_MAP)

    patterned, _ = add_leading_trailing_dense(baseline_tokens, pattern, length_axis=-1)
    patterned_out = sp.decode_tokens(patterned, TOKEN_MAP)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_decode_tokens_ragged_shape_matrix(pattern):
    seqs_list = ["ACGT", "AC", "GTTTT"]
    rag_bytes = make_ragged_baseline(seqs_list)
    rag_baseline_tokens = sp.tokenize(rag_bytes, TOKEN_MAP, unknown_token=-1)
    baseline_out = sp.decode_tokens(rag_baseline_tokens, TOKEN_MAP)

    rag_patterned = make_ragged_patterned(rag_baseline_tokens, pattern)
    patterned_out = sp.decode_tokens(rag_patterned, TOKEN_MAP)

    assert_broadcast_equal_ragged(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# reverse_complement (dense bytes + OHE)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_reverse_complement_bytes_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.reverse_complement(baseline, sp.DNA, length_axis=-1)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.reverse_complement(patterned, sp.DNA, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_reverse_complement_ohe_shape_matrix(pattern):
    # OHE baseline (L, A). Length axis = 0, ohe_axis = -1.
    baseline_ohe = sp.DNA.ohe(np.frombuffer(b"ACGTAC", dtype="S1"))
    baseline_out = sp.reverse_complement(
        baseline_ohe, sp.DNA, length_axis=0, ohe_axis=-1
    )

    patterned, new_la = add_leading_trailing_dense(baseline_ohe, pattern, length_axis=0)
    if pattern in ("trailing", "combined"):
        ohe_axis = -2
    else:
        ohe_axis = -1
    patterned_out = sp.reverse_complement(
        patterned, sp.DNA, length_axis=new_la, ohe_axis=ohe_axis
    )

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# k_shuffle (dense bytes only)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_k_shuffle_bytes_shape_matrix(pattern):
    # k_shuffle is stochastic — use a fixed seed so the same shuffle is
    # applied to every leading/trailing slice. The "applies independently"
    # property here is really "applies the SAME shuffle deterministically
    # to each slice when seeded identically."
    baseline = np.frombuffer(b"ACGTACGTAC", dtype="S1")  # length 10
    baseline_out = sp.k_shuffle(baseline, k=2, alphabet=sp.DNA, length_axis=-1, seed=0)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.k_shuffle(
        patterned, k=2, alphabet=sp.DNA, length_axis=new_la, seed=0
    )

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# bin_coverage (dense numeric)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_bin_coverage_shape_matrix(pattern):
    # baseline (L,) with L=6, bin_width=2 -> output (3,)
    baseline = np.arange(6, dtype=np.float64)
    baseline_out = sp.bin_coverage(baseline, bin_width=2, length_axis=-1)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.bin_coverage(patterned, bin_width=2, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# jitter (smoke shape test; existing test_jitter exercises multi-dim already)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_jitter_shape_matrix(pattern):
    # baseline (L,) numeric, length_axis=-1, jitter_axes=()
    baseline = np.arange(6, dtype=np.float64)
    max_jitter = 1
    seed = 0

    (baseline_out,) = sp.jitter(
        baseline,
        max_jitter=max_jitter,
        length_axis=-1,
        jitter_axes=(),
        seed=seed,
    )

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    (patterned_out,) = sp.jitter(
        patterned,
        max_jitter=max_jitter,
        length_axis=new_la,
        jitter_axes=(),
        seed=seed,
    )

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# gc_content / nucleotide_content (dense bytes + OHE)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_gc_content_bytes_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.gc_content(baseline, normalize=True, length_axis=-1)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.gc_content(patterned, normalize=True, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_gc_content_ohe_shape_matrix(pattern):
    baseline_ohe = sp.DNA.ohe(np.frombuffer(b"ACGTAC", dtype="S1"))  # (L, A)
    baseline_out = sp.gc_content(
        baseline_ohe,
        normalize=True,
        length_axis=0,
        alphabet=sp.DNA,
        ohe_axis=-1,
    )

    patterned, new_la = add_leading_trailing_dense(baseline_ohe, pattern, length_axis=0)
    if pattern in ("trailing", "combined"):
        ohe_axis = -2
    else:
        ohe_axis = -1
    patterned_out = sp.gc_content(
        patterned,
        normalize=True,
        length_axis=new_la,
        alphabet=sp.DNA,
        ohe_axis=ohe_axis,
    )

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_nucleotide_content_bytes_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.nucleotide_content(
        baseline, normalize=True, length_axis=-1, alphabet=sp.DNA
    )

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.nucleotide_content(
        patterned, normalize=True, length_axis=new_la, alphabet=sp.DNA
    )

    # nucleotide_content output appends an alphabet axis to the end.
    # baseline_out shape: (A,). Patterned (B, A) for leading; (A, M) for
    # trailing? Actually function does (*shape[:la], *shape[la+1:], A).
    # Just compare via the standard assertion — slicing leading/trailing
    # against alphabet axis should match.
    A = baseline_out.shape[-1]
    if pattern == "baseline":
        np.testing.assert_array_equal(patterned_out, baseline_out)
    elif pattern == "leading":
        # patterned_out should be (B, A)
        expected = np.broadcast_to(baseline_out[None, :], (B, A))
        np.testing.assert_array_equal(patterned_out, expected)
    elif pattern == "trailing":
        # patterned input shape was (L, M). After removing length axis -2
        # and appending A: output shape (M, A). Each m slice should equal
        # baseline_out.
        expected = np.broadcast_to(baseline_out[None, :], (M, A))
        np.testing.assert_array_equal(patterned_out, expected)
    elif pattern == "combined":
        # patterned input was (B, L, M). Output (B, M, A).
        expected = np.broadcast_to(baseline_out[None, None, :], (B, M, A))
        np.testing.assert_array_equal(patterned_out, expected)


# -----------------------------------------------------------------------------
# length (currently hardcoded axis -1; should fail loudly for extra dims)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_length_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_out = sp.length(baseline)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.length(patterned, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# translate (dense + ragged)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_translate_dense_shape_matrix(pattern):
    baseline = np.frombuffer(b"ATCGAT", dtype="S1")  # length 6 (2 codons)
    baseline_out = sp.AA.translate(baseline, length_axis=-1)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    patterned_out = sp.AA.translate(patterned, length_axis=new_la)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_translate_ragged_shape_matrix(pattern):
    seqs_list = ["ATCGAT", "GATCAT", "ATCGATCAT"]  # all divisible by 3
    rag_baseline = make_ragged_baseline(seqs_list)
    baseline_out = sp.AA.translate(rag_baseline)

    rag_patterned = make_ragged_patterned(rag_baseline, pattern)
    patterned_out = sp.AA.translate(rag_patterned)

    assert_broadcast_equal_ragged(patterned_out, baseline_out, pattern)


# -----------------------------------------------------------------------------
# Transform classes (smoke tests)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_reverse_complement_transform_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTAC", dtype="S1")
    baseline_t = spt.ReverseComplement("dna", length_axis=-1, ohe_axis=None)
    baseline_out = baseline_t(baseline)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    t = spt.ReverseComplement("dna", length_axis=new_la, ohe_axis=None)
    patterned_out = t(patterned)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_kshuffle_transform_shape_matrix(pattern):
    baseline = np.frombuffer(b"ACGTACGTAC", dtype="S1")
    baseline_t = spt.KShuffle(k=2, length_axis=-1, seed=0)
    baseline_out = baseline_t(baseline)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    t = spt.KShuffle(k=2, length_axis=new_la, seed=0)
    patterned_out = t(patterned)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)


@pytest.mark.parametrize("pattern", SHAPE_PATTERNS)
def test_jitter_transform_shape_matrix(pattern):
    baseline = np.arange(6, dtype=np.float64)
    baseline_t = spt.Jitter(max_jitter=1, length_axis=-1, jitter_axes=(), seed=0)
    baseline_out = baseline_t(baseline)

    patterned, new_la = add_leading_trailing_dense(baseline, pattern, length_axis=-1)
    t = spt.Jitter(max_jitter=1, length_axis=new_la, jitter_axes=(), seed=0)
    patterned_out = t(patterned)

    assert_broadcast_equal_dense(patterned_out, baseline_out, pattern)
