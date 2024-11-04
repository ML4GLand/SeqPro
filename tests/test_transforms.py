import numpy as np
import seqpro as sp
import seqpro.transforms as spt
from numpy.typing import NDArray
from pytest_cases import fixture


@fixture
def seqs():
    return sp.random_seqs((3, 5), sp.DNA)


def test_rc(seqs: NDArray):
    rc = spt.ReverseComplement("dna", length_axis=-1, ohe_axis=None)
    np.testing.assert_array_equal(sp.reverse_complement(seqs, sp.DNA, -1), rc(seqs))


def test_jitter(seqs: NDArray):
    jitter = spt.Jitter(1, length_axis=-1, jitter_axes=0, seed=0)
    np.testing.assert_array_equal(
        sp.jitter(seqs, max_jitter=1, length_axis=-1, jitter_axes=0, seed=0)[0],
        jitter(seqs),
    )


def test_random(seqs: NDArray):
    seed = 0
    rc = spt.ReverseComplement("dna", length_axis=-1, ohe_axis=None)
    rrc = spt.Random(0.5, rc, seed=seed)

    gen = np.random.default_rng(seed)
    to_rc = gen.random(size=len(seqs) * 100) < 0.5

    seqs = np.tile(seqs, (100, 1))
    rrc_seqs = rrc(seqs)

    np.testing.assert_equal(seqs[to_rc], rrc_seqs[to_rc], "Failed random complements.")
    np.testing.assert_equal(
        seqs[~to_rc], rrc_seqs[~to_rc], "Failed unchanged sequences."
    )
