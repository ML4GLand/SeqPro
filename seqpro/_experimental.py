import torch
import numpy as np
np.random.seed(13)
from seqdata import SeqData
from ._encoders import decode_seq
from ._helpers import _token2one_hot


# helper
def _find_distance(seq1, seq2) -> int:
    edits = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            edits += 1
    return edits

# analyzers?
def edit_distance(seq1 : str, seq2 : str, dual : bool = False) -> int:
    """
    Calculates the nucleotide edit distance between two sequences.

    Parameters
    ----------
    seq1 : str
        First nucleotide sequence expressed as a string.
    seq2 : str
        Second ucleotide sequence expressed as a string.
    dual : bool
        Whether to calculate the forwards and backwards edit distance, and return the lesser.
        Defaults to False.

    Returns
    -------
    edits : int
        Amount of edits between sequences.
    """
    assert len(seq1) == len(seq2), "Both sequences must be of same length."
    f_edits = _find_distance(seq1, seq2)
    b_edits = _find_distance(seq1, seq2[::-1]) if dual else len(seq1)
    return min(f_edits, b_edits)
