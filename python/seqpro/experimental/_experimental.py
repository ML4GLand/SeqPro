import numpy as np


# helper
def _find_distance(seq1, seq2) -> int:
    edits = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            edits += 1
    return edits


# analyzers?
def edit_distance(seq1: str, seq2: str, dual: bool = False) -> int:
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


# protein sequence handling
# vocabularies:
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
AMINO_ACIDS = [
    "A",
    "R",
    "N",
    "D",
    "B",
    "C",
    "E",
    "Q",
    "Z",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
CODONS = [
    "AAA",
    "AAC",
    "AAG",
    "AAT",
    "ACA",
    "ACC",
    "ACG",
    "ACT",
    "AGA",
    "AGC",
    "AGG",
    "AGT",
    "ATA",
    "ATC",
    "ATG",
    "ATT",
    "CAA",
    "CAC",
    "CAG",
    "CAT",
    "CCA",
    "CCC",
    "CCG",
    "CCT",
    "CGA",
    "CGC",
    "CGG",
    "CGT",
    "CTA",
    "CTC",
    "CTG",
    "CTT",
    "GAA",
    "GAC",
    "GAG",
    "GAT",
    "GCA",
    "GCC",
    "GCG",
    "GCT",
    "GGA",
    "GGC",
    "GGG",
    "GGT",
    "GTA",
    "GTC",
    "GTG",
    "GTT",
    "TAC",
    "TAT",
    "TCA",
    "TCC",
    "TCG",
    "TCT",
    "TGC",
    "TGG",
    "TGT",
    "TTA",
    "TTC",
    "TTG",
    "TTT",
]
STOP_CODONS = ["TAG", "TAA", "TGA"]

# want to one-hot encode AA sequence and codons
# https://github.com/gagneurlab/concise/blob/master/concise/preprocessing/sequence.py

# Performing motif anal


def kmer2seq(kmers):
    """
    Convert kmers to original sequence

    Arguments:
    kmers -- str, kmers separated by space.

    Returns:
    seq -- str, original sequence.

    """
    kmers_list = kmers.split(" ")
    bases = [kmer[0] for kmer in kmers_list[0:-1]]
    bases.append(kmers_list[-1])
    seq = "".join(bases)
    assert len(seq) == len(kmers_list) + len(kmers_list[0]) - 1
    return seq


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x : x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers
