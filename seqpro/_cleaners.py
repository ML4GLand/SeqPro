import numpy as np


def remove_N_seqs(seqs):
    """Removes sequences containing 'N' from a list of sequences.

    Parameters
    ----------
    seqs : list
        List of sequences to be filtered.

    Returns
    -------
    list
        List of sequences without 'N'.
    """
    return [seq for seq in seqs if "N" not in seq]


def remove_only_N_seqs(seqs):
    """Removes sequences consisting only of 'N' from a list of sequences.

    Parameters
    ----------
    seqs : list
        List of sequences to be filtered.

    Returns
    -------
    list
        List of sequences without only 'N'.
    """
    return [seq for seq in seqs if not all([x == "N" for x in seq])]


def sanitize_seq(seq):
    """Capitalizes and removes whitespace for single seq.

    Parameters
    ----------
    seq : str
        Sequence to be sanitized.

    Returns
    -------
    str
        Sanitized sequence.
    """
    return seq.strip().upper()


def sanitize_seqs(seqs):
    """Capitalizes and removes whitespace for a set of sequences.

    Parameters
    ----------
    seqs : list
        List of sequences to be sanitized.

    Returns
    -------
    numpy.ndarray
        Array of sanitized sequences.
    """
    return np.array([seq.strip().upper() for seq in seqs])
