from collections.abc import Sequence

import numpy as np

from ._utils import StrSeqType, cast_seqs, check_axes


def remove_N_seqs(seqs: Sequence[str]):
    """Removes sequences containing 'N' from a list of sequences.

    Parameters
    ----------
    seqs
        List of sequences to be filtered.

    Returns
    -------
    list[str]
        Sequences with no 'N' characters.
    """
    return [seq for seq in seqs if "N" not in seq]


def remove_only_N_seqs(seqs: Sequence[str]):
    """Removes sequences consisting only of 'N' from a list of sequences.

    Parameters
    ----------
    seqs
        List of sequences to be filtered.

    Returns
    -------
    list[str]
        Sequences that are not composed entirely of 'N' characters.
    """
    return [seq for seq in seqs if not all(x == "N" for x in seq)]


def remove_whitespace(seqs: StrSeqType):
    pass


def sanitize(seqs: StrSeqType, length_axis: int | None = None):
    """Capitalize characters, remove whitespace, and coerce to fixed length.

    Parameters
    ----------
    seqs
        List of sequences to be sanitized.

    Returns
    -------
    NDArray[np.bytes_]
        S1 byte array of sanitized sequences.
    """
    check_axes(seqs, length_axis, False)

    seqs = cast_seqs(seqs)

    (seqs != b"").sum(-1).max()
    seqs = np.char.strip(seqs.view(f"S{seqs.shape[-1]}"))
    seqs = np.char.upper(seqs.view("S1"))

    return seqs
