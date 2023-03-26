import numpy as np
np.random.seed(13)


# CLEANERS
def remove_only_N_seqs(seqs):
    return [seq for seq in seqs if not all([x == "N" for x in seq])]  

def sanitize_seq(seq):
    """Capitalizes and removes whitespace for single seq."""
    return seq.strip().upper()

def sanitize_seqs(seqs):
    """Capitalizes and removes whitespace for a set of sequences."""
    return np.array([seq.strip().upper() for seq in seqs])