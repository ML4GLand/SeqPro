import numpy as np
np.random.seed(13)

# my own
def gc_content_seq(seq, ohe=True):
    if ohe:
        return np.sum(seq[1:3, :])/seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C"))/len(seq)

# my own
def gc_content_seqs(seqs, ohe=True):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1)/seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])

# my own
def nucleotide_content_seq(seq, ohe=True, normalize=True):
    if ohe:
        if normalize:
            return np.sum(seq, axis=1)/seq.shape[1]
        else:
            return np.sum(seq, axis=1)
    else:
        if normalize:
            return np.array([seq.count(nuc)/len(seq) for nuc in "ACGT"])
        else:
            return np.array([seq.count(nuc) for nuc in "ACGT"])
            
# my own
def nucleotide_content_seqs(seqs, axis=0, ohe=True, normalize=True):
    if ohe:
        if normalize:
            return np.sum(seqs, axis=axis)/seqs.shape[0]
        else:
            return np.sum(seqs, axis=axis)
    else:
        print("Not implemented yet")

# haydens
def count_kmers_seq(seq : str, k : int, data = None) -> dict:
    """
    Counts k-mers in a given seq.
    Parameters
    ----------
    seq : str
        Nucleotide seq expressed as a string.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).
    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    assert len(seq) >= k, "Length of seq must be greater than that of k."
    data = {} if data is None else data
    for i in range(len(seq) - k + 1):
        kmer = seq[i: i + k]
        try:
            data[kmer] += 1
        except KeyError:
            data[kmer] = 1
    return data
