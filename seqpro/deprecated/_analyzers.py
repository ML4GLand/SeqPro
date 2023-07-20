import numpy as np


# my own
def len_seqs(seqs, ohe=False):
    """Calculate the length of each sequence in a list.
    
    Parameters
    ----------
    seqs : list
        List of sequences.
    ohe : bool, optional
        Whether to calculate the length of one-hot encoded sequences.
        Default is False.
        
    Returns
    -------
    np.array
        Array containing the length of each sequence.
    """
    if ohe:
        return np.array([seq.shape[1] for seq in seqs])
    else:
        return np.array([len(seq) for seq in seqs])

# my own
def gc_content_seq(seq, ohe=False):
    if ohe:
        return np.sum(seq[1:3, :])/seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C"))/len(seq)

# my own
def gc_content_seqs(seqs, ohe=False):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1)/seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])

# my own
def nucleotide_content_seq(seq, ohe=False, normalize=True):
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
def nucleotide_content_seqs(seqs, axis=0, ohe=False, normalize=True):
    if ohe:
        if normalize:
            return np.sum(seqs, axis=axis)/seqs.shape[0]
        else:
            return np.sum(seqs, axis=axis)
    else:
        if normalize:
            return np.array([np.array([seq.count(nuc)/len(seq) for nuc in "ACGT"]) for seq in seqs])
        else:
            return np.array([np.array([seq.count(nuc) for nuc in "ACGT"]) for seq in seqs])

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