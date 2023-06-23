import numpy as np


# my own
def random_base(alphabet=["A", "G", "C", "T"]):
    """Generate a random base from the AGCT alpahbet.
    
    Parameters
    ----------
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).
        
    Returns
    -------
    str
        Randomly chosen base.
    """
    return np.random.choice(alphabet)

# my own
def random_seq(seq_len, alphabet=["A", "G", "C", "T"]):
    """Generate a random sequence of length seq_len.
    
    Parameters
    ----------
    seq_len : int
        Length of sequence to return.
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).
        
    Returns
    -------
    str
        Randomly generated sequence.
    """
    return "".join([random_base(alphabet) for i in range(seq_len)])

# my own
def random_seqs(seq_num, seq_len, alphabet=["A", "G", "C", "T"]):
    """Generate seq_num random sequences of length seq_len.
    
    Parameters
    ----------
    seq_num (int):
        number of sequences to return
    seq_len (int):
        length of sequence to return
    alphabet : list, optional
        List of bases to choose from (default is ["A", "G", "C", "T"]).
        
    Returns
    -------
    numpy array
        Array of randomly generated sequences.
    """
    return np.array([random_seq(seq_len, alphabet) for i in range(seq_num)])
