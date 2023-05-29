import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union, Optional, Iterable
import matplotlib.pyplot as plt
from ._helpers import COMPLEMENT_DNA, COMPLEMENT_RNA
from ._helpers import _token2one_hot, _tokenize, _pad_sequences, _sequencize, _one_hot2token, _string_to_char_array, _one_hot2token, _char_array_to_string, _token2one_hot


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

# my own
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

# my own
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

# my own
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

# my own
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

# my own
def ascii_encode_seq(seq: str, pad: int = 0) -> np.array:
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.

    Parameters
    ----------
    seq : str
        Sequence to encode.
    pad : int
        Amount of padding to add to the end of the sequence. Defaults to 0.

    Returns
    -------
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

# my own
def ascii_encode_seqs(seqs: List[str], pad: int = 0) -> np.ndarray:
    """
    Converts a set of sequences to a NumPy array of byte-long ASCII codes.

    Parameters
    ----------
    seqs : List[str]
        Sequences to encode.
    pad : int
        Amount of padding to add to the end of the sequences. Defaults to 0.
    
    Returns
    -------
    np.ndarray
        Array of encoded sequences.
    """
    encode_seqs = np.array([ascii_encode_seq(seq, pad=pad) for seq in seqs], dtype=int)
    return encode_seqs

# my own
def ascii_decode_seq(seq: np.array) -> str:
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")

# my own
def ascii_decode_seqs(seqs: np.ndarray) -> np.array:
    """Convert a set of one-hot encoded arrays back to strings"""
    return np.array([ascii_decode_seq(seq) for seq in seqs], dtype=object)

# modifed concise
def ohe_seq(
    seq: str, 
    vocab: str = "DNA", 
    neutral_vocab: str = "N", 
    fill_value: int = 0
) -> np.array:
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value)

# modfied concise
def ohe_seqs(
    seqs: Iterable[str],
    vocab: str = "DNA",
    neutral_vocab: Union[str, List[str]] = "N",
    maxlen: Optional[int] = None,
    pad: bool = True,
    pad_value: str = "N",
    fill_value: Optional[str] = None,
    seq_align: str = "start",
    verbose: bool = True,
) -> np.ndarray:
    """Convert a set of sequences into one-hot-encoded array."""
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]
    if isinstance(seqs, str):
        raise ValueError("seq_vec should be an iterable not a string itself")
    assert len(vocab[0]) == len(pad_value)
    assert pad_value in neutral_vocab
    if pad:
        seqs_vec = _pad_sequences(seqs, maxlen=maxlen, align=seq_align, value=pad_value)
    arr_list = [
        ohe_seq(
            seq=seqs_vec[i],
            vocab=vocab,
            neutral_vocab=neutral_vocab,
            fill_value=fill_value,
        )
        for i in tqdm(
            range(len(seqs_vec)),
            total=len(seqs_vec),
            desc="One-hot encoding sequences",
            disable=not verbose,
        )
    ]
    if pad:
        return np.stack(arr_list)
    else:
        return np.array(arr_list, dtype=object)

# my own
def decode_seq(
    arr: np.ndarray,
    vocab: str = "DNA",
    neutral_value: int = -1,
    neutral_char: str = "N"
) -> str:
    """Convert a single one-hot encoded array back to string"""
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    return _sequencize(
        tvec=_one_hot2token(arr, neutral_value),
        vocab=vocab,
        neutral_value=neutral_value,
        neutral_char=neutral_char,
    )

# my own
def decode_seqs(
    arr: np.ndarray,
    vocab: str = "DNA",
    neutral_char: str = "N",
    neutral_value: int = -1,
    verbose: bool = True
) -> np.ndarray:
    """Convert a one-hot encoded array back to set of sequences"""
    arr_list: List[np.ndarray] = [
        decode_seq(
            arr=arr[i],
            vocab=vocab,
            neutral_value=neutral_value,
            neutral_char=neutral_char,
        )
        for i in tqdm(
            range(len(arr)),
            total=len(arr),
            desc="Decoding sequences",
            disable=not verbose,
        )
    ]
    return np.array(arr_list)

# my own
def reverse_complement_seq(seq, vocab="DNA"):
    """Reverse complement a single sequence."""
    if isinstance(seq, str):
        if vocab == "DNA":
            return "".join(COMPLEMENT_DNA.get(base, base) for base in reversed(seq))
        elif vocab == "RNA":
            return "".join(COMPLEMENT_RNA.get(base, base) for base in reversed(seq))
        else:
            raise ValueError("Invalid vocab, only DNA or RNA are currently supported")
    elif isinstance(seq, np.ndarray):
        return torch.from_numpy(np.flip(seq, axis=(0, 1)).copy()).numpy()

# my own
def reverse_complement_seqs(seqs, vocab="DNA", verbose=True):
    """Reverse complement set of sequences."""
    if isinstance(seqs[0], str):
        return np.array(
            [
                reverse_complement_seq(seq, vocab)
                for i, seq in tqdm(
                    enumerate(seqs),
                    total=len(seqs),
                    desc="Reverse complementing sequences",
                    disable=not verbose,
                )
            ]
        )
    elif isinstance(seqs[0], np.ndarray):
        return torch.from_numpy(np.flip(seqs, axis=(1, 2)).copy()).numpy()
    
# my own
def shuffle_seq(seq,  one_hot=False, seed=None):
    np.random.seed(seed)
    
    if one_hot:
        seq = np.argmax(seq, axis=-1)
        
    shuffled_idx = np.random.permutation(len(seq))
    shuffled_seq = np.array([seq[i] for i in shuffled_idx], dtype=seq.dtype)
    
    if one_hot:
        shuffled_seq = np.eye(4)[shuffled_seq]
        return shuffled_seq
    
    else:
        return np.array("".join(shuffled_seq))

# my own
def shuffle_seqs(seqs, one_hot=False, seed=None):
    return np.array([shuffle_seq(seq, one_hot=one_hot, seed=seed) for seq in seqs])

# modified dinuc_shuffle
def dinuc_shuffle_seq(
    seq, 
    num_shufs=None, 
    rng=None
):
    """
    Creates shuffles of the given sequence, in which dinucleotide frequencies
    are preserved.
    If `seq` is a string, returns a list of N strings of length L, each one
    being a shuffled version of `seq`. If `seq` is a 2D np array, then the
    result is an N x L x D np array of shuffled versions of `seq`, also
    one-hot encoded. If `num_shufs` is not specified, then the first dimension
    of N will not be present (i.e. a single string will be returned, or an L x D
    array).
    Parameters
    ----------
    seq : str
        The sequence to shuffle.
    num_shufs : int, optional
        The number of shuffles to create. If None, only one shuffle is created.
    rng : np.random.RandomState, optional
        The random number generator to use. If None, a new one is created.
    Returns
    -------
    list of str or np.array
        The shuffled sequences.
    Note
    ----
    This function comes from DeepLIFT's dinuc_shuffle.py.
    """
    if type(seq) is str or type(seq) is np.str_:
        arr = _string_to_char_array(seq)
    elif type(seq) is np.ndarray and len(seq.shape) == 2:
        seq_len, one_hot_dim = seq.shape
        arr = _one_hot2token(seq)
    else:
        raise ValueError("Expected string or one-hot encoded array")
    if not rng:
        rng = np.random.RandomState(rng)

    # Get the set of all characters, and a mapping of which positions have which
    # characters; use `tokens`, which are integer representations of the
    # original characters
    chars, tokens = np.unique(arr, return_inverse=True)

    # For each token, get a list of indices of all the tokens that come after it
    shuf_next_inds = []
    for t in range(len(chars)):
        mask = tokens[:-1] == t  # Excluding last char
        inds = np.where(mask)[0]
        shuf_next_inds.append(inds + 1)  # Add 1 for next token

    if type(seq) is str or type(seq) is np.str_:
        all_results = []
    else:
        all_results = np.empty(
            (num_shufs if num_shufs else 1, seq_len, one_hot_dim), dtype=seq.dtype
        )

    for i in range(num_shufs if num_shufs else 1):
        # Shuffle the next indices
        for t in range(len(chars)):
            inds = np.arange(len(shuf_next_inds[t]))
            inds[:-1] = rng.permutation(len(inds) - 1)  # Keep last index same
            shuf_next_inds[t] = shuf_next_inds[t][inds]

        counters = [0] * len(chars)

        # Build the resulting array
        ind = 0
        result = np.empty_like(tokens)
        result[0] = tokens[ind]
        for j in range(1, len(tokens)):
            t = tokens[ind]
            ind = shuf_next_inds[t][counters[t]]
            counters[t] += 1
            result[j] = tokens[ind]

        if type(seq) is str or type(seq) is np.str_:
            all_results.append(_char_array_to_string(chars[result]))
        else:
            all_results[i] = _token2one_hot(chars[result])
    return all_results if num_shufs else all_results[0]

# modified dinuc_shuffle
def dinuc_shuffle_seqs(seqs, num_shufs=None, rng=None):
    """
    Shuffle the sequences in `seqs` in the same way as `dinuc_shuffle_seq`.
    If `num_shufs` is not specified, then the first dimension of N will not be
    present (i.e. a single string will be returned, or an L x D array).
    Parameters
    ----------
    seqs : np.ndarray
        Array of sequences to shuffle
    num_shufs : int, optional
        Number of shuffles to create, by default None
    rng : np.random.RandomState, optional
        Random state to use for shuffling, by default None
    Returns
    -------
    np.ndarray
        Array of shuffled sequences
    Note
    -------
    This is taken from DeepLIFT
    """
    if not rng:
        rng = np.random.RandomState(rng)

    if type(seqs) is str or type(seqs) is np.str_:
        seqs = [seqs]

    all_results = []
    for i in range(len(seqs)):
        all_results.append(dinuc_shuffle_seq(seqs[i], num_shufs=num_shufs, rng=rng))
    return np.array(all_results)

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

# my own
def plot_gc_content(seqs, title="", ax=None, figsize=(10, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    gc_contents = gc_content_seqs(seqs, ohe=False)
    ax.hist(gc_contents, bins=100)
    ax.set_xlabel("GC content")
    ax.set_ylabel("Frequency")

# my own
def plot_nucleotide_content(seqs, title="", ax=None, figsize=(10, 5)):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    nuc_contents = nucleotide_content_seqs(seqs, axis=0, ohe=False, normalize=True)
    ax.plot(nuc_contents.T)
    ax.legend(["A", "C", "G", "T"])
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")