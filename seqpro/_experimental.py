import torch
import numpy as np
np.random.seed(13)
from tqdm.auto import tqdm


# VOCABS -- for DNA/RNA only for now
DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
COMPLEMENT_DNA = {"A": "T", "C": "G", "G": "C", "T": "A"}
COMPLEMENT_RNA = {"A": "U", "C": "G", "G": "C", "U": "A"}

# HELPERS - mostly for encoding/decoding
def _get_vocab(vocab):
    if vocab == "DNA":
        return DNA
    elif vocab == "RNA":
        return RNA
    else:
        raise ValueError("Invalid vocab, only DNA or RNA are currently supported")

# exact concise
def _get_vocab_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    Used in `_tokenize`.
    """
    return {l: i for i, l in enumerate(vocab)}

# exact concise
def _get_index_dict(vocab):
    """
    Returns a dictionary mapping each token to its index in the vocabulary.
    """
    return {i: l for i, l in enumerate(vocab)}

# modified concise
def _tokenize(seq, vocab="DNA", neutral_vocab=["N"]):
    """
    Convert sequence to integers based on a vocab

    Parameters
    ----------
    seq: 
        sequence to encode
    vocab: 
        vocabulary to use
    neutral_vocab: 
        neutral vocabulary -> assign those values to -1
    
    Returns
    -------
        List of length `len(seq)` with integers from `-1` to `len(vocab) - 1`
    """
    vocab = _get_vocab(vocab)
    if isinstance(neutral_vocab, str):
        neutral_vocab = [neutral_vocab]

    nchar = len(vocab[0])
    for l in vocab + neutral_vocab:
        assert len(l) == nchar
    assert len(seq) % nchar == 0  # since we are using striding

    vocab_dict = _get_vocab_dict(vocab)
    for l in neutral_vocab:
        vocab_dict[l] = -1

    # current performance bottleneck
    return [
        vocab_dict[seq[(i * nchar) : ((i + 1) * nchar)]]
        for i in range(len(seq) // nchar)
    ]

# my own
def _sequencize(tvec, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """
    Converts a token vector into a sequence of symbols of a vocab.
    """
    vocab = _get_vocab(vocab) 
    index_dict = _get_index_dict(vocab)
    index_dict[neutral_value] = neutral_char
    return "".join([index_dict[i] for i in tvec])

# modified concise
def _token2one_hot(tvec, vocab="DNA", fill_value=None):
    """
    Converts an L-vector of integers in the range [0, D] into an L x D one-hot
    encoding. If fill_value is not None, then the one-hot encoding is filled
    with this value instead of 0.

    Parameters
    ----------
    tvec : np.array
        L-vector of integers in the range [0, D]
    vocab_size : int
        D
    fill_value : float, optional
        Value to fill the one-hot encoding with. If None, then the one-hot
    """
    vocab = _get_vocab(vocab)
    vocab_size = len(vocab)
    arr = np.zeros((vocab_size, len(tvec)))
    tvec_range = np.arange(len(tvec))
    tvec = np.asarray(tvec)
    arr[tvec[tvec >= 0], tvec_range[tvec >= 0]] = 1
    if fill_value is not None:
        arr[:, tvec_range[tvec < 0]] = fill_value
    return arr.astype(np.int8) if fill_value is None else arr.astype(np.float16)

# modified dinuc_shuffle
def _one_hot2token(one_hot, neutral_value=-1, consensus=False):
    """
    Converts a one-hot encoding into a vector of integers in the range [0, D]
    where D is the number of classes in the one-hot encoding.

    Parameters
    ----------
    one_hot : np.array
        L x D one-hot encoding
    neutral_value : int, optional
        Value to use for neutral values.
    
    Returns
    -------
    np.array
        L-vector of integers in the range [0, D]
    """
    if consensus:
        return np.argmax(one_hot, axis=0)
    tokens = np.tile(neutral_value, one_hot.shape[1])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot.transpose()==1)
    tokens[seq_inds] = dim_inds
    return tokens

# pad and subset, exact concise
def _pad(seq, max_seq_len, value="N", align="end"):
    seq_len = len(seq)
    assert max_seq_len >= seq_len
    if align == "end":
        n_left = max_seq_len - seq_len
        n_right = 0
    elif align == "start":
        n_right = max_seq_len - seq_len
        n_left = 0
    elif align == "center":
        n_left = (max_seq_len - seq_len) // 2 + (max_seq_len - seq_len) % 2
        n_right = (max_seq_len - seq_len) // 2
    else:
        raise ValueError("align can be of: end, start or center")

    # normalize for the length
    n_left = n_left // len(value)
    n_right = n_right // len(value)

    return value * n_left + seq + value * n_right

# exact concise
def _trim(seq, maxlen, align="end"):
    seq_len = len(seq)

    assert maxlen <= seq_len
    if align == "end":
        return seq[-maxlen:]
    elif align == "start":
        return seq[0:maxlen]
    elif align == "center":
        dl = seq_len - maxlen
        n_left = dl // 2 + dl % 2
        n_right = seq_len - dl // 2
        return seq[n_left:n_right]
    else:
        raise ValueError("align can be of: end, start or center")

# modified concise
def _pad_sequences(
    seqs, 
    maxlen=None, 
    align="end", 
    value="N"
):
    """
    Pads sequences to the same length.

    Parameters
    ----------
    seqs : list of str
        Sequences to pad
    maxlen : int, optional
        Length to pad to. If None, then pad to the length of the longest sequence.
    align : str, optional
        Alignment of the sequences. One of "start", "end", "center"
    value : str, optional
        Value to pad with

    Returns
    -------
    np.array
        Array of padded sequences
    """

    # neutral element type checking
    assert isinstance(value, list) or isinstance(value, str)
    assert isinstance(value, type(seqs[0])) or type(seqs[0]) is np.str_
    assert not isinstance(seqs, str)
    assert isinstance(seqs[0], list) or isinstance(seqs[0], str)

    max_seq_len = max([len(seq) for seq in seqs])

    if maxlen is None:
        maxlen = max_seq_len
    else:
        maxlen = int(maxlen)

    if max_seq_len < maxlen:
        import warnings
        warnings.warn(
            f"Maximum sequence length ({max_seq_len}) is smaller than maxlen ({maxlen})."
        )
        max_seq_len = maxlen

    # check the case when len > 1
    for seq in seqs:
        if not len(seq) % len(value) == 0:
            raise ValueError("All sequences need to be dividable by len(value)")
    if not maxlen % len(value) == 0:
        raise ValueError("maxlen needs to be dividable by len(value)")

    padded_seqs = [
        _trim(_pad(seq, max(max_seq_len, maxlen), value=value, align=align), maxlen, align=align)
        for seq in seqs 
    ]
    return padded_seqs

# HELPERS -- misc
def _is_overlapping(a, b):
    """Returns True if two intervals overlap"""
    if b[0] >= a[0] and b[0] <= a[1]:
        return True
    else:
        return False

def _merge_intervals(intervals):
    """Merges a list of overlapping intervals"""
    if len(intervals) == 0:
        return None
    merged_list = []
    merged_list.append(intervals[0])
    for i in range(1, len(intervals)):
        pop_element = merged_list.pop()
        if _is_overlapping(pop_element, intervals[i]):
            new_element = pop_element[0], max(pop_element[1], intervals[i][1])
            merged_list.append(new_element)
        else:
            merged_list.append(pop_element)
            merged_list.append(intervals[i])
    return merged_list

def _hamming_distance(string1, string2):
    """Find hamming distance between two strings. Returns inf if they are different lengths"""
    distance = 0
    L = len(string1)
    if L != len(string2):
        return np.inf
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance

def _collapse_pos(positions):
    """Collapse neighbor positions of array to ranges"""
    ranges = []
    start = positions[0]
    for i in range(1, len(positions)):
        if positions[i - 1] == positions[i] - 1:
            continue
        else:
            ranges.append((start, positions[i - 1] + 2))
            start = positions[i]
    ranges.append((start, positions[-1] + 2))
    return ranges

# HELPERS -- next 4 are from dinuc shuffle in DeepLift package
def _string_to_char_array(seq):
    """
    Converts an ASCII string to a NumPy array of byte-long ASCII codes.
    e.g. "ACGT" becomes [65, 67, 71, 84].
    """
    return np.frombuffer(bytearray(seq, "utf8"), dtype=np.int8)

def _char_array_to_string(arr):
    """
    Converts a NumPy array of byte-long ASCII codes into an ASCII string.
    e.g. [65, 67, 71, 84] becomes "ACGT".
    """
    return arr.tostring().decode("ascii")

def _one_hot_to_tokens(one_hot):
    """
    Converts an L x D one-hot encoding into an L-vector of integers in the range
    [0, D], where the token D is used when the one-hot encoding is all 0. This
    assumes that the one-hot encoding is well-formed, with at most one 1 in each
    column (and 0s elsewhere).
    """
    tokens = np.tile(one_hot.shape[1], one_hot.shape[0])  # Vector of all D
    seq_inds, dim_inds = np.where(one_hot)
    tokens[seq_inds] = dim_inds
    return tokens

def _tokens_to_one_hot(tokens, one_hot_dim):
    """
    Converts an L-vector of integers in the range [0, D] to an L x D one-hot
    encoding. The value `D` must be provided as `one_hot_dim`. A token of D
    means the one-hot encoding is all 0s.
    """
    identity = np.identity(one_hot_dim + 1)[:, :-1]  # Last row is all 0s
    return identity[tokens]

# CLEANERS
def remove_only_N_seqs(seqs):
    return [seq for seq in seqs if not all([x == "N" for x in seq])]  

def sanitize_seq(seq):
    """Capitalizes and removes whitespace for single seq."""
    return seq.strip().upper()

def sanitize_seqs(seqs):
    """Capitalizes and removes whitespace for a set of sequences."""
    return np.array([seq.strip().upper() for seq in seqs])

# ENCODERS and DECODERS
def ascii_encode_seq(seq, pad=0):
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

def ascii_encode_seqs(seqs, pad=0):
    """
    Converts a set of sequences to a NumPy array of byte-long ASCII codes.
    """
    encode_seqs = np.array(
        [ascii_encode_seq(seq, pad=pad) for seq in seqs], dtype=int
    )
    return encode_seqs

def ascii_decode_seq(seq):
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")

def ascii_decode_seqs(seqs):
    """Convert a set of one-hot encoded arrays back to strings"""
    return np.array([ascii_decode_seq(seq) for seq in seqs], dtype=object)

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

def ohe_seq(
    seq, 
    vocab="DNA", 
    neutral_vocab="N", 
    fill_value=0
):
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value)

def ohe_seqs(
    seqs,
    vocab="DNA",
    neutral_vocab="N",
    maxlen=None,
    pad=True,
    pad_value="N",
    fill_value=None,
    seq_align="start",
    verbose=True,
):
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

def decode_seq(arr, vocab="DNA", neutral_value=-1, neutral_char="N"):
    """Convert a single one-hot encoded array back to string"""
    if isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    return _sequencize(
        tvec=_one_hot2token(arr, neutral_value),
        vocab=vocab,
        neutral_value=neutral_value,
        neutral_char=neutral_char,
    )

def decode_seqs(arr, vocab="DNA", neutral_char="N", neutral_value=-1, verbose=True):
    """Convert a one-hot encoded array back to set of sequences"""
    arr_list = [
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

# MODIFIERS -- shuffling
def shuffle_seq(seq,  one_hot=False, seed=None):
    np.random.seed(seed)
    
    if one_hot:
        seq = np.argmax(seq, axis=-1)
        
    shuffled_idx = np.random.permutation(len(seq))
    shuffled_seq = np.array([seq[i] for i in shuffled_idx], dtype=seq.dtype)
    
    if one_hot:
        shuffled_seq = np.eye(4)[shuffled_seq]
        
    return shuffled_seq

def shuffle_seqs(seqs, one_hot=False, seed=None):
    return np.array([shuffle_seq(seq, one_hot=one_hot, seed=seed) for seq in seqs])

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

# ANALYZERS -- for querying sequences
def gc_content_seq(seq, ohe=True):
    if ohe:
        return np.sum(seq[1:3, :])/seq.shape[1]
    else:
        return (seq.count("G") + seq.count("C"))/len(seq)
    
def gc_content_seqs(seqs, ohe=True):
    if ohe:
        seq_len = seqs[0].shape[1]
        return np.sum(seqs[:, 1:3, :], axis=1).sum(axis=1)/seq_len
    else:
        return np.array([gc_content_seq(seq) for seq in seqs])

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
            
def nucleotide_content_seqs(seqs, axis=0, ohe=True, normalize=True):
    if ohe:
        if normalize:
            return np.sum(seqs, axis=axis)/seqs.shape[0]
        else:
            return np.sum(seqs, axis=axis)
    else:
        print("Not implemented yet")

from .._settings import settings
from seqdata import SeqData
from ..models.base import BaseModel
import eugene.preprocess as pp
import torch
import numpy as np

def count_kmers(sequence : str, k : int, data = None) -> dict:
    """
    Counts k-mers in a given sequence.

    Parameters
    ----------
    sequence : str
        Nucleotide sequence expressed as a string.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).

    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    assert len(sequence) >= k, "Length of sequence must be greater than that of k."
    data = {} if data is None else data
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i: i + k]
        try:
            data[kmer] += 1
        except KeyError:
            data[kmer] = 1
    return data

def count_kmers_sdata(sdata : SeqData, k : int, frequency : bool = False) -> dict:
    """
    Counts k-mers in a given sequence from a SeqData object.

    Parameters
    ----------
    sdata : SeqData
        SeqData object containing sequences.
    k : int
        k value for k-mers (e.g. k=3 generates 3-mers).
    frequency : bool
        Whether to return relative k-mer frequency in place of count.
        Default is False.

    Returns
    -------
    kmers : dict
        k-mers and their counts expressed in a dictionary.
    """
    data = {}
    for seq in sdata.seqs:
        data = count_kmers(seq, k, data)
    if frequency:
        total = sum(data.values())
        for kmer in data:
            data[kmer] = data[kmer] / total
    return data

def _find_distance(seq1, seq2) -> int:
    edits = 0
    for i in range(len(seq1)):
        if seq1[i] != seq2[i]:
            edits += 1
    return edits

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

def edit_distance_sdata(sdata1 : SeqData, sdata2 : SeqData, dual : bool = False, average : bool = False) -> list:
    """
    Calculates the nucleotide edit distance between pairs of sequences from a SeqData object.

    Parameters
    ----------
    sdata1 : SeqData
        First SeqData object containing sequences.
    sdata2 : SeqData
        Second SeqData object containing sequences.
    dual : bool
        Whether to calculate the forwards and backwards edit distance, and return the lesser.
        Defaults to False.
    average : bool
        Whether to average all edit distances and return in place of a list.
        Defaults to False.

    Returns
    -------
    edits : list
        List containing itemized amounts of edits between sequences.
    """
    assert len(sdata1.seqs) == len(sdata2.seqs), "Both SeqData objects must be of the same length."
    distances = []
    for i in range(len(sdata1.seqs)):
        distances.append(edit_distance(sdata1.seqs[i], sdata2.seqs[i], dual))
    if average:
        return sum(distances) / len(distances)
    return distances

def latent_interpolation(latent_dim : int, samples : int, num_seqs : int = 1, model : BaseModel = None, normal : bool = False, inclusive : bool = True) -> list:
    """
    Linearly interpolates between two random latent points. Useful for visualizing generative models.

    Parameters
    ----------
    latent_dim : int
        Latent dimension of random latent space points.
    samples : int
        Number of samples to make between the two latent points. Higher numbers represent more sequences and should show smoother results.
    num_seqs : int
        Number of sequence channels to interpolate.
        Defaults to 1.
    model : BaseModel
        If provided, interpolated values will be passed through the given model and be returned in place of latent points.
        Default is None.
    normal : bool
        Whether to randomly sample points from a standard normal distibution rather than from 0 to 1.
        Defaults to False.
    inclusive : bool
        Whether to returm random latent points along with their interpolated samples.
        Defaults to True.

    Returns
    -------
    z_list : list
        List of latent space points represented as tensors.
    gen_seqs : list
        List of tokenized sequences expressed as strings.
        Returns in place of z_list when a model is provided.
    """
    if normal:
        z1 = torch.normal(0, 1, (num_seqs, latent_dim))
        z2 = torch.normal(0, 1, (num_seqs, latent_dim))
    else:
        z1 = torch.rand(num_seqs, latent_dim)
        z2 = torch.rand(num_seqs, latent_dim)

    z_list = []
    for n in range(samples):
        weight = (n + 1)/(samples + 1)
        z_interp = torch.lerp(z1, z2, weight)
        z_list.append(z_interp)
    if inclusive:
        z_list.insert(0, z1)
        z_list.append(z2)

    if model is None:
        return z_list

    gen_seqs = []
    for z in z_list:
        gen_seq = seqs_from_tensor(model(z))
        gen_seqs.append(gen_seq)
    return gen_seqs

def seqs_from_tensor(tensor : torch.tensor, num_seqs : int = 1) -> np.ndarray:
    """
    Decodes sequences represented by tensors into their string values.

    Parameters
    ----------
    tensor : torch.tensor
        Tensor to be decoded.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    tokens = np.argmax(tensor.detach().numpy(), axis=1).reshape(num_seqs, -1)
    seqs = np.array([pp.decode_seq(pp._utils._token2one_hot(token)) for token in tokens])
    return seqs

def generate_seqs_from_model(model : BaseModel, num_seqs : int = 1, normal : bool = False, device : str = "cpu"):
    """
    Generates random sequences from a generative model.

    Parameters
    ----------
    model : BaseModel
        Generative model used for sequence generation.
    num_seqs : int
        Number of sequences to decode.
        Default is 1.
    normal : bool
        Whether to sample from a normal distribution instead of a uniform distribution.
        Default is false.

    Returns
    -------
    seqs : np.ndarray
        Numpy array of decoded sequences.
    """
    if normal:
        z = torch.Tensor(np.random.normal(0, 1, (num_seqs, model.latent_dim)))
    else: 
        z = torch.rand(num_seqs, model.latent_dim)
    z = z.to(device)
    fake = model(z)
    return seqs_from_tensor(fake.cpu(), num_seqs)
