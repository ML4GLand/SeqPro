import torch
import numpy as np
from tqdm.auto import tqdm
from typing import List, Union, Optional, Iterable
from _helpers import _token2one_hot, _tokenize, _pad_sequences, _sequencize, _one_hot2token, _one_hot2token, _token2one_hot


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

