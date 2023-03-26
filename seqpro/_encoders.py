import torch
import numpy as np
np.random.seed(13)
from tqdm.auto import tqdm
from ._helpers import DNA, RNA, COMPLEMENT_DNA, COMPLEMENT_RNA
from ._helpers import _token2one_hot, _tokenize, _pad_sequences, _sequencize, _one_hot2token

# my own
def ascii_encode_seq(seq, pad=0):
    """
    Converts a string of characters to a NumPy array of byte-long ASCII codes.
    """
    encode_seq = np.array([ord(letter) for letter in seq], dtype=int)
    if pad > 0:
        encode_seq = np.pad(encode_seq, pad_width=(0, pad), mode="constant", constant_values=36)
    return encode_seq

# my own
def ascii_encode_seqs(seqs, pad=0):
    """
    Converts a set of sequences to a NumPy array of byte-long ASCII codes.
    """
    encode_seqs = np.array(
        [ascii_encode_seq(seq, pad=pad) for seq in seqs], dtype=int
    )
    return encode_seqs

# my own
def ascii_decode_seq(seq):
    """
    Converts a NumPy array of byte-long ASCII codes to a string of characters.
    """
    return "".join([chr(int(letter)) for letter in seq]).replace("$", "")

# my own
def ascii_decode_seqs(seqs):
    """Convert a set of one-hot encoded arrays back to strings"""
    return np.array([ascii_decode_seq(seq) for seq in seqs], dtype=object)

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

# modifed concise
def ohe_seq(
    seq, 
    vocab="DNA", 
    neutral_vocab="N", 
    fill_value=0
):
    """Convert a sequence into one-hot-encoded array."""
    seq = seq.strip().upper()
    return _token2one_hot(_tokenize(seq, vocab, neutral_vocab), vocab, fill_value=fill_value)

# modfied concise
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

# my own
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

# my own
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
