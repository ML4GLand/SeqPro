![PyPI - Downloads](https://img.shields.io/pypi/dm/seqpro)
![GitHub stars](https://img.shields.io/github/stars/ML4GLand/SeqPro)

# SeqPro (Sequence processing toolkit)
```python
import seqpro as sp
```

SeqPro is a Python package for processing DNA/RNA sequences, with limited support for protein sequences. SeqPro is fully functional on its own but is heavily utilized by other packages including [SeqData](https://github.com/ML4GLand/SeqData), [MotifData](https://github.com/ML4GLand/MotifData), [SeqExplainer](https://github.com/ML4GLand/SeqExplainer), and [EUGENe](https://github.com/ML4GLand/EUGENe).

All functions in SeqPro take as input a string, a list of strings, a NumPy array of strings, a NumPy array of single character bytes (S1) or a NumPy array of one-hot encoded strings. There is also emerging integration with XArray through the `seqpro.xr` submodule to integrate nicely with [SeqData](https://github.com/ML4GLand/SeqData).

Computational bottelnecks or code that is impossible to vectorize with NumPy alone are accelerated with Numba e.g. padding sequences, one-hot encoding, converting from one-hot encoding to nucleotides, etc.

# Installation

```bash
pip install seqpro
```

## API

```python
N = 2
L = 3

# Generating random sequences
seqs = sp.random_seqs(shape=(N, L), alphabet=sp.DNA, seed=1234)

# Padding
sp.pad_seqs(seqs, pad="right", pad_value="N", length=5, length_axis=-1)

# One-hot encoding and decoding
ohe = sp.ohe(seqs, alphabet=sp.DNA)
sp.decode_ohe(ohe, ohe_axis=-1, alphabet=sp.DNA, unknown_char="N")

# Tokenization
token_map = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
tokens = sp.tokenize(seqs, token_map=token_map, unknown_token=11)
sp.decode_tokens(tokens, token_map=token_map)

# Reverse complement
sp.reverse_complement(seqs, alphabet=sp.DNA)

# k-let preserving shuffling
sp.k_shuffle(seqs, k=2, length_axis=1, seed=1234)

# Calculating GC or nucleotide content
sp.gc_content(seqs, alphabet=sp.DNA)
sp.nucleotide_content(seqs, alphabet=sp.DNA)

# Randomly jittering sequences
sp.jitter(seqs, max_jitter=128, length_axis=1, seed=1234)

# Collapse coverage to a given bin width
sp.bin_coverage(coverage, bin_width=128, length_axis=1, normalize=False)
```

# More to come!

All contributions, including bug reports, documentation improvements, and enhancement suggestions are welcome. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md)
