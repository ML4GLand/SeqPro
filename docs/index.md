# SeqPro

[![PyPI - Downloads](https://img.shields.io/pypi/dm/seqpro)](https://pypi.org/project/seqpro/)
[![GitHub stars](https://img.shields.io/github/stars/ML4GLand/SeqPro)](https://github.com/ML4GLand/SeqPro)

SeqPro is a Python package for processing DNA/RNA sequences, with limited support for protein sequences. It makes almost zero compromises on speed — NumPy vectorization throughout, Numba JIT for bottlenecks, and a Rust extension for graph algorithms like k-mer shuffling.

All functions accept strings, lists of strings, NumPy arrays of strings or single-byte ASCII (`S1`), or one-hot encoded (`uint8`) arrays.

## Installation

```bash
pip install seqpro
```

## Quick Start

```python
import seqpro as sp

N, L = 2, 3

# Generate random sequences
seqs = sp.random_seqs(shape=(N, L), alphabet=sp.DNA, seed=1234)

# One-hot encode / decode
ohe = sp.ohe(seqs, alphabet=sp.DNA)
sp.decode_ohe(ohe, ohe_axis=-1, alphabet=sp.DNA, unknown_char="N")

# Tokenize
token_map = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
tokens = sp.tokenize(seqs, token_map=token_map, unknown_token=11)

# Reverse complement
sp.reverse_complement(seqs, alphabet=sp.DNA)

# k-let preserving shuffle
sp.k_shuffle(seqs, k=2, length_axis=1, seed=1234)

# GC / nucleotide content
sp.gc_content(seqs, alphabet=sp.DNA)
sp.nucleotide_content(seqs, alphabet=sp.DNA)
```

See the [API Reference](api/index.md) for full documentation.
