```{toctree}
:hidden: true
:caption: Contents
:maxdepth: 2

installation
api
```

# SeqPro -- Sequence processing toolkit

```{image} https://badge.fury.io/py/seqpro.svg
:alt: PyPI version
:target: https://badge.fury.io/py/seqpro
:class: inline-link
```

```{image} https://img.shields.io/pypi/dm/seqpro
:alt: PyPI - Downloads
:class: inline-link
```

```{image} https://img.shields.io/github/stars/ML4GLand/SeqPro
:alt: GitHub stars
:target: https://github.com/ML4GLand/SeqPro
:class: inline-link
```

SeqPro is a Python package for processing DNA/RNA sequences, with limited support for protein sequences. Some of the key features of SeqPro include:

- One-hot encoding and decoding of sequences
- Tokenization with custom token maps
- Padding and trimming sequences
- Reverse complement operations
- k-let preserving shuffling
- GC and nucleotide content calculation
- Coverage binning and jittering

SeqPro is fully functional on its own but is heavily utilized by other packages including [SeqData](https://github.com/ML4GLand/SeqData), [MotifData](https://github.com/ML4GLand/MotifData), [SeqExplainer](https://github.com/ML4GLand/SeqExplainer), and [EUGENe](https://github.com/ML4GLand/EUGENe).

Computational bottlenecks or code that is impossible to vectorize with NumPy alone are accelerated with Numba and Rust.

## Getting started

* {doc}`Install SeqPro <installation>`
* Browse the {doc}`API reference <api>`

## Quick example

```python
import seqpro as sp

# Generate random sequences
seqs = sp.random_seqs(shape=(2, 100), alphabet=sp.DNA, seed=1234)

# One-hot encode
ohe = sp.ohe(seqs, alphabet=sp.DNA)

# Reverse complement
rc = sp.reverse_complement(seqs, alphabet=sp.DNA)

# Calculate GC content
gc = sp.gc_content(seqs, alphabet=sp.DNA)
```

## Contributing

SeqPro is an open-source project and we welcome contributions. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md).
