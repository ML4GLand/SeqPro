![PyPI - Downloads](https://img.shields.io/pypi/dm/seqpro)
![GitHub stars](https://img.shields.io/github/stars/ML4GLand/SeqPro)

# SeqPro (Sequence processing toolkit)
```python
import seqpro as sp
```

SeqPro is a Python package for processing DNA/RNA sequences, with limited support for protein sequences. SeqPro is fully functional on its own but is heavily utilized by other packages including [SeqData](https://github.com/ML4GLand/SeqData), [MotifData](https://github.com/ML4GLand/MotifData), [SeqExplainer](https://github.com/ML4GLand/SeqExplainer), and [EUGENe](https://github.com/ML4GLand/EUGENe).

All functions in SeqPro take as input a string, a list of strings, a NumPy array of strings, a NumPy array of single character bytes (S1) or a NumPy array of one-hot encoded strings. There is also emerging integration with XArray through the `seqpro.xr` submodule.

Computational bottelnecks or code that is impossible to vectorize with NumPy alone are accelerated with Numba e.g. padding sequences, one-hot encoding, converting from one-hot encoding to nucleotides, etc.

## Manipulating sequences
```python

# Padding
sp.pad_seqs(seqs, pad="right", pad_value="N", max_length=None)

# One-hot encoding
sp.ohe(seqs, alphabet=sp.alphabets.DNA)

# Decode one-hot encoding
sp.decode_ohe(ohe, ohe_axis=1, alphabet=sp.alphabets.DNA, unknown__char="N")

# Reverse complement
sp.reverse_complement(seqs, alphabet=sp.alphabets.DNA)

# k-let preserving shuffling
sp.k_shuffle(seqs, k=2, length_axis=1, seed=1234)

# Calculating GC content
sp.gc_content(seqs, normalize=True)

# Generating random sequences
sp.random_seqs(shape=(N, L), alphabet=sp.alphabets.DNA, seed=1234)

# Randomly jittering sequences
sp.jitter(seqs, max_jitter=128, length_axis=1, seed=1234)
```

## Manipulating coverage
```python

# Collapse coverage to a given bin width
sp.bin_coverage(coverage, bin_width=128, length_axis=1, normalize=False)

# Can jitter coverage and sequences so they stay aligned
sp.jitter((seqs, coverage), max_jitter=128, length_axis=1, seed=1234)

# Normalize coverage using CPM or CPKM
sp.normalize_coverage(coverage, method="CPM", total_counts=200e6, length_axis=1)
```

# Requirements

```bash
python
numpy
numba
cython
ushuffle
```

# Contributing
This section was modified from https://github.com/pachterlab/kallisto.

All contributions, including bug reports, documentation improvements, and enhancement suggestions are welcome. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md)

As we work towards a stable v1.0.0 release, and we typically develop on branches. These are merged into `dev` once sufficiently tested. `dev` is the latest, stable, development branch. 

`main` is used only for official releases and is considered to be stable. If you submit a pull request, please make sure to request to merge into `dev` and NOT `main`.
