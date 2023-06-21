[![PyPI version](https://badge.fury.io/py/seqpro.svg)](https://badge.fury.io/py/seqpro)

# SeqPro (Sequence Processing toolkit in Python)
```python
import seqpro as sp
```

SeqPro is a Python package for processing genomic sequences. It provides a set of tools for cleaning, modifiying, encoding, and analyzing genomic sequences. SeqPro currently supports processing DNA/RNA sequences, with limited support for protein sequences. SeqPro is fully functional on its own but is also heavily utilized throughout the other packages in the ML4GLand project, including SeqData, EUGENe, MotifData, and SeqExplainer.

All functions in SeqPro take as input a string, a list of strings, a NumPy array of strings, or a NumPy array of single character bytes (S1) or one-hot encoded arrays. There is also emerging integration with XArray through the `seqpro.xr` submodule.

Computational bottelnecks or code that is impossible to vectorize with NumPy alone are accelerated with Numba e.g. padding sequences, one-hot encoding, converting from one-hot encoding to nucleotides, etc.

## Sequence cleaners (`cleaners`)

### Remove sequences with ambiguous bases

```python
sp.remove_N_seqs(seqs)
sp.sanitize_seqs(seqs)
```

## Sequence modifiers (`modifiers`)

### Reverse complement sequences

```python
sp.reverse_complement(seqs)
```

### K-let frequency preserving shuffles
```python
sp.k_shuffle(seqs, k=2)
```

## Sequence encoders (`encoders`)

## One-hot encoding

```python
sp.ohe(seqs)
```

## Sequence analysis (`analyzers`)

### Calculate sequence properties (e.g. GC content)

```python
sp.gc_content(seqs)
sp.nucleotide_content(seqs)
```

## Visaulize sequence properties (`visualizers`)

```python
sp.plot_nucleotide_content(seqs)
sp.plot_gc_content(seqs)
```

# Requirements

```bash
python
numpy
numba
```

# More to come!

## Tutorials

### Preparing sequences for sequence-to-function models

### Preparing features for MPRA activity prediction

### Motif enrichment analysis with HOMER/DEM/cisTarget