[![PyPI version](https://badge.fury.io/py/seqpro.svg)](https://badge.fury.io/py/seqpro)

# SeqPro (Sequence Processing toolkit in Python)
```python
import seqpro as sp
```

SeqPro is a Python package for processing genomic sequences. It provides a set of tools for cleaning, modifiying, encoding,  and analyzing genomic sequences. SeqPro currently supports processing DNA/RNA sequences, but will be extended to support protein sequences in the future. SeqPro is meant to be used in conjunction with other packages for sequence-to-function modeling in the ML4GLand project, including SeqData, EUGENe, MotifData, and SeqExplainer.

Most functions in SeqPro take in NumPy arrays of sequences as input and have two versions. One for acting on a single seq (`_seq()`) and one for acting on a list of sequences (`_seqs()`).

## Sequence cleaners (`cleaners`)

### Remove sequences with ambiguous bases

```python
sp.remove_N_seqs(seqs)
sp.sanitize_seqs(seqs)
```

## Sequence modifiers (`modifiers`)

### Reverse complement sequences

```python
sp.reverse_complement_seqs(seqs)
```

### Shuffle sequences
```python

sp.shuffle_seqs(seqs)
sp.dinucleotide_shuffle_seqs(seqs)
```

## Sequence encoders (`encoders`)

### Ascii encoding

```python
sp.ascii_encode_seqs(seqs)
sp.ascii_decode_seqs(seqs)
```

## One-hot encoding

```python
sp.ohe_seqs(seqs)
sp.ohe_seqs(seqs, order=2)
sp.decode_seqs(ohe_seqs)
```

## Sequence analysis (`analyzers`)

### Calculate sequence properties (e.g. GC content)

```python
sp.gc_content_seqs(seqs)
sp.nucleotide_content_seqs(seqs)
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
torch
```

# More to come!

## Tutorials

### Preparing sequences for sequence-to-function models

### Preparing features for MPRA activity prediction

### Motif enrichment analysis with HOMER/DEM/cisTarget

## Functionality

### Extract k-mers from sequences

```python
sp.extract_kmers(seqs, k)
sp.extract_gapped_kmers(seqs, k, g)
```


        
### Sequence annotations

  - Known genomic features
  - Overlap with different epigenomics data

# Acknowledgements

1. concise
2. dinuc shuffle
3. 

# References
1. 