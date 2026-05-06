![PyPI - Downloads](https://img.shields.io/pypi/dm/seqpro)
![GitHub stars](https://img.shields.io/github/stars/ML4GLand/SeqPro)

# SeqPro (Sequence processing toolkit)
```python
import seqpro as sp
```

SeqPro is a Python package for processing DNA/RNA sequences, with support for protein sequences. SeqPro is fully functional on its own but is heavily utilized by other packages including [SeqData](https://github.com/ML4GLand/SeqData), [MotifData](https://github.com/ML4GLand/MotifData), [SeqExplainer](https://github.com/ML4GLand/SeqExplainer), and [EUGENe](https://github.com/ML4GLand/EUGENe).

All functions in SeqPro take as input a string, a list of strings, a NumPy array of strings, a NumPy array of single character bytes (`S1`), or a NumPy array of one-hot encoded sequences (`uint8`). Variable-length sequence collections are supported via the `seqpro.rag` submodule (`Ragged` arrays). There is also experimental integration with XArray through the `seqpro.xr` submodule.

Computational bottlenecks or code that is impossible to vectorize with NumPy alone are accelerated with Numba (e.g. padding sequences, one-hot encoding, decoding) and Rust via PyO3/maturin (e.g. k-mer shuffling).

# Installation

```bash
pip install seqpro
```

## API

```python
N = 2
L = 9

# Generating random sequences
seqs = sp.random_seqs(shape=(N, L), alphabet=sp.DNA, seed=1234)

# Padding
sp.pad_seqs(seqs, pad="right", pad_value="N", length=12, length_axis=-1)

# One-hot encoding and decoding (via alphabet)
ohe = sp.DNA.ohe(seqs)
sp.DNA.decode_ohe(ohe, ohe_axis=-1, unknown_char="N")

# Tokenization
token_map = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
tokens = sp.tokenize(seqs, token_map=token_map, unknown_token=11)
sp.decode_tokens(tokens, token_map=token_map)

# Reverse complement (via alphabet or standalone)
sp.DNA.reverse_complement(seqs)
sp.reverse_complement(seqs, alphabet=sp.DNA)

# k-let preserving shuffling
sp.k_shuffle(seqs, k=2, length_axis=1, seed=1234)

# Calculating GC or nucleotide content
sp.gc_content(seqs, alphabet=sp.DNA)
sp.nucleotide_content(seqs, alphabet=sp.DNA)

# Randomly jittering sequences
sp.jitter(seqs, max_jitter=2, length_axis=1, seed=1234)

# Collapse coverage to a given bin width
sp.bin_coverage(coverage, bin_width=128, length_axis=1, normalize=False)

# Translation: DNA/RNA → amino acids
cds = sp.random_seqs(shape=(N, L), alphabet=sp.DNA, seed=42)
aa = sp.AA.translate(cds)

# One-hot encode amino acids
aa_ohe = sp.AA.ohe(aa)
sp.AA.decode_ohe(aa_ohe, ohe_axis=-1, unknown_char="X")
```

## Alphabets

SeqPro ships three built-in alphabets:

| Object | Type | Description |
|---|---|---|
| `sp.DNA` | `NucleotideAlphabet` | ACGT with reverse complement support |
| `sp.RNA` | `NucleotideAlphabet` | ACGU with reverse complement support |
| `sp.AA` | `AminoAlphabet` | Standard 20 amino acids + stop codon (`*`), with codon translation |

Each alphabet exposes `.ohe()`, `.decode_ohe()`, and `.reverse_complement()` (nucleotide alphabets) or `.translate()` (amino acid alphabet) directly on the object.

# More to come!

All contributions, including bug reports, documentation improvements, and enhancement suggestions are welcome. Everyone within the community is expected to abide by our [code of conduct](https://github.com/ML4GLand/EUGENe/blob/main/CODE_OF_CONDUCT.md)
