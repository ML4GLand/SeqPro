# Ragged Arrays

[`Ragged`][seqpro.rag.Ragged] is SeqPro's array type for collections of variable-length sequences. It stores data contiguously in a flat NumPy array alongside an offsets array that marks segment boundaries, following the Arrow/awkward-array layout. This avoids padding and lets you apply NumPy operations and SeqPro functions directly — without a Python loop. If you're familiar with [Awkward Array](https://awkward-array.org/doc/main/), [`Ragged`][seqpro.rag.Ragged] is a subclass and special case of `ak.Array`, where there is only one "awkward" axis.

```python
import numpy as np
import seqpro as sp
from seqpro.rag import Ragged
```

---

## Numeric data

### Construction

The two primary constructors are [`Ragged.from_lengths`][seqpro.rag.Ragged.from_lengths] (supply lengths for each element) and
[`Ragged.from_offsets`][seqpro.rag.Ragged.from_offsets] (supply a pre-computed offsets array).

```python
# per-position coverage for three intervals of different widths
data = np.array([
    0.1,
    0.5,
    0.3,  # interval 0 — length 3
    0.8,
    0.2,  # interval 1 — length 2
    0.4,
    0.6,
    0.9,
    0.1,
    0.7,  # interval 2 — length 5
])
lengths = np.array([3, 2, 5])
rag = Ragged.from_lengths(data, lengths)

rag.shape  # (3, None) (1)
rag.dtype  # dtype('float64')
rag.lengths  # array([3, 2, 5])
```

1. `None` marks the ragged dimension.


Access individual elements with standard indexing:

```python
rag[0]  # awkward array: [0.1, 0.5, 0.3]
rag[1]  # [0.8, 0.2]
```

### Arithmetic and NumPy ufuncs

NumPy ufuncs and arithmetic operators are dispatched element-wise across the flat data:

```python
scaled = rag * 2.0
shifted = rag + 1.0
normed = rag / rag.data.max()

log1p = np.log1p(rag)
rooted = np.sqrt(rag)
```

The result is always a new [`Ragged`][seqpro.rag.Ragged] with the same offsets — no copies of the offset structure.

---

## Sequence data

`Ragged[np.bytes_]` is SeqPro's representation of a collection of variable-length sequences.

### Building a sequence Ragged

```python
cds_seqs = ["ATGAAATAA", "ATGGGG", "ATCGAT"]

data = np.array(list("".join(cds_seqs)), dtype="S1")
lengths = np.array([len(s) for s in cds_seqs])
cds = Ragged.from_lengths(data, lengths)

cds.shape  # (3, None)
cds.dtype  # dtype('S1')
```

### Translation with `sp.AA.translate`

[`AminoAlphabet.translate`][seqpro.AminoAlphabet.translate] accepts `Ragged[np.bytes_]` and returns a new [`Ragged`][seqpro.rag.Ragged] of amino-acid
sequences — each output length is `input_length // 3`, so variable-length CDS stays variable
after translation with no extra bookkeeping:

```python
aa = sp.AA.translate(cds)

aa.shape  # (3, None)
aa.lengths  # array([3, 2, 2])
```

Pass `truncate_stop=True` to strip any codons after the first stop codon:

```python
# truncate_stop=False
# ATGTAAAAA → M * K
# truncate_stop=True
# ATGTAAAAA → M * (stop retained but truncated inclusive)
aa_trunc = sp.AA.translate(cds, truncate_stop=True)
```

### OHE Ragged translation

[`AminoAlphabet.translate`][seqpro.AminoAlphabet.translate] also accepts one-hot encoded ragged arrays (`Ragged[np.uint8]`). Provide
`nuc_alphabet` so SeqPro knows how to decode the OHE encoding:

```python
ohe_data = np.concatenate([sp.DNA.ohe(sp.cast_seqs(s)) for s in cds_seqs])
# ohe_data has shape (total_nucleotides, 4)

cds_ohe = Ragged.from_lengths(ohe_data, lengths)
aa_ohe = sp.AA.translate(cds_ohe, nuc_alphabet=sp.DNA)

aa_ohe.dtype  # dtype('uint8')  — output is OHE amino acids
```

---

## Record Ragged (structure-of-arrays)

A record [`Ragged`][seqpro.rag.Ragged] holds multiple named fields that share the same ragged structure. This is
the structure-of-arrays (SoA) pattern: one offsets array, multiple data arrays.

### Building a record Ragged

Use `ak.zip` on two or more [`Ragged`][seqpro.rag.Ragged] arrays of the same length and offsets:

```python
import awkward as ak

scores = np.array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.6, 0.9])
flags = np.array([1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
lengths = np.array([3, 2, 3])

r_scores = Ragged.from_lengths(scores, lengths)
r_flags = Ragged.from_lengths(flags, lengths)

rec = ak.zip({"score": r_scores, "flag": r_flags})
# ak.zip returns a Ragged automatically when inputs are Ragged
```

### Inspecting fields

```python
rec.shape  # (3, None)
rec.lengths  # array([3, 2, 3])

rec.dtype
# {"score": dtype('float64'), "flag": dtype('int8')}
```

### Field access

Fields are accessed by key or attribute. Both paths are zero-copy — the returned [`Ragged`][seqpro.rag.Ragged]
shares the parent's [`offsets`][seqpro.rag.Ragged.offsets] array:

```python
rec["score"]  # Ragged[float64]  — key access
rec.score  # same, attribute-style

# all fields share exactly the same offsets object
assert rec["score"].offsets is rec["flag"].offsets
```

### Flat data access

[`Ragged.data`][seqpro.rag.Ragged.data] on a record Ragged returns a dict of flat NumPy arrays, one per field:

```python
d = rec.data
d["score"]  # array([0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.6, 0.9])
d["flag"]  # array([1, 0, 1, 1, 0, 0, 1, 1], dtype=int8)
```

For per-field operations, access the field first, then use ufuncs:

```python
np.sqrt(rec["flag"].view(np.float32))
```

---

## API reference

See the [Ragged API reference](api/ragged.md) for the full method and property listing.
