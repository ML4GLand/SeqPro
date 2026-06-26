---
name: seqpro
description: Use when writing Python that processes biological sequences (DNA/RNA/protein) with the seqpro package — encoding, one-hot, k-mer shuffling, reverse complement, GC content, variable-length sequence batches, or anything involving seqpro's `Ragged` array. Covers the seqpro API surface and the conventions you need to use it correctly.
---

# seqpro

Python/Rust package for fast biological-sequence processing. Python+NumPy+Numba for hot loops, a small Rust extension (`src/kshuffle.rs`) for graph-algorithm ops, and a Rust-native `Ragged` array (`_core.Ragged`) for variable-length batches. Imported as `import seqpro as sp`.

## When to use

- Encoding/decoding DNA, RNA, or protein sequences (OHE, integer tokens, padding).
- Sequence augmentation: reverse complement, k-mer shuffle, jitter, random draws.
- Sequence stats: GC content, nucleotide composition, length.
- Variable-length batches (e.g. peaks, transcripts of different sizes) → `sp.Ragged`.
- Genomic interval I/O: `sp.bed`, `sp.gtf`.

## Conventions (load these into working memory)

- **Public API**: see `python/seqpro/__init__.py` for the full export list. Re-read it before assuming a symbol exists.
- **Input types** (`SeqType` in `python/seqpro/_utils.py`): str, bytes, nested str lists, or `ndarray` with dtype `str_`/`object_`/`bytes_`/`uint8`. `sp.cast_seqs(...)` normalizes string-like inputs to `|S1` bytes arrays; `uint8` (OHE) is left untouched.
- **Canonical in-memory dtypes**: `|S1` for string sequences, `uint8` for one-hot.
- **Axis arguments are required and explicit**: most functions take `length_axis` and (for OHE) `ohe_axis` as positional/keyword ints. Negative indices allowed. `check_axes()` validates and raises early — don't catch and paper over.
- **No Python loops over sequences in library code.** Hot paths use NumPy, Numba kernels in `_numba.py`, or the Rust extension. If you're tempted to write a `for` over residues, look for an existing vectorized op or a Numba kernel first.
- **Alphabets are singletons**: `sp.DNA`, `sp.RNA`, `sp.AA`. Construct custom ones via `sp.NucleotideAlphabet` / `sp.AminoAlphabet` (`python/seqpro/alphabets/_alphabets.py`).
- **`AminoAlphabet.translate(seqs, ..., validate=False, unknown="X")`**: translates nucleotides → amino acids. Case-insensitive: lowercase/soft-masked `acgt` always translate. `unknown=` controls non-canonical codons (anything outside `{A,C,G,T}`): a single character (default `"X"`) pads one marker per bad codon; the literal `"drop"` removes bad codons and returns a `Ragged` (even for dense input, since lengths then vary). `validate=True` is the single fast-fail path — it raises (case-insensitively) on `N`/IUPAC/non-one-hot input and, when it returns, guarantees exact translation. There is no separate `error` mode.
- **Transforms** (`python/seqpro/transforms/`) wrap functional ops as callables — use these in data pipelines instead of inline lambdas.

## Quick reference

| Task | Call | Notes |
|---|---|---|
| Normalize input | `sp.cast_seqs(x)` | → `|S1` bytes, or passthrough for OHE |
| One-hot encode | `sp.ohe(x, alphabet, length_axis=-1)` | last axis added for OHE dim |
| Decode OHE | `sp.decode_ohe(x, alphabet, ohe_axis=-1)` | |
| Tokenize / detokenize | `sp.tokenize` / `sp.decode_tokens` | integer ids; `parallel=True/False` forces/disables the parallel kernel (default `None` = size heuristic) |
| Pad | `sp.pad_seqs(x, pad_val, length=...)` | |
| Reverse complement | `sp.reverse_complement(x, alphabet, length_axis=-1)` | works on str/bytes/OHE |
| K-mer shuffle | `sp.k_shuffle(x, k, length_axis=-1, seed=...)` | calls Rust `_k_shuffle` |
| Jitter | `sp.jitter(x, max_jitter, length_axis=-1)` | |
| Random sequences | `sp.random_seqs(shape, alphabet, seed=...)` | |
| GC content | `sp.gc_content(x, length_axis=-1)` | |
| Nucleotide content | `sp.nucleotide_content(x, alphabet, length_axis=-1)` | |
| Coverage binning | `sp.bin_coverage(arr, bin_width, length_axis)` | |
| BED / GTF I/O | `sp.bed.read_bedlike(...)`, `sp.gtf.read_gtf(...)` | polars/pyranges-backed |
| Hash ragged strings | `rag.hash("sha256"\|"md5"\|"rapidhash")` |

For exact signatures and kwargs, read the docstring directly (`sp.<fn>?` in a REPL, or open the source — files are short).

## `Ragged` — variable-length sequence batches

`sp.Ragged` (backed by `python/seqpro/rag/_core.py`) is the canonical container for batches where sequences differ in length. It is a Rust-native class implementing `NDArrayOperatorsMixin` (NOT a subclass of `ak.Array`) with **exactly one ragged dimension**, plus zero-copy access to the underlying flat NumPy buffer and offsets.

### Mental model

A `Ragged` has three things:

- **`data`**: a flat contiguous `NDArray` of shape `(total_elements, *fixed_trailing_dims)`. Zero-copy access via `rag.data`.
- **`offsets`**: an `int64` array. Shape `(N+1,)` (contiguous, the common case) **or** `(2, N)` starts/stops (after some slices). Access via `rag.offsets`.
- **`shape`**: a tuple like `(batch, None, ohe_dim)` where exactly one entry is `None` — that's the ragged axis. `rag.rag_dim` gives its index.

`rag.lengths` derives segment lengths from offsets (cheap, returns an `ndarray`).

### Construction

```python
import numpy as np, seqpro as sp

# From lengths (most common — you have a flat numeric data buffer and per-segment lengths)
data = np.frombuffer(b"ACGTACGTACG", dtype=np.uint8)  # uint8 for char-as-number
lengths = np.array([4, 3, 4])
rag = sp.rag.Ragged.from_lengths(data, lengths)   # shape (3, None)

# From explicit offsets (also accepts S1 char arrays when shape has a None)
offsets = np.array([0, 4, 7, 11], dtype=np.int64)
char_data = np.frombuffer(b"ACGTACGTACG", dtype="S1")
rag = sp.rag.Ragged.from_offsets(char_data, shape=(3, None), offsets=offsets)  # shape (3, None)

# Empty with known shape
rag = sp.rag.Ragged.empty((10, None, 4), dtype=np.uint8)   # batch of 10 OHE seqs
```

`Ragged.empty(shape, dtype)` requires exactly one `None` in `shape`. Trailing fixed dims (e.g. the OHE axis) go after the `None`.

### Working with `Ragged` — do this, not that

| Task | Do | Don't |
|---|---|---|
| Bulk numeric op on the flat data | `rag.data[:] = ...` or `rag.data.view(...)` — zero-copy | Iterate `for seq in rag:` |
| Apply a `np.ufunc` | Just call it: `np.exp(rag)` — dispatched via `__array_ufunc__` (NDArrayOperatorsMixin) to return a `Ragged` | Manually unpack and rebuild |
| Count top-level rows | `len(rag)` — returns `shape[0]` (raises if `shape[0]` is the ragged axis) | `rag.shape[0]` with manual int-cast |
| Insert a leading size-1 axis | `rag[np.newaxis]` — returns `Ragged` with shape `(1, *old_shape)` | Manual `from_offsets` rebuild |
| Reinterpret bytes/dtype | `rag.view(np.uint8)` | `np.asarray(rag).view(...)` (loses ragged structure) |
| Reshape non-ragged axes | `rag.reshape(batch, None, k, 4)` | Touch `rag.data.shape` directly |
| Drop a size-1 axis | `rag.squeeze(axis)` (returns `ndarray` if collapses to 1D) | |
| Densify to NumPy | `rag.to_numpy()` (pads/raises per `allow_missing`) | Loop and stack |
| Pack into contiguous buffer | `rag.to_packed()` or `sp.rag.to_packed(rag)` — Numba-parallelized, safe on `np.memmap`; `copy=False` for zero-copy passthrough when already packed | `ak.to_packed(rag)` |
| Densify + right-pad to fixed length | `sp.rag.to_padded(rag, pad_value, *, length=None)` — flat-buffer numba kernel; `length=None` pads to batch max, explicit `length` pads/truncates; ragged-axis-last, non-record only | `rag.to_numpy()` with manual slicing or `ak_str.rpad` (~3× slower; the awkward path allocates extra intermediates) |
| Concatenate along ragged axis | `sp.rag.concatenate(rags, axis)` — concatenate a list of `Ragged` arrays along the ragged axis (`axis` must be the `None` dim, negative allowed); offset-arithmetic + buffered copy via Rust/rayon kernel; numeric dtypes (int32, float32, …) | `ak.concatenate(rags, axis=…)` |
| Strip to plain awkward | `rag.to_ak()` | |

### Record-layout `Ragged` (multi-field)

Build by calling `sp.rag.zip` (or equivalently `Ragged.from_fields`) with a dict of single-field `Ragged`s that share the **same offsets object**. The result is a `Ragged` with a record layout:

```python
import numpy as np, seqpro as sp
from seqpro.rag._utils import lengths_to_offsets

lengths = np.array([4, 3])
shared_offsets = lengths_to_offsets(lengths)

seq_rag   = sp.rag.Ragged.from_offsets(seq_flat,   shape=(2, None), offsets=shared_offsets)  # |S1
score_rag = sp.rag.Ragged.from_offsets(score_flat, shape=(2, None), offsets=shared_offsets)  # f4

batch = sp.rag.zip({"seq": seq_rag, "score": score_rag})   # → Ragged (record layout)
# equivalently: batch = sp.rag.Ragged.from_fields({"seq": seq_rag, "score": score_rag})
assert isinstance(batch, sp.rag.Ragged)

batch["score"].data[:] *= 2.0   # zero-copy mutation of the flat score buffer
```

The inputs **must share the same offsets object** (pass the same `shared_offsets` array to each `from_offsets` call) — that's what makes the result a single-ragged-dim record. Passing independently-constructed `from_lengths` results raises `ValueError` because their offsets objects differ even if lengths are equal.

- `rag.dtype` returns a NumPy *structured* dtype (e.g. `[("seq","S1"),("score","f4")]`), purely as a descriptor — memory is SoA, not AoS.
- `rag.data` returns a **dict keyed by field name**, not a single array. Always type-check before indexing.
- `rag["field"]` gives zero-copy single-field access and shares the parent's offsets object. Its `.data` is the flat NumPy buffer for that field.
- `rag.to_numpy()` on a record layout returns a **dict `{field: dense ndarray}`** (raises if any field is still jagged — lengths must be uniform for a dense conversion).
- `view` and `apply` are **not defined** on record layouts — operate per-field.

### Hashing strings

Hash each string in a `Ragged` (opaque-string or S1-chars leaf, any depth) with
a parallel Rust kernel:

```python
digests = rag.hash("sha256")        # (N, 32) uint8, one digest per string
md5s    = rag.hash("md5")           # (N, 16) uint8
fast    = rag.hash("rapidhash")     # (N,) uint64
seeded  = rag.hash("rapidhash", seed=42)   # seed valid for rapidhash only
# equivalently: sp.rag.hash(rag, "sha256")
```

Output mirrors the structure *above* the string level: a regular NumPy array
when strings aren't grouped (flat `(N, …)` / leading fixed dims), or a `Ragged`
reusing the outer offsets when they are (`(G, None, 16/32)` for md5/sha256,
`(G, None)` uint64 for rapidhash). Numeric and record-layout Rageds are
rejected.

### NumPy interop — what you can rely on

`_core.Ragged` implements `NDArrayOperatorsMixin` and `__array_ufunc__` directly (no awkward dependency):

- NumPy ufuncs (`np.add`, `np.exp`, etc.) on a non-record `Ragged` return a `Ragged`. Record layouts raise `NotImplementedError` — operate on individual fields.
- `rag.to_packed()` / `sp.rag.to_packed(rag)` is the canonical way to materialize a contiguous, zero-based buffer — Numba-parallelized and safe on `np.memmap`. Use `copy=False` for a zero-copy passthrough when the array is already packed (raises `ValueError` if not).
- Don't rely on awkward (`ak.*`) APIs on `_core.Ragged` — the backend no longer registers `ak.behavior`. Use `rag.to_ak()` to get an `ak.Array` if you need awkward interop, but prefer the native API.

When in doubt, read `python/seqpro/rag/_core.py` — it's the live backend and the docstrings are the source of truth. `_layout.py`, `_ops.py`, and `_utils.py` in the same dir contain supporting internals.

### Common pitfalls

- **Offsets layout drifts after slicing.** `rag.offsets` may become `(2, N)` starts/stops instead of `(N+1,)`. Check `rag.is_contiguous` / call `rag.to_packed()` before any code that assumes `(N+1,)`.
- **`rag.data` on a record layout is a dict.** Code like `rag.data.shape` will fail; branch on `isinstance(rag.data, dict)` or use `rag.parts` and inspect.
- **`Ragged` must have exactly one `None` in `shape`.** Constructing from data whose ragged structure doesn't match raises in `__init__`. Use `from_lengths` / `from_offsets` when in doubt.
- **The Rust k-shuffle expects contiguous `uint8` with the last axis as sequence length.** `sp.k_shuffle` handles this for you; if calling `seqpro._k_shuffle` directly, ensure layout.

## Where to look (don't memorize — read the source)

| Need | File |
|---|---|
| Public surface | `python/seqpro/__init__.py` |
| Input casting / axis helpers | `python/seqpro/_utils.py` |
| OHE / tokens / padding | `python/seqpro/_encoders.py` |
| Augmentations | `python/seqpro/_modifiers.py` |
| Stats | `python/seqpro/_analyzers.py` |
| Alphabets | `python/seqpro/alphabets/_alphabets.py` |
| Ragged | `python/seqpro/rag/_core.py` |
| Transforms (pipeline objects) | `python/seqpro/transforms/` |
| BED/GTF | `python/seqpro/bed.py`, `gtf.py` |
| Rust k-shuffle | `src/kshuffle.rs` |
| Tests as usage examples | `tests/` |
| Rendered docs | `site/` (built from `docs/`) |

## Don'ts

- Don't write Python `for` loops over residues or positions in library code. Look for a vectorized op, a Numba kernel in `_numba.py`, or extend one.
- Don't assume an axis — always pass `length_axis` (and `ohe_axis` where relevant) explicitly.
- Don't reach into `Ragged` internals (`_layout`, `_rl`, `__init__` shortcuts) from user code; use `data`, `offsets`, `fields`, `from_lengths`, `from_offsets`, `from_fields`, `empty`.
- Don't introduce strings into `Ragged`. ASCII bytes (`|S1`) only.
- Don't add a feature or change a public signature without updating this skill — see CLAUDE.md.
