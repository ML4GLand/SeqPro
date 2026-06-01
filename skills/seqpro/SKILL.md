---
name: seqpro
description: Use when writing Python that processes biological sequences (DNA/RNA/protein) with the seqpro package — encoding, one-hot, k-mer shuffling, reverse complement, GC content, variable-length sequence batches, or anything involving seqpro's `Ragged` array. Covers the seqpro API surface and the conventions you need to use it correctly.
---

# seqpro

Python/Rust package for fast biological-sequence processing. Python+NumPy+Numba for hot loops, a small Rust extension (`src/kshuffle.rs`) for graph-algorithm ops, and an awkward-backed `Ragged` array for variable-length batches. Imported as `import seqpro as sp`.

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
- **`AminoAlphabet.translate(seqs, ..., validate=False)`**: translates nucleotides → amino acids. Pass `validate=True` to raise on any input outside the alphabet — non-ACGT bytes (lowercase, `N`, IUPAC codes) for string/byte input, or any non-one-hot row for OHE input. When `validate=True` returns without raising, the translation is guaranteed exact; the default `validate=False` skips the check for speed and treats out-of-alphabet input as undefined.
- **Transforms** (`python/seqpro/transforms/`) wrap functional ops as callables — use these in data pipelines instead of inline lambdas.

## Quick reference

| Task | Call | Notes |
|---|---|---|
| Normalize input | `sp.cast_seqs(x)` | → `|S1` bytes, or passthrough for OHE |
| One-hot encode | `sp.ohe(x, alphabet, length_axis=-1)` | last axis added for OHE dim |
| Decode OHE | `sp.decode_ohe(x, alphabet, ohe_axis=-1)` | |
| Tokenize / detokenize | `sp.tokenize` / `sp.decode_tokens` | integer ids |
| Pad | `sp.pad_seqs(x, pad_val, length=...)` | |
| Reverse complement | `sp.reverse_complement(x, alphabet, length_axis=-1)` | works on str/bytes/OHE |
| K-mer shuffle | `sp.k_shuffle(x, k, length_axis=-1, seed=...)` | calls Rust `_k_shuffle` |
| Jitter | `sp.jitter(x, max_jitter, length_axis=-1)` | |
| Random sequences | `sp.random_seqs(shape, alphabet, seed=...)` | |
| GC content | `sp.gc_content(x, length_axis=-1)` | |
| Nucleotide content | `sp.nucleotide_content(x, alphabet, length_axis=-1)` | |
| Coverage binning | `sp.bin_coverage(arr, bin_width, length_axis)` | |
| BED / GTF I/O | `sp.bed.read_bedlike(...)`, `sp.gtf.read_gtf(...)` | polars/pyranges-backed |

For exact signatures and kwargs, read the docstring directly (`sp.<fn>?` in a REPL, or open the source — files are short).

## `Ragged` — variable-length sequence batches

`sp.Ragged` (in `python/seqpro/rag/_array.py`) is the canonical container for batches where sequences differ in length. It is a thin subclass of `ak.Array` with **exactly one ragged dimension**, plus zero-copy access to the underlying flat NumPy buffer and offsets.

### Mental model

A `Ragged` has three things:

- **`data`**: a flat contiguous `NDArray` of shape `(total_elements, *fixed_trailing_dims)`. Zero-copy access via `rag.data`.
- **`offsets`**: an `int64` array. Shape `(N+1,)` (contiguous, the common case) **or** `(2, N)` starts/stops (after some slices). Access via `rag.offsets`.
- **`shape`**: a tuple like `(batch, None, ohe_dim)` where exactly one entry is `None` — that's the ragged axis. `rag.rag_dim` gives its index.

`rag.lengths` derives segment lengths from offsets (cheap, returns an `ndarray`).

### Construction

```python
import numpy as np, seqpro as sp

# From lengths (most common — you have a flat data buffer and per-segment lengths)
data = np.frombuffer(b"ACGTACGTACG", dtype="S1")
lengths = np.array([4, 3, 4])
rag = sp.rag.Ragged.from_lengths(data, lengths)   # shape (3, None)

# From explicit offsets
offsets = np.array([0, 4, 7, 11], dtype=np.int64)
rag = sp.rag.Ragged.from_offsets(data, shape=(3, None), offsets=offsets)

# Empty with known shape
rag = sp.rag.Ragged.empty((10, None, 4), dtype=np.uint8)   # batch of 10 OHE seqs
```

`Ragged.empty(shape, dtype)` requires exactly one `None` in `shape`. Trailing fixed dims (e.g. the OHE axis) go after the `None`.

### Working with `Ragged` — do this, not that

| Task | Do | Don't |
|---|---|---|
| Bulk numeric op on the flat data | `rag.data[:] = ...` or `rag.data.view(...)` — zero-copy | Iterate `for seq in rag:` |
| Apply a `np.ufunc` | Just call it: `np.exp(rag)` — dispatched via awkward behavior to return a `Ragged` | Manually unpack and rebuild |
| Reinterpret bytes/dtype | `rag.view(np.uint8)` | `np.asarray(rag).view(...)` (loses ragged structure) |
| Reshape non-ragged axes | `rag.reshape(batch, None, k, 4)` | Touch `rag.data.shape` directly |
| Drop a size-1 axis | `rag.squeeze(axis)` (returns `ndarray` if collapses to 1D) | |
| Densify to NumPy | `rag.to_numpy()` (pads/raises per `allow_missing`) | Loop and stack |
| Pack into contiguous buffer | `rag.to_packed()` or `sp.rag.to_packed(rag)` — Numba-parallelized, safe on `np.memmap`; `copy=False` for zero-copy passthrough when already packed | `ak.to_packed(rag)` |
| Strip to plain awkward | `rag.to_ak()` | |

### Record-layout `Ragged` (multi-field)

Build by `ak.zip`-ing existing `Ragged`s that share offsets. The result is already a `Ragged` — no manual wrap needed (it's registered via `ak.behavior`):

```python
import awkward as ak
seq_rag   = sp.rag.Ragged.from_lengths(seq_flat,   lengths)  # |S1
score_rag = sp.rag.Ragged.from_lengths(score_flat, lengths)  # f4
batch = ak.zip({"seq": seq_rag, "score": score_rag})         # → Ragged (record layout)
assert isinstance(batch, sp.rag.Ragged)

batch["score"].data[:] *= 2.0   # zero-copy mutation of the flat score buffer
```

The two inputs **must share offsets** (same `lengths` / same `offsets` array) — that's what makes the result a single-ragged-dim record. Mixing mismatched offsets falls back to a plain `ak.Array`.

- `rag.dtype` returns a NumPy *structured* dtype (e.g. `[("seq","S1"),("score","f4")]`), purely as a descriptor — memory is SoA, not AoS.
- `rag.data` and `rag.parts` return **dicts keyed by field name**, not single arrays. Always type-check before indexing.
- `rag["field"]` gives zero-copy single-field access and shares the parent's offsets object. Its `.data` is the flat NumPy buffer for that field.
- `view`, `apply`, and `to_numpy` are **not defined** on record layouts — operate per-field.

### Awkward interop — what you can rely on

`Ragged` registers itself with `ak.behavior` so most awkward APIs Just Work:

- NumPy ufuncs (`np.add`, `np.exp`, etc.) on a `Ragged` return a `Ragged`.
- `ak.zip`, slicing, `ak.fields`, etc. work and produce `Ragged` when the result still has exactly one ragged dim. If a slice produces zero or >1 ragged dims, you get a plain `ak.Array` back.
- `rag.to_packed()` / `sp.rag.to_packed(rag)` is the canonical way to materialize a contiguous, zero-based buffer — Numba-parallelized and safe on `np.memmap`. Use `copy=False` for a zero-copy passthrough when the array is already packed (raises `ValueError` if not). Prefer this over `ak.to_packed(rag)`, which is slower and doesn't handle memmap inputs.
- Don't rely on awkward features outside this contract: strings (use `|S1` bytes), union types, and >1 ragged dim are unsupported by design.

When in doubt, read `python/seqpro/rag/_array.py` — it's ~800 lines and the docstrings are the source of truth. `_gufuncs.py` and `_utils.py` in the same dir are small.

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
| Ragged | `python/seqpro/rag/_array.py` |
| Transforms (pipeline objects) | `python/seqpro/transforms/` |
| BED/GTF | `python/seqpro/bed.py`, `gtf.py` |
| Rust k-shuffle | `src/kshuffle.rs` |
| Tests as usage examples | `tests/` |
| Rendered docs | `site/` (built from `docs/`) |

## Don'ts

- Don't write Python `for` loops over residues or positions in library code. Look for a vectorized op, a Numba kernel in `_numba.py`, or extend one.
- Don't assume an axis — always pass `length_axis` (and `ohe_axis` where relevant) explicitly.
- Don't reach into `Ragged` internals (`_parts`, `__init__` shortcuts) from user code; use `data`, `offsets`, `parts`, `from_lengths`, `from_offsets`, `empty`.
- Don't introduce strings into `Ragged`. ASCII bytes (`|S1`) only.
- Don't add a feature or change a public signature without updating this skill — see CLAUDE.md.
