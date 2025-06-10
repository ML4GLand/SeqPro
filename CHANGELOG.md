## 0.6.0 (2025-06-10)

### Fix

- compute lengths for non-contiguous offsets
- make Ragged.offsets layout match that of awkward for zero-copy conversion of non-contiguous data

## 0.5.0 (2025-05-24)

### Feat

- add all python ops and update repr to more clearly indicate type
- ufunc and add support

## 0.4.2 (2025-05-22)

### Fix

- keep view as S1 for ragged -> numpy

## 0.4.1 (2025-05-22)

### Fix

- expose gtf from top level

## 0.4.0 (2025-05-22)

### Feat

- rename bed functions to be shorter. add basic gtf functions.

## 0.3.2 (2025-05-07)

### Fix

- raise informative error for np.void ragged -> awk conversion, return correct value for 1D array result

## 0.3.1 (2025-04-25)

### Fix

- support gzipped bedlike files

## 0.3.0 (2025-04-25)

### Feat

- add __getitem__ for ragged that matches awkward

### Fix

- helper methods for empty ragged arrays

## 0.2.4 (2025-04-19)

### Fix

- maintain_order of bed rows with same chrom, start, and end

## 0.2.3 (2025-04-19)

### Fix

- move length_to_offsets to _ragged

## 0.2.2 (2025-04-19)

### Fix

- add and pass tests for len_to_offsets
- add experimental Ragged API

## v0.2.1 (2025-04-17)

### Fix

- adds bed.sort for natural sorting of chromosomes

## v0.2.0 (2025-03-17)

### Feat

- add from_pyranges and test roundtrip
- add from_pyranges
- bed functions and tests.
- let seed be int or generator everywhere.

## v0.1.16 (2025-01-09)

### Fix

- test full codon table
- translating DNA to AA

## v0.1.15 (2024-11-04)

### Fix

- add __version__ attribute to module

## v0.1.14 (2024-11-03)

### Fix

- bug in jitter that would not jitter the leftmost jitter axis.

## v0.1.13 (2024-08-19)

### Feat

- tokenize transform.
- rename _check_axes to not be private. feat!: change API of array_slice for dependency injection
- more convenient tokenizer API
- expose transforms API.
- initial transforms API.

### Fix

- add cdylib crate-type
- bump python ABI version
- cascade refactors. feat!: cast_seqs raises an error when given an empty string.
- test tokenizer
- imports and untokenize typo.

## v0.1.11 (2024-02-02)

### Fix

- relax dependency versions.

## v0.1.10 (2024-02-02)

### Feat

- python bindings to rust k_shuffle.
- kshuffle.rs passes tests

### Fix

- set version
- add k_shuffle python tests
- rust kshuffle working again.
- bump dependencies
- mark seqpro as typed.
- remove cython dependency.

## v0.1.9 (2023-12-20)

### Fix

- bug with jitter when using numba helper. Now uses a vectorized pure NumPy implementation.

## v0.1.8 (2023-11-20)

## v0.1.7 (2023-11-19)

### Feat

- add license.
- add license.
- initial support for one hot encoding amino acid sequences.

## v0.1.6 (2023-11-15)

### Fix

- convert length axes to positive values throughout.

## v0.1.5 (2023-11-06)

### Feat

- add tests for analyzers and make them all pass.
- bump version.
- numba accelerated ohe_to_bytes, vectorized random sequence generation. fix: bug in k_shuffle for ohe sequences.
- move alphabets to separate module for type annotated premade alphabets.
- remove shuffle() and merge the functionality into k_shuffle(), efficiently handling the case where k == 1.
- amino acid alphabets and translating from DNA to AA.
- expose cast_seqs at top level.
- bin_coverage() for binning per-base values along sequences into lower resolutions.
- first pass at vectorized refactor. first xarray port for ohe().
- first pass at vectorized refactor. first xarray port for ohe().

### Fix

- extraneous casting in gc_content and nucleotide_content. enh: xarray version of bin_coverage.
- k_shuffle() correct length in iterator. fix: _check_axes() for np.str_ input.
