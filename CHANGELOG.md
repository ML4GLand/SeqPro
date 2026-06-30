## 0.21.1 (2026-06-30)

### Fix

- **rag**: handle indexed empty row in Ragged.to_numpy

## 0.21.0 (2026-06-26)

### Feat

- **rag**: add Ragged.hash / sp.rag.hash (md5/sha256/rapidhash)
- **rust**: add _ragged_hash kernel (md5/sha256/rapidhash, rayon)

## 0.20.0 (2026-06-25)

### Feat

- **rag**: to_numpy(validate=False) skips the uniformity scan
- **rag**: contiguous-slice fast path for record R=2 getitem
- **rag**: contiguous-slice fast path for record R=1 getitem
- **rag**: contiguous-slice fast path for opaque-string getitem
- **rag**: contiguous-slice fast path for R=2 getitem
- **rag**: contiguous-slice fast path for R=1 getitem

### Fix

- **rag**: preserve middle fixed dim in record R=2 slice shape-tail

### Perf

- **rust**: release the GIL during PyO3 kernel compute
- **rag**: lean from_offsets — elide ascontiguousarray, gate size-check behind validate

## 0.19.1 (2026-06-23)

## 0.19.0 (2026-06-23)

### Feat

- **rag**: port reverse_complement to seqpro-core Rust; rag layer numba-free
- **rag**: port to_padded to seqpro-core Rust, drop numba kernel

### Refactor

- **rag**: truthful _ops docstring; guard reverse_complement_inplace empty offsets
- **rust**: extract pyo3-free seqpro-core crate owning the Ragged layout

## 0.18.0 (2026-06-23)

### Feat

- **rag**: reshape/squeeze/to_packed preserve Ragged subclass
- **rag**: __getitem__ preserves subclass on positional indexing
- **rag**: add _with_layout subclass-preserving constructor

### Fix

- **rag**: non-tuple record indexing matches numpy (A[x] == A[(x,)])

## 0.17.0 (2026-06-23)

### Feat

- **rag**: concatenate() along ragged axis (Rust kernel)
- **rag**: to_packed() on record and opaque-string-under-axis Ragged
- wire rag-gate pixi task; record throughput gate outcome
- string-conversion op cells for Ragged throughput gate
- nested R=2 op cells for Ragged throughput gate
- record (SoA) op cells for Ragged throughput gate
- single-level op cells for Ragged throughput gate
- harness core for rust-vs-awkward Ragged throughput gate
- _ingest bridge for R=2 + string-under-axis (oracle interop)
- nested record indexing + record-aware to_packed/to_padded/to_numpy
- nested + string-under-axis record fields sharing full offsets list
- R=2 lengths/squeeze/reshape
- per-axis nested to_padded + rectangular to_numpy (R=2); trailing-dim support in to_padded
- nested to_packed via nested_pack kernel
- nested_pack Rust kernel + binding for two-level pack
- per-group inner mask/int-array indexing via nested_gather
- nested_gather Rust kernel + binding for per-group middle selection
- per-group inner int/slice indexing (rag[:, k], rag[:, a:b])
- R=2 tuple indexing + leaf access via peel chaining
- R=2 outer-row indexing (lazy gather, peel to 1-level)
- string-under-axis leaf + nested to_chars/to_strings
- nested constructors from_offsets(list)/from_lengths(tuple) for R=2
- validate R=2 nested ragged layouts; cap at R<=2
- ingest/emit record layouts via awkward bridge (oracle interop)
- record to_numpy/to_padded dicts; raise view/ufunc on records
- record-aware to_packed (one shared packed offsets across fields)
- per-field squeeze/reshape on record Ragged
- record row-axis indexing (slice/mask -> record, int -> dict)
- record field access (key/attr) and __setitem__ mutation
- record-branch properties (data/dtype/offsets/shape/fields/state)
- add Ragged.from_fields record constructor and rag.zip alias
- add RecordLayout value object and validation arm
- add zero-copy to_chars/to_strings between opaque-string and char Ragged
- disambiguate opaque-string vs char layout by presence of None in shape
- report np.dtype('S') for opaque-string Ragged (string/char duality)
- **rust**: ragged_validate and ragged_select kernels
- **rag**: awkward ingestion and to_ak shim
- **rag**: to_numpy, to_packed, to_padded
- **rag**: squeeze and reshape on regular dims
- **rag**: element-wise ufunc interop
- **rag**: __getitem__ indexing and slicing
- **rag**: state predicates and view
- **rag**: Ragged constructors and core properties
- **rag**: RaggedLayout value object + validation
- **translate**: self-contained Rust OHE<->AA path with native drop
- **translate**: route truncate_stop through Rust
- **translate**: route unknown=drop compaction through Rust
- **translate**: route ragged bytes pad path through Rust
- **translate**: route dense pad path through Rust
- **rust**: add codon-stride translate kernels
- **tokenize**: route ragged path through Rust, drop Numba gather
- **tokenize**: route dense path through Rust _tokenize
- **rust**: add tokenize LUT gather kernel

### Fix

- **rag**: to_ak() on multi-leading-axis record Ragged
- **rag**: _core.Ragged/tokenize fixes found via genvarformer audit
- **rag**: add __len__, np.newaxis support, and element-wise fancy indexing to _core.Ragged
- **rag**: _ops.to_padded record detection works for both _array and _core backends
- **rag**: _core.to_numpy returns dict for records (restore designed contract); update _array-era test
- **rag**: correct _core getitem tuple routing for rag_dim==1 found via SeqPro own-suite audit
- **rag**: port _ops + _core to_packed/is_rag_dtype/to_numpy to _core object model
- **rag**: precise _core getitem routing — _core contract + genoray parity both green
- **rag**: repair _core.Ragged getitem regressions (string leaf, to_strings, r2 int array)
- **rag**: match _array is_base semantics for non-ndarray base
- **rag**: _core.Ragged fixes found via genoray audit
- **bench**: time to_packed on unpacked input so single-level pack compares equal work
- forward --repeats CLI flag into time_callable
- R=2 is_contiguous checks all offset levels; guard string-under-axis to_packed/to_numpy
- panic-safety in nested_pack (checked_mul, elem>0, o0-range guard)
- reject negative inner-slice bounds in rag[:, a:b]
- -O-safe record guards on is_string/to_chars/to_strings; note latent S4 over-acceptance
- raise TypeError on np.asarray of a record Ragged (-O-safe, was a bare assert)
- **rag**: reject mismatched boolean masks and bounds-check ragged_select
- **translate**: handle empty dense input in unknown=drop path

### Refactor

- **rag**: delete awkward _array backend; relocate interop to _ak_interop
- resolve clippy findings for -D warnings
- **rag**: route index/validate hot paths through Rust
- **rag**: use NDArrayOperatorsMixin for operator dunders
- **tokenize**: restore np.take/Numba path; Rust port regressed vs baseline

### Perf

- validate pack output size up-front; test parallel pack path
- Rust single-level pack kernel for to_packed (parallel gather)
- opt-in Ragged validation (default off) + slice indexing fast-path
- drop redundant per-field copy in record to_packed; align record is_base with single-level one-indirection rule
- **translate**: optimize LUT gather kernel (validity table + coarse chunking)
- **translate**: add rayon parallel path for large inputs

## 0.16.0 (2026-06-14)

### Feat

- **tokenize**: add parallel escape hatch overriding the size heuristic

## 0.15.2 (2026-06-13)

### Fix

- **tokenize**: accept readonly input on parallel path; guard int32 out=

### Perf

- **tokenize**: parallel Numba LUT gather with small-input np.take fast path
- **tokenize**: use 256-entry LUT gather instead of linear scan

## 0.15.1 (2026-06-08)

### Fix

- **rag**: traverse IndexedArray in unbox/_extract_list_offsets

## 0.15.0 (2026-06-07)

### Feat

- **translate**: Ragged pad + drop paths via unknown=
- **translate**: add unknown= param (pad path) and overloads
- **translate**: case-insensitive validate=True
- **translate**: add _nb_drop_unknown_codons compaction kernel

### Fix

- **translate**: update seqpro.xr translate call for marker_byte arg
- **translate**: restrict unknown marker to ASCII; document drop dense-table assumption
- **translate**: marker sentinel + case-folding in gufunc_translate_lut
- **translate**: marker sentinel + case-folding in gufunc_translate

## 0.14.0 (2026-06-01)

### Feat

- **rag**: add Ragged.to_packed method and export to_packed
- **rag**: add Numba-parallelized to_packed for flat layouts

### Fix

- **ci**: check out release tag and drop duplicate release trigger
- **rag**: produce canonical list-of-records layout in to_packed

### Refactor

- **rag**: make Ragged.to_packed copy keyword-only; add multibyte-trailing test
- **rag**: drop unused as_contiguous packing path in unbox

### Perf

- **rag**: microbenchmark to_packed throughput vs ak.to_packed
- **rag**: use to_packed at internal ak.to_packed call sites

## 0.13.0 (2026-06-01)

### Feat

- **rag**: guard to_padded for record/trailing-dim/non-contiguous
- **rag**: to_padded fixed-length pad and truncate
- **rag**: flat-buffer to_padded (pad to batch max)

## 0.12.1 (2026-05-31)

### Fix

- add attrs

### Perf

- **rag**: prototype flat-buffer ragged reverse_complement

## 0.12.0 (2026-05-29)

### Feat

- **translate**: validate OHE inputs are one-hot
- **translate**: add validate flag for nucleotide input checking

### Fix

- **ci**: merge runs when publish succeeded after a transitive skip
- **ci**: allow publish to run when bump was skipped
- **ci**: publish.yaml checks out the bumped tag, not the dispatch SHA

### Refactor

- **translate**: gate LUT build with predicate, drop exception control-flow
- **translate**: single source of truth for codon LUT index

### Perf

- **translate**: route Ragged path through codon LUT
- **translate**: O(1) LUT codon→AA lookup — 179× speedup on bench

## 0.11.1 (2026-05-21)

### Feat

- **ci**: add release-pipeline.yaml orchestrator
- add on_path flag to Vertex and path stack to ShuffleBuffers
- specialize k=2 path to skip LUT and codes lookup
- add ShuffleBuffers with sparse-reset LUT
- add KmerIndex trait with DirectLut and HashLut impls
- add kmer_encode module with rolling integer encoder
- ragged support for ohe, decode_ohe, tokenize, decode_tokens

### Fix

- **ci**: grant orchestrator permissions required by called workflows
- **ci**: make merge.yaml workflow_call-able, drop workflow_run
- **ci**: make publish.yaml workflow_call-able, drop workflow_run
- **ci**: make release.yaml workflow_call-able with dry_run
- **ci**: make bump.yaml workflow_call-able with increment and dry_run
- derive per-row seeds so batches get independent shuffles
- support arbitrary leading/trailing dims across public functions
- types

### Refactor

- pool LUT and reuse buffers via ShuffleBuffers
- rewrite k_shuffle1 with KmerIndex strategy + u32 Vertex

### Perf

- replace two-pass Wilson with single-pass loop-erased random walk
- add criterion benchmark for k_shuffle

## 0.11.0 (2026-05-07)

### Feat

- Ragged type string shows as `var * Ragged[dtype]`
- remove Ragged.apply method
- add polars-config-meta, set coordinate_system_zero_based in set_schema and bed.read()
- add config_meta to bed.read(), export detect_schema and set_schema from bed
- bed.to_pyr() accepts any narwhals-supported frame
- narwhalify bed.with_len()
- narwhalify bed.sort()
- add set_schema
- add detect_schema
- add CoordSchema, _SCHEMAS, and _resolve_schema
- add narwhals as hard dependency
- Ragged.reshape supports record layouts
- Ragged.squeeze supports record layouts
- clear NotImplementedError for view/apply/to_numpy on record Ragged
- Ragged.parts returns offsets-sharing field dict for records
- Ragged.data returns zero-copy field dict for record layouts
- Ragged.dtype returns field dict for record layouts
- Ragged[np.void] record array support with zero-copy field offsets
- support Ragged[np.void] record layout — offsets and data
- Ragged.__init__ skips unbox() for record layouts

### Fix

- import TypeIs from extensions
- Ragged dtype and typing behavior
- detect_schema call order in set_schema, sort docstring and unique pattern
- bed.sort() temp column name collision and null chrom handling
- type var rename, is_rag_dtype record support, is_contiguous/is_base record fixes
- lazy _parts init for Ragged created via ak behavior dispatch

### Refactor

- Ragged record dtype uses np.dtype structured dtype instead of dict

## 0.10.0 (2026-05-04)

### Feat

- extend AA.translate to accept Ragged[bytes] and Ragged[uint8] inputs

### Fix

- handling RecordArrays

## 0.9.0 (2025-11-09)

### Feat

- **perf**: faster reverse complementing and option to pass pre-alloc output

## 0.8.2 (2025-10-22)

### Fix

- wrong shape of Ragged[bytes].to_numpy()

## 0.8.1 (2025-10-22)

### Fix

- wrong shape of Ragged[bytes].to_numpy()

## 0.8.0 (2025-10-21)

### Feat

- k_shuffle for arbitrary alphabets

### Fix

- change ragged subtype display name to match class name
- default unknown character for AminoAlphabet.decode_ohe should be X not N
- Ragged from Array with parameters on non-List* content. perf: no RefCell in kshuffle

## 0.7.1 (2025-08-22)

### Fix

- add rlib crate type for more build flexibility

## 0.7.0 (2025-08-22)

### Feat

- is_rag_dtype. fix: 2d offsets length calc

### Fix

- catch depcreation warning from pyranges for having an old packaging workflow
- add informative errors for Ragged.from_offsets

## 0.6.1 (2025-06-21)

### Fix

- gtf attributes need not be quoted, fix regex to handle this flexibly.

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
