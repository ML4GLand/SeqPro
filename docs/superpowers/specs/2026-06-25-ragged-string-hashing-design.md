# Ragged string hashing (md5 / sha256 / rapidhash)

**Date:** 2026-06-25
**Status:** Approved (design)

## Summary

Add per-element cryptographic and non-cryptographic hashing of the strings held
in a `Ragged` container, backed by a single rayon-parallel Rust kernel. Each
string in the container is hashed independently into a fixed-size digest. The
result mirrors the input's structure above the string level: a regular NumPy
array when there is no outer grouping, or a `Ragged` (reusing the input's outer
offsets) when there is.

Three algorithms are supported: `md5` (16-byte digest), `sha256` (32-byte
digest), and `rapidhash` (8-byte / `uint64`).

## Motivation

Content-addressing, deduplication, and cache keys over large batches of
variable-length sequences are common. Doing this in Python (`hashlib` in a loop,
or per-element `bytes` slicing) violates the project's "no Python loops in hot
paths" rule. The strings are independent, so hashing is embarrassingly parallel
— a natural fit for a GIL-detached rayon kernel.

## Scope

In scope:

- Hashing the strings of a single-leaf string `Ragged`, in **both** string
  representations and at **any** ragged depth the class supports:
  - **Opaque string** leaf (`str_offsets is not None`, dtype `'S'`).
  - **Chars / S1** leaf (`str_offsets is None`, leaf dtype `|S1`) — the
    DNA-sequence form used by `reverse_complement`.
- Algorithms `md5`, `sha256`, `rapidhash`.
- A free function (canonical implementation) and a thin `Ragged` method.

Out of scope (raise, see Error handling):

- Record-layout (`RecordLayout`) Ragged arrays — hash fields individually.
- Numeric (non-byte) leaves.
- Hashing whole groups or the whole container into a single digest (the unit is
  always one string).

## Repo idiom established by this work

Public Ragged/sequence operations live as **free functions in the relevant
`_ops`-style module** (the implementation home); the corresponding `Ragged`
**method is a thin one-line delegator**. This matches `reverse_complement` and
`concatenate`. The convention is added to `CLAUDE.md` (Key Conventions) so future
operations follow it. Existing `to_packed` / `to_padded` are **not** refactored
here (out of scope).

## Public API

```python
# canonical implementation — python/seqpro/rag/_ops.py
def hash(
    rag: Ragged,
    algo: Literal["md5", "sha256", "rapidhash"],
    *,
    seed: int | None = None,
) -> NDArray | Ragged: ...

# thin delegator — python/seqpro/rag/_core.py (Ragged method)
def hash(self, algo, *, seed=None):
    from ._ops import hash as _hash
    return _hash(self, algo, seed=seed)
```

Exported from `python/seqpro/rag/__init__.py` (added to `__all__`).

- `algo`: `"md5" | "sha256" | "rapidhash"`.
- `seed`: valid **only** for `rapidhash`; passed to the portable seeded variant.
  `None` (default) → rapidhash default secrets (seed 0). Supplying a seed with
  `md5`/`sha256` raises `ValueError`.

## The unified delimiter rule

A "string" to hash is always the **innermost run of bytes**, independent of
representation and ragged depth. The input is first packed via `to_packed()`
(normalizing every offsets array to 1-D, zero-based, contiguous). Then:

- **Opaque string** (`_rl.str_offsets is not None`):
  - `delimiters = str_offsets`
  - `outer_offsets = list(_layout.offsets)`  (all of them; the string level sits
    below the offsets levels)
- **Chars / S1** (`str_offsets is None`, leaf dtype `|S1`):
  - `delimiters = _layout.offsets[-1]`  (the innermost char-delimiting axis is
    the string level)
  - `outer_offsets = list(_layout.offsets[:-1])`

In both cases the byte buffer is `_rl.data` viewed as `uint8`, and `delimiters`
index it directly (1 byte per char/element). `N = len(delimiters) - 1` strings
are hashed in packed order. This single contract — `(uint8 buffer, int64
delimiters)` — covers opaque strings, chars, and arbitrary ragged depth with one
kernel.

## Output shape

The output mirrors the input's structure **above** the string level, collapsing
each string to a fixed-size digest leaf (16 bytes for md5, 32 for sha256, scalar
`uint64` for rapidhash).

| Input structure | Output |
|---|---|
| No outer ragged level (flat opaque `(N,)`, or chars R=1 `(N, None)`) | Regular `NDArray`: `(*leading_fixed, N, 16/32)` `uint8`, or `(*leading_fixed, N)` `uint64` |
| One+ outer ragged level (opaque `(G, None)`, chars R=2 `(G, None, None)`, deeper) | `Ragged` reusing `outer_offsets`, digest as fixed-size leaf: crypto → `uint8` leaf with trailing fixed dim `(<outer dims…>, 16/32)`; rapidhash → scalar `uint64` leaf `(<outer dims…>)` |

Construction: the kernel always returns the flat `(N, 16/32)` `uint8` (crypto) or
`(N,)` `uint64` (rapidhash) digest buffer. The wrapper then either:

- returns it directly (reshaped to `(*leading_fixed, N, …)`) when `outer_offsets`
  is empty, or
- wraps it via `Ragged.from_offsets(digest_buffer, out_shape, outer_offsets)`
  when `outer_offsets` is non-empty. The outer offsets already index the `N`
  strings, so they are reused unchanged (zero recomputation). This composes for
  R=3+ into a nested `Ragged`.

**Return type:** `NDArray | Ragged` (a union, matching `to_padded`).

Edge cases:

- Empty container (`N == 0`) → correctly-shaped empty array / Ragged.
- Empty strings (zero-length elements) → hashed normally (well-defined for all
  three algorithms).

## Data flow

1. Python validates input is a byte-leaf string Ragged (see Error handling).
2. `rag.to_packed()` if not already contiguous/zero-based → flat `|S1`/`'S'`
   `data` + 1-D zero-based offsets.
3. Compute `delimiters` and `outer_offsets` per the unified rule.
4. Call the Rust kernel `_ragged_hash(data.view(uint8), delimiters, algo, seed)`.
5. Reshape (regular case) or wrap in `Ragged` (outer-ragged case); return.

## Rust kernel (Approach A: one generic kernel + algo dispatch)

A single `#[pyfunction]` registered in `src/lib.rs`, with core logic in a new
`src/hashing.rs` (and, if shared logic is warranted, `crates/seqpro-core`):

```rust
#[pyfunction]
fn _ragged_hash<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, u8>,
    str_offsets: PyReadonlyArray1<'py, i64>,
    algo: &str,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyAny>>
```

PyO3 boundary rules (per project conventions):

- While attached: take `data: &[u8]` and `offsets: &[i64]` slices.
- Do all compute inside `py.detach(|| ...)`, capturing only the `Ungil` slices.
- Re-attach, then `into_pyarray`.

Compute:

- **md5 / sha256** share one generic function using the RustCrypto `digest::Digest`
  trait:

  ```rust
  fn hash_elems<D: Digest>(data: &[u8], offsets: &[i64]) -> Vec<u8> {
      let n = offsets.len() - 1;
      let out_size = <D as Digest>::output_size();
      let mut out = vec![0u8; n * out_size];
      out.par_chunks_mut(out_size)
         .zip(offsets.par_windows(2))
         .for_each(|(chunk, w)| {
             let d = D::digest(&data[w[0] as usize..w[1] as usize]);
             chunk.copy_from_slice(&d);
         });
      out
  }
  ```

  Returned to Python as a 2-D `Array2<u8>` of shape `(N, out_size)`.

- **rapidhash** — uses the crate's portable/stable hashing path so digests are
  reproducible across runs/machines (matching md5/sha256). Exact symbol pinned to
  the locked crate version (the `rapidhash_v3` family, e.g.
  `rapidhash_v3_seeded(bytes, &secrets)` with secrets derived from `seed`, or the
  unseeded portable one-shot when `seed is None`):

  ```rust
  let hashes: Vec<u64> = (0..n).into_par_iter()
      .map(|i| rapidhash_portable(&data[off[i]..off[i+1]], seed))
      .collect();
  ```

  Returned as a 1-D `Array1<u64>`.

- Return type differs by algorithm, so the function returns `Bound<'py, PyAny>`
  (a 2-D `uint8` array for crypto, a 1-D `uint64` array for rapidhash). The
  Python wrapper knows the expected digest width per algo and handles shaping.
- Unknown `algo` → `PyValueError` (also guarded earlier in Python).

### Dependencies (`Cargo.toml`)

- `md-5 = "0.10"` (RustCrypto)
- `sha2 = "0.10"` (RustCrypto)
- `digest = "0.10"` (the generic `Digest` trait bound)
- `rapidhash = "4"` (portable/stable v3 path)

Strict lint/format gates kept clean (project hygiene); no new `clippy`/`rustfmt`
violations.

## Error handling

| Condition | Result |
|---|---|
| Numeric (non-byte) leaf, e.g. `int32`/`float32` | `ValueError` ("hashing requires string/char data, not numeric") |
| Not a string collection (flat S1, no ragged axis, no `str_offsets`) | `ValueError` |
| Record-layout (`RecordLayout`) Ragged | `NotImplementedError` (hash fields individually) |
| Unknown `algo` | `ValueError` listing valid options |
| `seed` provided for `md5`/`sha256` | `ValueError` |

## Testing (`tests/test_ragged_hash.py`)

Correctness:

- **md5 / sha256** compared element-wise against Python `hashlib`, for:
  - opaque-string input and chars/S1 input;
  - R=1, R=2, and R=3 nesting.
- **rapidhash** via known-answer vectors lifted from the crate's test suite, plus
  determinism properties: identical bytes → identical hash, seed changes output,
  duplicate strings match, distinct strings differ.

Structure preservation:

- R≤1 input → regular `NDArray` of the expected shape.
- R≥2 input → `Ragged` whose output offsets are **identical to** the input's
  `outer_offsets`, with group-by-group digest correctness.
- Crypto leaf carries trailing fixed dim 16/32; rapidhash leaf is scalar `uint64`.

Edge cases:

- Empty container (`N == 0`).
- Empty-string elements (zero-length).
- Single-element container.
- Unpacked / non-contiguous input (exercises the `to_packed()` path).
- Arbitrary / non-UTF8 bytes.
- Leading fixed dims present.

Error cases: numeric ragged, non-collection S1, record ragged, unknown algo,
seed-on-crypto.

## Documentation

- `skills/seqpro/SKILL.md` updated in the same PR (required by `CLAUDE.md` for new
  public features): document `Ragged.hash(...)` / `sp.rag.hash(...)`, the three
  algorithms, output-shape rule, and string-representation coverage.
- `CLAUDE.md` Key Conventions gains the free-function/delegator idiom bullet.
```

## Files touched

- `Cargo.toml` — add `md-5`, `sha2`, `digest`, `rapidhash`.
- `src/lib.rs` — register `_ragged_hash`.
- `src/hashing.rs` (new) — the kernel (and `crates/seqpro-core` if shared logic
  is warranted).
- `python/seqpro/rag/_ops.py` — `hash()` free function (canonical impl).
- `python/seqpro/rag/_core.py` — `Ragged.hash()` delegator.
- `python/seqpro/rag/__init__.py` — export `hash`.
- `tests/test_ragged_hash.py` (new) — tests.
- `skills/seqpro/SKILL.md` — document the feature.
- `CLAUDE.md` — add the idiom convention.
