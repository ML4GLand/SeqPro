# Design: Unified translate non-canonical codon handling (combine PR #40 + #41)

**Date:** 2026-06-05
**Status:** Approved (design), pending implementation plan
**Supersedes:** PR #40 (`fix/translate-handles-non-canonical-codons`) and PR #41 (`feat/on-unknown-policy`) — these are combined into one branch.

## Problem

`AminoAlphabet.translate`'s two Numba kernels (`gufunc_translate` generic scan,
`gufunc_translate_lut` LUT path) **silently corrupt output** for any codon
containing a byte outside `{A, C, G, T}`:

- `gufunc_translate` exits the inner loop without writing `res[0]` on a no-match
  codon. Because `guvectorize` allocates outputs via `np.empty`, the emitted byte
  is whatever was on the page (typically NUL) — silently wrong AA buffers.
- `gufunc_translate_lut` hashes each byte with `(byte >> 1) & 3`, a bijection on
  `{A,C,G,T}` that collides for off-alphabet bytes: `NNN → T`, `\x00\x00\x00 → K`,
  with no signal the input was bad.

This was found processing 1KG-3202 via GenVarLoader → SeqPro translation: 15
protein haplotypes contained NUL bytes mid-sequence that would have been silently
embedded into downstream pLM caches.

PR #40 fixed the corruption (emit `'X'`). PR #41 layered on case-insensitivity
plus a four-mode `on_unknown` policy (`error`/`pad`/`collapse`/`shorten`) and an
`unknown_marker` param, defaulting to `error` (a breaking change).

## Goal

Combine both into one PR while **minimizing API surface** and honoring Python's
"one obvious way to do it." Fix the corruption; add only the parts that genuinely
earn their place; avoid duplicating the existing `validate=` mechanism.

## Decisions

### 1. Case-insensitivity is unconditional (no flag)

Soft-masked lowercase (`acgt`) from RepeatMasker, lowercase FASTA, and GVL
haplotype injection should always translate. There is no use case where
soft-masked nucleotides should *not* translate. Both kernels apply a single
`& 0xDF` upper-casing mask before the canonical-range check. This is one ALU op,
no measurable perf cost, and adds no parameter.

### 2. No `error` mode — `validate=True` is the single fast-fail path

`on_unknown="error"` from #41 duplicated `validate=True` (both raise on bad
input). Per the seqpro convention that **validation is opt-in and front-loaded
via `validate=`**, the `error` mode is dropped. Callers who want deterministic
failure on non-canonical input pass `validate=True`. This is documented, not
re-implemented per feature.

`validate=True` becomes **case-insensitive**: `_check_nuc_bytes` upper-cases via
`& 0xDF` before the alphabet membership test, so lowercase `acgt` passes (it
translates exactly) while `N` and other off-alphabet bytes still raise. This
keeps `validate=True` coherent with what `translate` actually tolerates:
"validation passed ⇒ translation is exact."

### 3. One parameter, invalid states unrepresentable: `unknown: str = "X"`

The only new parameter versus `main`. `unknown_marker`, `collapse`, and `error`
from #41 are removed. Semantics:

| `unknown` value          | behavior                                          | applies to       |
|--------------------------|---------------------------------------------------|------------------|
| any **single char**      | **pad**: emit that char per non-canonical codon   | dense + Ragged   |
| literal **`"drop"`**     | drop non-canonical codons entirely                | dense + Ragged*  |
| anything else            | `ValueError` at the wrapper                       | —                |

\* `"drop"` changes output length, so it always yields a `Ragged` (see §4).

- **Default `"X"`** reproduces PR #40 exactly ⇒ the combined PR is **non-breaking**
  (pure bugfix + additive opt-ins).
- A marker char and "drop" can't be specified together — the single string makes
  the useless `(drop, marker=...)` combination unrepresentable.

### 4. `"drop"` always returns `Ragged`; overloads reflect it

Dropping codons makes per-sequence lengths vary, so `"drop"` returns a `Ragged`
even for dense input. Dense input (only `StrSeqType → NDArray[bytes]`; OHE enters
only via Ragged) + `"drop"`: move `length_axis` to the end, reshape to `(-1, L)`
so every non-length position becomes one Ragged sequence (`n = product of
non-length dims`), run the drop compaction, return `Ragged[np.bytes_]`. Pad
preserves shape, so dense pad stays dense.

Overloads (the `Literal["drop"]` overload precedes the general `str` overload so
a literal `"drop"` call resolves to `Ragged`):

```python
@overload  # dense + drop → Ragged
def translate(self, seqs: StrSeqType, length_axis=..., *,
              nuc_alphabet=..., truncate_stop=..., validate=...,
              unknown: Literal["drop"]) -> Ragged[np.bytes_]: ...
@overload  # dense + pad → dense
def translate(self, seqs: StrSeqType, length_axis=..., *,
              nuc_alphabet=..., truncate_stop=..., validate=...,
              unknown: str = "X") -> NDArray[np.bytes_]: ...
@overload  # Ragged bytes → Ragged bytes (pad or drop)
def translate(self, seqs: Ragged[np.bytes_], length_axis=..., *,
              nuc_alphabet=..., truncate_stop=..., validate=...,
              unknown: str = "X") -> Ragged[np.bytes_]: ...
@overload  # Ragged OHE → Ragged OHE (pad or drop)
def translate(self, seqs: Ragged[np.uint8], length_axis=..., *,
              nuc_alphabet: NucleotideAlphabet, truncate_stop=..., validate=...,
              unknown: str = "X") -> Ragged[np.uint8]: ...
```

Known type-checker limitation: a *runtime* `str` variable (not a literal) on
dense input resolves to the `NDArray` overload. Runtime behavior is still
correct; this is documented.

### 5. `"drop"` compaction is a single Numba kernel

#41's drop/collapse used a Python `for`-loop over sequences plus naive
`np.concatenate` — unacceptable for seqpro hot paths. Replace with **one `@njit`
kernel** that takes `codons`, the translated flat buffer, `offsets`, and
`valid_upper_bytes`; re-checks each codon's canonical-ness inline (`& 0xDF` +
membership), copies kept AAs into one pre-sized output buffer, and emits new
per-sequence lengths in a single pass. New offsets via `cumsum` of those lengths.

This eliminates #41's `_codon_unknown_mask` (`np.isin`), `_shrink_ragged_unknowns`
(Python loop + concat), and the `collapse`/`error` branches entirely. **Pad needs
no mask and no second pass** — the translate kernel already emits the marker for
non-canonical codons.

## Final signature

```python
def translate(
    self,
    seqs,
    length_axis=None,
    *,
    nuc_alphabet=None,
    truncate_stop=False,
    validate=False,
    unknown: str = "X",   # single char = pad with it; "drop" = drop codons (→ Ragged)
) -> NDArray[np.bytes_] | Ragged[np.bytes_] | Ragged[np.uint8]: ...
```

## Implementation outline

### Kernels (`python/seqpro/_numba.py`)
- Keep #40's sentinel fix in both kernels.
- Add `& 0xDF` upper-casing before the canonical check (case-insensitivity).
- Add `marker_byte: uint8` argument to both kernels (replaces hardcoded `88`);
  the wrapper passes `ord(unknown)` for pad, or any byte for drop (dropped anyway).
- New `@njit` drop-compaction kernel (§5).

### Wrapper (`python/seqpro/alphabets/_alphabets.py`)
- Parse `unknown` once at the top:
  - `unknown == "drop"` → drop policy.
  - `len(unknown) == 1` → `marker_byte = ord(unknown)`, pad policy.
  - else → `ValueError`.
- pad (default): run translate kernel with `marker_byte`; done. Dense stays dense.
- drop: run translate kernel, then the compaction kernel; return `Ragged`. For
  dense input, reshape to `(-1, L)` first (§4).
- `_check_nuc_bytes` (and OHE check path) upper-case via `& 0xDF` before the
  membership test, making `validate=True` case-insensitive.
- Update the three overloads to the four in §4.

### Removed code (from #41's branch)
- `OnUnknown` literal type with `collapse`/`error`, `_ON_UNKNOWN_VALUES`,
  `_validate_on_unknown`, `_validate_unknown_marker`, `_codon_unknown_mask`,
  `_shrink_ragged_unknowns`.

## Conventions to record in CLAUDE.md

1. **Validation is opt-in and front-loaded.** Add fast-fail/validation via a
   `validate=` flag (or equivalent opt-in), not per-feature error modes. One
   obvious way to ask "is this input clean?"
2. **No naive numpy in hot paths.** Never use raw Python loops or naive numpy
   (e.g. per-segment `np.concatenate`) where Numba is faster/leaner — unless the
   numpy version is *verifiably* ~equivalent in time and memory. When Numba is a
   poor fit (graph algorithms like k-shuffle), use the Rust/PyO3 extension.

## Docs

- `skills/seqpro/SKILL.md` (repo rule: required for public-signature changes):
  document `unknown=` (single char = pad; `"drop"` = drop, returns Ragged),
  automatic case-insensitivity, and `validate=True` as the fast-fail path.

## Tests (`tests/test_translate.py`)

Port the worthwhile coverage from #40 + #41; drop tests for removed features.

- **Keep:** kernel-level NUL/`N`/mixed/lowercase → marker; every canonical + stop
  codon still correct through both kernels; pad back-compat
  (`unknown="X"` == #40 output); dense + Ragged + OHE paths.
- **New:** `unknown="drop"` on Ragged **and** dense (→ Ragged, reshape semantics);
  drop numba kernel correctness; arbitrary pad char; invalid `unknown` (`"xy"`,
  `""`) → `ValueError`; `validate=True` accepts lowercase, rejects `N`.
- **Delete:** all `error`/`collapse` tests; `unknown_marker` validation tests.

## CHANGELOG

Single combined entry, `fix` + `feat`, **no `BREAKING CHANGE`** (default `"X"`
matches #40):
- `fix(translate)`: non-canonical codons no longer silently corrupt output.
- `feat(translate)`: unconditional case-insensitive codons; `unknown=` param
  (pad char / `"drop"`); `validate=True` now case-insensitive.

## PR mechanics

Combine into one branch off `main`, based on #40's branch
(`fix/translate-handles-non-canonical-codons`, the clean kernel fix), layering
the simplified feature on top. Close #41 and its draft dependency, pointing at
the unified PR. Suggested title: `feat(translate): case-insensitive + unknown codon policy`.

## Non-goals

- No `error`/`collapse` modes, no separate `unknown_marker` parameter.
- No change to `truncate_stop`, `nuc_alphabet`, or `length_axis` semantics.
- No new public functions beyond the one `unknown=` parameter.
