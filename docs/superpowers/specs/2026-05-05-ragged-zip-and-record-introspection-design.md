# Ragged: `ak.zip` Support and Record-Layout Introspection

Date: 2026-05-05
Status: Approved (ready for implementation plan)

## Context

`Ragged[np.void]` (record-layout) support landed recently (see `2026-05-04-ragged-record-array-design.md`). That work made `Ragged.__init__` accept record layouts and gave `__getitem__` zero-copy field access. However:

1. The class docstring still warns that records, fields, and `ak.zip` are untested/experimental.
2. The introspection trio (`dtype`, `data`, `parts`) raises `TypeError` on record layouts, forcing callers to branch on internal state.
3. Shape-only transforms (`squeeze`, `reshape`) currently raise on records even though their semantics extend cleanly across fields.
4. There is no test coverage for `ak.zip` of `Ragged` inputs, despite that being the natural way to construct a record-layout `Ragged`.

## Goals

- Make `ak.zip` of `Ragged` inputs a supported, tested operation — both auto-coerced and explicit-wrap forms.
- Replace the "raises on records" pattern in `dtype`/`data`/`parts` with field-keyed dicts so the public API is not state-conditional.
- Extend `squeeze` and `reshape` to record layouts (cheap: offsets shared, shape adjusted).
- Remove the experimental disclaimer from the class docstring.

## Non-Goals

- `apply`, `view`, and `to_numpy` on record layouts. These have ambiguous or heterogeneous semantics across fields and remain `NotImplementedError` (with a message pointing to per-field access).
- Union types — still unsupported, still untested.
- `ak.zip` of mixed `Ragged` + plain `ak.Array` inputs — out of scope; document as untested.

## API Changes

### Introspection trio (record layout only)

| Property | Non-record (unchanged) | Record (new) |
|---|---|---|
| `dtype` | `np.dtype` | `dict[str, np.dtype]` |
| `data` | `NDArray` | `dict[str, NDArray]` (zero-copy field views) |
| `parts` | `RagParts` | `dict[str, RagParts]` (all sharing one `offsets` ndarray) |

Field order in the returned dicts matches the awkward `RecordArray.fields` order (insertion order from layout).

### Methods on record layout

| Method | Record behavior |
|---|---|
| `squeeze`, `reshape` | Works. Compute once on shared offsets/shape; apply field-wise zero-copy. |
| `view`, `apply`, `to_numpy` | `NotImplementedError` with message pointing to per-field access. |
| `__getitem__(field)` | Already works — unchanged. |
| `__setitem__(field, value)` | Inherited from `ak.Array`; documented as the supported way to update a field (e.g., `rag["field0"] = rag.field0.view(np.uint32)`). |

### `ak.zip` paths

- **Path A (auto):** `ak.zip({"a": r1, "b": r2})` returns a `Ragged` directly via awkward's behavior dispatch, if dispatch survives the operation. To be verified during implementation; if it doesn't survive cleanly, drop A and document B as canonical.
- **Path B (explicit):** `Ragged(ak.zip({"a": r1, "b": r2}))` — works today via existing `_is_record_layout` detection in `__init__`. Always supported.

## Internal Model

- Keep `_parts = None` as the record sentinel internally.
- Build the `dtype`/`data`/`parts` dicts on demand by walking the `RecordArray` once, then cache the dict on the instance (analogous to existing `_offsets_cache`). Use `object.__setattr__` to bypass any `ak.Array` setattr interception.
- All `RagParts` in the cached `parts` dict share the same `offsets` ndarray returned by the `offsets` property — no copies.

## Tests

Add to `tests/test_ragged.py`:

### `TestRecordRagged` (extend existing)
- `dtype` returns dict with correct field order and per-field dtypes.
- `data` returns dict of zero-copy ndarrays (assert `arr.base is not None` and field shape consistent with offsets).
- `parts` returns dict of `RagParts`, each sharing `offsets` with the parent (`is` identity).
- `squeeze` and `reshape` operate on a multi-axis record `Ragged` and preserve field structure + zero-copy offsets.
- `view`, `apply`, `to_numpy` raise `NotImplementedError` on records.
- Field reassignment: `rag["field0"] = rag["field0"].view(...)` round-trips correctly.

### `TestZip` (new)
- `ak.zip({"a": r1, "b": r2})` produces a `Ragged` (auto path) — if behavior dispatch carries through; otherwise mark as `xfail` with note.
- `Ragged(ak.zip({"a": r1, "b": r2}))` always produces a `Ragged` (explicit path).
- Three-field zip works.
- `ak.zip` with `depth_limit=1` works for nested-Ragged inputs.
- Field-order in resulting `Ragged` matches input dict order.
- Length-mismatched inputs: capture current awkward behavior (likely raises during zip).

## Docstring

Remove from `Ragged` class docstring:

> Ragged arrays are not tested with support for Awkward records/fields or union types. Functionality that appears to work with these features may be experimental. Recommended to use depth_limit=1 when using ak.zip with one or more Ragged arrays as input.

Replace with:

> Record-layout Ragged arrays (produced by `ak.zip` of Ragged inputs or by passing a record-layout `ak.Array`) return field-keyed dicts from `dtype`, `data`, and `parts`. Use `rag["field"]` for zero-copy single-field access. `view`, `apply`, and `to_numpy` are not defined on record layouts; access individual fields. Union types remain unsupported.

## Risks / Open Questions

- Whether awkward's behavior dispatch carries `__list__=Ragged` through `ak.zip`. Resolved during implementation by writing the test first; fall back to documenting Path B only if A fails.
- `ak.Array.__setattr__` interception — already encountered in the offsets cache; `object.__setattr__` is the workaround pattern.
- Whether existing `_is_record_layout` detection covers all layouts produced by `ak.zip` (e.g., `with_name` variants). Verify with the test suite.
