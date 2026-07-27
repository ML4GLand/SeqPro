# String-under-axis integer indexing (issue #71) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make integer indexing on a string-under-axis `Ragged` return one element per string instead of silently concatenating the whole group into one blob.

**Architecture:** Integer indexing peels one real ragged level. For a string-under-axis leaf (`offsets` non-empty **and** `str_offsets` set) the peel lands on the standalone opaque-string layout (`offsets == []`, `str_offsets` set, `shape == (k,)`) — which Spec C already defines as the zero-real-level special case of the same layout. Two call sites construct this: the plain `__getitem__` integer branch and the record-row integer branch. Both mirror the narrowing that `_slice_contig_string` already performs for slices.

**Tech Stack:** Python 3.9+, NumPy, pytest. Pure Python layer — no Rust (`src/`, `crates/`) changes.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-ragged-string-getitem-design.md`.
- Data must stay **zero-copy**: the result's `data` is a view of the parent's buffer. Only the small `(k+1,)` offsets slice is copied (to rebase to zero, which `is_contiguous` requires for opaque strings — `python/seqpro/rag/_core.py:319-321`).
- The **standalone/flat** opaque-string case (`offsets == []`) is unchanged: `flat[0]` still returns `bytes`. The `self._layout.offsets` guard distinguishes it.
- No Python loops over elements — this is a per-batch-item accessor (repo rule: "No naive NumPy in hot paths").
- Breaking change. `major_version_zero = true` in `pyproject.toml:64`, so a `!` conventional commit on 0.x produces a **minor** bump (0.21.2 → 0.22.0), which is what the spec calls for. Do **not** hand-edit `CHANGELOG.md` — commitizen generates it on bump.
- Test command (run from the worktree root; the worktree has no `.pixi` of its own, so borrow the main checkout's dev env and point `PYTHONPATH` at the worktree source):

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/test_ragged_core.py -q
```

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `python/seqpro/rag/_core.py:722-731` | `Ragged.__getitem__` integer branch — plain string-under-axis | Modify (Task 1) |
| `python/seqpro/rag/_core.py:1216-1226` | `_getitem_record_rows` integer branch — opaque-string record fields | Modify (Task 2) |
| `tests/test_ragged_core.py` | Existing home of the string-under-axis test section (`test_string_under_axis_integer_index` at line 733) | Modify + add (Tasks 1, 2) |
| `skills/seqpro/SKILL.md` | Public API skill doc; CLAUDE.md requires an update for any breaking change | Modify (Task 3) |

`_getitem_record_rows_r2` (`_core.py:1243`) delegates to `Ragged(fl)[where]` and inherits Task 1's fix — no separate change, but Task 2 covers it with a test.

---

### Task 1: Plain string-under-axis integer indexing

**Files:**
- Modify: `python/seqpro/rag/_core.py:722-731`
- Test: `tests/test_ragged_core.py` (replace `test_string_under_axis_integer_index` at line 733; add new tests after it)

**Interfaces:**
- Consumes: `RaggedLayout(data=..., offsets=..., shape=..., str_offsets=...)` from `python/seqpro/rag/_layout.py` (already imported at `_core.py:10`).
- Produces: `Ragged.__getitem__(int)` on a string-under-axis `Ragged` returns `Ragged` with `offsets == []`, `str_offsets` set, `shape == (k,)`, `is_string is True`. Task 2 relies on this same construction shape.

- [ ] **Step 1: Replace the test that pins the old behavior**

`tests/test_ragged_core.py:732-741` currently reads:

```python
def test_string_under_axis_integer_index():
    rag = Ragged.from_offsets(
        np.frombuffer(b"TTGG", "S1"),
        (2, None),
        np.array([0, 1, 2]),
        str_offsets=np.array([0, 2, 4]),
    )
    assert rag[0] == b"TT"
    assert rag[1] == b"GG"
```

It uses exactly one string per group, so it can never distinguish concatenation from per-string indexing — that is why the bug survived. Replace it with:

```python
def test_string_under_axis_integer_index():
    """Peeling one group yields the standalone opaque-string layout (Spec C)."""
    rag = Ragged.from_offsets(
        np.frombuffer(b"TTGG", "S1"),
        (2, None),
        np.array([0, 1, 2]),
        str_offsets=np.array([0, 2, 4]),
    )
    row = rag[0]
    assert isinstance(row, Ragged)
    assert row.is_string and row.shape == (1,)
    assert len(row) == 1
    assert row[0] == b"TT"
    assert rag[1][0] == b"GG"
```

- [ ] **Step 2: Add the failing tests from the issue**

Append immediately after the test above:

```python
def _issue71_pair():
    """String-under-axis and numeric Ragged sharing one offsets object.

    Groups: 0 -> ('A', 'GG'), 1 -> ('TC',).
    """
    data = np.frombuffer(b"AGGTC", dtype="S1")
    outer = np.array([0, 2, 3], dtype=OFFSET_TYPE)  # group  -> string index
    inner = np.array([0, 1, 3, 5], dtype=OFFSET_TYPE)  # string -> byte index
    s = Ragged.from_offsets(data, (2, None), outer, str_offsets=inner)
    n = Ragged.from_offsets(np.array([10, 20, 30], dtype=np.int32), (2, None), outer)
    return s, n


def test_string_under_axis_index_preserves_boundaries():
    """Issue #71: interior str_offsets boundaries must survive an integer index."""
    s, _ = _issue71_pair()
    row = s[0]
    assert len(row) == 2
    assert row[0] == b"A"
    assert row[1] == b"GG"
    assert list(s[1]) == [b"TC"]


def test_string_under_axis_index_matches_lengths():
    """len(s[i]) must equal s.lengths[i] and the numeric row length."""
    s, n = _issue71_pair()
    for i in range(len(s)):
        assert len(s[i]) == int(s.lengths[i])
        assert len(s[i]) == len(n[i])


def test_string_under_axis_index_is_zero_copy():
    s, _ = _issue71_pair()
    assert np.shares_memory(s[0].data, s.data)


def test_string_under_axis_index_empty_group():
    """A group holding zero strings peels to a length-0 result."""
    rag = Ragged.from_offsets(
        np.frombuffer(b"AC", "S1"),
        (2, None),
        np.array([0, 0, 2], dtype=OFFSET_TYPE),  # group 0 empty
        str_offsets=np.array([0, 1, 2], dtype=OFFSET_TYPE),
    )
    assert len(rag[0]) == 0
    assert len(rag[1]) == 2


def test_string_under_axis_index_agrees_with_to_chars():
    s, _ = _issue71_pair()
    chars = s.to_chars()
    for i in range(len(s)):
        for j in range(len(s[i])):
            assert s[i][j] == chars[i][j].tobytes()


def test_string_under_axis_index_negative_and_oob():
    s, _ = _issue71_pair()
    assert list(s[-1]) == [b"TC"]
    with pytest.raises(IndexError):
        s[5]


def test_string_under_axis_index_multidim():
    """(batch, ploidy, ~variants): reaches the flat branch via _getitem_multidim."""
    data = np.frombuffer(b"AGGTCNNAC", dtype="S1")
    o0 = np.array([0, 2, 3, 4, 6], dtype=OFFSET_TYPE)  # 4 segments -> string idx
    i0 = np.array([0, 1, 3, 5, 7, 9], dtype=OFFSET_TYPE)  # 6 boundaries -> bytes
    rag = Ragged.from_offsets(data, (2, 2, None), o0, str_offsets=i0)
    row = rag[0]  # -> (2, None) string-under-axis
    assert list(row[0]) == [b"A", b"GG"]
    assert list(row[1]) == [b"TC"]


def test_standalone_string_index_still_returns_bytes():
    """Regression guard: the zero-real-level case is the terminal peel."""
    flat = Ragged.from_offsets(
        np.frombuffer(b"cathithere", "S1"), (3,), np.array([0, 3, 5, 10])
    )
    assert flat[0] == b"cat"
    assert flat[-1] == b"there"
    assert list(flat) == [b"cat", b"hi", b"there"]
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/test_ragged_core.py -q -k "string_under_axis or standalone_string"
```

Expected: `test_string_under_axis_index_preserves_boundaries` fails (`len(row)` raises `TypeError`/returns 3 rather than 2 — the current return is `bytes`), `test_string_under_axis_integer_index` fails on `isinstance(row, Ragged)`, and the other new per-string tests fail. `test_standalone_string_index_still_returns_bytes` must PASS already.

- [ ] **Step 4: Implement the fix**

In `python/seqpro/rag/_core.py`, replace the integer branch at lines 722-731:

```python
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            if self._rl.str_offsets is not None and self._layout.offsets:
                # string-under-axis: outer offsets index variants -> map to bytes via str_offsets
                so = self._rl.str_offsets
                return self._rl.data[int(so[lo]) : int(so[hi])].tobytes()
            row = self._rl.data[lo:hi]
            if self._rl.is_string:
                return row.tobytes()
            return row
```

with:

```python
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            if self._rl.str_offsets is not None and self._layout.offsets:
                # string-under-axis: peel the real level -> standalone opaque
                # string (k,), preserving the per-string boundaries that live in
                # str_offsets. Concatenating here would drop them (issue #71).
                return Ragged(_peel_string_row(self._rl, lo, hi))
            row = self._rl.data[lo:hi]
            if self._rl.is_string:
                return row.tobytes()
            return row
```

Then add this module-level helper next to the other layout helpers — place it immediately above `class Ragged` in `python/seqpro/rag/_core.py` (Task 2 reuses it, which is why it is a free function rather than a method: `_getitem_record_rows` operates on per-field `RaggedLayout`s, not on `self`):

```python
def _peel_string_row(
    rl: "RaggedLayout[Any]", lo: int, hi: int
) -> "RaggedLayout[Any]":
    """Peel strings ``[lo, hi)`` off a string-under-axis leaf.

    Returns the standalone opaque-string layout (``offsets == []``,
    ``str_offsets`` set, ``shape == (hi - lo,)``) — the zero-real-level special
    case of string-under-axis (Spec C Section 2). The data buffer is a view;
    only the ``(k + 1,)`` offsets slice is copied, rebased to zero as
    ``is_contiguous`` requires.
    """
    so = rl.str_offsets
    assert so is not None  # caller guarantees a string leaf
    b0 = int(so[lo])
    return RaggedLayout(
        data=rl.data[b0 : int(so[hi])],
        offsets=[],
        shape=(hi - lo,),
        str_offsets=so[lo : hi + 1] - b0,
    )
```

Note `lo`/`hi` are indices in **string** space, so `so[lo : hi + 1]` is the right slice whether `offsets[0]` is 1-D canonical or a lazy `(2, M)` gather layout — `_starts_stops()` normalizes both.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/test_ragged_core.py -q -k "string_under_axis or standalone_string"
```

Expected: all PASS.

- [ ] **Step 6: Run the full ragged suite for regressions**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/ -q
```

Expected: no new failures. If another test asserts the old concatenating behavior, judge it the same way as `test_string_under_axis_integer_index` — update it to index one level deeper, and note it in the commit body.

- [ ] **Step 7: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core.py
git commit -m "fix(rag)!: preserve string boundaries when indexing string-under-axis

Integer indexing on a string-under-axis Ragged concatenated the whole
group into one bytes, dropping the per-string boundaries already held in
str_offsets. It now peels to the standalone opaque-string layout (k,),
so len(s[i]) == s.lengths[i] and s[i][j] is one string.

BREAKING CHANGE: Ragged.__getitem__ with an integer on a string-under-axis
array returns a Ragged of strings, not one concatenated bytes. Use
b\"\".join(s[i]) for the old value.

Refs #71"
```

---

### Task 2: Record-layout string fields

**Files:**
- Modify: `python/seqpro/rag/_core.py:1216-1226`
- Test: `tests/test_ragged_core_records.py`

**Interfaces:**
- Consumes: `_peel_string_row(rl, lo, hi) -> RaggedLayout` from Task 1.
- Produces: `_getitem_record_rows` integer branch returns `dict[str, NDArray | Ragged]` where opaque-string fields are `Ragged` (standalone opaque-string layout) and numeric/char fields stay `ndarray`; every entry has length `hi - lo`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ragged_core_records.py`:

```python
def _issue71_record():
    """Record with an opaque-string field and a numeric field sharing offsets.

    Groups: 0 -> ('A', 'GG') / starts (1, 2), 1 -> ('TC',) / starts (3,).
    Group 0's second allele is multi-byte, so a 1-byte coincidence cannot
    mask a regression.
    """
    outer = np.array([0, 2, 3], dtype=OFFSET_TYPE)
    alt = Ragged.from_offsets(
        np.frombuffer(b"AGGTC", dtype="S1"),
        (2, None),
        outer,
        str_offsets=np.array([0, 1, 3, 5], dtype=OFFSET_TYPE),
    )
    start = Ragged.from_offsets(np.array([1, 2, 3], dtype=np.int32), (2, None), outer)
    return Ragged.from_fields({"alt": alt, "start": start})


def test_record_row_string_field_preserves_boundaries():
    """Issue #71: a string field peeled from a record row keeps its boundaries."""
    row = _issue71_record()[0]
    assert isinstance(row, dict)
    assert list(row["alt"]) == [b"A", b"GG"]
    np.testing.assert_array_equal(row["start"], np.array([1, 2], dtype=np.int32))


def test_record_row_fields_have_matching_lengths():
    """Every field of a peeled row must have the same length, so zip aligns."""
    rec = _issue71_record()
    for i in range(len(rec)):
        row = rec[i]
        assert len(row["alt"]) == len(row["start"])
    row0 = rec[0]
    assert list(zip(row0["start"], row0["alt"])) == [(1, b"A"), (2, b"GG")]


def test_record_row_string_field_is_zero_copy():
    rec = _issue71_record()
    assert np.shares_memory(rec[0]["alt"].data, rec["alt"].data)


def test_record_multidim_row_string_field_preserves_boundaries():
    """(batch, ploidy, ~variants) record: rec[0][h] routes via _getitem_record_rows_r2."""
    outer = np.array([0, 2, 3, 4, 6], dtype=OFFSET_TYPE)
    alt = Ragged.from_offsets(
        np.frombuffer(b"AGGTCNNAC", dtype="S1"),
        (2, 2, None),
        outer,
        str_offsets=np.array([0, 1, 3, 5, 7, 9], dtype=OFFSET_TYPE),
    )
    start = Ragged.from_offsets(
        np.arange(6, dtype=np.int32), (2, 2, None), outer
    )
    rec = Ragged.from_fields({"alt": alt, "start": start})
    row = rec[0][0]
    assert list(row["alt"]) == [b"A", b"GG"]
    assert len(row["alt"]) == len(row["start"])
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/test_ragged_core_records.py -q -k "issue71 or record_row or record_multidim"
```

Expected: FAIL. Today `rec[0]["alt"]` is `array([b'A', b'G', b'G'], dtype='|S1')` — a 3-element `S1` char array next to a 2-element numeric array.

- [ ] **Step 3: Implement the fix**

In `python/seqpro/rag/_core.py`, replace the integer branch at lines 1216-1226:

```python
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            out: dict[str, Any] = {}
            for name, fl in rec.fields.items():
                if fl.str_offsets is not None:
                    so = fl.str_offsets
                    row = fl.data[int(so[lo]) : int(so[hi])]
                else:
                    row = fl.data[lo:hi]
                out[name] = row
            return out
```

with:

```python
        if isinstance(where, (int, np.integer)):
            lo, hi = int(starts[where]), int(stops[where])
            out: dict[str, Any] = {}
            for name, fl in rec.fields.items():
                if fl.str_offsets is not None:
                    # Each field carries its own str_offsets (Spec C Section 5);
                    # peel it against the shared lo/hi so every field of the row
                    # has the same length (issue #71).
                    out[name] = Ragged(_peel_string_row(fl, lo, hi))
                else:
                    out[name] = fl.data[lo:hi]
            return out
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/test_ragged_core_records.py -q -k "issue71 or record_row or record_multidim"
```

Expected: all PASS.

- [ ] **Step 5: Run the full suite**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/ -q
```

Expected: no new failures.

- [ ] **Step 6: Commit**

```bash
git add python/seqpro/rag/_core.py tests/test_ragged_core_records.py
git commit -m "fix(rag)!: preserve string boundaries in peeled record rows

_getitem_record_rows returned a string field as the raw concatenated S1
buffer, so a peeled row mixed a 3-char array with a 2-element numeric
array. String fields now peel to the standalone opaque-string layout,
giving every field of the row the same length.

BREAKING CHANGE: peeling a record row returns opaque-string fields as a
Ragged of strings, not a concatenated S1 array.

Refs #71"
```

---

### Task 3: Lint, typecheck, and skill docs

**Files:**
- Modify: `skills/seqpro/SKILL.md`

**Interfaces:**
- Consumes: the public behavior established in Tasks 1 and 2.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Document the behavior in the "do this, not that" table**

In `skills/seqpro/SKILL.md`, add this row to the `### Working with `Ragged` — do this, not that` table (the table starting at line ~92), after the `Count top-level rows` row:

```markdown
| Index one group of an opaque-string `Ragged` | `rag[i]` → `Ragged` of `bytes`, one per string (`len(rag[i]) == rag.lengths[i]`); `rag[i][j]` is one `bytes` | `b"".join(rag[i])`-style concatenation — that was the pre-0.22 behavior and it dropped the per-string boundaries |
```

- [ ] **Step 2: Document the layout rule in the record section**

In `skills/seqpro/SKILL.md`, append to the bullet list at the end of the `### Record-layout `Ragged` (multi-field)` section (after the `view` and `apply` bullet):

```markdown
- Peeling a row (`rag[i]` where `i` is an integer) returns a **dict** whose entries all have the same length: numeric/char fields as `ndarray`, opaque-string fields as a `Ragged` of `bytes`. This is what makes `zip(row["start"], row["alt"])` correct.
```

- [ ] **Step 3: Run lint and typecheck**

```bash
cd /carter/users/dlaub/projects/ML4GLand/SeqPro/.claude/worktrees/issue-71-string-under-axis-getitem
PY=/carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin
$PY/ruff check python/ tests/ && $PY/ruff format --check python/ tests/
$PY/pyrefly check python
```

Expected: clean. Fix anything reported before committing.

- [ ] **Step 4: Run the full suite one last time**

```bash
PYTHONPATH=python /carter/users/dlaub/projects/ML4GLand/SeqPro/.pixi/envs/dev/bin/python -m pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add skills/seqpro/SKILL.md
git commit -m "docs(skill): document string-under-axis integer indexing

Refs #71"
```

- [ ] **Step 6: Push and open a draft PR**

```bash
git push -u origin worktree-issue-71-string-under-axis-getitem
gh pr create --draft --title "fix(rag)!: preserve string boundaries when indexing a string-under-axis Ragged" --body "$(cat <<'EOF'
Closes #71.

Integer indexing on a string-under-axis `Ragged` concatenated a whole group
into one blob, dropping the per-string boundaries already sitting in
`str_offsets`. `s.lengths[0] == 2` but `len(s[0]) == 3`.

Two sites had the defect:

- `Ragged.__getitem__` integer branch — the one reported.
- `_getitem_record_rows` integer branch — worse, and the one GenVarLoader
  hits: a peeled row mixed a concatenated `S1` char array (not even `bytes`)
  with a per-variant numeric array in the same dict.

Both now peel to the standalone opaque-string layout (`offsets == []`,
`str_offsets` set, `shape == (k,)`), which Spec C already defines as the
zero-real-level special case of string-under-axis. Zero-copy on data; only
the small offsets slice is copied.

This makes `zip(rv.start[0][h], rv.alt[0][h])` correct in GenVarLoader
(mcvickerlab/GenVarLoader#330).

**Breaking:** the integer index now returns a `Ragged` of strings rather than
one concatenated `bytes`. `b"".join(s[i])` recovers the old value.
`major_version_zero` is set, so this bumps 0.21.2 → 0.22.0.

The existing `test_string_under_axis_integer_index` pinned the old behavior
but used one string per group, so it could never distinguish the two
interpretations — that is why the bug survived. It has been rewritten.

Spec: `docs/superpowers/specs/2026-07-27-ragged-string-getitem-design.md`
Plan: `docs/superpowers/plans/2026-07-27-ragged-string-getitem.md`
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec item | Task |
|---|---|
| Site 1 — `__getitem__` integer branch | 1 |
| Site 2 — `_getitem_record_rows` | 2 |
| `_getitem_record_rows_r2` inherits site 1 | 2 (Step 1, `test_record_multidim_row_string_field_preserves_boundaries`) |
| Standalone/flat case unchanged | 1 (`test_standalone_string_index_still_returns_bytes`) |
| Test 1 — issue repro | 1 |
| Test 2 — numeric/string parity | 1 |
| Test 3 — multi-dim | 1 |
| Test 4 — record row with indel | 2 |
| Test 5 — zero-copy | 1 and 2 |
| Test 6 — empty group | 1 |
| Test 7 — `to_chars()` agreement | 1 |
| Test 8 — negative index / OOB | 1 |
| Test 9 — standalone regression | 1 |
| Minor bump via `!` commit | 1, 2 (commit messages) |
| `skills/seqpro/SKILL.md` update | 3 |

**Type consistency:** `_peel_string_row(rl: RaggedLayout, lo: int, hi: int) -> RaggedLayout` is defined in Task 1 Step 4 and used with that exact signature in Task 2 Step 3. Both call sites wrap it in `Ragged(...)`.

**Imports:** the new tests use `Ragged`, `OFFSET_TYPE`, `np`, and `pytest`. `tests/test_ragged_core.py` already imports all four. `tests/test_ragged_core_records.py` does **not** import `OFFSET_TYPE` — Task 2 Step 1 must extend its existing `from seqpro.rag._utils import lengths_to_offsets` (line 6) to `from seqpro.rag._utils import OFFSET_TYPE, lengths_to_offsets`.
