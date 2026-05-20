# Release Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine the four release workflows (`bump`, `release`, `publish`, `merge`) into a single dispatch-driven orchestrator with strict ordering, while keeping every existing workflow independently dispatchable.

**Architecture:** Introduce a new `release-pipeline.yaml` whose only job is to chain four `workflow_call` invocations via `needs:`. Each existing workflow is amended to add a `workflow_call:` trigger alongside its current triggers. The fragile `workflow_run` triggers on `publish.yaml` and `merge.yaml` are deleted. New dispatch inputs (`increment`, `skip_bump`, `tag`, `dry_run`) are surfaced on the orchestrator and plumbed through.

**Tech Stack:** GitHub Actions YAML, commitizen / commitizen-action, maturin-action, softprops/action-gh-release, PyPI Trusted Publishing (OIDC).

**Spec:** `docs/superpowers/specs/2026-05-20-release-pipeline-design.md`

---

## File Structure

Files modified or created (all paths relative to repo root):

| Path | Action | Responsibility |
|---|---|---|
| `.github/workflows/release-pipeline.yaml` | **Create** | The top-level orchestrator. `workflow_dispatch` only. Holds the four dispatch inputs and chains the four reusable workflows. |
| `.github/workflows/bump.yaml` | Modify | Add `workflow_call:` (with `increment`, `dry_run`); plumb `increment` to commitizen-action; expose `outputs.tag`. Keep `workflow_dispatch`. |
| `.github/workflows/release.yaml` | Modify | Add `workflow_call:` (with `tag`, `dry_run`); extend "Resolve tag" step to read `inputs.tag` in call mode; skip softprops in dry-run and dump body to step summary. Keep `push: tags:` and `workflow_dispatch`. |
| `.github/workflows/publish.yaml` | Modify | Add `workflow_call:` (with `dry_run`); delete `on.workflow_run`; delete the per-job `if:` guards that referenced `workflow_run.conclusion`; gate `uv publish` on `dry_run`. Add banner about hand-modifications. |
| `.github/workflows/merge.yaml` | Modify | Add `workflow_call:` (with `dry_run`); delete `on.workflow_run`; gate `git push origin stable` on `dry_run`. Keep `workflow_dispatch`. |

No source code is touched. No test files (these workflows are tested via dispatch on a feature branch — see Task 6).

Order of work: refactor each leaf workflow to be `workflow_call`-able with dry-run support **first** (Tasks 1–4), then write the orchestrator that calls them (Task 5), then end-to-end test on a branch (Task 6).

---

### Task 1: Make `bump.yaml` workflow_call-able with increment override and dry-run

**Files:**
- Modify: `.github/workflows/bump.yaml`

**Context:** Today `bump.yaml` is `workflow_dispatch` only. It runs commitizen-action which auto-detects the increment from commit prefixes. We need to expose an `increment` override (the original motivating ask from the user) and a `dry_run` flag, and emit the resulting tag as a job output so the orchestrator can pass it downstream.

The commitizen-action upstream supports an `increment` input that accepts `"PATCH"`, `"MINOR"`, `"MAJOR"`, or empty (auto). We pass empty when our input is `"auto"`.

`cz version --project` reads the version from `pyproject.toml` / `Cargo.toml` and is the canonical way to learn what the new tag is *after* a bump completes.

- [ ] **Step 1: Replace `bump.yaml` with the call-able version**

Full file:

```yaml
name: Bump version

on:
  workflow_call:
    inputs:
      increment:
        description: "Force a specific bump (auto|PATCH|MINOR|MAJOR)"
        type: string
        required: false
        default: "auto"
      dry_run:
        description: "Compute the bump but do not commit/push"
        type: boolean
        required: false
        default: false
    outputs:
      tag:
        description: "The version string after the bump (or projected version in dry-run)"
        value: ${{ jobs.bump.outputs.tag }}
  workflow_dispatch:
    inputs:
      increment:
        description: "Force a specific bump"
        type: choice
        required: false
        default: "auto"
        options: ["auto", "PATCH", "MINOR", "MAJOR"]
      dry_run:
        description: "Compute the bump but do not commit/push"
        type: boolean
        required: false
        default: false

jobs:
  bump:
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.emit.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          token: "${{ secrets.GH_TOKEN }}"

      - name: Install commitizen
        run: pip install commitizen --quiet

      - name: Detect partial bump
        id: detect
        run: |
          PROJECT_VERSION=$(cz version --project)
          LATEST_TAG=$(git tag --sort=-version:refname | grep -E '^[0-9]+\.[0-9]+\.[0-9]+$' | head -1)
          if [ -n "$LATEST_TAG" ] && [ "$LATEST_TAG" != "$PROJECT_VERSION" ]; then
            echo "partial=true" >> $GITHUB_OUTPUT
            echo "tag=$LATEST_TAG" >> $GITHUB_OUTPUT
          else
            echo "partial=false" >> $GITHUB_OUTPUT
          fi

      - name: Bump version (real)
        if: steps.detect.outputs.partial != 'true' && inputs.dry_run == false
        uses: commitizen-tools/commitizen-action@master
        with:
          github_token: ${{ secrets.GH_TOKEN }}
          increment: ${{ inputs.increment != 'auto' && inputs.increment || '' }}

      - name: Bump version (dry-run)
        if: steps.detect.outputs.partial != 'true' && inputs.dry_run == true
        run: |
          ARGS="--dry-run --yes"
          if [ "${{ inputs.increment }}" != "auto" ]; then
            ARGS="$ARGS --increment ${{ inputs.increment }}"
          fi
          cz bump $ARGS

      - name: Fix partial bump (tag exists on remote, update files only)
        if: steps.detect.outputs.partial == 'true' && inputs.dry_run == false
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          NEW_VERSION="${{ steps.detect.outputs.tag }}"
          cz bump --files-only --yes
          BUMPED_VERSION=$(cz version --project)
          git add pyproject.toml CHANGELOG.md
          git commit -m "bump: version $NEW_VERSION"
          git push origin main
          git push origin "$BUMPED_VERSION"

      - name: Emit resulting tag
        id: emit
        run: |
          TAG=$(cz version --project)
          echo "tag=$TAG" >> $GITHUB_OUTPUT
          echo "Resolved tag: $TAG"
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/bump.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 3: Sanity-check with actionlint if available**

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/bump.yaml || echo "actionlint not installed, skipping"`
Expected: no errors reported (or skip message).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/bump.yaml
git commit -m "fix(ci): make bump.yaml workflow_call-able with increment and dry_run

Adds workflow_call trigger with increment/dry_run inputs and a tag
output so the orchestrator can pass it downstream. Keeps the existing
workflow_dispatch entry point intact. Plumbs the increment input into
commitizen-action.
"
```

---

### Task 2: Make `release.yaml` workflow_call-able with dry-run

**Files:**
- Modify: `.github/workflows/release.yaml`

**Context:** Today `release.yaml` triggers on tag push or `workflow_dispatch` with a `tag` input. The "Resolve tag" step branches on `github.event_name`. We need a third branch for `workflow_call` and a dry-run mode where we skip `softprops/action-gh-release` (because in dry-run no tag exists on the remote) and instead dump the changelog body to the run summary.

- [ ] **Step 1: Replace `release.yaml` with the call-able version**

Full file:

```yaml
name: Create release

on:
  push:
    tags:
      - '[0-9]+.[0-9]+.[0-9]+'
  workflow_dispatch:
    inputs:
      tag:
        description: 'Tag to release (e.g. 0.10.0)'
        required: true
  workflow_call:
    inputs:
      tag:
        description: 'Tag to release'
        type: string
        required: true
      dry_run:
        description: 'Build the release body but do not create a GH release'
        type: boolean
        required: false
        default: false

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Resolve tag
        id: tag
        run: |
          case "${{ github.event_name }}" in
            push)          echo "tag=${GITHUB_REF#refs/tags/}" >> $GITHUB_OUTPUT ;;
            workflow_dispatch) echo "tag=${{ inputs.tag }}" >> $GITHUB_OUTPUT ;;
            workflow_call) echo "tag=${{ inputs.tag }}" >> $GITHUB_OUTPUT ;;
          esac

      - name: Extract changelog section
        run: |
          TAG="${{ steps.tag.outputs.tag }}"
          awk -v tag="$TAG" '
            $0 ~ "^## " tag " " { flag=1; next }
            flag && /^## / { exit }
            flag { print }
          ' CHANGELOG.md > body.md

      - name: Dry-run summary
        if: github.event_name == 'workflow_call' && inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: release ${{ steps.tag.outputs.tag }}"
            echo
            echo "Would create a GitHub Release with the following body:"
            echo
            echo '```markdown'
            cat body.md
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"

      - name: Create GitHub release
        if: ${{ !(github.event_name == 'workflow_call' && inputs.dry_run == true) }}
        uses: softprops/action-gh-release@v3
        with:
          tag_name: ${{ steps.tag.outputs.tag }}
          body_path: body.md
          token: ${{ secrets.GH_TOKEN }}
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/release.yaml
git commit -m "fix(ci): make release.yaml workflow_call-able with dry_run

Adds a workflow_call trigger that accepts tag and dry_run inputs. In
dry-run mode the changelog body is emitted to GITHUB_STEP_SUMMARY and
softprops/action-gh-release is skipped (the tag does not exist on the
remote when bump ran dry).
"
```

---

### Task 3: Make `publish.yaml` workflow_call-able, delete workflow_run trigger

**Files:**
- Modify: `.github/workflows/publish.yaml`

**Context:** `publish.yaml` is maturin-generated boilerplate (matrix of build jobs across Linux/musllinux/Windows/macOS/sdist, plus a final `release` job that calls `uv publish`). Today every job has the same guard: `if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}`. With `workflow_run` going away, those guards become incorrect — `workflow_call` invocations should reach the jobs unconditionally. We delete the guards entirely.

The `uv publish` step needs a `dry_run` gate. When dry-running, the wheels still upload as Action artifacts (so they can be inspected); only the PyPI push is skipped.

**Important:** PyPI Trusted Publishing trust policy targets the called workflow's filename (`publish.yaml`) and environment (`pypi`). The OIDC token's `job_workflow_ref` claim resolves to the *called* workflow — so wrapping this in `workflow_call` from the orchestrator does **not** require any PyPI side changes. (See spec, "PyPI Trusted Publisher impact".)

Banner: regen via `maturin generate-ci -o .github/workflows/publish.yaml github` will overwrite the trigger block and `if:` conditions; whoever bumps maturin in the future must re-apply.

- [ ] **Step 1: Read the current file as the baseline**

The current file is 218 lines; only the header, triggers, and per-job `if:` guards need editing. The bodies of the matrix jobs (build steps, upload-artifact, etc.) are unchanged.

Run: `wc -l .github/workflows/publish.yaml`
Expected: `218 .github/workflows/publish.yaml` (or similar — the exact count just confirms we're looking at the right file).

- [ ] **Step 2: Replace the header (lines 1–16) with the new trigger block**

Replace the top 16 lines (everything before the first `jobs:` line) with:

```yaml
# This file was autogenerated by maturin v1.13.1 and has been hand-modified
# to support workflow_call from release-pipeline.yaml.
#
# To regenerate, run:
#
#    maturin generate-ci -o .github/workflows/publish.yaml github
#
# Regenerating will overwrite the trigger block and the per-job `if:`
# conditions below. Re-apply the workflow_call inputs (`dry_run`), delete
# the `on.workflow_run` block, remove all `if: ${{ github.event_name ... }}`
# guards, and wrap the `uv publish` step with the dry_run gate before
# committing.
name: Publish

on:
  workflow_call:
    inputs:
      dry_run:
        description: 'Build wheels but skip uv publish'
        type: boolean
        required: false
        default: false
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Build wheels but skip uv publish'
        type: boolean
        required: false
        default: false

permissions:
  contents: read
```

- [ ] **Step 3: Delete every per-job `if:` guard that references `workflow_run`**

There are five — one on each of `linux`, `musllinux`, `windows`, `macos`, `release` (the `sdist` job already has no such guard but verify). The pattern to remove (with surrounding two-space indent):

```yaml
    if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}
```

After this step those jobs run unconditionally (which is correct: this workflow only fires via `workflow_call` or `workflow_dispatch` now — both should always run all jobs).

Use a sanity grep:

```bash
grep -n "workflow_run" .github/workflows/publish.yaml
```

Expected: no output. If any line still mentions `workflow_run`, remove it.

- [ ] **Step 4: Gate the `uv publish` step on `dry_run`**

Find the existing `Publish to PyPI` step (in the `release` job near the end of the file). It currently looks like:

```yaml
      - name: Publish to PyPI
        run: uv publish 'wheels-*/*'
```

Replace it with:

```yaml
      - name: Publish to PyPI
        if: inputs.dry_run == false
        run: uv publish 'wheels-*/*'

      - name: Dry-run summary (skipped uv publish)
        if: inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: would have published the following wheels"
            echo
            ls -1 wheels-*/* | sed 's/^/- /'
          } >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 5: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/publish.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 6: Re-confirm no `workflow_run` lingers**

Run: `grep -n "workflow_run" .github/workflows/publish.yaml || echo OK`
Expected: `OK`.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/publish.yaml
git commit -m "fix(ci): make publish.yaml workflow_call-able, drop workflow_run

Adds workflow_call trigger with a dry_run input. Deletes the
on.workflow_run trigger (the new release-pipeline.yaml orchestrator
calls this workflow directly via needs:). Removes the per-job if:
guards that referenced workflow_run.conclusion since both workflow_call
and workflow_dispatch should always run all jobs. Gates uv publish on
dry_run; wheels still upload as Action artifacts for inspection.

Adds a banner explaining that maturin generate-ci will overwrite these
hand-modifications and how to re-apply them.

PyPI trusted publishing is unaffected: OIDC job_workflow_ref still
resolves to publish.yaml when invoked via workflow_call.
"
```

---

### Task 4: Make `merge.yaml` workflow_call-able with dry-run, drop workflow_run

**Files:**
- Modify: `.github/workflows/merge.yaml`

**Context:** Today `merge.yaml` triggers on `workflow_run` (after `Create release` succeeds) or `workflow_dispatch`. The job rebases `stable` onto `main` and pushes. In the new pipeline, the orchestrator calls this via `workflow_call`, so the `workflow_run` trigger is deleted. Dry-run logs the rebase but skips the push.

- [ ] **Step 1: Replace `merge.yaml` with the call-able version**

Full file:

```yaml
name: Merge main -> stable

on:
  workflow_call:
    inputs:
      dry_run:
        description: 'Compute the rebase but do not push'
        type: boolean
        required: false
        default: false
  workflow_dispatch:
    inputs:
      dry_run:
        description: 'Compute the rebase but do not push'
        type: boolean
        required: false
        default: false

jobs:
  merge:
    environment: pypi
    permissions:
      id-token: write
    runs-on: ubuntu-latest
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          ref: stable
          fetch-depth: 0
          token: "${{ secrets.GH_TOKEN }}"
      - name: Config git
        run: |
          git config --global user.name "${GITHUB_ACTOR}"
          git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
      - name: Rebase stable onto main
        run: git rebase origin/main
      - name: Push to stable
        if: inputs.dry_run == false
        run: git push origin stable
      - name: Dry-run summary (skipped push)
        if: inputs.dry_run == true
        run: |
          {
            echo "## Dry-run: would push the following to stable"
            echo
            echo '```'
            git log --oneline origin/stable..HEAD
            echo '```'
          } >> "$GITHUB_STEP_SUMMARY"
```

- [ ] **Step 2: Validate the YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/merge.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 3: Confirm no `workflow_run` lingers**

Run: `grep -n "workflow_run" .github/workflows/merge.yaml || echo OK`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/merge.yaml
git commit -m "fix(ci): make merge.yaml workflow_call-able, drop workflow_run

The new release-pipeline.yaml orchestrator now drives this workflow
via needs: instead of relying on the workflow_run trigger fired by
release.yaml. Adds a dry_run input that logs the rebase plan to
GITHUB_STEP_SUMMARY but skips git push origin stable.
"
```

---

### Task 5: Create the orchestrator `release-pipeline.yaml`

**Files:**
- Create: `.github/workflows/release-pipeline.yaml`

**Context:** The orchestrator is `workflow_dispatch` only. It runs a small `resolve` job that validates inputs, then calls each reusable workflow in order. Job dependencies enforce strict ordering: `release` needs `[resolve, bump]`, `publish` needs `release`, `merge` needs `publish`.

When `skip_bump=true`, the `bump` job is skipped (GitHub treats it as `result == 'skipped'`); the `release` job's condition must accept that explicitly via `if: always() && (needs.bump.result == 'success' || needs.bump.result == 'skipped')`.

The canonical `tag` downstream jobs read is: `needs.bump.outputs.tag` if bump ran, else `needs.resolve.outputs.tag`.

- [ ] **Step 1: Create the file**

```yaml
name: Release pipeline

on:
  workflow_dispatch:
    inputs:
      increment:
        description: "Force a specific bump (auto = let commitizen decide)"
        type: choice
        required: false
        default: "auto"
        options: ["auto", "PATCH", "MINOR", "MAJOR"]
      skip_bump:
        description: "Skip bump; release/publish/merge an existing tag instead"
        type: boolean
        required: false
        default: false
      tag:
        description: "Existing tag to release (required when skip_bump=true)"
        type: string
        required: false
        default: ""
      dry_run:
        description: "Run end-to-end without external side effects"
        type: boolean
        required: false
        default: false

jobs:
  resolve:
    runs-on: ubuntu-latest
    outputs:
      tag: ${{ steps.out.outputs.tag }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Validate inputs
        id: out
        run: |
          if [ "${{ inputs.skip_bump }}" = "true" ]; then
            TAG="${{ inputs.tag }}"
            if [ -z "$TAG" ]; then
              echo "::error::skip_bump=true requires a tag input"
              exit 1
            fi
            if ! echo "$TAG" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
              echo "::error::tag '$TAG' does not match X.Y.Z"
              exit 1
            fi
            git fetch --tags --quiet
            if ! git rev-parse --verify --quiet "refs/tags/$TAG" >/dev/null; then
              echo "::error::tag '$TAG' does not exist on the remote"
              exit 1
            fi
            echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          else
            echo "tag=" >> "$GITHUB_OUTPUT"
          fi

  bump:
    needs: resolve
    if: inputs.skip_bump == false
    uses: ./.github/workflows/bump.yaml
    secrets: inherit
    with:
      increment: ${{ inputs.increment }}
      dry_run: ${{ inputs.dry_run }}

  release:
    needs: [resolve, bump]
    if: |
      always() &&
      needs.resolve.result == 'success' &&
      (needs.bump.result == 'success' || needs.bump.result == 'skipped')
    uses: ./.github/workflows/release.yaml
    secrets: inherit
    with:
      tag: ${{ needs.bump.outputs.tag || needs.resolve.outputs.tag }}
      dry_run: ${{ inputs.dry_run }}

  publish:
    needs: release
    uses: ./.github/workflows/publish.yaml
    secrets: inherit
    with:
      dry_run: ${{ inputs.dry_run }}

  merge:
    needs: publish
    uses: ./.github/workflows/merge.yaml
    secrets: inherit
    with:
      dry_run: ${{ inputs.dry_run }}
```

- [ ] **Step 2: Validate YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/release-pipeline.yaml'))"`
Expected: no output, exit 0.

- [ ] **Step 3: Sanity-check with actionlint if available**

Run: `command -v actionlint >/dev/null && actionlint .github/workflows/release-pipeline.yaml || echo "actionlint not installed, skipping"`
Expected: no errors reported (or skip message).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release-pipeline.yaml
git commit -m "feat(ci): add release-pipeline.yaml orchestrator

Chains bump -> release -> publish -> merge via reusable workflow_call
with strict needs: ordering. Surfaces increment (auto/PATCH/MINOR/MAJOR),
skip_bump, tag, and dry_run inputs on workflow_dispatch. The resolve
job validates skip_bump/tag combinations before any side-effect step
runs. Each step's existing workflow_dispatch is preserved for single-
step recovery; this orchestrator is the canonical happy-path entry.
"
```

---

### Task 6: End-to-end dry-run verification on a branch

**Files:** none modified; this task is verification only.

**Context:** Before merging the workflow changes to `main`, run the full pipeline in dry-run mode on the feature branch and confirm: no tag is pushed, no GH release is created, no wheels go to PyPI, `stable` does not move. Then test each step's standalone dispatch to confirm independent re-runs still work.

Branch name assumed: `ci/release-pipeline` (rename to whatever the implementer used).

- [ ] **Step 1: Push the branch to GitHub**

```bash
git push -u origin HEAD
```

- [ ] **Step 2: Dispatch the orchestrator in dry-run mode (default-path: with bump)**

```bash
gh workflow run release-pipeline.yaml \
  --ref "$(git branch --show-current)" \
  -f dry_run=true \
  -f increment=PATCH
```

Then watch:

```bash
gh run watch
```

Expected: all four called workflows succeed. Verify in the GH UI that:
- `bump` step summary shows the projected next version, no commit/tag was pushed (`git fetch && git log origin/main -5` shows no new bump commit).
- `release` step summary contains the changelog slice; no GH Release exists for the projected tag.
- `publish` matrix builds wheels and uploads them as Action artifacts; PyPI is untouched.
- `merge` step summary shows the rebase plan; `git fetch && git log origin/stable -5` is unchanged.

- [ ] **Step 3: Dispatch with `skip_bump=true` against an existing tag**

Pick a real tag (`gh release list --limit 5`) and dispatch:

```bash
gh workflow run release-pipeline.yaml \
  --ref "$(git branch --show-current)" \
  -f skip_bump=true \
  -f tag=<existing-tag> \
  -f dry_run=true
```

Expected: `bump` job is skipped (visible in GH UI as gray), `release`/`publish`/`merge` all succeed with the chosen tag, no external side effects.

- [ ] **Step 4: Dispatch with `skip_bump=true` and a bad tag**

```bash
gh workflow run release-pipeline.yaml \
  --ref "$(git branch --show-current)" \
  -f skip_bump=true \
  -f tag=999.999.999 \
  -f dry_run=true
```

Expected: `resolve` job fails with an error message about the tag not existing. Downstream jobs do not start.

- [ ] **Step 5: Confirm each standalone workflow still dispatches**

```bash
gh workflow run bump.yaml --ref "$(git branch --show-current)" -f dry_run=true -f increment=PATCH
gh workflow run merge.yaml --ref "$(git branch --show-current)" -f dry_run=true
gh workflow run publish.yaml --ref "$(git branch --show-current)" -f dry_run=true
```

Expected: each runs to completion in dry-run mode.

- [ ] **Step 6: Confirm a single tag push still triggers `release.yaml` alone**

This is the documented escape hatch. Don't actually push a real tag from the branch — just confirm the trigger is still declared:

```bash
grep -A3 "^on:" .github/workflows/release.yaml | head -10
```

Expected: output includes both `push:` with `tags: ['[0-9]+.[0-9]+.[0-9]+']` and `workflow_dispatch:` and `workflow_call:`.

- [ ] **Step 7: Open a PR**

```bash
gh pr create --title "ci: combined release pipeline orchestrator" --body "$(cat <<'EOF'
## Summary
- New `release-pipeline.yaml` chains bump → release → publish → merge via reusable `workflow_call` with strict `needs:` ordering.
- Adds `increment` (auto/PATCH/MINOR/MAJOR), `skip_bump`, `tag`, and `dry_run` inputs.
- Deletes the flaky `workflow_run` triggers on `publish.yaml` and `merge.yaml`.
- Every existing workflow keeps its `workflow_dispatch` entry point for single-step recovery.
- PyPI trusted publishing trust policy is untouched: `uv publish` still runs inside `publish.yaml`, so `job_workflow_ref` continues to resolve there.

## Test plan
- [x] End-to-end dry-run via `gh workflow run release-pipeline.yaml -f dry_run=true -f increment=PATCH`
- [x] Dry-run with `skip_bump=true` against an existing tag
- [x] Negative test: `skip_bump=true` with a non-existent tag fails in `resolve`
- [x] Each standalone workflow still dispatches alone
- [ ] First post-merge real release: confirm PyPI publish succeeds (proves trust policy unaffected)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-review

**Spec coverage:**
- Architecture (orchestrator + reusable workflows + tag-push escape hatch) → Tasks 1–5
- Inputs `increment`, `skip_bump`, `tag`, `dry_run` → Task 5 declares them; Tasks 1/2/3/4 receive them
- Resolve job that validates `skip_bump`/`tag` → Task 5
- Strict `needs:` failure semantics → Task 5
- Dry-run semantics per step (bump/release/publish/merge) → Tasks 1, 2, 3, 4 each implement the matching gate
- `release.yaml` tag-push escape hatch preserved → Task 2 keeps `push: tags:` in the new file
- Delete `workflow_run` triggers from publish/merge → Tasks 3 and 4 explicitly delete + grep-verify
- PyPI trust policy unchanged → captured in Task 3's commit body and Step 4 of the verification plan asserts it implicitly (real publish on first post-merge release)
- Banner on hand-edited `publish.yaml` → Task 3 Step 2
- Testing plan (dry-run, skip_bump, increment override, single-step dispatch survival) → Task 6 covers all four

**Placeholder scan:** none — every step has a concrete command or full code block.

**Type consistency:** input names (`increment`, `skip_bump`, `tag`, `dry_run`) match across orchestrator and called workflows. `outputs.tag` from `bump.yaml` matches what `release` reads via `needs.bump.outputs.tag`. The `softprops/action-gh-release` step is referenced by the same name (`Create GitHub release`) before and after the dry-run wrap.
