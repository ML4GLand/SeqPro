# Combined release pipeline

**Status:** Design  
**Date:** 2026-05-20  
**Scope:** `.github/workflows/`

## Problem

The current release flow is four independent workflows chained by GitHub's `workflow_run` triggers:

```
bump.yaml  (workflow_dispatch)
  └─ pushes tag
     └─ release.yaml  (push: tags)
          └─ workflow_run → publish.yaml   ┐
          └─ workflow_run → merge.yaml     ┘  (parallel)
```

Pain points:

1. **`publish` and `merge` run in parallel.** If `uv publish` fails after `merge` has already fast-forwarded `stable`, the `stable` branch advertises a release that never reached PyPI. Recovery is manual.
2. **`workflow_run` triggers are slow and opaque.** They typically delay ~1–2 minutes and only check `conclusion == 'success'`, not whether the right artifacts actually exist.
3. **No way to override commitizen's increment.** A perf-only PR landed with `feat:` prefixes bumps MINOR even though the project is pre-1.0 and the surface didn't change.
4. **Hard to do partial re-runs cleanly.** Re-running `publish` for an existing tag works via `workflow_dispatch`, but `merge` can race the recovery.

## Goals

- One canonical dispatch entry point that runs **bump → release → publish → merge** in strict order.
- Each step's existing `workflow_dispatch` survives so single-step recovery still works.
- A `dry_run` mode that exercises the pipeline without side effects on git/PyPI/the `stable` branch.
- An `increment` input that overrides commitizen's auto-detection (the original motivating ask).
- A `skip_bump` + `tag` input pair that enters the pipeline at the release step against an existing tag.

## Non-goals

- Changing what each individual step *does*. `bump` still runs commitizen, `publish` still builds maturin wheels via the matrix, `merge` still rebases `stable` onto `main`.
- Replacing PyPI Trusted Publishing or rotating its trust policy. The `uv publish` step continues to run inside `publish.yaml`, so the OIDC `job_workflow_ref` claim still resolves there and the existing PyPI policy (targeting `publish.yaml` + `pypi` environment) keeps working.

## Architecture

**Pattern:** reusable workflows + a thin orchestrator.

```
release-pipeline.yaml    NEW   (workflow_dispatch — the entry point)
├─ resolve              job   computes the canonical tag + dry_run flag
├─ bump                 calls bump.yaml         (skipped if skip_bump=true)
├─ release              calls release.yaml      (needs: bump or resolve)
├─ publish              calls publish.yaml      (needs: release)
└─ merge                calls merge.yaml        (needs: publish)
```

Each existing workflow is amended to add a `workflow_call:` trigger alongside its current trigger(s). No jobs are relocated.

### Tag-push escape hatch

`release.yaml`'s existing `push: tags: [0-9]+.[0-9]+.[0-9]+` trigger stays. Pushing a tag outside the pipeline (e.g., a local hotfix bump) creates a GitHub Release as today, but **does not** auto-trigger publish or merge — `workflow_run` triggers are deleted. For full pipeline behavior on a manually-pushed tag, dispatch `release-pipeline.yaml` with `skip_bump=true, tag=<x.y.z>`.

## Inputs (release-pipeline.yaml)

| Input         | Type                                  | Default | Used by              |
|---------------|---------------------------------------|---------|----------------------|
| `increment`   | choice: `auto`, `PATCH`, `MINOR`, `MAJOR` | `auto`  | bump                 |
| `skip_bump`   | boolean                               | `false` | gates `bump`         |
| `tag`         | string                                | `""`    | release/publish/merge when `skip_bump=true` |
| `dry_run`     | boolean                               | `false` | all steps            |

### Resolve job

Runs first. Outputs:

- `tag`: the canonical version string downstream jobs reference.
  - if `skip_bump=true`: validate `tag` matches `^[0-9]+\.[0-9]+\.[0-9]+$` and that the tag exists on the remote; pass through.
  - else: empty at this point; the `bump` job's `outputs.tag` is what downstream `needs: bump` jobs read.
- `dry_run`: passes through the input for use in downstream `with:`.

### Job dependencies

```
resolve
  ↓
bump   (if: skip_bump == false)
  ↓
release  (needs: [resolve, bump]; tag = needs.bump.outputs.tag || needs.resolve.outputs.tag)
  ↓
publish  (needs: release)
  ↓
merge    (needs: publish)
```

`needs:` enforces strict ordering. If any step fails, downstream jobs are skipped. Re-entry is via either re-dispatching the orchestrator (clean state) or dispatching the failed step on its own.

## Dry-run semantics

Each step honors `dry_run` differently:

| Step    | Real run                                  | Dry run                                                   |
|---------|-------------------------------------------|-----------------------------------------------------------|
| bump    | `cz bump --yes`, push commit + tag        | `cz bump --dry-run` (prints plan, no commit, no tag push) |
| release | publishes GH release from CHANGELOG slice | extracts the changelog slice, prints it to `GITHUB_STEP_SUMMARY`, skips `softprops/action-gh-release` (which requires the tag to exist — which it doesn't in dry-run since bump skipped the tag push) |
| publish | builds wheels, `uv publish` to PyPI       | builds wheels (uploads as artifacts), skips `uv publish`  |
| merge   | `git rebase origin/main && git push stable` | logs the rebase plan, no push                            |

## Required changes per file

### `release-pipeline.yaml` (new, ~80 lines)

- `on: workflow_dispatch:` with the four inputs above.
- `jobs.resolve` (single runner, fast): validates inputs, emits `tag` + `dry_run` outputs.
- `jobs.bump`: `uses: ./.github/workflows/bump.yaml`, `if: inputs.skip_bump == false`, passes `increment`, `dry_run`. Has `outputs.tag`.
- `jobs.release`: `uses: ./.github/workflows/release.yaml`, `needs: [resolve, bump]`, passes resolved `tag`, `dry_run`. Must handle the bump-skipped case (`needs.bump` is the GH "skipped" state — use `always() && (needs.bump.result == 'success' || needs.bump.result == 'skipped')`).
- `jobs.publish`: `uses: ./.github/workflows/publish.yaml`, `needs: release`, passes `dry_run`. Inherits the `pypi` environment via the reusable workflow's own job definition.
- `jobs.merge`: `uses: ./.github/workflows/merge.yaml`, `needs: publish`, passes `dry_run`.

### `bump.yaml`

- Add `on.workflow_call:` with inputs `increment` (string), `dry_run` (boolean).
- Keep `on.workflow_dispatch:` for single-step runs — add the same inputs there.
- In the "Bump version" step, pass `increment: ${{ inputs.increment != 'auto' && inputs.increment || '' }}` to `commitizen-tools/commitizen-action`.
- Add `outputs.tag` on the bump job by capturing `cz version --project` after the bump step.
- Honor `dry_run`: skip the action call entirely and run `cz bump --dry-run` instead; in that branch, `outputs.tag` becomes the *projected* version (still useful for downstream dry-run wiring).

### `release.yaml`

- Add `on.workflow_call:` with inputs `tag` (string, required), `dry_run` (boolean).
- Keep `on.push.tags` and `on.workflow_dispatch` (existing behavior preserved).
- In the "Resolve tag" step, add a third branch for `workflow_call` reading `inputs.tag`.
- In `workflow_call` mode with `dry_run=true`, skip the `softprops/action-gh-release` step entirely and instead `cat body.md >> $GITHUB_STEP_SUMMARY`. (Tag won't exist on the remote in this case because `bump` ran with `--dry-run`.)

### `publish.yaml`

- Add `on.workflow_call:` with input `dry_run` (boolean, default `false`).
- Keep `on.workflow_dispatch`. **Delete** `on.workflow_run` — the new orchestrator replaces it.
- Replace all per-job `if: ${{ github.event_name == 'workflow_dispatch' || github.event.workflow_run.conclusion == 'success' }}` guards with the simpler condition that the job is reachable (i.e., delete the guards entirely — `workflow_call` and `workflow_dispatch` both reach the job unconditionally now).
- In the final `release` job, wrap the `uv publish` step with `if: inputs.dry_run == false || github.event_name == 'workflow_dispatch'`. When dry-running, the wheels still upload as Action artifacts so they can be inspected.
- Add a banner comment at the top: this file has been hand-modified since `maturin generate-ci` last ran; future regen via `maturin generate-ci` will overwrite the trigger block and `if:` conditions — re-apply by hand.

### `merge.yaml`

- Add `on.workflow_call:` with input `dry_run` (boolean).
- Keep `on.workflow_dispatch`. **Delete** `on.workflow_run`.
- In the merge step, gate `git push origin stable` on `inputs.dry_run == false`; otherwise log the planned rebase.

## PyPI Trusted Publisher impact

None. The `uv publish` step continues to execute inside `publish.yaml`, so the OIDC token's `job_workflow_ref` claim still points at `publish.yaml` and the existing PyPI policy (workflow filename `publish.yaml` + environment `pypi`) keeps matching. Moving the publish step into `release-pipeline.yaml` *would* require re-issuing the trust policy; we explicitly do not do that.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `maturin generate-ci` overwrites the workflow_call additions on `publish.yaml`. | Banner comment at the top of `publish.yaml`. Regen happens rarely (only on maturin upgrades). |
| Reusable workflows can't share OIDC tokens in a way that breaks PyPI trust. | Step runs inside the called workflow file; trust claim still resolves there. Verified above. |
| `skip_bump=true, tag=<unknown>` would silently no-op. | `resolve` job validates the tag format and that it exists on the remote (`git fetch --tags && git rev-parse "${tag}"`). |
| A user dispatching only `merge.yaml` after a failed `publish` could ship `stable` without PyPI. | This was true before; the orchestrator's strict ordering improves the common case. Single-step dispatch remains a manual escape hatch — keep it but document the responsibility. |

## Testing plan

- **Dry run end-to-end**: dispatch `release-pipeline.yaml` with `dry_run=true` on a feature branch. Verify no tag is pushed, draft release is created, wheels upload as artifacts, `stable` is unchanged.
- **`skip_bump` path**: pick an existing tag, dispatch with `skip_bump=true, tag=<x.y.z>, dry_run=true`. Verify `release` and downstream jobs use that tag.
- **`increment` override**: on the next real release, dispatch with `increment=PATCH` against a tree that has `feat:` commits; verify the resulting tag is a PATCH bump.
- **Single-step dispatch survival**: dispatch each of `bump.yaml`, `release.yaml`, `publish.yaml`, `merge.yaml` individually with appropriate inputs in dry-run mode; confirm they still behave as today.
- **Trusted publishing**: after the first real (non-dry) release through the orchestrator, confirm PyPI publish succeeds — proves the trust policy is unaffected.

## Out of scope (for follow-ups)

- A "dispatch on PR merge to main" path that auto-runs the pipeline. Today bumping is a deliberate manual choice; that stays.
- Hotfix releases from a non-main branch. The current `merge.yaml` always rebases `stable` onto `main`; making that branch-aware is a separate effort.
