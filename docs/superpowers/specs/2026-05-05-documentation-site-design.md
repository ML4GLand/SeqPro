# Documentation Site Design

**Date:** 2026-05-05  
**Status:** Approved

## Overview

Add a public documentation site to SeqPro using Zensical (a modern static site
generator built on MkDocs/Material). The API reference is auto-generated from
NumPy docstrings at build time via mkdocstrings-python, with no manual page
maintenance required. The site is deployed to GitHub Pages on every push to
`main`.

## Architecture

Zensical reads `zensical.toml` at the repo root and sources Markdown from
`docs/`. A `docs/gen_ref_pages.py` script runs at build time via
`mkdocs-gen-files` and writes a single virtual `docs/api/index.md` containing
one `:::seqpro` mkdocstrings directive. mkdocstrings-python imports the live
`seqpro` package (including the compiled Rust extension), parses NumPy
docstrings via Griffe, and renders all public exports from `seqpro.__init__`
into formatted HTML. `mkdocs-literate-nav` auto-builds the API section
navigation from the generated file tree.

## File Structure

```
SeqPro/
├─ zensical.toml                  # site config (new)
├─ docs/
│  ├─ index.md                    # landing page mirroring README (new)
│  ├─ gen_ref_pages.py            # build-time API page generator (new)
│  └─ api/                        # generated at build time, gitignored
└─ .github/
   └─ workflows/
      └─ docs.yml                 # GitHub Pages deployment (new)
```

`docs/api/` is added to `.gitignore`.

## Configuration (`zensical.toml`)

Key settings:
- `paths = ["python"]` so mkdocstrings resolves `seqpro` from the source tree
- `docstring_style = "numpy"` matching the existing docstring convention
- `show_source = true` (links to source on GitHub)
- `navigation.tabs` feature so "API Reference" sits as a top-level tab
- Plugins: `mkdocstrings`, `mkdocs-gen-files`, `mkdocs-literate-nav`

## API Reference Generation (`gen_ref_pages.py`)

The script runs at `zensical build` time under `mkdocs-gen-files`. It writes
one virtual file:

```
docs/api/index.md
```

Contents:
```markdown
# API Reference

:::seqpro
    options:
      members: true
      inherited_members: false
      show_source: true
```

This renders all public symbols exported from `seqpro.__init__` — functions,
classes, and constants — in a single organized page with NumPy-style docstring
formatting.

## Dependencies

Added to pixi `dev` dependency group:

| Package | Purpose |
|---|---|
| `zensical` | Site generator |
| `mkdocstrings-python` | Docstring → HTML rendering via Griffe |
| `mkdocs-gen-files` | Run `gen_ref_pages.py` at build time |
| `mkdocs-literate-nav` | Auto-nav from generated file tree |

## Deployment (GitHub Actions)

File: `.github/workflows/docs.yml`  
Trigger: push to `main`

Steps:
1. Checkout + setup Python 3.12
2. Install `uv` via `astral-sh/setup-uv@v5`
3. `uv pip install zensical mkdocstrings-python mkdocs-gen-files mkdocs-literate-nav`
4. `maturin build --release` to compile the Rust extension (wheel lands in `target/wheels/`)
5. `uv pip install target/wheels/*.whl` so mkdocstrings can import `seqpro`
6. `zensical build --clean`
7. `actions/upload-pages-artifact` + `actions/deploy-pages`

The Rust extension must be compiled before `zensical build` because
mkdocstrings imports the live package to resolve type annotations.

Site URL: `https://ml4gland.github.io/SeqPro`

## Out of Scope

- Hand-written narrative guides (can be added as future `docs/*.md` pages)
- Versioned docs (can be added later with mike)
- Search beyond Zensical's built-in English search
