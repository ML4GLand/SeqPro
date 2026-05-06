# Documentation Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Zensical documentation site with auto-generated API reference from NumPy docstrings, deployed to GitHub Pages on every push to `main`.

**Architecture:** Zensical reads `zensical.toml` at the repo root and sources Markdown from `docs/`. A static `docs/api/index.md` contains a single `:::seqpro` mkdocstrings directive that renders all public exports from `seqpro.__init__` using NumPy docstring style. GitHub Actions builds and deploys the site via the standard Pages workflow.

**Tech Stack:** Zensical, mkdocstrings-python (Griffe), uv, maturin, GitHub Actions Pages

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `pixi.toml` | Modify | Replace Sphinx deps in `[feature.doc]` with Zensical deps; update doc task |
| `zensical.toml` | Create | Site config: mkdocstrings plugin, nav, theme, repo link |
| `docs/index.md` | Create | Landing page (content from README) |
| `docs/api/index.md` | Create | API reference page with `:::seqpro` directive |
| `.github/workflows/docs.yml` | Create | Build + deploy to GitHub Pages |

---

### Task 1: Replace Sphinx with Zensical in pixi.toml

**Files:**
- Modify: `pixi.toml` — `[feature.doc]` section

The existing `[feature.doc]` block has Sphinx dependencies. Replace the entire block.

- [ ] **Step 1: Open pixi.toml and find the doc feature block**

Current block to replace (lines roughly matching):
```toml
[feature.doc.dependencies]
sphinx = "*"
sphinx-book-theme = "*"
sphinx-autobuild = "*"
sphinx-autodoc-typehints = "*"
myst-parser = "*"

[feature.doc.tasks]
doc = { cmd = "make clean && make html", cwd = "docs" }
```

- [ ] **Step 2: Replace with Zensical deps and tasks**

```toml
[feature.doc.pypi-dependencies]
zensical = "*"
mkdocstrings-python = "*"

[feature.doc.tasks]
doc = "zensical serve"
build-doc = "zensical build --clean"
```

Note: `[feature.doc.dependencies]` (conda deps) becomes `[feature.doc.pypi-dependencies]` since zensical and mkdocstrings-python are PyPI-only.

- [ ] **Step 3: Install the doc environment to verify pixi accepts the config**

```bash
pixi install -e doc
```

Expected: resolves without errors. If pixi complains about `[feature.doc.dependencies]` being empty after removal, delete the header entirely (the pypi-dependencies table is enough).

- [ ] **Step 4: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build: replace sphinx with zensical in doc environment"
```

---

### Task 2: Create zensical.toml

**Files:**
- Create: `zensical.toml`

- [ ] **Step 1: Create `zensical.toml` at the repo root**

```toml
[project]
site_name = "SeqPro"
site_url = "https://ml4gland.github.io/SeqPro"
site_description = "Fast biological sequence processing for Python"
site_author = "David Laub, Adam Klie"
repo_url = "https://github.com/ML4GLand/SeqPro"
repo_name = "ML4GLand/SeqPro"
edit_uri = "edit/main/docs/"
docs_dir = "docs"

nav = [
  { "Home" = "index.md" },
  { "API Reference" = "api/index.md" },
]

[project.theme]
variant = "modern"
features = [
  "navigation.tabs",
  "navigation.top",
  "navigation.footer",
  "toc.follow",
  "content.action.edit",
  "content.action.view",
  "search.highlight",
]

[project.plugins.mkdocstrings.handlers.python]
paths = ["python"]
inventories = ["https://docs.python.org/3/objects.inv"]

[project.plugins.mkdocstrings.handlers.python.options]
docstring_style = "numpy"
inherited_members = false
show_source = true
members_order = "alphabetical"
show_root_heading = true
show_root_full_path = false
```

- [ ] **Step 2: Verify Zensical accepts the config (dry run)**

```bash
pixi run -e doc zensical build --clean 2>&1 | head -30
```

Expected: no `Config error` lines. The build will fail because `docs/index.md` doesn't exist yet — that's fine at this step, just check there are no TOML parse errors.

- [ ] **Step 3: Commit**

```bash
git add zensical.toml
git commit -m "docs: add zensical.toml site configuration"
```

---

### Task 3: Create docs/index.md

**Files:**
- Create: `docs/index.md`

- [ ] **Step 1: Create the landing page**

```markdown
# SeqPro

[![PyPI - Downloads](https://img.shields.io/pypi/dm/seqpro)](https://pypi.org/project/seqpro/)
[![GitHub stars](https://img.shields.io/github/stars/ML4GLand/SeqPro)](https://github.com/ML4GLand/SeqPro)

SeqPro is a Python package for processing DNA/RNA sequences, with limited support for protein sequences. It makes almost zero compromises on speed — NumPy vectorization throughout, Numba JIT for bottlenecks, and a Rust extension for graph algorithms like k-mer shuffling.

All functions accept strings, lists of strings, NumPy arrays of strings or single-byte ASCII (`S1`), or one-hot encoded (`uint8`) arrays.

## Installation

```bash
pip install seqpro
```

## Quick Start

```python
import seqpro as sp

N, L = 2, 3

# Generate random sequences
seqs = sp.random_seqs(shape=(N, L), alphabet=sp.DNA, seed=1234)

# One-hot encode / decode
ohe = sp.ohe(seqs, alphabet=sp.DNA)
sp.decode_ohe(ohe, ohe_axis=-1, alphabet=sp.DNA, unknown_char="N")

# Tokenize
token_map = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
tokens = sp.tokenize(seqs, token_map=token_map, unknown_token=11)

# Reverse complement
sp.reverse_complement(seqs, alphabet=sp.DNA)

# k-let preserving shuffle
sp.k_shuffle(seqs, k=2, length_axis=1, seed=1234)

# GC / nucleotide content
sp.gc_content(seqs, alphabet=sp.DNA)
sp.nucleotide_content(seqs, alphabet=sp.DNA)
```

See the [API Reference](api/index.md) for full documentation.
```

- [ ] **Step 2: Commit**

```bash
git add docs/index.md
git commit -m "docs: add landing page"
```

---

### Task 4: Create docs/api/index.md

**Files:**
- Create: `docs/api/index.md`

This is a static file (not generated at build time) containing a single mkdocstrings directive. It renders everything exported in `seqpro.__all__`.

- [ ] **Step 1: Create `docs/api/index.md`**

```markdown
# API Reference

::: seqpro
    options:
      members: true
      inherited_members: false
      show_source: true
      members_order: alphabetical
      show_root_heading: false
```

- [ ] **Step 2: Build the site locally to verify rendering**

```bash
pixi run -e doc zensical build --clean 2>&1
```

Expected: `INFO - Documentation built in X.X seconds` with no `ERROR` lines. If mkdocstrings can't import `seqpro` (ImportError on the Rust extension), run `maturin develop` first to compile `seqpro.abi3.so` — the `.so` is gitignored and must exist locally.

- [ ] **Step 3: Spot-check the built output**

```bash
grep -l "gc_content\|reverse_complement\|ohe" site/api/index.html
```

Expected: `site/api/index.html` — confirms the API symbols were rendered into the page.

- [ ] **Step 4: Commit**

```bash
git add docs/api/index.md
git commit -m "docs: add API reference page"
```

---

### Task 5: Create GitHub Actions deployment workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Create the workflow file**

```yaml
name: Documentation

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/configure-pages@v5

      - uses: astral-sh/setup-uv@v5

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install doc dependencies
        run: uv pip install --system zensical mkdocstrings-python

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Build Rust extension
        run: |
          uv pip install --system maturin
          maturin build --release

      - name: Install seqpro wheel
        run: uv pip install --system target/wheels/*.whl

      - name: Install remaining seqpro Python dependencies
        run: uv pip install --system numba numpy polars pyranges pandera pandas pyarrow natsort narwhals awkward polars-config-meta

      - name: Build documentation
        run: zensical build --clean

      - uses: actions/upload-pages-artifact@v4
        with:
          path: site

      - uses: actions/deploy-pages@v4
        id: deployment
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/docs.yml
git commit -m "ci: add GitHub Pages documentation deployment workflow"
```

---

### Task 6: Enable GitHub Pages in repository settings

This is a manual step in the GitHub UI — it cannot be done from the CLI.

- [ ] **Step 1: Go to `https://github.com/ML4GLand/SeqPro/settings/pages`**

- [ ] **Step 2: Under "Build and deployment", set Source to "GitHub Actions"**

- [ ] **Step 3: Push to `main` and verify the workflow runs**

```bash
git push origin main
```

Then check `https://github.com/ML4GLand/SeqPro/actions` — the "Documentation" workflow should appear and pass.

- [ ] **Step 4: Verify the live site**

Open `https://ml4gland.github.io/SeqPro` and confirm:
- Landing page loads
- "API Reference" tab is visible
- `gc_content`, `ohe`, `reverse_complement` etc. appear with rendered NumPy docstrings

---

## Notes

- The `seqpro.abi3.so` Rust extension must be importable when `zensical build` runs, because mkdocstrings imports the live package to resolve type annotations. In the CI workflow this is handled by the `maturin build --release` + wheel install steps. Locally, ensure you're in the `dev` pixi environment where the editable install is active.
- The `site/` directory is already in `.gitignore` (existing entry `/site`), so the build output is never committed.
- To add narrative guide pages in the future, create `docs/guides/*.md` and add entries to the `nav` list in `zensical.toml`.
