# Installation

## From PyPI

The easiest way to install SeqPro is via pip:

```bash
pip install seqpro
```

## From source

To install the latest development version from GitHub:

```bash
git clone https://github.com/ML4GLand/SeqPro.git
cd SeqPro
pip install -e .
```

## Development setup with Pixi

For development, we recommend using [Pixi](https://pixi.sh/) for environment management:

```bash
git clone https://github.com/ML4GLand/SeqPro.git
cd SeqPro
pixi install
pixi run install
```

This will set up a development environment with all dependencies including test and documentation tools.

## Dependencies

SeqPro requires Python 3.9 or later. Core dependencies include:

- numpy
- numba
- polars
- pandas
- pyarrow
- pyranges
- awkward

Optional dependencies for development:

- pytest (testing)
- hypothesis (property-based testing)
- sphinx (documentation)
