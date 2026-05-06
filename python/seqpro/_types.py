from pathlib import Path
from typing import TypeAlias

PathLike: TypeAlias = str | Path
"""A file-system path accepted as either a plain string or a `pathlib.Path`."""
