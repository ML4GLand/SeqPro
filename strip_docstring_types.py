"""Strip inline type annotations from NumPy-style docstrings.

Replaces `name : type` with `name` in Parameters/Attributes sections and
`name : type` with `name` in Returns sections. Unnamed (type-only) return
entries are renamed `result`. Raises/Notes/Examples sections are untouched.

Usage:
    python scripts/strip_docstring_types.py python/seqpro/ [--dry-run]
"""

import ast
import re
import sys
from pathlib import Path

STRIP_SECTIONS = {"Parameters", "Attributes", "Other Parameters"}
RETURN_SECTIONS = {"Returns", "Yields"}
LEAVE_ALONE = {"Raises", "Warns", "See Also", "Notes", "References", "Examples"}
ALL_SECTIONS = STRIP_SECTIONS | RETURN_SECTIONS | LEAVE_ALONE

SECTION_HEADER_RE = re.compile(r"^(\s*)(\S.*\S|\S)\s*$")
UNDERLINE_RE = re.compile(r"^\s*-{3,}\s*$")
PARAM_WITH_TYPE_RE = re.compile(r"^(\s+)(\*?\w+)\s*:.+$")
RETURN_NAMED_RE = re.compile(r"^(\s+)(\w+)\s*:.+$")


def get_docstring_ranges(source: str) -> list[tuple[int, int]]:
    """Return 1-indexed (start_line, end_line) for every docstring in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            start, end = first.lineno, first.end_lineno
            # Exclude the closing """ line so it's never treated as content
            ranges.append((start, end - 1 if end > start else end))
    return ranges


def in_any_range(lineno: int, ranges: list[tuple[int, int]]) -> bool:
    for start, end in ranges:
        if start <= lineno <= end:
            return True
    return False


def process_source(source: str) -> tuple[str, int]:
    """Return (modified_source, num_lines_changed)."""
    lines = source.splitlines(keepends=True)
    ranges = get_docstring_ranges(source)

    result: list[str] = []
    changes = 0

    section: str | None = None
    item_indent: int | None = None
    pending_header: str | None = None  # section header line not yet confirmed

    for i, line in enumerate(lines):
        lineno = i + 1

        if not in_any_range(lineno, ranges):
            section = None
            item_indent = None
            pending_header = None
            result.append(line)
            continue

        stripped = line.rstrip("\n\r")
        content = stripped.lstrip()
        indent = len(stripped) - len(content)

        # Detect section underline following a header
        if pending_header is not None:
            if UNDERLINE_RE.match(stripped):
                section = pending_header
                item_indent = None
                pending_header = None
                result.append(line)
                continue
            else:
                pending_header = None

        # Detect potential section header (content must be a known section name)
        if content in ALL_SECTIONS:
            pending_header = content
            result.append(line)
            continue

        # Blank line resets section
        if not content:
            section = None
            item_indent = None
            result.append(line)
            continue

        if section is None or section in LEAVE_ALONE:
            result.append(line)
            continue

        # Establish item indent from first content line after the underline
        if item_indent is None:
            item_indent = indent

        if indent != item_indent:
            # Description line — leave alone
            result.append(line)
            continue

        # Item line at item indent
        if section in STRIP_SECTIONS:
            m = PARAM_WITH_TYPE_RE.match(stripped)
            if m:
                new_line = m.group(1) + m.group(2) + "\n"
                result.append(new_line)
                changes += 1
                continue

        elif section in RETURN_SECTIONS:
            named_m = RETURN_NAMED_RE.match(stripped)
            if named_m:
                # Named return: strip type
                new_line = named_m.group(1) + named_m.group(2) + "\n"
                result.append(new_line)
                changes += 1
                continue
            elif content and not content.endswith(":"):
                # Type-only unnamed return: replace with `result`
                new_line = " " * indent + "result\n"
                if new_line.rstrip() != stripped:
                    result.append(new_line)
                    changes += 1
                    continue

        result.append(line)

    return "".join(result), changes


def process_file(path: Path, dry_run: bool) -> int:
    source = path.read_text()
    modified, changes = process_source(source)
    if changes:
        print(f"  {path}: {changes} line(s) changed")
        if not dry_run:
            path.write_text(modified)
    return changes


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    paths = [a for a in args if not a.startswith("--")]

    if not paths:
        print("Usage: strip_docstring_types.py <dir_or_file> [--dry-run]")
        sys.exit(1)

    total = 0
    files_changed = 0
    for p in paths:
        root = Path(p)
        py_files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for f in py_files:
            n = process_file(f, dry_run)
            if n:
                total += n
                files_changed += 1

    action = "Would change" if dry_run else "Changed"
    print(f"\n{action} {total} line(s) across {files_changed} file(s).")


if __name__ == "__main__":
    main()
