#!/usr/bin/env python
"""Remove local absolute paths from tracked result artifacts before publishing.

This rewrites JSON path values to repository-relative paths and replaces the
repository/home prefixes in Markdown/CSV/TXT. It is intentionally conservative:
only files under ``outputs/`` are touched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from _bootstrap import REPO_ROOT


def _relative(value: str, root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except Exception:
            home = Path.home()
            try:
                return "~/" + path.resolve().relative_to(home.resolve()).as_posix()
            except Exception:
                return path.name
    return value


def _clean_json(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {k: _clean_json(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v, root) for v in value]
    if isinstance(value, str):
        # Replace every occurrence of the repo root inside a string (error
        # messages and command logs contain paths in the middle of a sentence).
        for prefix in (str(root), root.as_posix(), str(root).replace("\\", "/")):
            value = value.replace(prefix, ".")
        value = value.replace(".\\", "").replace("./", "")
        # Anything outside the repo but under the user's home is shortened.
        home = Path.home()
        value = value.replace(str(home), "~").replace(home.as_posix(), "~")
        return value
    return value


def _clean_text(text: str, root: Path) -> str:
    home = Path.home()
    # Longest / most specific prefix first; home replacement must come last or
    # repository paths would turn into ~/Documents/... instead of relative paths.
    replacements = [
        (str(root), "."),
        (root.as_posix(), "."),
        (str(root).replace("\\", "/"), "."),
        (str(home), "~"),
        (home.as_posix(), "~"),
    ]
    for old, new in replacements:
        if old and old != ".":
            text = text.replace(old, new)
    # Collapse the common "./" prefix produced by the replacement above.
    text = text.replace(".\\", "").replace("./", "")
    return text


def sanitize(root: Path, dry_run: bool = False) -> list[str]:
    changed: list[str] = []
    outputs = root / "outputs"
    if not outputs.is_dir():
        return changed
    for path in sorted(outputs.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".json", ".md", ".csv", ".txt", ".yaml", ".yml"}:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if suffix == ".json":
            try:
                data = json.loads(original)
                updated = json.dumps(_clean_json(data, root), indent=2, ensure_ascii=False) + "\n"
            except json.JSONDecodeError:
                updated = _clean_text(original, root)
        else:
            updated = _clean_text(original, root)
        if updated != original:
            changed.append(str(path.relative_to(root)))
            if not dry_run:
                path.write_text(updated, encoding="utf-8")
    return changed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Sanitize local paths in outputs/.")
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    changed = sanitize(Path(args.root), dry_run=args.dry_run)
    print(f"{'Would sanitize' if args.dry_run else 'Sanitized'} {len(changed)} files")
    for item in changed:
        print(f"  {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
