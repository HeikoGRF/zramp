#!/usr/bin/env python3
"""Fail if a repository file reaches GitHub's regular-Git warning threshold."""

from __future__ import annotations

import sys
from pathlib import Path


WARNING_BYTES = 50 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    files = [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts]
    oversized = [path for path in files if path.stat().st_size >= WARNING_BYTES]
    caches = [
        path
        for path in files
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}
    ]
    total = sum(path.stat().st_size for path in files)
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:10]
    print(f"Files: {len(files):,}")
    print(f"Logical size: {total / 1024**2:.1f} MiB")
    print("Largest files:")
    for path in largest:
        print(f"  {path.stat().st_size / 1024**2:7.1f} MiB  {path.relative_to(ROOT)}")
    if caches:
        print(f"ERROR: found {len(caches)} cache files", file=sys.stderr)
    if oversized:
        for path in oversized:
            print(f"ERROR: file is at least 50 MiB: {path.relative_to(ROOT)}", file=sys.stderr)
    return 1 if caches or oversized else 0


if __name__ == "__main__":
    raise SystemExit(main())

