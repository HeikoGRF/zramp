"""Activate the non-paper-facing runtime shipped beside ``final``.

The archive keeps experiment entry points in ``final`` and inherited simulator
machinery in ``shared_runtime``. Entry points call :func:`activate` before
importing that machinery so the layout works without installing a package.
"""

from __future__ import annotations

import sys
from pathlib import Path


FINAL_ROOT = Path(__file__).resolve().parent
SHARED_RUNTIME_ROOT = FINAL_ROOT.parent / "shared_runtime"


def activate() -> Path:
    """Add the archived shared runtime to ``sys.path`` and return its path."""
    if not SHARED_RUNTIME_ROOT.is_dir():
        raise RuntimeError(
            f"archived shared runtime is missing: {SHARED_RUNTIME_ROOT}"
        )
    path = str(SHARED_RUNTIME_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return SHARED_RUNTIME_ROOT
