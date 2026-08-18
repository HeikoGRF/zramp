"""Budgeted utility-selection zRAMP and legacy comparison variants.

The historical modules retain their original package layout. Common simulator
modules live in ``paper_experiments/code/shared_runtime``; activating that
directory here keeps the recovered entry points importable without duplicating
the maintained implementation inside the project archive.
"""

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_SHARED_RUNTIME = _REPOSITORY_ROOT / "paper_experiments" / "code" / "shared_runtime"
if _SHARED_RUNTIME.is_dir() and str(_SHARED_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_SHARED_RUNTIME))
