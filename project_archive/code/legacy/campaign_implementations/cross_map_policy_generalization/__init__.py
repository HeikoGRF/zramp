"""Recovered source for the documented cross-map policy experiments.

Only modules tied to results retained in ``results/legacy_experiments`` are
included. Shared predictor and radio-simulation code is loaded from the
repository's maintained ``paper_experiments/code/shared_runtime`` directory.
"""

from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_SHARED_RUNTIME = _REPOSITORY_ROOT / "paper_experiments" / "code" / "shared_runtime"
if _SHARED_RUNTIME.is_dir() and str(_SHARED_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_SHARED_RUNTIME))
