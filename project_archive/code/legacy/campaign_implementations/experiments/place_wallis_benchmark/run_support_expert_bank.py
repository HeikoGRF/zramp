#!/usr/bin/env python3
"""Legacy-compatible entry point for the archived expert-bank simulations.

The maintained implementation lives in
``paper_experiments/code/shared_runtime``. Keeping this small adapter at the
original path lets the historical launchers remain reviewable and runnable
without preserving a second, diverging implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
SHARED_RUNTIME_ROOT = REPOSITORY_ROOT / "paper_experiments" / "code" / "shared_runtime"
if not SHARED_RUNTIME_ROOT.is_dir():
    raise RuntimeError(
        f"archived shared runtime is missing: {SHARED_RUNTIME_ROOT}"
    )
if str(SHARED_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_RUNTIME_ROOT))

from zramp_runtime.support_expert_bank import *  # noqa: E402,F403
from zramp_runtime.support_expert_bank import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
