#!/usr/bin/env python3
"""Paper-facing entry point for the archived ZRAMP simulations."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from archive_paths import activate as activate_shared_runtime  # noqa: E402

activate_shared_runtime()

from zramp_runtime.support_expert_bank import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
