"""
Helper that injects the parent FROMTHEGROUND directory into sys.path so the
fixed modules (`model`, `build_map`, `node`, `sim_seeding`) can be imported
without making this package depend on the rest of the parent simulation code.

Import this module once at the top of any entry-point file in the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

PARENT_DIR = Path(__file__).resolve().parent.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
