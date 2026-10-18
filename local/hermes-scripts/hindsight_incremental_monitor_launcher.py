#!/usr/bin/env python3
"""Read-only monitoring launcher for the active Hindsight scripts."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / 'hindsight_incremental_monitor.py'
if not SCRIPT.is_file():
    raise SystemExit('active read-only monitor unavailable; no repair or fallback is attempted')
sys.path.insert(0, str(ROOT))
runpy.run_path(str(SCRIPT), run_name='__main__')
