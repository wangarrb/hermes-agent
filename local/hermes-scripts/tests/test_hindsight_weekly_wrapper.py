from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


WRAPPER = Path(__file__).resolve().parents[1] / "hindsight_weekly_wrapper.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("hindsight_weekly_wrapper_test", WRAPPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_weekly_wrapper_builds_deterministic_no_agent_command():
    module = load_wrapper()

    command = module.build_command("/venv/python")

    assert command == [
        "/venv/python",
        "/home/wyr/.hermes/scripts/hindsight_daily_noagent.py",
        "--mode",
        "full",
        "--include-wiki",
    ]
