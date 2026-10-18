import importlib.util
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    name = "hindsight_offline_cron_runner_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "hindsight_offline_cron_runner.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def args(**overrides):
    values = {
        "weekly_window": "all-history",
        "week_mode": "current",
        "backfill_missing_daily": False,
        "_offline_reflect_llm_args": [],
        "llm_profile": "oneapi-auto-cheep",
        "poll": 60,
        "timeout": 0,
        "prefilter": "safe",
        "weekly_budget_max_pending_units": 12,
        "weekly_budget_max_pending_chars": 500000,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cron_weekly_budget_and_submit_do_not_authorize_pre_checkpoint_repair_by_default():
    module = load_module()
    current = args()
    assert "--backfill-missing-daily" not in module.weekly_budget_cmd(current, "2026-W41")
    assert "--backfill-missing-daily" not in module.weekly_submit_cmd(current, "2026-W41")


def test_cron_weekly_submit_carries_enforced_configured_budget_limits():
    module = load_module()
    cmd = module.weekly_submit_cmd(args(), "2026-W41")

    assert "--enforce-budget" in cmd
    unit_flag = cmd.index("--budget-max-pending-units")
    char_flag = cmd.index("--budget-max-pending-chars")
    assert cmd[unit_flag + 1] == "12"
    assert cmd[char_flag + 1] == "500000"


def test_cron_weekly_stops_before_submit_if_budget_preflight_blocks(monkeypatch):
    import io

    module = load_module()
    calls = []
    monkeypatch.setattr(module, "status", lambda *_args: None)
    monkeypatch.setattr(module, "run", lambda cmd, _log: calls.append(list(cmd)))
    monkeypatch.setattr(module, "run_capture_json", lambda *_args: {"budget_decision": "blocked_budget_exceeded"})

    try:
        module.weekly(args(), io.StringIO())
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("cron weekly must fail closed before submit when budget is blocked")

    assert calls
    assert all("offline-reflect-llm" not in cmd for cmd in calls)


def test_cron_weekly_pre_checkpoint_repair_requires_explicit_opt_in():
    module = load_module()
    requested = args(backfill_missing_daily=True)
    assert "--backfill-missing-daily" in module.weekly_budget_cmd(requested, "2026-W41")
    assert "--backfill-missing-daily" in module.weekly_submit_cmd(requested, "2026-W41")
