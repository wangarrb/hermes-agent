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
        "backfill_missing_daily": False,
        "_offline_reflect_llm_args": [],
        "llm_profile": "oneapi-auto-cheep",
        "poll": 60,
        "timeout": 0,
        "prefilter": "safe",
        "weekly_budget_max_pending_units": 200,
        "weekly_budget_max_pending_chars": 10_000_000,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cron_weekly_budget_and_submit_do_not_backfill_by_default():
    module = load_module()
    current = args()
    assert "--backfill-missing-daily" not in module.weekly_budget_cmd(current, "2026-W41")
    assert "--backfill-missing-daily" not in module.weekly_submit_cmd(current, "2026-W41")


def test_cron_weekly_backfill_requires_explicit_opt_in():
    module = load_module()
    requested = args(backfill_missing_daily=True)
    assert "--backfill-missing-daily" in module.weekly_budget_cmd(requested, "2026-W41")
    assert "--backfill-missing-daily" in module.weekly_submit_cmd(requested, "2026-W41")
