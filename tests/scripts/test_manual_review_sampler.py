"""Tests for the manual_review sampler's daily review-call cap (2026-09-11).

The daily pipeline (00:01) and the Sunday weekly full run (05:00) both invoke
the sampler; the cap must span both invocations so a calendar day never exceeds
the configured number of review-LLM calls (user constraint: <=5/day).
Read-only: no Hindsight writes, no LLM calls.
"""
from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPTS = Path("/home/wyr/.hermes/scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestDailyReviewCap:
    def test_default_cap_is_5(self):
        smp = _load("hindsight_manual_review_sampler")
        assert smp.DEFAULT_MAX_DAILY_REVIEWS == 5

    def test_counts_utc_history_on_local_day(self):
        smp = _load("hindsight_manual_review_sampler")
        tz = timezone(timedelta(hours=8))
        now = datetime.now(tz)
        today_utc = now.astimezone(timezone.utc).isoformat()
        yesterday_utc = (now - timedelta(days=1)).astimezone(timezone.utc).isoformat()
        state = {
            "reviewed": {},
            "history": [
                {"at": yesterday_utc, "decisions": [{"x": 1}, {"x": 2}]},
                {"at": today_utc, "decisions": [{"x": 1}]},
            ],
        }
        assert smp.reviews_done_today(state, tz) == 1

    def test_ignores_malformed_entries(self):
        smp = _load("hindsight_manual_review_sampler")
        assert smp.reviews_done_today({}) == 0
        state = {"history": [{"at": "bogus", "decisions": [1]}, {"no": "at"}]}
        assert smp.reviews_done_today(state) == 0
