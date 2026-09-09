"""Tests for hindsight backfill tools (gap sessions + codex rollouts).

Covers the pure logic of both one-off migration scripts added 2026-09-04 to
backfill the 08-27~09-04 ingest blackout. Read-only against real state.db /
rollout files; no Hindsight writes.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import json

import pytest

SCRIPTS = Path("/home/wyr/.hermes/scripts")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bgs():
    return _load("hindsight_backfill_gap_sessions")


@pytest.fixture(scope="module")
def bcr():
    return _load("hindsight_backfill_codex_rollouts")


# ── gap sessions (state.db -> session JSON) ──────────────────────────────


class TestGapSessions:
    def test_gap_sessions_respects_min_messages(self, bgs):
        sessions = bgs.gap_sessions(Path("/home/wyr/.hermes/profiles/designer/state.db"))
        assert sessions, "expected real gap sessions in designer state.db"
        assert all(s["ua_msgs"] >= bgs.MIN_MESSAGES for s in sessions)

    def test_gap_sessions_windows_sane(self, bgs):
        sessions = bgs.gap_sessions(Path("/home/wyr/.hermes/profiles/designer/state.db"))
        assert all(s["t0"] <= s["t1"] for s in sessions)

    def test_export_messages_filters_roles_and_has_epoch_ts(self, bgs):
        sessions = bgs.gap_sessions(Path("/home/wyr/.hermes/profiles/designer/state.db"))
        msgs = bgs.export_messages(
            Path("/home/wyr/.hermes/profiles/designer/state.db"), sessions[0]["session_id"]
        )
        assert msgs
        assert all(m["role"] in ("user", "assistant") for m in msgs)
        assert all(str(m["timestamp"]).isdigit() for m in msgs)

    def test_build_snapshot_schema(self, bgs):
        session = {"session_id": "x", "t0": 1788500000, "t1": 1788500001, "model": "m"}
        messages = [{"role": "user", "content": "hi", "timestamp": "1788500000"}]
        snap = bgs.build_snapshot(session, messages)
        assert snap["message_count"] == 1
        assert snap["platform"] == "backfill"
        assert snap["messages"] == messages
        assert snap["session_start"].startswith("2026-")

    def test_codex_backfill_files_exist(self):
        files = list(Path("/home/wyr/.hermes/sessions").glob("session_codex-*.json"))
        files += Path("/home/wyr/.hermes/profiles/reviewer/sessions").glob("session_codex-*.json")
        assert len(files) >= 20, "codex backfill snapshots missing"


# ── codex rollouts (JSONL -> session JSON) ───────────────────────────────


class TestCodexRollouts:
    def _real_rollout(self) -> Path:
        base = Path("/home/wyr/.codex-kanban/reviewer/sessions")
        for p in base.rglob("rollout-2026-09-02T00-22-21*.jsonl"):
            return p
        pytest.skip("sample rollout not present")

    def test_convert_filters_scaffolding(self, bcr):
        snap = bcr.convert_rollout(str(self._real_rollout()))
        assert snap and snap["message_count"] >= 3
        bad_prefixes = ("# AGENTS.md instructions", "<INSTRUCTIONS>", "<environment_context>")
        assert not any(m["content"].startswith(bad_prefixes) for m in snap["messages"])

    def test_convert_roles_and_id(self, bcr):
        snap = bcr.convert_rollout(str(self._real_rollout()))
        assert all(m["role"] in ("user", "assistant") for m in snap["messages"])
        assert snap["session_id"].startswith("codex-")

    def test_default_profile_maps_to_main_sessions_dir(self, bcr):
        assert str(bcr.OUT_MAIN) == "/home/wyr/.hermes/sessions"

    def test_kanban_backfill_files_exist(self):
        count = 0
        for profile in ("designer", "coordinator", "planner"):
            base = Path("/home/wyr/.hermes/profiles") / profile / "sessions"
            for f in base.glob("session_*.json"):
                try:
                    if json.loads(f.read_text()).get("platform") == "backfill":
                        count += 1
                except Exception:
                    pass
        assert count >= 20, "kanban backfill snapshots missing"
