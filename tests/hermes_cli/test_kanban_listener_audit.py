"""Tests for rule-based /listen-kanban coordinator audit behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
import hermes_cli.kanban_listener as listener_mod
from hermes_cli.kanban_listener import (
    ListenerState,
    READY_UNCLAIMED_SECONDS,
    _audit_routing_guard,
    _find_ready_task,
    _maybe_listener_health_check,
    _repair_global_running_claims,
    _live_profile_pids_by_profile,
)


class _NoColor:
    GREEN = ""
    YELLOW = ""
    RESET = ""


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_ready_old(conn, task_id: str, *, age_seconds: int) -> None:
    old_ts = int(time.time()) - age_seconds
    conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (old_ts, task_id))
    conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))


def test_live_profile_detection_counts_codex_and_deepseek_bridge_listeners(monkeypatch):
    """Interactive bridge watchers are real Kanban listeners for their profile."""
    cmdlines = {
        1001: [
            "python3",
            "/home/wyr/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py",
            "--watch-only",
            "--profile",
            "implementer",
            "--board",
            "egomotion4d",
        ],
        1002: [
            "python3",
            "/home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/codex_kanban_interactive.py",
            "--profile=planner",
            "--board",
            "egomotion4d",
        ],
        1003: [
            "/home/wyr/.local/bin/hermes",
            "kanban",
            "--board",
            "egomotion4d",
            "list",
        ],
    }

    monkeypatch.setattr(listener_mod.os, "listdir", lambda path: [*(str(k) for k in cmdlines), "self"])
    monkeypatch.setattr(listener_mod, "_proc_cmdline", lambda pid: cmdlines.get(pid, []))
    monkeypatch.setattr(listener_mod, "_proc_environ_profile", lambda pid: "")

    live = _live_profile_pids_by_profile({"implementer", "planner"})

    assert live == {
        "implementer": {1001},
        "planner": {1002},
    }


def test_live_profile_detection_counts_bridge_claim_assignees(monkeypatch):
    """Assist-mode bridge watchers count as live listeners for every claim assignee."""
    cmdlines = {
        1001: [
            "python3",
            "/home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/codex_kanban_interactive.py",
            "--watch-child",
            "--profile",
            "planner",
            "--claim-assignees",
            "planner,implementer",
            "--board",
            "egomotion4d",
        ],
        1002: [
            "python3",
            "/home/wyr/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py",
            "--profile=critic",
            "--claim-assignees=critic,implementer",
            "--board",
            "egomotion4d",
        ],
    }

    monkeypatch.setattr(listener_mod.os, "listdir", lambda path: [*(str(k) for k in cmdlines), "self"])
    monkeypatch.setattr(listener_mod, "_proc_cmdline", lambda pid: cmdlines.get(pid, []))
    monkeypatch.setattr(listener_mod, "_proc_environ_profile", lambda pid: "")

    live = _live_profile_pids_by_profile({"planner", "critic", "implementer"})

    assert live == {
        "planner": {1001},
        "critic": {1002},
        "implementer": {1001, 1002},
    }


def test_routing_guard_treats_assist_listener_as_known_implementer(kanban_home):
    """An explicit assist listener for implementer must not create routing noise."""
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="Implementer work", assignee="implementer")
        _make_ready_old(conn, ready_id, age_seconds=READY_UNCLAIMED_SECONDS + 60)

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator", "planner"},
            live_pids={"planner": {1001}, "implementer": {1001}},
            C=_NoColor,
        )

        assert repairs == 0
        followups = conn.execute(
            "SELECT id FROM tasks WHERE assignee='coordinator' AND status!='archived'"
        ).fetchall()
        assert followups == []
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='needs-worker-health'",
            (ready_id,),
        ).fetchone()
        assert event is not None
        payload = json.loads(event["payload"])
        assert payload["listener_alive"] is True
        assert payload["live_profile_pids"] == [1001]


def test_routing_guard_suppresses_busy_listener_backlog(kanban_home):
    """A live profile already running one task may legitimately queue ready work.

    This is the T3-running / T4-ready false-alarm pattern: do not create a
    coordinator routing follow-up just because a later ready task is older than
    READY_UNCLAIMED_SECONDS.
    """
    with kb.connect() as conn:
        running_id = kb.create_task(conn, title="T3 active", assignee="implementer")
        assert kb.claim_task(conn, running_id, claimer="host:12345") is not None
        ready_id = kb.create_task(conn, title="T4 queued", assignee="implementer")
        _make_ready_old(conn, ready_id, age_seconds=READY_UNCLAIMED_SECONDS + 60)

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator", "implementer"},
            live_pids={"implementer": {12345}},
            C=_NoColor,
        )

        assert repairs == 0
        followups = conn.execute(
            "SELECT id FROM tasks WHERE assignee='coordinator' AND status!='archived'"
        ).fetchall()
        assert followups == []


def test_routing_guard_records_health_event_for_live_but_idle_stale_ready_task(kanban_home):
    """Live process with no running kanban task is health noise, not routing.

    This catches panes restored with --continue but not actually listening, but
    it should not call the coordinator model for a deterministic health issue.
    """
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="T4 queued", assignee="implementer")
        _make_ready_old(conn, ready_id, age_seconds=READY_UNCLAIMED_SECONDS + 60)

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator", "implementer"},
            live_pids={"implementer": {12345}},
            C=_NoColor,
        )

        assert repairs == 0
        followups = conn.execute(
            "SELECT title, body FROM tasks WHERE assignee='coordinator' AND status!='archived'"
        ).fetchall()
        assert followups == []

        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='needs-worker-health'",
            (ready_id,),
        ).fetchone()
        assert event is not None
        payload = json.loads(event["payload"])
        assert payload["listener_alive"] is True
        assert payload["active_running_task_ids"] == []
        assert payload["health_kind"] == "live_idle_unclaimed"


def test_routing_guard_records_health_event_for_missing_known_listener(kanban_home):
    """Known profiles without a live listener should not invoke coordinator."""
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="T4 queued", assignee="implementer")
        _make_ready_old(conn, ready_id, age_seconds=READY_UNCLAIMED_SECONDS + 60)

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator", "implementer"},
            live_pids={},
            C=_NoColor,
        )

        assert repairs == 0
        followups = conn.execute(
            "SELECT id FROM tasks WHERE assignee='coordinator' AND status!='archived'"
        ).fetchall()
        assert followups == []
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='needs-worker-health'",
            (ready_id,),
        ).fetchone()
        assert event is not None
        payload = json.loads(event["payload"])
        assert payload["listener_alive"] is False
        assert payload["health_kind"] == "no_live_listener"


def test_routing_guard_still_creates_followup_for_unknown_assignee(kanban_home):
    """Unknown assignees remain routing decisions for coordinator."""
    with kb.connect() as conn:
        ready_id = kb.create_task(conn, title="Needs route", assignee="unknown-lane")

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator", "implementer"},
            live_pids={},
            C=_NoColor,
        )

        assert repairs == 1
        followup = conn.execute(
            "SELECT title, body FROM tasks WHERE assignee='coordinator' AND status!='archived'"
        ).fetchone()
        assert followup is not None
        assert "Coordinator routing follow-up" in followup["title"]
        assert "assignee 'unknown-lane' has no profile on disk" in followup["body"]
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='needs-routing'",
            (ready_id,),
        ).fetchone()
        assert event is not None


def test_routing_guard_does_not_rewrap_coordinator_audit_followup(kanban_home):
    """Coordinator audit follow-ups must not recursively create follow-ups."""
    with kb.connect() as conn:
        followup_id = kb.create_task(
            conn,
            title="Coordinator routing follow-up: T4 queued",
            body="Rule-based coordinator routing guard created this follow-up.",
            assignee="coordinator",
            created_by="coordinator-audit",
        )
        _make_ready_old(conn, followup_id, age_seconds=READY_UNCLAIMED_SECONDS + 60)

        repairs = _audit_routing_guard(
            conn,
            known_profiles={"coordinator"},
            live_pids={"coordinator": {12345}},
            C=_NoColor,
        )

        assert repairs == 0
        nested = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE 'Coordinator routing follow-up: Coordinator routing follow-up:%'"
        ).fetchall()
        assert nested == []


def test_hermes_listener_delays_assist_claim_until_ready_age(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="implementer gets first chance", assignee="implementer")

    task = _find_ready_task("", "implementer", primary_assignee="critic", assist_claim_delay_s=60)
    assert task is None

    with kb.connect() as conn:
        old_ts = int(time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    task = _find_ready_task("", "implementer", primary_assignee="critic", assist_claim_delay_s=60)
    assert task is not None
    assert task["id"] == task_id


def test_hermes_listener_uses_per_assignee_assist_claim_delay(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="critic waits for implementer lane", assignee="implementer")

    task = _find_ready_task(
        "",
        "implementer",
        primary_assignee="critic",
        assist_claim_delay_s=0,
        assist_claim_delays={"implementer": 60},
    )
    assert task is None

    with kb.connect() as conn:
        old_ts = int(time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    task = _find_ready_task(
        "",
        "implementer",
        primary_assignee="critic",
        assist_claim_delay_s=0,
        assist_claim_delays={"implementer": 60},
    )
    assert task is not None
    assert task["id"] == task_id


def test_hermes_listener_uses_profile_qualified_assist_claim_delay(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="backup profile waits for implementer lane", assignee="implementer")

    task = _find_ready_task(
        "",
        "implementer",
        primary_assignee="backup_immplementer",
        assist_claim_delay_s=0,
        assist_claim_profile_delays={("backup_immplementer", "implementer"): 60},
    )
    assert task is None

    with kb.connect() as conn:
        old_ts = int(time.time()) - 120
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, task_id))

    task = _find_ready_task(
        "",
        "implementer",
        primary_assignee="backup_immplementer",
        assist_claim_delay_s=0,
        assist_claim_profile_delays={("backup_immplementer", "implementer"): 60},
    )
    assert task is not None
    assert task["id"] == task_id


def test_hermes_listener_profile_delay_shorthand_defaults_to_implementer(monkeypatch):
    monkeypatch.setenv(
        "HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS",
        "backup_immplementer:60,45",
    )

    delays = listener_mod._resolve_assist_claim_profile_delays()

    assert delays[("backup_immplementer", "implementer")] == 60
    assert delays[("implementer", "implementer")] == 45


def test_coordinator_health_tick_uses_fast_interval_even_when_quiet(monkeypatch):
    """Coordinator rule audit must not wait quiet-hour 30min cadence."""
    state = ListenerState(board="egomotion4d", assignee="coordinator")
    state._last_health_check = 1000.0
    calls = []
    clock = {"t": 1059.0}

    monkeypatch.setattr(listener_mod.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(listener_mod, "_daytime_health_seconds", lambda: 1800.0)
    monkeypatch.setattr(listener_mod, "_listener_health_check", lambda cli_ref, state: calls.append(state.assignee))

    _maybe_listener_health_check(object(), state)
    assert calls == []

    clock["t"] = 1060.1
    _maybe_listener_health_check(object(), state)
    assert calls == ["coordinator"]


def test_coordinator_reclaims_idle_deepseek_pane_despite_recent_heartbeat(kanban_home, monkeypatch):
    """Passive heartbeats must not hide a zellij-restarted idle pane forever."""
    now = 10_000
    pid = 123456
    host = "test-host"
    lock = f"{host}:{pid}:deepseek-interactive"

    monkeypatch.setattr(listener_mod.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_pid_alive", lambda claimed_pid: int(claimed_pid) == pid)
    monkeypatch.setattr(
        listener_mod,
        "_proc_cmdline",
        lambda claimed_pid: [
            "python3",
            "deepseek_kanban_interactive.py",
            "--watch-child",
            "--profile",
            "critic",
            "--zellij-session",
            "kanban-egomotion4d",
            "--zellij-pane-id",
            "3",
        ],
    )
    monkeypatch.setattr(
        listener_mod,
        "_zellij_dump_screen",
        lambda *, session, pane_id: "Hi，有什么想继续的？\n编写任务或使用 /。\n",
    )

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="stale interactive task", assignee="critic")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=lock)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, pid)  # type: ignore[attr-defined]
        old_started = now - 61
        conn.execute("UPDATE tasks SET started_at=?, workspace_path=? WHERE id=?", (old_started, "/tmp/repo", task_id))
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (old_started, task_id))
        kb._append_event(
            conn,
            task_id,
            "heartbeat",
            {"note": "deepseek-interactive waiting for complete/block from DeepSeek TUI"},
            run_id=claimed.current_run_id,
        )
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=? AND kind='heartbeat'", (now, task_id))

        repairs = _repair_global_running_claims(conn, current_task_id="", C=_NoColor)

        assert repairs == 1
        row = conn.execute("SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='coordinator_audit_reclaimed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert "idle interactive pane" in payload["reason"]


def test_coordinator_reclaims_stalled_busy_deepseek_pane_despite_recent_heartbeat(kanban_home, monkeypatch):
    """A frozen 'working' DeepSeek pane with only passive heartbeats is stale."""
    now = 20_000
    pid = 234567
    host = "test-host"
    lock = f"{host}:{pid}:deepseek-interactive"

    monkeypatch.setattr(listener_mod.time, "time", lambda: now)
    monkeypatch.setattr(kb, "_pid_alive", lambda claimed_pid: int(claimed_pid) == pid)
    monkeypatch.setattr(
        listener_mod,
        "_proc_cmdline",
        lambda claimed_pid: [
            "python3",
            "deepseek_kanban_interactive.py",
            "--watch-child",
            "--profile",
            "implementer",
            "--zellij-session",
            "kanban-egomotion4d",
            "--zellij-pane-id",
            "1",
        ],
    )
    monkeypatch.setattr(
        listener_mod,
        "_zellij_dump_screen",
        lambda *, session, pane_id: "KANBAN_TASK_BOUNDARY\nHermes Kanban 已领取任务 t_demo\n工作中\n",
    )
    monkeypatch.setattr(listener_mod, "_interactive_tui_has_external_child", lambda pid, workspace_path: False)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="stalled busy interactive task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=lock)
        assert claimed is not None
        kb._set_worker_pid(conn, task_id, pid)  # type: ignore[attr-defined]
        old_started = now - 601
        conn.execute("UPDATE tasks SET started_at=?, workspace_path=? WHERE id=?", (old_started, "/tmp/repo", task_id))
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (old_started, task_id))
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_started, task_id))
        kb._append_event(
            conn,
            task_id,
            "heartbeat",
            {"note": "deepseek-interactive waiting for complete/block from DeepSeek TUI"},
            run_id=claimed.current_run_id,
        )
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=? AND kind='heartbeat'", (now, task_id))

        repairs = _repair_global_running_claims(conn, current_task_id="", C=_NoColor)

        assert repairs == 1
        row = conn.execute("SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None
        event = conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='coordinator_audit_reclaimed' ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
        payload = json.loads(event["payload"])
        assert "stalled busy interactive pane" in payload["reason"]
