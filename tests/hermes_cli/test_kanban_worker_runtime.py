"""Unit tests for shared interactive Kanban worker runtime helpers."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_worker_runtime as runtime


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _args(**kwargs) -> argparse.Namespace:
    defaults = {
        "profile": "planner",
        "claim_assignees": None,
        "assist_claim_delay_s": 0,
        "assist_claim_delay_for": None,
        "assist_claim_profile_delay": None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_claim_assignees_from_args_dedupes_profile_and_csv_values():
    args = _args(profile="critic", claim_assignees="critic,implementer, planner ,implementer")

    assert runtime.claim_assignees_from_args(args, default_profile="planner") == [
        "critic",
        "implementer",
        "planner",
    ]


def test_claim_policy_from_args_merges_global_and_profile_qualified_delays(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_ASSIST_CLAIM_DELAYS", "implementer=30,critic=45")
    monkeypatch.setenv("HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS", "backup_immplementer:implementer=300")
    args = _args(
        profile="backup_immplementer",
        claim_assignees=["backup_immplementer,implementer"],
        assist_claim_delay_s=90,
        assist_claim_delay_for=["planner=12"],
        assist_claim_profile_delay=["backup_immplementer:planner=60", "120"],
    )

    policy = runtime.claim_policy_from_args(args, default_profile="planner")

    assert policy.profile == "backup_immplementer"
    assert policy.claim_assignees == ["backup_immplementer", "implementer"]
    assert policy.assist_claim_delay_s == 90
    assert policy.assist_claim_delays == {
        "implementer": 30,
        "critic": 45,
        "planner": 12,
    }
    assert policy.assist_claim_profile_delays == {
        ("backup_immplementer", "implementer"): 300,
        ("backup_immplementer", "planner"): 60,
        ("implementer", "implementer"): 120,
    }
    assert runtime.assist_claim_delay_for(policy, "backup_immplementer") == 0
    assert runtime.assist_claim_delay_for(policy, "implementer") == 30
    assert runtime.assist_claim_delay_for(policy, "planner") == 12
    assert runtime.assist_claim_delay_for(policy, "unknown") == 90


def test_default_self_poll_owner_is_pane_scoped(monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_SELF_POLL_OWNER", raising=False)

    assert runtime.default_self_poll_owner(
        profile="critic",
        listener_kind="deepseek-self-poll",
        pane_id="pane/9",
        pid=123,
    ) == "critic-pane_9"

    monkeypatch.setenv("HERMES_KANBAN_SELF_POLL_OWNER", "manual owner")
    assert runtime.default_self_poll_owner(
        profile="critic",
        listener_kind="deepseek-self-poll",
        pane_id="pane/9",
        pid=123,
    ) == "manual_owner"


def test_select_ready_candidate_respects_assignee_order_and_assist_ready_age(kanban_home):
    workspace_board = "runtime-select-test"
    with kb.connect(board=workspace_board) as conn:
        implementer_id = kb.create_task(conn, title="waited implementer", assignee="implementer")
        planner_id = kb.create_task(conn, title="primary planner", assignee="planner")
        recent_ts = int(runtime.time.time())
        old_ts = recent_ts - 120
        conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (recent_ts, implementer_id))
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (recent_ts, implementer_id))
        conn.execute("UPDATE tasks SET created_at=? WHERE id=?", (recent_ts, planner_id))
        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (recent_ts, planner_id))

        policy = runtime.ClaimPolicy(
            profile="planner",
            claim_assignees=["implementer", "planner"],
            assist_claim_delay_s=60,
        )
        assert runtime.select_ready_candidate(conn, policy=policy).id == planner_id

        conn.execute("UPDATE task_events SET created_at=? WHERE task_id=?", (old_ts, implementer_id))
        assert runtime.select_ready_candidate(conn, policy=policy).id == implementer_id


def test_reset_interactive_claims_filters_board_workspace_and_listener_kind(kanban_home, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()
    other_workspace = tmp_path / "other"
    other_workspace.mkdir()
    board = "runtime-reset-test"
    host = socket.gethostname()

    with kb.connect(board=board) as conn:
        match_id = kb.create_task(conn, title="matching", assignee="planner")
        other_workspace_id = kb.create_task(conn, title="other workspace", assignee="planner")
        other_kind_id = kb.create_task(conn, title="other listener", assignee="planner")
        for task_id, ws, kind in [
            (match_id, workspace, "codex-interactive"),
            (other_workspace_id, other_workspace, "codex-interactive"),
            (other_kind_id, workspace, "deepseek-interactive"),
        ]:
            claimed = kb.claim_task(conn, task_id, ttl_seconds=3600, claimer=f"{host}:99999999:{kind}")
            assert claimed is not None
            kb.set_workspace_path(conn, task_id, ws)
            kb._set_worker_pid(conn, task_id, 99999999)  # type: ignore[attr-defined]

    reset_ids = runtime.reset_interactive_claims(
        board=board,
        profile="planner",
        claim_assignees=["planner"],
        workspace=workspace,
        listener_kind="codex-interactive",
        reason="operator reset-kanban",
    )

    assert reset_ids == [match_id]
    with kb.connect(board=board) as conn:
        rows = {
            row["id"]: row
            for row in conn.execute(
                "SELECT id, status, claim_lock, worker_pid FROM tasks ORDER BY id"
            ).fetchall()
        }
    assert rows[match_id]["status"] == "ready"
    assert rows[match_id]["claim_lock"] is None
    assert rows[match_id]["worker_pid"] is None
    assert rows[other_workspace_id]["status"] == "running"
    assert rows[other_workspace_id]["claim_lock"].endswith(":codex-interactive")
    assert rows[other_kind_id]["status"] == "running"
    assert rows[other_kind_id]["claim_lock"].endswith(":deepseek-interactive")
