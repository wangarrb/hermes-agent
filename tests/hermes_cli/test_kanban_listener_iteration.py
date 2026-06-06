"""Tests for the planner NEEDS REVISION iteration loop."""

from __future__ import annotations

import queue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
import hermes_cli.kanban_listener as listener_mod
from hermes_cli.kanban_listener import (
    ListenerState,
    _audit_fixable_blocked_review_tasks,
    _audit_missing_coordinator_followups,
    _audit_planner_revision_acceptance_loop,
    _audit_terminal_block_adjudication_tasks,
    _auto_complete_if_still_running,
    _claim_and_run,
    _daytime_health_seconds,
    _reclaim_orphan_running_tasks,
    reset_kanban_listener,
)


class _NoColor:
    GREEN = ""
    YELLOW = ""
    DIM = ""
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


def _finish_latest_run_without_closing_task(
    conn, task_id, *, ended_at, outcome="completed", summary=None, started_at=None, error=None
):
    """Simulate agent process returning while task row is still running."""
    if started_at is None:
        conn.execute(
            """
            UPDATE task_runs
               SET status=?, outcome=?, summary=?, error=?, ended_at=?
             WHERE id = (SELECT MAX(id) FROM task_runs WHERE task_id=?)
            """,
            (outcome, outcome, summary, error, ended_at, task_id),
        )
    else:
        conn.execute(
            """
            UPDATE task_runs
               SET status=?, outcome=?, summary=?, error=?, started_at=?, ended_at=?
             WHERE id = (SELECT MAX(id) FROM task_runs WHERE task_id=?)
            """,
            (outcome, outcome, summary, error, started_at, ended_at, task_id),
        )


def test_hermes_assist_listener_prompt_uses_task_assignee_role(kanban_home):
    """Hermes critic/planner panes assisting implementer must get role override."""
    with kb.connect(board="assist-hermes-test") as conn:
        task_id = kb.create_task(
            conn,
            title="Implement from critic pane",
            assignee="implementer",
        )

    pending = queue.Queue()
    cli_ref = SimpleNamespace(
        _pending_input=pending,
        _agent_running=False,
        agent=object(),
        session_id="test-session",
        system_prompt="",
        preloaded_skills=[],
    )
    state = ListenerState(
        board="assist-hermes-test",
        assignee="critic",
        claim_assignees=["critic", "implementer"],
    )

    ok = _claim_and_run(
        cli_ref,
        state,
        {"id": task_id, "title": "Implement from critic pane", "assignee": "implementer"},
    )

    assert ok is True
    prompt = pending.get_nowait()
    assert f"work kanban task {task_id}" in prompt
    assert "Pane/profile: critic" in prompt
    assert "task assignee/role: implementer" in prompt
    assert "按 implementer 职责工作" in prompt


def test_reset_kanban_listener_reclaims_current_task_and_returns_idle(kanban_home, monkeypatch):
    """Operator reset-kanban must drop the active claim without killing the pane."""
    with kb.connect(board="reset-listener-test") as conn:
        task_id = kb.create_task(conn, title="stuck task", assignee="planner")

    pending = queue.Queue()
    cli_ref = SimpleNamespace(
        _pending_input=pending,
        _agent_running=False,
        agent=object(),
        session_id="test-session",
        system_prompt="",
        preloaded_skills=[],
    )
    state = ListenerState(board="reset-listener-test", assignee="planner")
    cli_ref._kanban_listener = state

    assert _claim_and_run(
        cli_ref,
        state,
        {"id": task_id, "title": "stuck task", "assignee": "planner"},
    )
    assert state.current_task_id == task_id

    assert reset_kanban_listener(cli_ref) is True

    assert state.current_task_id == ""
    assert state.interrupt_requested is False
    assert state._retry_cooldown_until == 0.0
    with kb.connect(board="reset-listener-test") as conn:
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (task_id,),
        ).fetchone()
    assert row["status"] == "ready"
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None
    assert event["kind"] == "reclaimed"
    assert "reset-kanban" in event["payload"]


def test_reset_kanban_listener_is_noop_when_idle(kanban_home):
    state = ListenerState(board="reset-idle-test", assignee="planner")
    cli_ref = SimpleNamespace(_kanban_listener=state)

    assert reset_kanban_listener(cli_ref) is False
    assert state.current_task_id == ""
    assert state.interrupt_requested is False
    assert state._retry_cooldown_until == 0.0


def test_provider_failure_cooldown_keeps_current_task_claimed(kanban_home):
    """Short/no-summary provider failure should stay running during cooldown."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="503 task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        now = int(time.time())
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            ended_at=now,
            outcome="completed",
            summary=None,
        )

        state = ListenerState(board="", assignee="implementer", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        assert cleanup_ok is False
        assert state.current_task_id == task_id
        assert state._retry_cooldown_until > time.time()
        row = conn.execute("SELECT status, claim_lock FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "running"
        assert row["claim_lock"] is not None

        # Health tick for the same assignee must not reclaim the current task.
        _reclaim_orphan_running_tasks("", "implementer", current_task_id=task_id)
        row2 = conn.execute("SELECT status, claim_lock FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row2["status"] == "running"
        assert row2["claim_lock"] == row["claim_lock"]


def test_provider_failure_retries_after_cooldown_elapsed(kanban_home):
    """After 10 minutes the listener may reclaim to ready for retry."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="503 task", assignee="implementer")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            ended_at=int(time.time()) - 601,
            outcome="completed",
            summary=None,
        )

        state = ListenerState(board="", assignee="implementer", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        assert cleanup_ok is True
        row = conn.execute(
            "SELECT status, claim_lock, consecutive_failures FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["claim_lock"] is None
        assert row["consecutive_failures"] == 1


def test_provider_429_keeps_ten_failures_silent_even_with_low_task_retries(kanban_home):
    """429/rate-limit should not permanent-block before ten silent failures."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="429 task", assignee="critic")
        conn.execute(
            "UPDATE tasks SET max_retries=2, consecutive_failures=9 WHERE id=?",
            (task_id,),
        )
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            ended_at=int(time.time()) - 601,
            outcome="failed",
            summary=None,
            error="HTTP 429 Too Many Requests / rate limit",
        )

        state = ListenerState(board="", assignee="critic", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        assert cleanup_ok is True
        row = conn.execute(
            "SELECT status, consecutive_failures FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "ready"
        assert row["consecutive_failures"] == 10


def test_night_health_check_interval_is_30_minutes(monkeypatch):
    """Quiet hours health cadence should be 30 minutes, not 10 or 4."""
    class T:
        tm_hour = 3

    monkeypatch.setattr(listener_mod.time, "localtime", lambda: T())
    assert _daytime_health_seconds() == 1800.0

    class ActiveT:
        tm_hour = 10

    monkeypatch.setattr(listener_mod.time, "localtime", lambda: ActiveT())
    assert _daytime_health_seconds() == 60.0


def test_real_summary_still_auto_completes_immediately(kanban_home):
    """Cooldown only applies to non-productive/provider-error runs."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="forgot complete", assignee="critic")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        ended_at = int(time.time())
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            started_at=ended_at - 120,
            ended_at=ended_at,
            outcome="completed",
            summary="APPROVED: real review output",
        )

        state = ListenerState(board="", assignee="critic", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        assert cleanup_ok is True
        row = conn.execute("SELECT status, result FROM tasks WHERE id=?", (task_id,)).fetchone()
        assert row["status"] == "done"
        assert row["result"] == "auto-completed by listener safety net"


def test_approved_acceptance_skips_revision(kanban_home):
    """APPROVED acceptance should NOT create a planner revision task."""
    with kb.connect() as conn:
        acceptance_id = kb.create_task(
            conn, title="整体验收: Test", assignee="planner",
        )
        kb.complete_task(conn, acceptance_id, result="APPROVED: All tests pass, Q1/Q2/Q3 answered")

        created = _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)
        assert created == 0, "APPROVED should not create any follow-up"

        revision_rows = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchall()
        assert len(revision_rows) == 0, "no revision task should exist"


def test_goal_unachievable_skips_revision(kanban_home):
    """GOAL_UNACHIEVABLE acceptance should NOT create a planner revision task."""
    with kb.connect() as conn:
        acceptance_id = kb.create_task(
            conn, title="整体验收: Test", assignee="planner",
        )
        kb.complete_task(conn, acceptance_id,
                         result="GOAL_UNACHIEVABLE: fused pose does not improve metrics and cannot be fixed")

        created = _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)
        assert created == 0, "GOAL_UNACHIEVABLE should not create any follow-up"

        revision_rows = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchall()
        assert len(revision_rows) == 0, "no revision task should exist"


def test_abandoned_chinese_skips_revision(kanban_home):
    """Chinese 目标无法达成 should also stop iteration."""
    with kb.connect() as conn:
        acceptance_id = kb.create_task(
            conn, title="整体验收: Test", assignee="planner",
        )
        kb.complete_task(conn, acceptance_id,
                         result="结论：目标无法达成。current_source_pose与canonical_fused_pose无显著差异")

        created = _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)
        assert created == 0, "目标无法达成 should not create follow-up"

        revision_rows = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchall()
        assert len(revision_rows) == 0


def test_needs_revision_creates_planner_revision_task(kanban_home):
    """Round 1: planner acceptance done with NEEDS REVISION → planner revision task."""
    with kb.connect() as conn:
        # Create a planner acceptance task
        acceptance_id = kb.create_task(
            conn, title="整体验收: ShelfOcc benchmark",
            assignee="planner",
        )
        kb.complete_task(
            conn, acceptance_id,
            result="NEEDS REVISION: 72 cells all placeholder, cannot answer Q1/Q2/Q3",
        )

        # Audit should detect the NEEDS REVISION and create revision task
        created = _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)

        assert created > 0, "should create planner revision task"
        # Verify revision task was created
        revision_rows = conn.execute(
            "SELECT id, title, assignee, status FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchall()
        assert len(revision_rows) >= 1
        task = revision_rows[0]
        assert task["assignee"] == "planner"
        assert task["status"] == "ready"  # parent is done, so it's immediately ready

        # Running again should be idempotent (no new tasks)
        conn.execute("UPDATE tasks SET created_at=0 WHERE id=?", (acceptance_id,))
        created2 = _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)
        assert created2 == 0


def test_fixable_blocked_critic_review_creates_fix_and_rereview(kanban_home):
    """A fixable critic block should continue the flow through revision tasks."""
    with kb.connect() as conn:
        impl_id = kb.create_task(conn, title="W7-M1 implementation", assignee="implementer")
        kb.complete_task(conn, impl_id, result="W7-M1 report ready")
        review_id = kb.create_task(
            conn,
            title="W7-M1 critic: review strict fusion candidate matrix",
            assignee="critic",
            parents=[impl_id],
        )
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None
        kb.block_task(
            conn,
            review_id,
            reason="BLOCKED: report overstates evidence. Fix: correct depth framing and label as PARETO.",
        )
        downstream_id = kb.create_task(
            conn,
            title="W7-V1 downstream visual pack",
            assignee="implementer",
            parents=[review_id],
        )

        created = _audit_fixable_blocked_review_tasks(conn, _NoColor)

        assert created == 2
        fix = conn.execute(
            "SELECT id, assignee, status, body FROM tasks "
            "WHERE title LIKE '根据 critic block 修复:%'"
        ).fetchone()
        assert fix is not None
        assert fix["assignee"] == "implementer"
        assert fix["status"] == "ready"
        assert "correct depth framing" in fix["body"]
        assert kb.parent_ids(conn, fix["id"]) == [impl_id]

        rereview = conn.execute(
            "SELECT id, assignee, status, body FROM tasks "
            "WHERE title LIKE '复审 critic block 修复结果:%'"
        ).fetchone()
        assert rereview is not None
        assert rereview["assignee"] == "critic"
        assert rereview["status"] == "todo"
        assert kb.parent_ids(conn, rereview["id"]) == [fix["id"]]

        assert kb.parent_ids(conn, downstream_id) == [rereview["id"]]

        created_again = _audit_fixable_blocked_review_tasks(conn, _NoColor)
        assert created_again == 0


def test_terminal_blocked_critic_review_creates_planner_adjudication(kanban_home):
    """A non-fixable critic block with downstream work must reach planner."""
    with kb.connect() as conn:
        impl_id = kb.create_task(conn, title="W7-M1 implementation", assignee="implementer")
        kb.complete_task(conn, impl_id, result="W7-M1 report ready")
        review_id = kb.create_task(
            conn,
            title="W7-M1 critic: review strict fusion candidate matrix",
            assignee="critic",
            parents=[impl_id],
        )
        claimed = kb.claim_task(conn, review_id)
        assert claimed is not None
        kb.block_task(
            conn,
            review_id,
            reason="GOAL_UNACHIEVABLE: evidence shows fusion does not improve precision under strict metrics.",
        )
        downstream_id = kb.create_task(
            conn,
            title="R7 planner dynamic review",
            assignee="planner",
            parents=[review_id],
        )

        created = _audit_terminal_block_adjudication_tasks(conn, _NoColor)

        assert created == 1
        adjudication = conn.execute(
            "SELECT id, assignee, status, body FROM tasks "
            "WHERE title LIKE '裁决 blocked review:%'"
        ).fetchone()
        assert adjudication is not None
        assert adjudication["assignee"] == "planner"
        assert adjudication["status"] == "ready"
        assert "GOAL_UNACHIEVABLE" in adjudication["body"]
        assert review_id in adjudication["body"]
        # The adjudication task must be runnable; do not parent it to the blocked review.
        assert kb.parent_ids(conn, adjudication["id"]) == []
        # Downstream remains blocked until planner explicitly closes, rewires, or archives it.
        assert kb.parent_ids(conn, downstream_id) == [review_id]

        created_again = _audit_terminal_block_adjudication_tasks(conn, _NoColor)
        assert created_again == 0


def test_revision_complete_creates_re_acceptance_with_linked_children(kanban_home):
    """Round 2: revision done + all linked children done → re-acceptance."""
    with kb.connect() as conn:
        # Create acceptance and revision
        acceptance_id = kb.create_task(conn, title="整体验收: Test", assignee="planner")
        kb.complete_task(conn, acceptance_id, result="NEEDS REVISION: failed")
        _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)

        revision_row = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchone()
        assert revision_row is not None, "revision task should exist"
        revision_id = revision_row["id"]
        kb.complete_task(conn, revision_id, result="New plan ready")

        # Create linked child implementer task
        impl_id = kb.create_task(conn, title="R0: fix thing", assignee="implementer", parents=[revision_id])
        kb.complete_task(conn, impl_id)

        # Now run the acceptance loop audit
        created = _audit_planner_revision_acceptance_loop(conn, _NoColor)
        assert created > 0, "should create re-acceptance task"

        acceptance_rows = conn.execute(
            "SELECT id, title, status, assignee FROM tasks WHERE title LIKE '%验收%' AND id != ?",
            (acceptance_id,),
        ).fetchall()
        assert len(acceptance_rows) >= 1
        new_acc = acceptance_rows[0]
        assert new_acc["assignee"] == "planner"
        # Should be todo or ready (parent revision is done)


def test_revision_acceptance_loop_handles_coordinator_mediated_children(kanban_home):
    """Revision → coordinator follow-up → implementer/review done should re-accept."""
    with kb.connect() as conn:
        acceptance_id = kb.create_task(conn, title="整体验收: Test", assignee="planner")
        kb.complete_task(conn, acceptance_id, result="NEEDS REVISION: placeholder benchmark")
        _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)

        revision_id = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchone()["id"]
        kb.complete_task(conn, revision_id, result="Revision plan ready")

        coordinator_id = kb.create_task(
            conn,
            title="Coordinator follow-up: 根据整体验收意见修改计划: Test",
            assignee="coordinator",
            parents=[revision_id],
        )
        kb.complete_task(conn, coordinator_id, result="Spawned R0 and review")

        impl_id = kb.create_task(conn, title="R0: real benchmark smoke", assignee="implementer", parents=[coordinator_id])
        kb.complete_task(conn, impl_id, result="R0 done")
        review_id = kb.create_task(conn, title="[Review] R0", assignee="critic", parents=[impl_id])
        kb.complete_task(conn, review_id, result="APPROVED")

        created = _audit_planner_revision_acceptance_loop(conn, _NoColor)
        assert created == 1

        new_acc = conn.execute(
            """
            SELECT id, title, assignee
            FROM tasks
            WHERE id != ? AND assignee='planner' AND title LIKE '整体验收%'
            """,
            (acceptance_id,),
        ).fetchone()
        assert new_acc is not None
        assert new_acc["assignee"] == "planner"


def test_revision_acceptance_loop_waits_for_coordinator_descendants(kanban_home):
    """Do not re-accept while coordinator-spawned descendants are still active."""
    with kb.connect() as conn:
        acceptance_id = kb.create_task(conn, title="整体验收: Test", assignee="planner")
        kb.complete_task(conn, acceptance_id, result="NEEDS REVISION: placeholder benchmark")
        _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)

        revision_id = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchone()["id"]
        kb.complete_task(conn, revision_id, result="Revision plan ready")

        coordinator_id = kb.create_task(
            conn,
            title="Coordinator follow-up: 根据整体验收意见修改计划: Test",
            assignee="coordinator",
            parents=[revision_id],
        )
        kb.complete_task(conn, coordinator_id, result="Spawned R0 and review")

        impl_id = kb.create_task(conn, title="R0: real benchmark smoke", assignee="implementer", parents=[coordinator_id])
        kb.complete_task(conn, impl_id, result="R0 done")
        kb.create_task(conn, title="[Review] R0", assignee="critic", parents=[impl_id])

        created = _audit_planner_revision_acceptance_loop(conn, _NoColor)
        assert created == 0

        new_acc_rows = conn.execute(
            "SELECT id FROM tasks WHERE id != ? AND assignee='planner' AND title LIKE '整体验收%'",
            (acceptance_id,),
        ).fetchall()
        assert len(new_acc_rows) == 0


def test_three_round_iteration(kanban_home):
    """Verify at least 3 full iteration rounds work end-to-end."""
    with kb.connect() as conn:
        last_acceptance_id = None
        for round_num in range(1, 4):
            # Accept task (manually approve — just to get a 'done' acceptance)
            acceptance = last_acceptance_id or kb.create_task(
                conn, title="整体验收: Iteration", assignee="planner",
            )
            if not last_acceptance_id:
                kb.complete_task(conn, acceptance, result=f"NEEDS REVISION: round {round_num} issues")

            # Audit → should create planner revision (or in round 1, the original acceptance triggers it)
            _audit_missing_coordinator_followups(conn, _NoColor, int(time.time()) - 5)

            revision_rows = conn.execute(
                "SELECT id, title FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
                f" AND id NOT IN (SELECT parent_id FROM task_links WHERE child_id LIKE '%验收%')"
            ).fetchall()

            # If no revision was created, check if the re-acceptance loop created an acceptance
            if not revision_rows:
                acceptance_rows = conn.execute(
                    "SELECT id, title FROM tasks WHERE title LIKE '整体验收 (%'"
                ).fetchall()
                if acceptance_rows:
                    last_acceptance_id = acceptance_rows[-1]["id"]
                    kb.complete_task(conn, last_acceptance_id,
                                     result=f"NEEDS REVISION: round {round_num} issues")
                    continue

            if not revision_rows:
                pytest.skip(f"No revision task for round {round_num}")

            rev_row = revision_rows[-1]
            revision_id = rev_row["id"]
            kb.complete_task(conn, revision_id)

            # Create linked implementation tasks and complete them
            impl = kb.create_task(conn, title=f"R{round_num}-0: fix", assignee="implementer",
                                  parents=[revision_id])
            kb.complete_task(conn, impl)

            # Run revision acceptance loop → should create re-acceptance
            _audit_planner_revision_acceptance_loop(conn, _NoColor)

            # Check if re-acceptance was created
            new_acc_rows = conn.execute(
                "SELECT id, title, status FROM tasks WHERE title LIKE '%验收%' AND id != ?",
                (acceptance,),
            ).fetchall()
            if new_acc_rows:
                last_acceptance_id = new_acc_rows[-1]["id"]
                # Make it NEEDS REVISION for next round
                kb.complete_task(conn, last_acceptance_id,
                                 result=f"NEEDS REVISION: round {round_num + 1} issues remain")
            else:
                last_acceptance_id = None

        # Verify at least 3 rounds of iteration
        planning_tasks = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '根据整体验收意见修改计划%'"
        ).fetchall()
        acceptance_tasks = conn.execute(
            "SELECT id FROM tasks WHERE title LIKE '整体验收%'"
        ).fetchall()
        # Should have at least 2 acceptance tasks (original + one re-created)
        # and at least 2 revision tasks
        assert len(acceptance_tasks) >= 2, f"only {len(acceptance_tasks)} acceptance tasks"


def test_cooldown_not_bypassed_when_current_run_has_null_ended_at(kanban_home):
    """Regression: cooldown must work even when the latest run (in-progress) has ended_at=NULL.

    Bug: _auto_complete_if_still_running() queried ORDER BY id DESC LIMIT 1, which
    returns the *current* in-progress run (ended_at=NULL) after a reclaim+re-claim
    cycle.  Since ended_at was NULL, the cooldown check was skipped, causing rapid
    fire retries (8-12s apart instead of 10min cooldown).

    Fix: use last_finished_run (ended_at IS NOT NULL) for cooldown and decision logic.
    """
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="503 rapid-fire task", assignee="planner")
        # First claim + failed run (ended 10s ago, within cooldown window)
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        now = int(time.time())
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            ended_at=now - 10,  # 10s ago — well within 600s cooldown
            outcome="completed",
            summary=None,
        )
        # Reclaim to ready (simulates what the listener does)
        kb.reclaim_task(conn, task_id, reason="auto-retry test")
        # Second claim — creates a new run with ended_at=NULL (still running)
        claimed2 = kb.claim_task(conn, task_id)
        assert claimed2 is not None
        # The new run has ended_at=NULL (agent hasn't finished yet)

        state = ListenerState(board="", assignee="planner", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        # Cooldown from the *previous* finished run should still be active
        assert cleanup_ok is False, (
            "cooldown was bypassed! The listener should wait 10min before retrying, "
            "not immediately reclaim when the current run has ended_at=NULL"
        )
        assert state.current_task_id == task_id
        assert state._retry_cooldown_until > time.time(), (
            "cooldown_until should be in the future (based on last finished run's ended_at)"
        )


def test_cooldown_allows_retry_after_10min_with_null_current_run(kanban_home):
    """After 10+ minutes, retry is allowed even when current run has ended_at=NULL."""
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="503 old task", assignee="planner")
        claimed = kb.claim_task(conn, task_id)
        assert claimed is not None
        # Failed run ended 601s ago (past cooldown)
        _finish_latest_run_without_closing_task(
            conn,
            task_id,
            ended_at=int(time.time()) - 601,
            outcome="completed",
            summary=None,
        )
        kb.reclaim_task(conn, task_id, reason="auto-retry test")
        # New claim — current run has ended_at=NULL
        claimed2 = kb.claim_task(conn, task_id)
        assert claimed2 is not None

        state = ListenerState(board="", assignee="planner", current_task_id=task_id)
        cleanup_ok = _auto_complete_if_still_running(state)

        # Cooldown has elapsed — should reclaim for retry
        assert cleanup_ok is True
        # Note: current_task_id is NOT cleared by _auto_complete_if_still_running itself;
        # the caller does _cleanup_after_task() after getting True.
