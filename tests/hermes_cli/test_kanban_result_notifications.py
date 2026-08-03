"""Durable publisher-pane result notifications."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _subscriptions(conn, task_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT target_profile FROM kanban_result_subscriptions "
        "WHERE task_id = ? AND active = 1 ORDER BY target_profile",
        (task_id,),
    ).fetchall()
    return [str(row["target_profile"]) for row in rows]


def test_create_task_persists_one_explicit_result_subscriber(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review this",
            assignee="reviewer",
            result_subscriber="Planner",
        )
        assert _subscriptions(conn, task_id) == ["planner"]


def test_idempotent_create_ensures_requested_subscription(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="review this",
            assignee="reviewer",
            idempotency_key="review-once",
        )
        same_id = kb.create_task(
            conn,
            title="review this retry",
            assignee="reviewer",
            idempotency_key="review-once",
            result_subscriber="planner",
        )
        assert same_id == task_id
        assert _subscriptions(conn, task_id) == ["planner"]


def test_reassign_preserves_original_result_subscriber(kanban_home: Path) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="cross-role work",
            assignee="implementer",
            result_subscriber="reviewer",
        )
        assert kb.reassign_task(conn, task_id, "designer")
        assert _subscriptions(conn, task_id) == ["reviewer"]


def _create_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "title": "cross-role work",
        "body": None,
        "assignee": "reviewer",
        "created_by": "planner",
        "workspace": "scratch",
        "branch": None,
        "base_commit": None,
        "target_branch": None,
        "project": None,
        "tenant": None,
        "priority": 0,
        "parent": None,
        "triage": False,
        "idempotency_key": None,
        "max_runtime": None,
        "skills": None,
        "max_retries": None,
        "goal_mode": False,
        "goal_max_turns": None,
        "initial_status": "running",
        "json": False,
        "notify_origin": None,
        "notify_profile": None,
        "origin_profile": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.mark.parametrize(
    ("origin", "assignee", "notify_origin", "notify_profile", "expected"),
    [
        ("planner", "reviewer", None, None, ["planner"]),
        ("planner", "planner", None, None, []),
        ("planner", "planner", True, None, ["planner"]),
        ("planner", "reviewer", False, None, []),
        ("planner", "reviewer", None, "coordinator", ["coordinator"]),
        ("custom-a", "custom-b", None, None, ["custom-a"]),
    ],
)
def test_cli_create_result_subscription_policy(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
    assignee: str,
    notify_origin: bool | None,
    notify_profile: str | None,
    expected: list[str],
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_ORIGIN_PROFILE", origin)
    monkeypatch.setattr(kanban_cli, "_check_dispatcher_presence", lambda: (True, ""))
    args = _create_args(
        assignee=assignee,
        notify_origin=notify_origin,
        notify_profile=notify_profile,
    )
    assert kanban_cli._cmd_create(args) == 0
    with kb.connect() as conn:
        task_id = kb.list_tasks(conn)[0].id
        assert _subscriptions(conn, task_id) == expected


def test_cli_explicit_origin_overrides_environment(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_ORIGIN_PROFILE", "planner")
    monkeypatch.setattr(kanban_cli, "_check_dispatcher_presence", lambda: (True, ""))
    assert kanban_cli._cmd_create(_create_args(origin_profile="coordinator")) == 0
    with kb.connect() as conn:
        task_id = kb.list_tasks(conn)[0].id
        assert _subscriptions(conn, task_id) == ["coordinator"]


def test_cli_uses_existing_listener_profile_env_as_origin_fallback(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HERMES_KANBAN_ORIGIN_PROFILE", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_PROFILE", "planner")
    monkeypatch.setattr(kanban_cli, "_check_dispatcher_presence", lambda: (True, ""))
    assert kanban_cli._cmd_create(_create_args(assignee="reviewer")) == 0
    with kb.connect() as conn:
        task_id = kb.list_tasks(conn)[0].id
        assert _subscriptions(conn, task_id) == ["planner"]


def test_disabled_feature_skips_implicit_but_keeps_explicit_subscription(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_KANBAN_ORIGIN_PROFILE", "planner")
    monkeypatch.setenv("HERMES_KANBAN_RESULT_NOTIFICATIONS", "0")
    monkeypatch.setattr(kanban_cli, "_check_dispatcher_presence", lambda: (True, ""))

    assert kanban_cli._cmd_create(_create_args(title="implicit")) == 0
    assert kanban_cli._cmd_create(
        _create_args(title="explicit", notify_profile="planner")
    ) == 0

    with kb.connect() as conn:
        by_title = {task.title: task.id for task in kb.list_tasks(conn)}
        assert _subscriptions(conn, by_title["implicit"]) == []
        assert _subscriptions(conn, by_title["explicit"]) == ["planner"]


def _queue_rows(conn, profile: str = "planner"):
    return conn.execute(
        "SELECT * FROM kanban_result_queue WHERE target_profile = ? ORDER BY id",
        (profile,),
    ).fetchall()


def test_actionable_task_events_enqueue_once_without_terminal_io(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        completed = kb.create_task(
            conn, title="complete", assignee="reviewer", result_subscriber="planner"
        )
        blocked = kb.create_task(
            conn, title="block", assignee="reviewer", result_subscriber="planner"
        )
        dependency = kb.create_task(
            conn, title="dependency", assignee="reviewer", result_subscriber="planner"
        )
        gave_up = kb.create_task(
            conn, title="give up", assignee="reviewer", result_subscriber="planner"
        )

        assert kb.complete_task(conn, completed, summary="done")
        assert kb.block_task(conn, blocked, reason="needs decision")
        assert kb.block_task(conn, dependency, reason="wait", kind="dependency")
        assert kb._record_task_failure(
            conn,
            gave_up,
            error="terminal failure",
            outcome="crashed",
            force_trip=True,
        )

        assert [(row["task_id"], row["event_kind"]) for row in _queue_rows(conn)] == [
            (completed, "completed"),
            (blocked, "blocked"),
            (gave_up, "gave_up"),
        ]


def test_return_for_rework_notifies_root_once_and_subscribed_descendant_once(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        root = kb.create_task(
            conn, title="root", assignee="planner", result_subscriber="reviewer"
        )
        child = kb.create_task(
            conn,
            title="child",
            assignee="implementer",
            parents=(root,),
            result_subscriber="reviewer",
        )
        kb.return_task_for_rework(
            conn, root, actor="reviewer", reason="replace route"
        )
        rows = _queue_rows(conn, "reviewer")
        assert [(row["task_id"], row["event_kind"]) for row in rows] == [
            (child, "invalidated_for_rework"),
            (root, "returned_for_rework"),
        ]


def test_fifo_lease_stops_at_unexpired_foreign_head_and_reclaims_expired(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        first = kb.create_task(
            conn, title="first", assignee="reviewer", result_subscriber="planner"
        )
        second = kb.create_task(
            conn, title="second", assignee="reviewer", result_subscriber="planner"
        )
        assert kb.complete_task(conn, first)
        assert kb.complete_task(conn, second)

        leased = kb.lease_result_notifications(
            conn,
            target_profile="planner",
            lease_owner="watcher-a",
            limit=1,
            lease_seconds=60,
            now=100,
        )
        assert [item.task_id for item in leased] == [first]
        assert kb.lease_result_notifications(
            conn,
            target_profile="planner",
            lease_owner="watcher-b",
            limit=8,
            lease_seconds=60,
            now=120,
        ) == []

        reclaimed = kb.lease_result_notifications(
            conn,
            target_profile="planner",
            lease_owner="watcher-b",
            limit=8,
            lease_seconds=60,
            now=161,
        )
        assert [item.task_id for item in reclaimed] == [first, second]
        assert not kb.mark_result_notifications_delivered(
            conn, [item.id for item in reclaimed], lease_owner="watcher-a", now=162
        )
        assert kb.release_result_notification_lease(
            conn, [item.id for item in reclaimed], lease_owner="watcher-b"
        )


def test_result_wait_state_excludes_current_task_but_reports_queue(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        current = kb.create_task(
            conn, title="current", assignee="planner", result_subscriber="planner"
        )
        watched = kb.create_task(
            conn, title="watched", assignee="reviewer", result_subscriber="planner"
        )
        state = kb.result_wait_state(conn, "planner", exclude_task_id=current)
        assert state.watched_task_ids == [watched]
        assert state.queue_ids == []

        assert kb.complete_task(conn, watched)
        state = kb.result_wait_state(conn, "planner", exclude_task_id=current)
        assert state.watched_task_ids == []
        assert len(state.queue_ids) == 1


def test_hard_delete_removes_result_subscription_and_queue_rows(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn, title="delete", assignee="reviewer", result_subscriber="planner"
        )
        assert kb.complete_task(conn, task_id)
        assert kb.delete_task(conn, task_id)
        assert conn.execute(
            "SELECT 1 FROM kanban_result_subscriptions WHERE task_id = ?", (task_id,)
        ).fetchone() is None
        assert conn.execute(
            "SELECT 1 FROM kanban_result_queue WHERE task_id = ?", (task_id,)
        ).fetchone() is None
