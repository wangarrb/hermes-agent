"""Durable cooperative controls for all interactive Kanban listeners."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _pending_pause(conn) -> tuple[str, int]:
    task_id = kb.create_task(conn, title="review", assignee="reviewer")
    assert kb.claim_task(conn, task_id, claimer="review-pane")
    returned = kb.return_task_for_rework(
        conn, task_id, actor="reviewer", reason="old contract is invalid"
    )
    return task_id, returned.control_ids[0]


class DummyListener(bl.BaseInteractiveListener):
    agent_name = "Dummy"
    agent_slug = "dummy"
    idle_markers = ("ready>",)
    busy_markers = ("working",)

    def build_tui_cmd(self, workspace, **kwargs):
        return []

    def has_saved_sessions(self, workspace):
        return False

    def inject_text(self, task_id, title, assignee, profile, prompt_path, board):
        return str(prompt_path)

    def pane_label(self, task_id=None):
        return f"dummy:{task_id or 'idle'}"


def _listener_args(workspace: Path) -> Namespace:
    return Namespace(
        profile="reviewer",
        claim_assignees="reviewer",
        assist_role=None,
        zellij_session="test-session",
        zellij_pane_id="7",
        workspace=str(workspace),
        board="default",
    )


def _listener(workspace: Path) -> DummyListener:
    listener = DummyListener()
    listener._profile = "reviewer"
    listener._board = "default"
    listener._workspace = workspace
    listener._log_path = workspace / "listener.log"
    return listener


def test_control_lease_is_exclusive_retryable_and_ack_idempotent(kanban_home):
    with kb.connect() as conn:
        _, control_id = _pending_pause(conn)
        first = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="pane-a", lease_seconds=30
        )
        assert first.id == control_id
        assert first.status == "delivering"
        assert kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="pane-b"
        ) is None

        assert kb.release_control_lease(conn, control_id, receiver="pane-a")
        second = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="pane-b"
        )
        assert second.id == control_id
        assert kb.mark_control_delivered(conn, control_id, receiver="pane-b")
        assert kb.ack_control_message(conn, control_id, receiver="reviewer")
        assert kb.ack_control_message(conn, control_id, receiver="reviewer")

        message = kb.list_control_messages(conn)[0]

    assert message.status == "acked"


def test_expired_delivery_lease_is_adopted_after_restart(kanban_home):
    with kb.connect() as conn:
        _, control_id = _pending_pause(conn)
        assert kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="dead-listener"
        )
        conn.execute(
            "UPDATE task_control_messages SET delivery_expires = 0 WHERE id = ?",
            (control_id,),
        )
        conn.commit()

        adopted = kb.lease_control_message(
            conn, profiles=["reviewer"], receiver="new-listener"
        )

    assert adopted.id == control_id
    assert adopted.delivery_owner == "new-listener"


def test_busy_pane_keeps_control_pending(kanban_home, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    listener = _listener(workspace)
    injected: list[str] = []
    titles: list[str] = []
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: "working")
    monkeypatch.setattr(
        bl, "zellij_inject", lambda **kwargs: injected.append(kwargs["text"]) or True
    )
    monkeypatch.setattr(
        bl, "zellij_rename_pane", lambda **kwargs: titles.append(kwargs["name"]) or True
    )

    with kb.connect() as conn:
        _, control_id = _pending_pause(conn)
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        message = kb.list_control_messages(conn)[0]

    assert message.id == control_id
    assert message.status == "pending"
    assert injected == []
    assert titles[-1] == f"[PAUSE {message.task_id}]"


@pytest.mark.parametrize("screen", [None, "", "   \n"])
def test_unknown_pane_state_never_injects_control(
    kanban_home, tmp_path, monkeypatch, screen,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    listener = _listener(workspace)
    injected: list[str] = []
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: screen)
    monkeypatch.setattr(
        bl, "zellij_inject", lambda **kwargs: injected.append(kwargs["text"]) or True
    )
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **kwargs: True)

    with kb.connect() as conn:
        _pending_pause(conn)
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        message = kb.list_control_messages(conn)[0]

    assert injected == []
    assert message.status == "pending"


def test_control_second_probe_failure_is_fail_closed(
    kanban_home, tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    listener = _listener(workspace)
    screens = iter(["working", None])
    injected: list[str] = []
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: next(screens))
    monkeypatch.setattr(
        bl, "zellij_inject", lambda **kwargs: injected.append(kwargs["text"]) or True
    )
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **kwargs: True)

    with kb.connect() as conn:
        _pending_pause(conn)
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        message = kb.list_control_messages(conn)[0]

    assert injected == []
    assert message.status == "pending"


def test_idle_pane_receives_control_once_and_blocks_claim_until_ack(
    kanban_home, tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    listener = _listener(workspace)
    injected: list[str] = []
    titles: list[str] = []
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: "ready>")
    monkeypatch.setattr(
        bl, "zellij_inject", lambda **kwargs: injected.append(kwargs["text"]) or True
    )
    monkeypatch.setattr(
        bl, "zellij_rename_pane", lambda **kwargs: titles.append(kwargs["name"]) or True
    )

    with kb.connect() as conn:
        task_id, control_id = _pending_pause(conn)
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        message = kb.list_control_messages(conn)[0]
        prompt = bl.prompt_dir(
            workspace, "default", "reviewer", agent_slug="dummy"
        ) / f"task-{task_id}.md"

    assert len(injected) == 1
    assert "\n" not in injected[0]
    assert f"control-ack {control_id}" in injected[0]
    assert task_id in injected[0]
    assert message.status == "delivered"
    assert titles[-1] == f"[PAUSE {task_id}]"
    assert "SUPERSEDED" in prompt.read_text(encoding="utf-8")


def test_failed_listener_injection_releases_lease(kanban_home, tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    listener = _listener(workspace)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: "ready>")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kwargs: False)

    with kb.connect() as conn:
        _pending_pause(conn)
        assert listener.pump_control_messages(args, conn, workspace / "listener.log")
        message = kb.list_control_messages(conn)[0]

    assert message.status == "pending"


def test_shared_claim_path_delivers_control_before_claiming_another_task(
    kanban_home, tmp_path, monkeypatch,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = _listener_args(workspace)
    args.ttl = 900
    listener = _listener(workspace)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **kwargs: "ready>")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kwargs: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **kwargs: True)

    with kb.connect() as conn:
        _pending_pause(conn)
        waiting = kb.create_task(conn, title="next", assignee="reviewer")
        claimed, run_id = listener.claim_and_inject_one(
            args, log_path=workspace / "listener.log", conn=conn
        )
        waiting_after = kb.get_task(conn, waiting)

    assert claimed is None
    assert run_id is None
    assert waiting_after.status == "ready"


def test_codewhale_steering_dismiss_ignores_system_controls(monkeypatch, tmp_path):
    from plugins.kanban.deepseek_listener import deepseek_kanban_interactive as ds

    monkeypatch.setattr(
        ds.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("system control must not be dismissed"),
    )
    assert not ds._auto_dismiss_steering(
        session="session",
        pane_id="1",
        screen="[SYSTEM CONTROL 7] stop old contract and control-ack 7",
        profile="reviewer",
        log_path=tmp_path / "listener.log",
    )


def test_task_prompt_contains_role_guidance_and_generation_fences(tmp_path):
    prompt = bl.build_interactive_prompt(
        agent_name="Codex",
        board="project",
        profile="reviewer",
        task_id="t_abcd1234",
        task_assignee="implementer",
        task_title="implement",
        context="context",
        workspace=tmp_path,
        run_id=42,
        generation=3,
    )

    assert "--run-id 42 --generation 3" in prompt
    assert "implementer：负责落地执行" in prompt
    assert "HERMES_KANBAN_GENERATION=3" in prompt
