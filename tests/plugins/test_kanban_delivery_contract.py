"""Semantic acknowledgement contract for interactive Kanban delivery."""

from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl


class _Listener(bl.BaseInteractiveListener):
    agent_name = "Test"
    agent_slug = "test"
    idle_markers = ("❯",)

    def build_tui_cmd(self, workspace, **kwargs):
        return []

    def has_saved_sessions(self, workspace):
        return True

    def inject_text(self, *args, **kwargs):
        return "prompt"

    def pane_label(self, task_id=None):
        return "test"

    def on_post_inject(self, args, *, zellij_session, zellij_pane_id,
                       log_path, injected_marker=None,
                       pre_write_composer=None):
        return "known_unsubmitted"


class _StrictListener(_Listener):
    semantic_delivery_required = True

    def on_post_inject(self, *args, **kwargs):
        return None


def test_required_semantic_backend_never_infers_confirmation(tmp_path):
    listener = _StrictListener()
    listener.read_pane_screen = lambda **_: "❯"
    state = listener._post_injection_contract(
        Namespace(), zellij_session="s", zellij_pane_id="0",
        log_path=tmp_path / "watch.log", injected_marker="prompt",
        pre_write_composer=None, correlation="task:t_1",
    )
    assert state == "unknown"


def test_post_inject_receives_marker_and_pre_write_composer(tmp_path):
    seen = {}

    class Capture(_Listener):
        def on_post_inject(self, args, **kwargs):
            seen.update(kwargs)
            return "confirmed"

    Capture()._post_injection_contract(
        Namespace(), zellij_session="s", zellij_pane_id="0",
        log_path=tmp_path / "watch.log", injected_marker="m",
        pre_write_composer="before", correlation="control:3",
    )
    assert seen["injected_marker"] == "m"
    assert seen["pre_write_composer"] == "before"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    monkeypatch.setattr(bl, "zellij_submit", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_rename_pane", lambda **_: True)
    return home


def _args(tmp_path):
    return Namespace(profile="reviewer", claim_assignees="reviewer",
                      assist_role=None, zellij_session="s",
                      zellij_pane_id="0", workspace=str(tmp_path), board="default")


def test_nonconfirmed_control_ack_is_not_marked_delivered(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "❯")
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="review", assignee="reviewer")
        kb.claim_task(conn, task_id, claimer="review-pane")
        returned = kb.return_task_for_rework(conn, task_id, actor="reviewer", reason="redo")
        assert listener.pump_control_messages(_args(tmp_path), conn, tmp_path / "watch.log")
        row = kb.list_control_messages(conn)[0]
    assert row.status == "pending"


def test_nonconfirmed_result_ack_releases_lease(kanban_home, tmp_path, monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(bl, "zellij_inject", lambda **_: True)
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="result", assignee="reviewer", result_subscriber="reviewer")
        kb.complete_task(conn, task_id, summary="done")
        assert listener.pump_result_notifications(_args(tmp_path), conn, tmp_path / "watch.log")
        row = conn.execute("SELECT status, lease_owner FROM kanban_result_queue").fetchone()
    assert tuple(row) == ("pending", None)
