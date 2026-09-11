from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from plugins.kanban import base_listener as bl


def _run_stub(*, panes, calls, **kwargs):
    calls.append(kwargs["args"])
    args = kwargs["args"]
    if "list-panes" in args:
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(panes), stderr="")
    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")


def test_injection_rejects_nonexistent_pane_even_when_zellij_returns_zero(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **kw: _run_stub(panes=[], calls=calls, args=a[0], **kw))

    assert not bl.zellij_inject(
        session="s", pane_id="99", text="hello", expected_pane_prefix="claude-kanban",
        log_path=tmp_path / "inject.log",
    )
    assert len(calls) == 1
    assert not any("write-chars" in c for c in calls)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("claude-kanban", True),
        ("claude-kanban [PAUSE t_1]", True),
        ("claude-kanban extra", True),
        ("claude-kanbanish", False),
        ("hermes-kanban", False),
    ],
)
def test_pane_identity_uses_exact_title_token_boundary(tmp_path, monkeypatch, title, expected):
    calls = []
    panes = [{"pane_id": "7", "title": title, "is_plugin": False, "exited": False}]
    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **kw: _run_stub(panes=panes, calls=calls, args=a[0], **kw))
    assert bl.zellij_inject(
        session="s", pane_id="7", text="hello", expected_pane_prefix="claude-kanban",
        log_path=tmp_path / "inject.log",
    ) is expected


@pytest.mark.parametrize("exc", [OSError("missing zellij"), subprocess.TimeoutExpired("zellij", 5), subprocess.CalledProcessError(1, "zellij")])
def test_transport_failures_are_logged_and_return_false(tmp_path, monkeypatch, exc):
    def run(*_a, **_kw):
        raise exc

    monkeypatch.setattr(bl.subprocess, "run", run)
    assert not bl.zellij_inject(
        session="s", pane_id="7", text="hello", log_path=tmp_path / "inject.log",
    )
    assert (tmp_path / "inject.log").exists()


@pytest.mark.parametrize("text", ["a\nb", "a\rb", "a\tb", "a\x00b", "a\x7fb"])
def test_transport_rejects_control_bytes_before_subprocess(tmp_path, monkeypatch, text):
    monkeypatch.setattr(bl.subprocess, "run", lambda *_a, **_kw: pytest.fail("must not run"))
    assert not bl.zellij_inject(
        session="s", pane_id="7", text=text, log_path=tmp_path / "inject.log",
    )


def test_text_then_single_raw_submit_sequence(tmp_path, monkeypatch):
    calls = []
    panes = [{"pane_id": "7", "title": "claude-kanban", "is_plugin": False, "exited": False}]
    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **kw: _run_stub(panes=panes, calls=calls, args=a[0], **kw))
    assert bl.zellij_inject(session="s", pane_id="7", text="hello", expected_pane_prefix="claude-kanban", log_path=tmp_path / "inject.log")
    assert bl.zellij_submit(session="s", pane_id="7", expected_pane_prefix="claude-kanban", log_path=tmp_path / "inject.log")
    actions = [c[4:] for c in calls if "action" in c]
    assert any("write-chars" in c for c in actions)
    assert any(c[-4:] == ["write", "-p", "7", "13"] for c in actions)


def test_title_matching_is_casefolded_and_command_tokens_reject_foreign_role(tmp_path, monkeypatch):
    calls = []
    panes = [{"pane_id": "7", "title": " CLAUDE-KANBAN [PAUSE t] ", "is_plugin": False, "exited": False,
              "terminal_command": "claude --continue"}]
    monkeypatch.setattr(bl.subprocess, "run", lambda *a, **kw: _run_stub(panes=panes, calls=calls, args=a[0], **kw))
    assert bl.zellij_inject(session="s", pane_id="7", text="hello", expected_pane_prefix="claude-kanban", log_path=tmp_path / "inject.log")
    panes[0]["terminal_command"] = "hermes --continue"
    assert not bl.zellij_inject(session="s", pane_id="7", text="hello", expected_pane_prefix="claude-kanban", log_path=tmp_path / "inject.log")
    panes[0]["terminal_command"] = "[conda] <defunct> hermes"
    assert bl.zellij_inject(session="s", pane_id="7", text="hello", expected_pane_prefix="claude-kanban", log_path=tmp_path / "inject.log")


def test_validator_handles_none_subprocess_result(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.subprocess, "run", lambda *_a, **_kw: None)
    assert not bl.zellij_inject(session="s", pane_id="7", text="hello", log_path=tmp_path / "inject.log")
