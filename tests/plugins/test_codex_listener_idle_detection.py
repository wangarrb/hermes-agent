from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from plugins.kanban.codex_listener import codex_kanban_interactive as codex


def _args() -> argparse.Namespace:
    return argparse.Namespace(zellij_session="kanban-test", zellij_pane_id="2")


def test_codex_post_inject_retries_raw_enter_while_prompt_is_queued(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    commands: list[list[str]] = []
    screen = "› 请读取 /tmp/task.md 中的 Kanban 任务并执行。\n"
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
    )

    expected = [
        "zellij", "--session", "kanban-test", "action",
        "write", "-p", "2", "13",
    ]
    assert commands == [expected] * listener._POST_INJECT_MAX_RETRIES


def test_codex_post_inject_stops_when_agent_is_busy(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    commands: list[list[str]] = []
    monkeypatch.setattr(
        codex,
        "zellij_dump_screen",
        lambda **_: "› 请读取 task 中的 Kanban 任务\n• Working (1s)\n",
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
    )

    assert commands == []
