from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from plugins.kanban import base_listener as base
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


def test_codex_claim_precheck_rejects_working_view_with_idle_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    """The persistent composer prompt must not override current Working state."""
    listener = codex.CodexInteractiveListener()
    working_screen = """\
• Working (3s • esc to interrupt) · 1 background terminal running

› Run /review on my current changes

  gpt-5.6-sol high · master · Context 64% used
"""
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: working_screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")


def test_codex_claim_precheck_accepts_idle_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    idle_screen = """\
─ Worked for 22m 15s ─

› Run /review on my current changes

  gpt-5.6-sol high · master · Context 64% used
"""
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: idle_screen)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    assert listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == [10.0, 10.0, 10.0]


def test_codex_claim_precheck_resets_stability_when_composer_changes(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()

    def screen(text: str) -> str:
        return f"""\
─ Worked for 1m ─

› {text}

  gpt-5.6-sol high · master · Context 64% used
"""

    screens = iter(
        [
            screen("draft A"),
            screen("draft B"),
            screen("draft B"),
            screen("draft B"),
            screen("draft B"),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: next(screens))
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    assert listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == [10.0, 10.0, 10.0, 10.0]


def test_codex_claim_precheck_accepts_empty_composer_without_stability_delay(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    idle_screen = (
        "─ Worked for 1m ─\n\n"
        "› \n\n"
        "  gpt-5.6-sol high · master · Context 64% used\n"
    )
    sleeps: list[float] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: idle_screen)
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    assert listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == []


def test_codex_claim_precheck_ignores_stale_busy_words_in_completed_output(
    tmp_path: Path, monkeypatch,
) -> None:
    """Transcript prose must not override the live idle composer state."""
    listener = codex.CodexInteractiveListener()
    idle_screen = """\
• The previous goal remains running in its working directory.
─ Worked for 6m 35s ─

› Run /review on my current changes

  gpt-5.6-sol high · master · Context 22% used
"""
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: idle_screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    assert listener.on_claim_pre_check(_args(), tmp_path / "listener.log")


def test_capacity_error_uses_bounded_retry_path(tmp_path: Path, monkeypatch) -> None:
    """Model-capacity failures must retry the same live session after backoff."""
    listener = codex.CodexInteractiveListener()
    calls: list[dict] = []
    clock = [1000.0]
    screen = "Selected model is at capacity. Please try a different model.\n› \n"

    runtime_base = sys.modules.get("base_listener", base)
    monkeypatch.setattr(runtime_base.time, "time", lambda: clock[0])
    monkeypatch.setattr(runtime_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        listener,
        "wait_for_stable_composer_input",
        lambda **_: True,
    )
    monkeypatch.setattr(runtime_base, "zellij_inject", lambda **kwargs: calls.append(kwargs))

    assert listener.check_api_failure_retry(
        session="kanban-test",
        pane_id="2",
        screen=screen,
        task_id="t_capacity",
        log_path=tmp_path / "listener.log",
    )
    assert calls == []

    clock[0] += listener.API_CAPACITY_RETRY_BACKOFF[0]
    assert listener.check_api_failure_retry(
        session="kanban-test",
        pane_id="2",
        screen=screen,
        task_id="t_capacity",
        log_path=tmp_path / "listener.log",
    )
    assert listener._api_retry_count == 1
    assert len(calls) == 2
