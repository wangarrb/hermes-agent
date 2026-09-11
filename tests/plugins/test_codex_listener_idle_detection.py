from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from plugins.kanban import base_listener as base
from plugins.kanban.codex_listener import codex_kanban_interactive as codex


def _args() -> argparse.Namespace:
    return argparse.Namespace(zellij_session="kanban-test", zellij_pane_id="2")


def test_codex_requires_semantic_delivery_acknowledgement() -> None:
    listener = codex.CodexInteractiveListener()
    assert listener.semantic_delivery_required is True


def test_codex_idle_detection_accepts_bare_prompt_and_role_status_forms() -> None:
    listener = codex.CodexInteractiveListener()

    assert listener.pane_is_idle("›\n")
    assert listener.pane_is_idle("› reviewer · Context 20% used\n")
    assert listener.pane_is_idle("›\n  reviewer · Context 20% used\n")


def test_codex_composer_parser_preserves_marker_before_inline_status() -> None:
    listener = codex.CodexInteractiveListener()

    assert listener.composer_input_text(
        "› marker text  gpt-5.6-sol high · Context 20% used\n",
    ) == "marker text"
    assert listener.composer_input_text(
        "› reviewer · Context 20% used\n",
    ) is None


def test_codex_claim_precheck_rejects_nonempty_composer_draft(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    draft_screen = "› existing draft\n  gpt-5.6-sol · Context 20% used\n"
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: draft_screen)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleeps.append(seconds))

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == []


def test_codex_post_inject_confirms_marker_transition_via_current_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    marker = "请读取 task.md 中的 Kanban 任务并执行。 [任务 t1: title]"
    screens = iter([
        f"› {marker}\n  reviewer · Context 20% used\n",
        "›\n  reviewer · Context 20% used\n",
    ])
    enters: list[dict] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: next(screens))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **kwargs: enters.append(kwargs) or True)

    state = listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker=marker,
        pre_write_composer=None,
    )

    assert state == "confirmed"
    assert len(enters) == 1
    assert enters[0]["expected_pane_prefix"] == "codex-kanban"
    assert enters[0]["correlation"].split(":", 1)[0] in {"task", "control", "result"}


def test_codex_post_inject_ignores_marker_in_transcript_tail(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    marker = "请读取 task.md 中的 Kanban 任务并执行。 [任务 t1: title]"
    screen = f"previous output {marker}\n› unrelated draft\n  reviewer · Context 20% used\n"
    enters: list[dict] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **kwargs: enters.append(kwargs) or True)

    state = listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker=marker,
        pre_write_composer=None,
    )

    assert state == "unknown"
    assert enters == []


def test_codex_post_inject_returns_known_unsubmitted_after_bounded_retries(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    marker = "kanban_task_boundary [任务 control-3]"
    screen = f"› {marker}\n  reviewer · Context 20% used\n"
    enters: list[dict] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **kwargs: enters.append(kwargs) or True)

    state = listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker=marker,
        pre_write_composer=None,
    )

    assert state == "known_unsubmitted"
    assert len(enters) == listener._POST_INJECT_MAX_RETRIES


def test_codex_post_inject_returns_confirmed_on_live_busy_transition(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    marker = "kanban_task_boundary [任务 result-7]"
    screens = iter([
        f"› {marker}\n  reviewer · Context 20% used\n",
        "• Working (1s)\n",
    ])
    enters: list[dict] = []
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: next(screens))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **kwargs: enters.append(kwargs) or True)

    state = listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker=marker,
        pre_write_composer=None,
    )

    assert state == "confirmed"
    assert len(enters) == 1


def test_codex_post_inject_confirms_when_busy_is_first_observation(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: "• Working (1s)\n")
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **_: True)

    assert listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker="marker",
        pre_write_composer=None,
    ) == "confirmed"


def test_codex_post_inject_ignores_stale_busy_line_above_current_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    marker = "kanban_task_boundary [任务 t1]"
    screen = (
        "• Working (stale transcript)\n"
        f"› {marker}\n"
        "  reviewer · Context 20% used\n"
    )
    enters: list[dict] = []
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(codex, "zellij_submit_enter", lambda **kwargs: enters.append(kwargs) or True)

    assert listener.on_post_inject(
        _args(), zellij_session="kanban-test", zellij_pane_id="2",
        log_path=tmp_path / "listener.log", injected_marker=marker,
        pre_write_composer=None,
    ) == "known_unsubmitted"
    assert len(enters) == listener._POST_INJECT_MAX_RETRIES


def test_codex_post_inject_returns_unknown_for_empty_or_unsupported_screen(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = codex.CodexInteractiveListener()
    monkeypatch.setattr(time, "sleep", lambda _: None)
    for screen in ("", "  status only\n"):
        monkeypatch.setattr(codex, "zellij_dump_screen", lambda screen=screen, **_: screen)
        monkeypatch.setattr(codex, "zellij_submit_enter", lambda **_: True)
        assert listener.on_post_inject(
            _args(), zellij_session="kanban-test", zellij_pane_id="2",
            log_path=tmp_path / "listener.log", injected_marker="marker",
            pre_write_composer=None,
        ) == "unknown"


def test_codex_claim_precheck_rejects_working_view_with_idle_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    """The persistent composer prompt must not override current Working state."""
    listener = codex.CodexInteractiveListener()
    working_screen = """\
• Working (3s • esc to interrupt) · 1 background terminal running

›

  gpt-5.6-sol high · master · Context 64% used
"""
    monkeypatch.setattr(codex, "zellij_dump_screen", lambda **_: working_screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")


def test_codex_claim_precheck_rejects_nonempty_idle_composer(
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

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == []


def test_codex_claim_precheck_rejects_changing_composer_draft(
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

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")
    assert sleeps == []


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

›

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
    assert len(calls) == 1
