from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from plugins.kanban.hermes_listener import hermes_kanban_interactive as hermes


def _args() -> argparse.Namespace:
    return argparse.Namespace(zellij_session="kanban-test", zellij_pane_id="2")


def test_hermes_requires_semantic_delivery_acknowledgement() -> None:
    assert hermes.HermesInteractiveListener().semantic_delivery_required is True


def test_hermes_composer_parser_distinguishes_empty_from_unknown() -> None:
    listener = hermes.HermesInteractiveListener()

    assert listener.composer_input_text("planner ❯\n") == ""
    assert listener.composer_input_text("planner ❯ draft text\n") == "draft text"
    assert listener.composer_input_text("transcript only\n") is None


def test_hermes_claim_precheck_rejects_nonempty_composer_draft(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    monkeypatch.setattr(
        hermes,
        "zellij_dump_screen",
        lambda **_: "planner ❯ existing draft\n────────────────\n",
    )
    monkeypatch.setattr(time, "sleep", lambda _: None)

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")


def test_hermes_claim_precheck_fails_closed_for_unknown_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: "status only\n")

    assert not listener.on_claim_pre_check(_args(), tmp_path / "listener.log")


def test_hermes_post_inject_confirms_marker_transition_via_current_composer(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    marker = "请读取 task.md 中的 Kanban 任务并执行。 [任务 t1: title]"
    screens = iter([
        f"planner ❯ {marker}\n────────────────\n",
        "planner ❯\n────────────────\n",
    ])
    enters: list[dict] = []
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: next(screens))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **kwargs: enters.append(kwargs) or True,
    )

    state = listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="task:t1:run:1:generation:1",
    )

    assert state == "confirmed"
    assert len(enters) == 1
    assert enters[0]["expected_pane_prefix"] == "hermes-kanban"
    assert enters[0]["correlation"] == "task:t1:run:1:generation:1"


def test_hermes_post_inject_returns_known_unsubmitted_after_bounded_retries(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    marker = "kanban_task_boundary [任务 control-3]"
    screen = f"planner ❯ {marker}\n────────────────\n"
    enters: list[dict] = []
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **kwargs: enters.append(kwargs) or True,
    )

    state = listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="control:control-3",
    )

    assert state == "known_unsubmitted"
    assert len(enters) == listener._POST_INJECT_MAX_RETRIES


def test_hermes_post_inject_confirms_on_live_busy_transition(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    marker = "kanban_task_boundary [任务 result-7]"
    screens = iter([
        f"planner ❯ {marker}\n────────────────\n",
        "└ 💻 preparing terminal…\n⚕ ❯ msg=interrupt · /queue\n",
    ])
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: next(screens))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(hermes, "zellij_submit_enter", lambda **_: True)

    assert listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="result:7",
    ) == "confirmed"


def test_hermes_post_inject_confirms_consumed_marker_before_api_error(
    tmp_path: Path, monkeypatch,
) -> None:
    """An API failure after submit is not an unsent composer prompt."""
    listener = hermes.HermesInteractiveListener()
    marker = "请读取 task-run.md 中的 Kanban 任务并执行。 [任务 t1: title] [by watcher]"
    screen = (
        f"● {marker}\n"
        "────────────────────────\n"
        "⚠ API call failed (attempt 1/3): HTTP 400\n"
        "planner ❯\n"
        "────────────────────────\n"
    )
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **_: pytest.fail("consumed marker must not be resubmitted"),
    )

    assert listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="task:t1:run:1:generation:1",
    ) == "confirmed"


def test_hermes_post_inject_confirms_wrapped_consumed_marker(
    tmp_path: Path, monkeypatch,
) -> None:
    """Terminal line wrapping must not turn a consumed prompt into a retry."""
    listener = hermes.HermesInteractiveListener()
    marker = "请读取 task-run.md 中的 Kanban 任务并执行。 [任务 t1: title] [by watcher]"
    wrapped_screen = (
        "● 请读取 task-\n"
        "run.md 中的 Kanban 任务并执行。 [任务 t1: title] [by watcher]\n"
        "⚠ API call failed (attempt 1/3): HTTP 400\n"
        "planner ❯\n"
        "────────────────────────\n"
    )
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: wrapped_screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **_: pytest.fail("wrapped consumed marker must not be resubmitted"),
    )

    assert listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="task:t1:run:2:generation:1",
    ) == "confirmed"


def test_hermes_post_inject_ignores_marker_in_transcript_tail(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    marker = "请读取 task.md 中的 Kanban 任务并执行。 [任务 t1: title]"
    screen = f"previous output {marker}\nplanner ❯ unrelated draft\n────────────────\n"
    enters: list[dict] = []
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **kwargs: enters.append(kwargs) or True,
    )

    assert listener.on_post_inject(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker=marker,
        pre_write_composer="",
        correlation="task:t1:run:1:generation:1",
    ) == "unknown"
    assert enters == []


def test_hermes_api_retry_ignores_stale_transcript_error(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    stale = "api call failed while handling an earlier turn\n" + "output\n" * 8
    screen = stale + "planner ❯\n────────────────\n"
    now = [100.0]
    retries: list[dict] = []
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(hermes.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **kwargs: retries.append(kwargs) or True,
    )
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)

    listener.on_task_running_monitor(_args(), object(), "t1", tmp_path / "watch.log")
    now[0] += listener.API_RETRY_BACKOFF[0] + 1
    listener.on_task_running_monitor(_args(), object(), "t1", tmp_path / "watch.log")

    assert retries == []


def test_hermes_api_retry_is_bounded_per_task_and_error_kind(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    screen = "Error: api call failed\nplanner ❯\n────────────────\n"
    now = [100.0]
    injections: list[dict] = []
    monkeypatch.setattr(hermes, "zellij_dump_screen", lambda **_: screen)
    monkeypatch.setattr(hermes.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes.time, "sleep", lambda _: None)
    monkeypatch.setattr(listener, "wait_for_stable_composer_input", lambda **_: True)
    monkeypatch.setattr(hermes, "zellij_inject", lambda **kwargs: injections.append(kwargs) or True)
    monkeypatch.setattr(
        hermes,
        "zellij_submit_enter",
        lambda **kwargs: injections.append(kwargs) or True,
    )

    for _ in range(listener.API_RETRY_MAX + 1):
        listener.on_task_running_monitor(_args(), object(), "t1", tmp_path / "watch.log")
        index = min(listener._api_retry_count, len(listener.API_RETRY_BACKOFF) - 1)
        now[0] += listener.API_RETRY_BACKOFF[index] + 1

    assert len([item for item in injections if "api:" in item.get("correlation", "")]) <= listener.API_RETRY_MAX


def test_base_post_injection_contract_forwards_hermes_correlation(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = hermes.HermesInteractiveListener()
    received: dict[str, object] = {}

    def capture(*args, **kwargs):
        received.update(kwargs)
        return "confirmed"

    monkeypatch.setattr(listener, "on_post_inject", capture)
    state = listener._post_injection_contract(
        _args(),
        zellij_session="kanban-test",
        zellij_pane_id="2",
        log_path=tmp_path / "listener.log",
        injected_marker="payload",
        pre_write_composer="",
        correlation="task:t1:run:7:generation:3",
        run_id=7,
        generation=3,
        task_id="t1",
    )

    assert state == "confirmed"
    assert received["correlation"] == "task:t1:run:7:generation:3"
