"""Interactive watchers keep unfinished goal/reviewer tasks moving."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from plugins.kanban import base_listener as bl
from plugins.kanban.hermes_listener.hermes_kanban_interactive import (
    HermesInteractiveListener,
)
from plugins.kanban.codex_listener.codex_kanban_interactive import (
    CodexInteractiveListener,
)


@pytest.fixture(autouse=True)
def daytime_goal_check_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep legacy idle-episode tests outside the overnight throttle window."""
    monkeypatch.setattr(bl.time, "localtime", lambda _: SimpleNamespace(tm_hour=12))


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class _Listener(bl.BaseInteractiveListener):
    agent_name = "Test"
    agent_slug = "test"
    idle_markers = ("❯",)
    busy_markers = ("working",)

    def build_tui_cmd(self, workspace: Path, **kwargs):
        return []

    def has_saved_sessions(self, workspace: Path) -> bool:
        return True

    def inject_text(self, *args, **kwargs) -> str:
        return ""

    def pane_label(self, task_id: str | None = None) -> str:
        return "test"


def _args() -> argparse.Namespace:
    return argparse.Namespace(zellij_session="s", zellij_pane_id="0")


def _running_task(*, assignee: str, goal_mode: bool) -> str:
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="large objective",
            assignee=assignee,
            goal_mode=goal_mode,
            goal_max_turns=10 if goal_mode else None,
        )
        assert kb.claim_task(conn, task_id, claimer="watcher") is not None
    return task_id


def _reviewer_checkpoint(origin_task_id: str, *, status: str = "running") -> str:
    with kb.connect() as conn:
        reviewer_task_id = kb.create_task(
            conn,
            title="load-bearing decision",
            assignee="reviewer",
        )
        if status == "running":
            assert kb.claim_task(conn, reviewer_task_id, claimer="reviewer-watcher") is not None
        kb.add_comment(
            conn,
            origin_task_id,
            author="planner",
            body=f"REVIEWER_CHECKPOINT_PENDING {reviewer_task_id}",
        )
    return reviewer_task_id


def test_goal_task_gets_one_completion_check_per_idle_episode(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = _Listener()
    now = [100.0]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "planner ❯\n")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    checks = [text for text in injected if "GOAL_COMPLETION_CHECK" in text]
    assert len(checks) == 1
    assert "任务都完成了吗？" in checks[0]
    assert "不得向用户列出普通技术选项" in checks[0]
    assert "第一个未满足的承重 gate" in checks[0]
    assert "Continue the same goal" not in checks[0]
    assert checks[0].endswith("[by watcher]")


def test_goal_completion_check_is_throttled_to_2_minutes_outside_1_to_9(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = _Listener()
    now = [100.0]
    screen = ["planner ❯\n"]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: screen[0])
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        screen[0] = "working\n"
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        screen[0] = "planner ❯\n"
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

        assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1
        now[0] += listener.DAYTIME_GOAL_COMPLETION_INTERVAL_S
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 2


def test_goal_completion_check_is_throttled_to_30_minutes_from_1_to_9(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = _Listener()
    now = [100.0]
    screen = ["planner ❯\n"]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(bl.time, "localtime", lambda _: SimpleNamespace(tm_hour=2))
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: screen[0])
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

        screen[0] = "working\n"
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        screen[0] = "planner ❯\n"
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 60
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1

        now[0] += listener.OVERNIGHT_GOAL_COMPLETION_INTERVAL_S
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 2


def test_codex_working_view_never_gets_reviewer_lifecycle_followup(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="reviewer", goal_mode=False)
    listener = CodexInteractiveListener()
    codex_base = sys.modules[listener.on_task_running_monitor.__module__]
    now = [100.0]
    injected: list[str] = []
    working_screen = """\
• Working (18s • esc to interrupt) · 1 background terminal running

› Run /review on my current changes

  gpt-5.6-sol high · master · Context 64% used
"""
    monkeypatch.setattr(codex_base.time, "time", lambda: now[0])
    monkeypatch.setattr(codex_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(codex_base, "zellij_dump_screen", lambda **_: working_screen)
    monkeypatch.setattr(
        codex_base, "zellij_inject", lambda **kw: injected.append(kw["text"])
    )

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert not [text for text in injected if "REVIEW_LIFECYCLE" in text]


def test_goal_completion_check_is_suppressed_while_reviewer_checkpoint_is_open(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    _reviewer_checkpoint(task_id)
    listener = _Listener()
    now = [100.0]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "planner ❯\n")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert not [text for text in injected if "GOAL_COMPLETION_CHECK" in text]


def test_goal_completion_check_resumes_after_reviewer_checkpoint_closes(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    reviewer_task_id = _reviewer_checkpoint(task_id)
    listener = _Listener()
    now = [100.0]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "designer ❯\n")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        with conn:
            conn.execute(
                "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
                (int(now[0]), reviewer_task_id),
            )
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1


def test_idle_reviewer_gets_one_lifecycle_nudge(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="reviewer", goal_mode=False)
    listener = _Listener()
    now = [100.0]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(bl, "zellij_dump_screen", lambda **_: "reviewer ❯\n")
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "REVIEW_LIFECYCLE" in text]) == 1


def test_hermes_strict_idle_path_also_checks_goal_completion(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = HermesInteractiveListener()
    hermes_base = sys.modules[HermesInteractiveListener.__mro__[1].__module__]
    now = [100.0]
    injected: list[str] = []
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(hermes_base.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.time.time",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.zellij_dump_screen",
        lambda **_: (
            "Task status: running\n"
            "└ 💻 $ hermes kanban --board egomotion4d show t_example  0.1s\n"
            "⚕ xopglm52 │ 49% │ 1.1d\n"
            "planner ❯\n"
            "────────────────\n"
        ),
    )
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))
    monkeypatch.setattr(
        hermes_base, "zellij_inject", lambda **kw: injected.append(kw["text"]),
    )

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1


def test_hermes_role_prompt_above_decorative_border_checks_goal_completion(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A bottom pane border must not hide an idle Hermes role prompt."""
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = HermesInteractiveListener()
    hermes_base = sys.modules[HermesInteractiveListener.__mro__[1].__module__]
    now = [100.0]
    injected: list[str] = []
    screen = (
        "The next execution step is still pending.\n"
        "planner ❯\n"
        "────────────────\n"
    )
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(hermes_base.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.time.time",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.zellij_dump_screen",
        lambda **_: screen,
    )
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))
    monkeypatch.setattr(
        hermes_base, "zellij_inject", lambda **kw: injected.append(kw["text"]),
    )

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1


def test_hermes_stale_busy_marker_outside_recent_tail_does_not_block_goal_followup(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Old tool activity may stay visible after Hermes has returned idle."""
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = HermesInteractiveListener()
    hermes_base = sys.modules[HermesInteractiveListener.__mro__[1].__module__]
    now = [100.0]
    injected: list[str] = []
    screen = (
        "┊ 💻 preparing terminal…\n"
        "tool output line 1\n"
        "tool output line 2\n"
        "tool output line 3\n"
        "tool output line 4\n"
        "tool output line 5\n"
        "planner ❯\n"
        "────────────────\n"
    )
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(hermes_base.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.time.time",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.zellij_dump_screen",
        lambda **_: screen,
    )
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))
    monkeypatch.setattr(
        hermes_base, "zellij_inject", lambda **kw: injected.append(kw["text"]),
    )

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert len([text for text in injected if "GOAL_COMPLETION_CHECK" in text]) == 1


def test_hermes_active_interrupt_status_is_not_a_safe_idle_boundary() -> None:
    listener = HermesInteractiveListener()

    assert not listener._is_truly_idle_line(
        "⚕ ❯ msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
    )
    assert not listener._is_truly_idle_line(
        "⚕ msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
    )


@pytest.mark.parametrize(
    "activity_line",
    [
        "(´･_･`) ruminating...",
        "(｡•︿•｡) processing...",
        "└ ⚙ preparing process…",
        "⚙ wait proc_c915e817910 180s  (02m08s)",
        "└ 💻 preparing terminal…",
    ],
)
def test_hermes_status_bar_does_not_look_idle_while_activity_is_visible(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    activity_line: str,
) -> None:
    task_id = _running_task(assignee="planner", goal_mode=True)
    listener = HermesInteractiveListener()
    hermes_base = sys.modules[HermesInteractiveListener.__mro__[1].__module__]
    now = [100.0]
    injected: list[str] = []
    screen = (
        f"{activity_line}\n"
        "⚕ xopglm52 │ 49% │ ⚙ 4 │ 1.1d\n"
        "────────────────\n"
        "⚕ ❯ msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel\n"
    )
    monkeypatch.setattr(bl.time, "time", lambda: now[0])
    monkeypatch.setattr(bl.time, "sleep", lambda _: None)
    monkeypatch.setattr(hermes_base.time, "time", lambda: now[0])
    monkeypatch.setattr(hermes_base.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.time.time",
        lambda: now[0],
    )
    monkeypatch.setattr(
        "plugins.kanban.hermes_listener.hermes_kanban_interactive.zellij_dump_screen",
        lambda **_: screen,
    )
    monkeypatch.setattr(bl, "zellij_inject", lambda **kw: injected.append(kw["text"]))
    monkeypatch.setattr(
        hermes_base, "zellij_inject", lambda **kw: injected.append(kw["text"]),
    )

    with kb.connect() as conn:
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")
        now[0] += listener.IDLE_FOLLOWUP_GRACE_S + 1
        listener.on_task_running_monitor(_args(), conn, task_id, tmp_path / "watch.log")

    assert not [text for text in injected if "GOAL_COMPLETION_CHECK" in text]
