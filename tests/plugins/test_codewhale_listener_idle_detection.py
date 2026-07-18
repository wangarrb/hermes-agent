"""CodeWhale composer detection for interactive Kanban injection."""

import json
from argparse import Namespace
from pathlib import Path

from plugins.kanban import base_listener
from plugins.kanban.deepseek_listener import deepseek_kanban_interactive


def _can_accept(screen: str) -> bool:
    return base_listener._pane_can_accept_new_kanban_task(
        screen,
        deepseek_kanban_interactive._DEEPSEEK_IDLE_MARKERS,
        deepseek_kanban_interactive._DEEPSEEK_BUSY_MARKERS,
        deepseek_kanban_interactive._DEEPSEEK_QUEUED_INPUT_MARKERS,
    )


def test_codewhale_suggested_composer_is_idle_only_at_safe_boundary() -> None:
    completed_screen = """\
✓ 完成
───────────────────────────────────────────────────────────
❯ Coordinate parallel tasks...
"""
    historical_prompt = """\
❯ Coordinate parallel tasks...
activity: thinking
"""
    active_screen = """\
kanban_task_boundary 请读取任务并执行。
⣤ 工作中
───────────────────────────────────────────────────────────
❯ Coordinate parallel tasks...
"""

    assert _can_accept(completed_screen)
    assert not _can_accept(historical_prompt)
    assert not _can_accept(active_screen)


def test_codewhale_idle_status_line_allows_injection() -> None:
    resumed_idle_screen = """\
cw auto · act · Full Access
────────────────────────────────────────────
previous assistant response
────────────────────────────────────────────
❯ 编写任务或使用 /。
· 空闲                             F1:快捷键
"""

    assert _can_accept(resumed_idle_screen)


def test_codewhale_post_inject_submits_queued_prompt_with_raw_enter(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = deepseek_kanban_interactive.CodeWhaleInteractiveListener()
    commands: list[list[str]] = []
    screen = """\
cw auto · operate · Full Access
────────────────────────────────────────────
❯ kanban_task_boundary 请读取 /tmp/task.md 中的 Kanban 任务并执行。
"""
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "zellij_dump_screen",
        lambda **kwargs: screen,
    )
    monkeypatch.setattr(deepseek_kanban_interactive.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        deepseek_kanban_interactive.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    listener.on_post_inject(
        Namespace(),
        zellij_session="kanban-test",
        zellij_pane_id="7",
        log_path=tmp_path / "listener.log",
    )

    assert commands == [
        [
            "zellij",
            "--session",
            "kanban-test",
            "action",
            "write",
            "-p",
            "7",
            "13",
        ]
    ]


def test_codewhale_post_inject_does_not_resubmit_busy_prompt(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = deepseek_kanban_interactive.CodeWhaleInteractiveListener()
    commands: list[list[str]] = []
    screen = """\
kanban_task_boundary 请读取任务并执行。
⣤ 工作中
────────────────────────────────────────────
❯ Coordinate parallel tasks...
"""
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "zellij_dump_screen",
        lambda **kwargs: screen,
    )
    monkeypatch.setattr(deepseek_kanban_interactive.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        deepseek_kanban_interactive.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )

    listener.on_post_inject(
        Namespace(),
        zellij_session="kanban-test",
        zellij_pane_id="7",
        log_path=tmp_path / "listener.log",
    )

    assert commands == []


def test_full_access_launch_can_resume_matching_saved_session() -> None:
    assert deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        other_active=False,
        has_sessions=True,
    )


def test_non_full_access_launch_can_resume_saved_session() -> None:
    assert deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        other_active=False,
        has_sessions=True,
    )


def test_launch_does_not_resume_without_match_or_with_same_role_active() -> None:
    assert not deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        other_active=False,
        has_sessions=False,
    )
    assert not deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        other_active=True,
        has_sessions=True,
    )


def test_unreadable_child_environment_uses_listener_profile_role_home(
    tmp_path: Path, monkeypatch,
) -> None:
    parent_by_pid = {200: 100, 100: 1}
    cmdline_by_pid = {
        100: [
            "python3",
            "/repo/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py",
            "--profile",
            "coordinator",
        ],
        1: ["/sbin/init"],
    }
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "_proc_environment_value",
        lambda pid, name: None,
    )
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "_proc_parent_pid",
        lambda pid: parent_by_pid.get(int(pid)),
    )
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "_proc_cmdline",
        lambda pid: cmdline_by_pid.get(int(pid), []),
    )

    assert deepseek_kanban_interactive._codewhale_process_role_home(
        200, home=tmp_path,
    ) == tmp_path / ".codewhale-kanban-coordinator"


def test_saved_session_must_match_project_workspace(tmp_path: Path) -> None:
    role_home = tmp_path / ".codewhale-kanban-implementer"
    sessions = role_home / "sessions"
    sessions.mkdir(parents=True)
    project_workspace = tmp_path / "Egomotion4D"
    old_workspace = project_workspace / ".ds-sessions" / "implementer"
    (sessions / "old.json").write_text(
        json.dumps({"metadata": {"workspace": str(old_workspace)}}),
        encoding="utf-8",
    )

    assert not deepseek_kanban_interactive._has_saved_session_for_workspace(
        role_home=role_home,
        workspace=project_workspace,
    )

    (sessions / "current.json").write_text(
        json.dumps({"metadata": {"workspace": str(project_workspace)}}),
        encoding="utf-8",
    )

    assert deepseek_kanban_interactive._has_saved_session_for_workspace(
        role_home=role_home,
        workspace=project_workspace,
    )


def test_restricted_idle_pane_requests_full_access_before_claim(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = deepseek_kanban_interactive.CodeWhaleInteractiveListener()
    sent: list[tuple[str, str]] = []
    screen = """\
cw DeepSeek · operate · never
───────────────────────────
❯ 编写任务或使用 /。
"""
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "zellij_dump_screen",
        lambda **kwargs: screen,
    )
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "_send_full_access_key",
        lambda *, session, pane_id, log_path: sent.append((session, pane_id)) or True,
    )
    args = Namespace(
        yolo=True,
        zellij_session="kanban-test",
        zellij_pane_id="4",
        profile="coordinator",
    )

    assert not listener._ensure_full_access(args, tmp_path / "listener.log")
    assert sent == [("kanban-test", "4")]


def test_full_access_pane_passes_startup_gate(tmp_path: Path, monkeypatch) -> None:
    listener = deepseek_kanban_interactive.CodeWhaleInteractiveListener()
    screen = """\
cw DeepSeek · operate · Full Access
───────────────────────────
❯ 编写任务或使用 /。
"""
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "zellij_dump_screen",
        lambda **kwargs: screen,
    )
    args = Namespace(
        yolo=True,
        zellij_session="kanban-test",
        zellij_pane_id="1",
        profile="implementer",
    )

    assert listener._ensure_full_access(args, tmp_path / "listener.log")


def test_restricted_busy_pane_does_not_receive_mode_key(
    tmp_path: Path, monkeypatch,
) -> None:
    listener = deepseek_kanban_interactive.CodeWhaleInteractiveListener()
    sent: list[tuple[str, str]] = []
    screen = """\
cw DeepSeek · operate · never
───────────────────────────
⣤ 工作中
"""
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "zellij_dump_screen",
        lambda **kwargs: screen,
    )
    monkeypatch.setattr(
        deepseek_kanban_interactive,
        "_send_full_access_key",
        lambda *, session, pane_id, log_path: sent.append((session, pane_id)) or True,
    )
    args = Namespace(
        yolo=True,
        zellij_session="kanban-test",
        zellij_pane_id="1",
        profile="implementer",
    )

    assert not listener._ensure_full_access(args, tmp_path / "listener.log")
    assert sent == []


def test_role_sessions_do_not_replace_the_project_tool_workspace(tmp_path: Path) -> None:
    project_workspace = tmp_path / "Egomotion4D"

    role_home, tool_workspace = deepseek_kanban_interactive._role_runtime_paths(
        workspace=project_workspace,
        profile="implementer",
        home=tmp_path,
    )

    assert role_home == tmp_path / ".codewhale-kanban-implementer"
    assert tool_workspace == project_workspace


def test_role_config_symlink_is_migrated_to_regular_file(tmp_path: Path) -> None:
    source_home = tmp_path / ".codewhale"
    role_home = tmp_path / ".codewhale-kanban-coordinator"
    source_home.mkdir()
    role_home.mkdir()
    source_config = source_home / "config.toml"
    source_config.write_text('model = "deepseek-v4-pro"\n', encoding="utf-8")
    role_config = role_home / "config.toml"
    role_config.symlink_to(source_config)

    migrated = deepseek_kanban_interactive._prepare_role_config(
        role_home=role_home,
        source_home=source_home,
    )

    assert migrated == role_config
    assert role_config.is_file()
    assert not role_config.is_symlink()
    assert role_config.read_text(encoding="utf-8") == source_config.read_text(
        encoding="utf-8"
    )
    assert (role_home / ".config.toml.symlink-backup").is_symlink()


def test_existing_regular_role_config_is_preserved(tmp_path: Path) -> None:
    source_home = tmp_path / ".codewhale"
    role_home = tmp_path / ".codewhale-kanban-implementer"
    source_home.mkdir()
    role_home.mkdir()
    (source_home / "config.toml").write_text('model = "source"\n', encoding="utf-8")
    role_config = role_home / "config.toml"
    role_config.write_text('model = "role-specific"\n', encoding="utf-8")

    deepseek_kanban_interactive._prepare_role_config(
        role_home=role_home,
        source_home=source_home,
    )

    assert role_config.read_text(encoding="utf-8") == 'model = "role-specific"\n'
