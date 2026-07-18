"""CodeWhale composer detection for interactive Kanban injection."""

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


def test_full_access_launch_never_resumes_saved_session() -> None:
    assert not deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        full_access_requested=True,
        other_active=False,
        has_sessions=True,
    )


def test_non_full_access_launch_can_resume_saved_session() -> None:
    assert deepseek_kanban_interactive._should_continue_session(
        continue_requested=True,
        full_access_requested=False,
        other_active=False,
        has_sessions=True,
    )


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
