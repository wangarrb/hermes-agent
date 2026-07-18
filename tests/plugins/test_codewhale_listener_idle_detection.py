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
