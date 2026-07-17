"""CodeWhale composer detection for interactive Kanban injection."""

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
