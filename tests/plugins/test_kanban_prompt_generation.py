from __future__ import annotations

from pathlib import Path

from plugins.kanban import base_listener as bl


def _write(tmp_path: Path, *, run_id: int, generation: int) -> Path:
    return bl.write_task_prompt(
        agent_name="Codex",
        agent_slug="codex",
        board="default",
        profile="planner",
        task_id="t_same",
        task_assignee="planner",
        task_title="same task",
        context=f"run={run_id} generation={generation}",
        workspace=tmp_path,
        run_id=run_id,
        generation=generation,
    )


def test_repeated_task_runs_write_distinct_prompt_paths(tmp_path: Path) -> None:
    first = _write(tmp_path, run_id=11, generation=1)
    second = _write(tmp_path, run_id=12, generation=2)

    assert first != second
    assert first.name == "task-t_same-run-11-generation-1.md"
    assert second.name == "task-t_same-run-12-generation-2.md"
    assert first.read_text(encoding="utf-8").find("run=11") >= 0
    assert second.read_text(encoding="utf-8").find("run=12") >= 0


def test_legacy_task_prompt_path_remains_readable(tmp_path: Path) -> None:
    legacy = bl.task_prompt_path(
        tmp_path, "default", "planner", "codex", "t_same",
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy prompt", encoding="utf-8")

    resolved = bl.resolve_task_prompt_path(
        tmp_path, "default", "planner", "codex", "t_same",
        run_id=99, generation=3,
    )
    assert resolved == legacy
    assert resolved.read_text(encoding="utf-8") == "legacy prompt"
