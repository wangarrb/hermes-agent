from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "local"
    / "bin"
    / "hermes-kanban-role-context-listener.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("hermes_role_context_listener_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_same_role_without_task_skills_adds_no_context(tmp_path: Path) -> None:
    module = load_module()

    assert module.render_role_context(
        board="board",
        workspace=tmp_path,
        pane_profile="planner",
        task_assignee="planner",
        task_skills=[],
        profiles_root=tmp_path / "profiles",
        shared_skills_root=tmp_path / "skills",
    ) == ""


def test_assist_role_loads_declared_profile_and_project_prompt(tmp_path: Path) -> None:
    module = load_module()
    profiles = tmp_path / "profiles"
    implementer = profiles / "implementer"
    (implementer / "skills" / "delivery").mkdir(parents=True)
    (implementer / "role-context.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "implementer",
                "description": "Implement the frozen delivery contract.",
                "skills": [
                    {"name": "delivery", "path": "skills/delivery/SKILL.md"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (implementer / "skills" / "delivery" / "SKILL.md").write_text(
        "# Delivery\nRun focused tests.\n", encoding="utf-8"
    )
    prompt = (
        tmp_path
        / ".hermes-kanban"
        / "board"
        / "implementer"
        / "kanban-system-prompt.md"
    )
    prompt.parent.mkdir(parents=True)
    prompt.write_text("Implementer project contract.\n", encoding="utf-8")

    rendered = module.render_role_context(
        board="board",
        workspace=tmp_path,
        pane_profile="coordinator",
        task_assignee="implementer",
        task_skills=[],
        profiles_root=profiles,
        shared_skills_root=tmp_path / "shared-skills",
    )

    assert "effective_role=implementer assist=true" in rendered
    assert "Implement the frozen delivery contract." in rendered
    assert "Run focused tests." in rendered
    assert "Implementer project contract." in rendered
    assert "BLOCK_ASSIST_ROLE_CONTEXT" not in rendered
