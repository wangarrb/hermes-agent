from __future__ import annotations

import importlib.util
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


def test_wrapper_reexports_shared_effective_role_renderer() -> None:
    module = load_module()

    assert module.render_role_context is module.render_effective_role_context


def test_wrapper_delegates_directly_to_upstream_listener(tmp_path, monkeypatch) -> None:
    module = load_module()
    called = []
    fake = type("Upstream", (), {"main": lambda self, args: called.append(args) or 7})()
    monkeypatch.setitem(sys.modules, "hermes_kanban_interactive", fake)
    monkeypatch.setattr(module, "DEFAULT_HERMES_KANBAN_ROOT", tmp_path)

    assert module.main(["--profile", "planner"]) == 7
    assert called == [["--profile", "planner"]]
