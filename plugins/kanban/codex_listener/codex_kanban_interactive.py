#!/usr/bin/env python3
"""CodexInteractiveListener — Hermes Kanban listener for OpenAI Codex CLI.

Codex TUI idle markers:
  - Idle:  "> " prompt prefix (user input line)
  - Busy:  "thinking" / "running" / "⠋" spinner chars
  - Queued: (no specific marker; Codex processes input sequentially)

Injection strategy: write prompt to a .md file, inject a single-line
command that reads and executes the prompt.  No \\n in injected text.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Base class import ──
# PLUGIN_DIR = hermes-agent/plugins/kanban/<listener>/  →  we need hermes-agent/ on sys.path
HERMES_AGENT_ROOT = Path(__file__).resolve().parents[3]
if str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))
PLUGIN_KANBAN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_KANBAN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_KANBAN_DIR))

from base_listener import (  # noqa: E402
    BaseInteractiveListener,
    claim_assignees,
    log,
    log_line,
    now_s,
    prompt_dir,
    role_guidance,
    zellij_dump_screen,
    _pane_can_accept_new_kanban_task,
    _tail_nonempty_lines,
)


class CodexInteractiveListener(BaseInteractiveListener):
    agent_name = "Codex"
    agent_slug = "codex"

    # ── Idle/busy markers ──
    # Codex CLI uses › (U+203A) as its prompt symbol since v0.9+
    idle_markers: tuple[str, ...] = ("› ",)
    busy_markers: tuple[str, ...] = ("thinking", "running", "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    queued_input_markers: tuple[str, ...] = ()

    # ── Abstract method implementations ──

    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = ["codex"]
        if continue_session:
            cmd.append("--continue")
        if model:
            cmd.extend(["--model", model])
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        cmd.extend(extra_args or [])
        cmd.append(str(workspace))
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check if Codex has saved sessions for this workspace.

        Codex stores sessions in ~/.codex/projects/<encoded-cwd>/.
        """
        codex_dir = Path.home() / ".codex" / "projects"
        if not codex_dir.is_dir():
            return False
        # Encode workspace path the way Codex does (URL-safe base64 of path)
        import base64
        encoded = base64.urlsafe_b64encode(str(workspace).encode()).decode().rstrip("=")
        project_dir = codex_dir / encoded
        return project_dir.is_dir() and any(project_dir.iterdir())

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build single-line injection for Codex.

        Codex accepts freeform text at the '>' prompt.
        We tell it to read the task prompt file and execute.
        """
        return (
            f"请读取 {prompt_path} 中的 Kanban 任务并执行。"
            f" [任务 {task_id}: {title}]"
        )

    def pane_label(self, task_id: str | None = None) -> str:
        if task_id:
            return f"codex-kanban [{task_id}]"
        return "codex-kanban"


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = CodexInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
