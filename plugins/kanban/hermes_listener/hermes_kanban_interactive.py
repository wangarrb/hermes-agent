#!/usr/bin/env python3
"""HermesInteractiveListener — Hermes Kanban listener for the Hermes CLI itself.

Unlike Codex/CodeWhale/Claude which are external TUI agents, Hermes is the
native agent.  The watcher runs as a separate process and injects task
prompts into a running `hermes --continue` session via zellij.

Key differences from other listeners:
  - No TUI launch: `hermes --continue` is already running in the pane.
  - The watcher is "watch-only" by default — it only claims and injects.
  - Idle markers: "›" or "❯" (hermes idle prompt).
  - Injection: single-line instruction to read the task prompt file.

Usage:
  # Watcher-only (recommended for zellij layout):
  python3 hermes_kanban_interactive.py \\
      --watch-only --auto-start \\
      --profile implementer --claim-assignees implementer \\
      --board egomotion4d --workspace /home/wyr/code/Egomotion4D \\
      --zellij-session kanban-egomotion4d --zellij-pane-id 1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

# ── Base class import ──
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
    zellij_inject,
    zellij_rename_pane,
)

# ── Hermes-specific imports ──
HERMES_REPO = HERMES_AGENT_ROOT
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402


# ──────────────────────────────────────────────
# Hermes-specific constants
# ──────────────────────────────────────────────

_HERMES_IDLE_MARKERS = (
    "›",
    "❯",
    "msg=interrupt",
    "implementer ❯",
    "critic ❯",
    "planner ❯",
    "coordinator ❯",
)

_HERMES_BUSY_MARKERS = (
    "activity: thinking",
    "running",
    "executing",
    "preparing terminal",
    "💻 $",
    "work kanban",
    "kanban --board",
)

_HERMES_QUEUED_INPUT_MARKERS = ()


# ──────────────────────────────────────────────
# Hermes subclass
# ──────────────────────────────────────────────

class HermesInteractiveListener(BaseInteractiveListener):
    agent_name = "Hermes"
    agent_slug = "hermes"

    idle_markers = _HERMES_IDLE_MARKERS
    busy_markers = _HERMES_BUSY_MARKERS
    queued_input_markers = _HERMES_QUEUED_INPUT_MARKERS

    # ── Build TUI command (not used in watch-only mode) ──
    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        cmd = ["hermes"]
        if continue_session:
            cmd.append("--continue")
        cmd.extend(extra_args or [])
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        # Hermes sessions are managed internally; always use --continue
        return True

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build single-line injection for Hermes CLI.

        Hermes reads the prompt file and executes the task.
        No \\n in injected text (safe for PTY raw mode).
        """
        return (
            f"请读取 {prompt_path} 中的 Kanban 任务并执行。"
            f" [任务 {task_id}: {title}]"
        )

    def pane_label(self, task_id: str | None = None) -> str:
        if task_id:
            return f"hermes-kanban [{task_id}]"
        return "hermes-kanban"

    # ── Override launcher_main: hermes is already running in the pane ──
    def launcher_main(self, args: argparse.Namespace) -> int:
        """For Hermes, launcher_main just runs watcher_main.

        The hermes --continue session is already running in the zellij pane
        (started via hermes-kanban-continue).  We only need the watcher
        process to claim tasks and inject prompts.
        """
        return self.watcher_main(args)


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = HermesInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
