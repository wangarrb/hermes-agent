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

import argparse
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
        if continue_session:
            # Codex v0.100+ uses 'codex resume --last' instead of 'codex --continue'
            cmd = ["codex", "resume", "--last"]
        else:
            cmd = ["codex"]
        if model:
            cmd.extend(["--model", model])
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        cmd.extend(extra_args or [])
        cmd.append(str(workspace))
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check if Codex has saved sessions for this workspace.

        Codex v0.100+ stores sessions in ~/.codex/sessions/YYYY/MM/DD/.
        Older versions used ~/.codex/projects/<encoded-cwd>/.
        Check both locations.
        """
        # v0.100+ path: ~/.codex/sessions/
        sessions_dir = Path.home() / ".codex" / "sessions"
        if sessions_dir.is_dir():
            try:
                # Check if any session files exist (walk all date subdirs)
                for child in sessions_dir.rglob("rollout-*.jsonl"):
                    return True
            except OSError:
                pass

        # Legacy path: ~/.codex/projects/<base64-cwd>/
        codex_dir = Path.home() / ".codex" / "projects"
        if codex_dir.is_dir():
            import base64
            encoded = base64.urlsafe_b64encode(str(workspace).encode()).decode().rstrip("=")
            project_dir = codex_dir / encoded
            if project_dir.is_dir() and any(project_dir.iterdir()):
                return True

        return False

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

    # ── Override on_claim_pre_check: check last 5 lines, not just last line ──
    # Codex TUI layout puts the "›" prompt 2-3 lines above the bottom status
    # bar (model name, workspace path).  The base class only checks the very
    # last non-empty line, which is the status bar and never contains "›",
    # so the pane is never deemed ready.  We check the last 5 non-empty lines
    # for an idle marker and ensure no busy marker is present in the same
    # window.  Requires TWO consecutive checks, 2 s apart, for stability
    # (same pattern as HermesInteractiveListener).
    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        if not self.idle_markers:
            return True
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        for attempt in range(2):
            screen = zellij_dump_screen(
                session=session, pane_id=str(pane_id), log_path=log_path,
            )
            if not screen:
                return False
            tail_lines = _tail_nonempty_lines(screen, limit=5)
            if not tail_lines:
                return False
            tail = "\n".join(tail_lines).lower()
            has_idle = any(m.lower() in tail for m in self.idle_markers)
            has_busy = any(m.lower() in tail for m in self.busy_markers)
            if not has_idle or has_busy:
                last_line = tail_lines[-1].lower() if tail_lines else ""
                log_line(log_path, (
                    f"on_claim_pre_check attempt {attempt+1}/2: "
                    f"not ready (idle={has_idle} busy={has_busy} "
                    f"last={last_line[:60]})"
                ))
                return False
            if attempt == 0:
                import time as _t
                _t.sleep(2.0)
        return True


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = CodexInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
