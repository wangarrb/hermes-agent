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
import re
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
    role_context_backend = "codex"

    # ── Idle/busy markers ──
    # Codex CLI uses › (U+203A) as its prompt symbol since v0.9+
    idle_markers: tuple[str, ...] = ("› ",)
    # Text busy states are live Codex status rows, not arbitrary transcript
    # prose (for example, a completed answer may say a goal remains running).
    busy_markers: tuple[str, ...] = ("• thinking", "• working", "• running", "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    queued_input_markers: tuple[str, ...] = ()

    # ── Abstract method implementations ──

    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        provider: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        if continue_session:
            # Codex v0.100+ uses 'codex resume --last' instead of 'codex --continue'
            cmd = ["codex", "resume", "--last"]
        else:
            cmd = ["codex"]
        if provider:
            cmd.extend(["-c", f'model_provider="{provider}"'])
        if model:
            cmd.extend(["--model", model])
        if sandbox:
            cmd.extend(["--sandbox", sandbox])
        cmd.extend(extra_args or [])
        cmd.append(str(workspace))
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check if Codex has saved sessions for this workspace.

        Codex v0.100+ stores sessions in CODEX_HOME/sessions/YYYY/MM/DD/.
        When using per-role CODEX_HOME (e.g. ~/.codex-kanban/<role>/),
        check that path first, then fall back to ~/.codex/sessions/.
        Older versions used ~/.codex/projects/<encoded-cwd>/.
        """
        import os

        # Determine CODEX_HOME: env var takes priority, then ~/.codex
        codex_home = os.environ.get("CODEX_HOME")
        if codex_home:
            codex_home_path = Path(codex_home)
        else:
            codex_home_path = Path.home() / ".codex"

        # v0.100+ path: CODEX_HOME/sessions/
        sessions_dir = codex_home_path / "sessions"
        if sessions_dir.is_dir():
            try:
                for child in sessions_dir.rglob("rollout-*.jsonl"):
                    return True
            except OSError:
                pass

        # Also check ~/.codex/sessions/ (global sessions when using per-role CODEX_HOME)
        global_sessions = Path.home() / ".codex" / "sessions"
        if global_sessions != sessions_dir and global_sessions.is_dir():
            try:
                for child in global_sessions.rglob("rollout-*.jsonl"):
                    return True
            except OSError:
                pass

        # Legacy path: CODEX_HOME/projects/<base64-cwd>/
        projects_dir = codex_home_path / "projects"
        if projects_dir.is_dir():
            import base64
            encoded = base64.urlsafe_b64encode(str(workspace).encode()).decode().rstrip("=")
            project_dir = projects_dir / encoded
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

    def pane_is_idle(self, screen: str) -> bool:
        """Require an idle composer with no busy marker in the live viewport."""
        tail_lines = _tail_nonempty_lines(screen, limit=5)
        if not tail_lines:
            return False

        viewport = "\n".join(tail_lines).lower()
        if any(marker.lower() in viewport for marker in self.busy_markers):
            return False

        last_line = tail_lines[-1].lower()
        is_status_bar = "context" in last_line and "%" in last_line
        # Codex TUI may render the composer prompt and status bar on the same
        # line (e.g. "› Implement {feature}  gpt-5.6-sol · Context 20% used").
        # In that case the idle marker is on the status-bar line itself, so
        # check the last line rather than skipping to the second-to-last.
        if is_status_bar and not any(
            marker.lower() in last_line for marker in self.idle_markers
        ):
            check_line = (
                tail_lines[-2].lower()
                if len(tail_lines) >= 2
                else last_line
            )
        else:
            check_line = last_line
        return any(marker.lower() in check_line for marker in self.idle_markers)

    _COMPOSER_PROMPT_RE = re.compile(r"^\s*›(?:\s?(.*))?$")

    def composer_input_text(self, screen: str) -> str | None:
        """Extract the current Codex composer buffer, excluding its status bar."""
        lines = screen.splitlines()
        prompt_index = None
        first_text = ""
        for index in range(len(lines) - 1, -1, -1):
            match = self._COMPOSER_PROMPT_RE.match(lines[index])
            if match:
                prompt_index = index
                first_text = (match.group(1) or "").strip()
                break
        if prompt_index is None:
            return None

        parts = [first_text] if first_text else []
        for line in lines[prompt_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if "context" in lowered and ("%" in lowered or "used" in lowered):
                break
            parts.append(stripped)
        payload = "\n".join(parts).strip()
        return payload or None

    # ── Override on_claim_pre_check: check last 5 lines, not just last line ──
    # Codex TUI layout puts the "›" prompt 2-3 lines above the bottom status
    # bar (model name, workspace path).  The base class only checks the very
    # last non-empty line, which is the status bar and never contains "›",
    # so the pane is never deemed ready. We check the live viewport and, when
    # the persistent composer contains text, require three unchanged 10-second
    # intervals. Content changes reset the count.
    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        if not self.idle_markers:
            return True
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        screen = zellij_dump_screen(
            session=session, pane_id=str(pane_id), log_path=log_path,
        )
        if not screen:
            return False
        tail_lines = _tail_nonempty_lines(screen, limit=5)
        if not tail_lines or not self.pane_is_idle(screen):
            last_line = tail_lines[-1].lower() if tail_lines else ""
            log_line(log_path, (
                "on_claim_pre_check: not ready (live viewport is not idle; "
                f"last={last_line[:60]})"
            ))
            return False
        return self.wait_for_stable_composer_input(
            session=session,
            pane_id=str(pane_id),
            log_path=log_path,
            initial_screen=screen,
            screen_reader=zellij_dump_screen,
        )

    def on_claim_post_confirm(
        self, args: argparse.Namespace, log_path: Path,
    ) -> bool:
        """Close the race where typing starts after the pre-claim probe."""
        session = getattr(args, "zellij_session", "")
        pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not session or not pane_id:
            return False
        return self.wait_for_stable_composer_input(
            session=session,
            pane_id=pane_id,
            log_path=log_path,
            screen_reader=zellij_dump_screen,
        )

    # ── Post-inject: retry Enter if prompt stays in composer ──
    _POST_INJECT_CONFIRM_S = 1.5
    _POST_INJECT_MAX_RETRIES = 3

    def on_post_inject(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
    ) -> None:
        """Submit a Codex prompt that remained queued after the first Enter.

        Codex TUI sometimes does not submit on the first CR (raw byte 13).
        Retry up to _POST_INJECT_MAX_RETRIES times, checking the screen each
        time for the queued prompt.  Stop as soon as the prompt is gone (sent)
        or a busy marker appears (agent started processing).
        """
        import time as _t
        import subprocess as _sp
        cmd_base = (
            ["zellij", "--session", zellij_session, "action"]
            if zellij_session
            else ["zellij", "action"]
        )
        for attempt in range(1, self._POST_INJECT_MAX_RETRIES + 1):
            _t.sleep(self._POST_INJECT_CONFIRM_S)
            screen = zellij_dump_screen(
                session=zellij_session,
                pane_id=zellij_pane_id,
                log_path=log_path,
            )
            if not screen:
                return
            tail = _tail_nonempty_lines(screen, limit=20)
            tail_lower = "\n".join(tail).lower()
            # Check if any busy marker appeared (agent started)
            if any(m.lower() in tail_lower for m in self.busy_markers):
                return
            # Check if the injected prompt is still visible (unsent)
            has_queued = any(
                "请读取" in line and "kanban" in line.lower()
                for line in tail
            )
            if not has_queued:
                if attempt > 1:
                    log_line(
                        log_path,
                        f"codex post-inject: queued prompt cleared after {attempt} Enter(s)",
                    )
                return
            # Prompt still queued — send another Enter
            _sp.run(
                cmd_base + ["write", "-p", zellij_pane_id, "13"],
                check=False,
                stdout=_sp.DEVNULL,
                stderr=_sp.PIPE,
                text=True,
                timeout=5,
            )
            log_line(
                log_path,
                f"codex post-inject: queued Kanban prompt remained; sent raw Enter (attempt {attempt}/{self._POST_INJECT_MAX_RETRIES})",
            )


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = CodexInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
