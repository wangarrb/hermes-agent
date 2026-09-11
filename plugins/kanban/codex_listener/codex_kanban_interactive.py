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
import time
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
    zellij_submit_enter,
    _pane_can_accept_new_kanban_task,
    _tail_nonempty_lines,
)
from session_scope import latest_codex_thread  # noqa: E402


class CodexInteractiveListener(BaseInteractiveListener):
    agent_name = "Codex"
    agent_slug = "codex"
    role_context_backend = "codex"

    # ── Idle/busy markers ──
    # Codex CLI uses › (U+203A) as its prompt symbol since v0.9+
    idle_markers: tuple[str, ...] = ("›",)
    # Text busy states are live Codex status rows, not arbitrary transcript
    # prose (for example, a completed answer may say a goal remains running).
    busy_markers: tuple[str, ...] = ("• thinking", "• working", "• running", "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    queued_input_markers: tuple[str, ...] = ()
    semantic_delivery_required = True

    # ── Abstract method implementations ──

    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        provider: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        resume_session_id = getattr(self, "_resume_session_id", None)
        if continue_session and resume_session_id:
            cmd = ["codex", "resume", str(resume_session_id)]
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
        """Resolve an explicit saved thread for this exact workspace."""
        codex_home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        self._resume_session_id = latest_codex_thread(codex_home, workspace)
        return self._resume_session_id is not None

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

        # The prompt may be bare, followed by role/status text, or share a
        # row with the status bar.  Only a prompt glyph at the start of a
        # current line counts; transcript prose containing › is ignored.
        return any(re.match(r"^\s*›(?:\s|$)", line) for line in tail_lines)

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

        # Strip an inline role/status suffix while preserving composer text,
        # e.g. ``› marker  gpt-5.6-sol high · Context 20% used``.  A row made
        # entirely of role/status metadata is therefore treated as empty.
        if first_text and "context" in first_text.lower() and "%" in first_text:
            inline_status = re.search(
                r"(?:\s{2,}|^)(?:[\w.-]+(?:\s+\w+)*\s*)?·.*?context\b.*$",
                first_text,
                flags=re.IGNORECASE,
            )
            if inline_status:
                first_text = first_text[:inline_status.start()].rstrip()
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

    def wait_for_stable_composer_input(
        self, *, session: str, pane_id: str, log_path: Path,
        initial_screen: str | None = None,
        screen_reader: object | None = None,
    ) -> bool:
        """Allow automation only when Codex's composer is currently empty.

        Codex keeps user drafts in a persistent composer; waiting for a stable
        draft would still overwrite it.  Automatic task/control/result paths
        therefore require an idle prompt with no composer text.
        """
        read_screen = screen_reader or zellij_dump_screen
        screen = initial_screen
        if screen is None:
            screen = read_screen(
                session=session, pane_id=pane_id, log_path=log_path,
            )
        return bool(screen and self.pane_is_idle(screen) and not self.composer_input_text(screen))

    # ── Override on_claim_pre_check: check last 5 lines, not just last line ──
    # Codex TUI layout puts the "›" prompt above the bottom status bar (model
    # name, workspace path).  The base class only checks the very last
    # non-empty line, so use the live viewport and refuse to overwrite any
    # persistent composer draft.
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
        if self.composer_input_text(screen):
            log_line(log_path, "on_claim_pre_check: not ready (composer has draft text)")
            return False
        return True

    def on_claim_post_confirm(
        self, args: argparse.Namespace, log_path: Path,
    ) -> bool:
        """Close the race where typing starts after the pre-claim probe."""
        session = getattr(args, "zellij_session", "")
        pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(
            session=session, pane_id=pane_id, log_path=log_path,
        )
        if not screen or not self.pane_is_idle(screen):
            return False
        if self.composer_input_text(screen):
            log_line(log_path, "on_claim_post_confirm: composer has draft text")
            return False
        return True

    # ── Post-inject: retry Enter if prompt stays in composer ──
    _POST_INJECT_CONFIRM_S = 1.5
    _POST_INJECT_MAX_RETRIES = 3

    def on_post_inject(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
        injected_marker: str | None = None,
        pre_write_composer: str | None = None,
        correlation: str | None = None,
    ) -> str:
        """Confirm that the injected prompt left Codex's current composer.

        Only the live composer is inspected.  A marker must first be observed
        there, then disappear (or transition to a live busy state) before
        returning ``confirmed``.  A marker that remains queued after bounded
        retries is ``known_unsubmitted``; missing/unsupported screens are
        ``unknown``.
        """
        del args, pre_write_composer
        marker = str(injected_marker or "").strip()
        if not marker:
            return "unknown"
        saw_marker = False
        submit_correlation = correlation or (
            f"task:{getattr(self, '_active_task_id', '')}"
            if getattr(self, "_active_task_id", None)
            else (
                f"control:{getattr(self, '_active_control_id', '')}"
                if getattr(self, "_active_control_id", None) is not None
                else "result:"
            )
        )

        def _marker_present(composer: str | None) -> bool:
            if not composer:
                return False
            compact = " ".join(composer.split())
            target = " ".join(marker.split())
            return target in compact

        def _busy(screen: str) -> bool:
            live_lines = _tail_nonempty_lines(screen, limit=5)
            for index, line in enumerate(live_lines):
                lowered = line.lower()
                if not any(item.lower() in lowered for item in self.busy_markers):
                    continue
                # A busy-looking transcript row above the current composer is
                # stale output, not a live transition.
                if any(re.match(r"^\s*›(?:\s|$)", tail)
                       for tail in live_lines[index + 1 :]):
                    continue
                return True
            return False

        for attempt in range(1, self._POST_INJECT_MAX_RETRIES + 1):
            time.sleep(self._POST_INJECT_CONFIRM_S)
            screen = zellij_dump_screen(
                session=zellij_session,
                pane_id=zellij_pane_id,
                log_path=log_path,
            )
            if not screen:
                return "unknown"
            composer = self.composer_input_text(screen)
            if _busy(screen):
                return "confirmed"
            if _marker_present(composer):
                saw_marker = True
                zellij_submit_enter(
                    session=zellij_session,
                    pane_id=zellij_pane_id,
                    expected_pane_prefix=self.expected_pane_prefix(),
                    correlation=submit_correlation,
                    log_path=log_path,
                )
                log_line(log_path, f"codex post-inject: queued prompt remained; sent Enter (attempt {attempt}/{self._POST_INJECT_MAX_RETRIES})")
                continue
            if saw_marker:
                return "confirmed"
            # Non-empty unrelated composer or a screen without a supported
            # composer is inconclusive; never scan transcript tail for marker.
            return "unknown"
        return "known_unsubmitted" if saw_marker else "unknown"


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = CodexInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
