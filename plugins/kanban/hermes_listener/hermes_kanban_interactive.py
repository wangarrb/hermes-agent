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
import re
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
    _tail_nonempty_lines,
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

import time

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
    "implementer ❯",
    "critic ❯",
    "planner ❯",
    "coordinator ❯",
    "reviewer ❯",
)

_HERMES_BUSY_MARKERS = (
    "activity: thinking",
    "running",
    "executing",
    "preparing terminal",
    "💻 $",
    "💻 preparing terminal",
    "work kanban",
    "kanban --board",
    "msg=interrupt",
)

_HERMES_QUEUED_INPUT_MARKERS = ()


# ──────────────────────────────────────────────
# Hermes subclass
# ──────────────────────────────────────────────

class HermesInteractiveListener(BaseInteractiveListener):
    agent_name = "Hermes"
    agent_slug = "hermes"
    role_context_backend = "hermes"

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

    # ── Override on_task_running_monitor: stricter API error detection ──
    # Hermes pane shows ❯ prompt even while working (ghost state), so the
    # base listener's idle+error detection causes false "继续" injections.
    # We require BOTH:
    #   1. Pane is truly idle (idle marker in last 5 lines, no busy marker)
    #   2. API error appears in the last 5 lines (not just anywhere in 20 lines)
    # This prevents matching error words in normal scrollback/output.
    #
    # Markers are deliberately narrow — only match concrete transport/HTTP/protocol
    # errors.  DO NOT include broad markers like "⚠", "error", "failed", "timeout"
    # etc.: these appear in normal Hermes output (tool stderr, user content, warning
    # messages) and cause false "继续" injections that interrupt working tasks.
    _HERMES_STRICT_ERROR_MARKERS: tuple[str, ...] = (
        "api call failed", "api request failed",
        "xunfei request failed",
        "notenoughcv", "engineinternalerror", "system is busy",
        "connection refused", "connection reset",
        "connection aborted", "connection broken",
        "connection closed by remote",
        "connect timeout", "connection timeout",
        "read timeout",
        "proxy error",
        "ssl error", "broken pipe",
        "remote end closed connection",
        "network is unreachable",
        "http 429", "http 502", "http 503", "http 504",
        "http 400",
        "code: 429", "code: 502", "code: 503", "code: 504",
        "error code: 429", "error code: 502", "error code: 503", "error code: 504",
        # Hermes error box markers (span 15+ lines, need wider detection window)
        "non-retryable",
        "aborting",
        "invalid character",
        "inference failed",
        "param validation error",
    )

    def on_task_running_monitor(
        self, args: argparse.Namespace, conn: Any,
        task_id: str, log_path: Path,
    ) -> None:
        """Stricter monitoring: only inject '继续' when error is in the
        very last lines (not scrollback) AND pane is truly idle."""
        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not zellij_session or not zellij_pane_id:
            return

        screen = zellij_dump_screen(session=zellij_session, pane_id=zellij_pane_id, log_path=log_path)
        if not screen:
            return

        # Use last 20 lines for error detection (Hermes error boxes span 15+ lines)
        tail_lines = _tail_nonempty_lines(screen, limit=20)
        tail = "\n".join(tail_lines).lower()

        # Idle marker must be in the LAST line (prompt at bottom), not just
        # anywhere in tail — › can appear in tool output, scrollback, etc.
        # Additionally, the line must be a BARE prompt (marker + whitespace only).
        # If the user is typing, the line looks like "❯ some text…" and is NOT idle.
        last_line = tail_lines[-1] if tail_lines else ""
        has_idle = self._is_truly_idle_line(last_line)
        has_busy = any(m.lower() in tail for m in self.busy_markers)
        if not has_idle or has_busy:
            # Pane is busy or not showing idle prompt — reset retry state
            self._api_retry_count = 0
            self._api_retry_first_at = None
            return

        # Check for API error in the tail (Hermes error boxes are wide)
        has_error = any(m in tail for m in self._HERMES_STRICT_ERROR_MARKERS)
        if not has_error:
            self._api_retry_count = 0
            self._api_retry_first_at = None
            return

        # API error confirmed in last 5 lines — retry with backoff
        if self._api_retry_count >= self.API_RETRY_MAX:
            return

        now = time.time()
        if self._api_retry_first_at is None:
            self._api_retry_first_at = now
            log_line(log_path, f"api-error-idle observed for task {task_id} (retry {self._api_retry_count}/{self.API_RETRY_MAX})")

        elapsed = now - self._api_retry_first_at
        backoff = self.API_RETRY_BACKOFF[self._api_retry_count] if self._api_retry_count < len(self.API_RETRY_BACKOFF) else 60.0

        if elapsed < backoff:
            return

        self._api_retry_count += 1
        self._api_retry_first_at = None
        log_line(log_path, f"api-error-retry {self._api_retry_count}/{self.API_RETRY_MAX} for task {task_id}: injecting 继续 after {elapsed:.0f}s")

        zellij_inject(session=zellij_session, pane_id=zellij_pane_id, text="继续", log_path=log_path)
        time.sleep(0.5)
        zellij_inject(session=zellij_session, pane_id=zellij_pane_id, text="\r", log_path=log_path)

    # ── Override on_claim_pre_check: only last line → idle ──
    # Hermes shows the › prompt between every turn.  Checking 40 lines
    # for idle markers (base class) is too wide — a stray › from 5+
    # lines ago makes the pane look idle.  We only check the LAST
    # non-empty line for a leading idle marker pattern: the prompt always
    # ends the last visible line (e.g. "coordinator ❯ " or "› ").
    # Requires TWO consecutive checks, 2s apart, for stability.
    #
    # BUGFIX: zellij often draws a horizontal border line (─────)
    # below the idle prompt, making it the last non-empty line.
    # We skip "decorative" lines (pure box-drawing chars) so the
    # actual prompt line is found.
    #
    # BUGFIX 2: The idle marker must match ONLY when the prompt line
    # contains nothing after the marker except whitespace.  When the
    # user is actively typing, the line looks like "❯ some text…" —
    # the marker is present but the pane is NOT idle.  We must NOT
    # inject into a pane where the user is composing input.
    _DECORATIVE_LINE_RE = re.compile(r'^[─═│┃┤├┬┴┼┌┐└┘╭╰╮╯╚╝─┄┈╶╨╺╻╼╽╾╿┣┡┢┥┙┛┝┟┠┞]+$')

    # Pattern: idle marker at start of line, followed by optional whitespace only.
    # Matches: "❯ ", "› ", "planner ❯ ", "coordinator ❯  "
    # Does NOT match: "❯ some text", "❯/steer 记住…"
    _IDLE_ONLY_RE = re.compile(
        r'^(?:'
        r'(?:coordinator|planner|implementer|critic|reviewer)\s*'
        r')?'
        r'[›❯]'
        r'\s*$'
    )

    def _is_truly_idle_line(self, line: str) -> bool:
        """Return True only if the line is a bare prompt with no user input."""
        return bool(self._IDLE_ONLY_RE.match(line.strip()))

    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        if not self.idle_markers:
            return True
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        for attempt in range(2):
            screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
            if not screen:
                return False
            # Find last non-empty, non-decorative line
            last_line = ""
            for line in reversed(screen.splitlines()):
                line = line.strip()
                if line and not self._DECORATIVE_LINE_RE.match(line):
                    last_line = line
                    break
            # Strict idle check: prompt marker with NO user input after it
            if not self._is_truly_idle_line(last_line):
                log_line(log_path, f"on_claim_pre_check attempt {attempt+1}/2: last line NOT truly idle ({last_line[:80]})")
                return False
            # Also check busy markers in the tail — Hermes shows ❯ even
            # between turns while executing tools; busy markers (💻, msg=interrupt)
            # indicate the agent is still working and must not be interrupted.
            # Only check the LAST 5 non-empty lines (viewport scope) to avoid
            # false positives from scrollback: Hermes tool output boxes (┊ 💻 …)
            # linger in scrollback long after the tool finishes.
            tail = "\n".join(_tail_nonempty_lines(screen, limit=5)).lower()
            has_busy = any(m.lower() in tail for m in self.busy_markers)
            if has_busy:
                log_line(log_path, f"on_claim_pre_check attempt {attempt+1}/2: busy marker detected, NOT idle")
                return False
            if attempt == 0:
                time.sleep(2.0)
        return True

    # ── Override on_claim_post_confirm: don't steal other roles' tasks ──
    # If this watcher claimed a task whose assignee differs from the
    # watcher's profile, reclaim it immediately.  This prevents the
    # coordinator from stealing implementer tasks.
    def on_claim_post_confirm(self, args: argparse.Namespace, log_path: Path,
                              task_id: str | None = None) -> bool:
        return True  # identity: we override claim_and_inject_one instead

    def claim_and_inject_one(
        self, args: argparse.Namespace, *, log_path: Path, conn: Any | None = None,
    ) -> tuple[str | None, int | None]:
        result = super().claim_and_inject_one(args, log_path=log_path, conn=conn)
        if result[0] is None:
            return result  # nothing claimed
        task_id, run_id = result
        # Check: did we claim a task for a role we're NOT authorized to assist?
        # If the watcher's claim_assignees includes the task's assignee, it's an
        # intended assist (e.g. coordinator assisting implementer), not stealing.
        pane_profile = self._profile
        authorized_assignees = set(claim_assignees(args))
        try:
            conn2 = conn or kb.connect(board=self._board)
            task = kb.get_task(conn2, task_id)
            if task and task.assignee and task.assignee != pane_profile:
                if task.assignee in authorized_assignees:
                    # Intended assist — keep the claim and proceed with injection.
                    log_line(log_path, f"assisting {task_id}: assignee={task.assignee} in claim_assignees={sorted(authorized_assignees)} (assist-role)")
                else:
                    log_line(log_path, f"reclaiming {task_id}: assignee={task.assignee} not in claim_assignees={sorted(authorized_assignees)} (task stealing guard)")
                    from base_listener import _reclaim_task_without_signaling_worker
                    _reclaim_task_without_signaling_worker(
                        conn2, task_id,
                        reason=f"{self.agent_slug}-interactive task stealing guard: claimed task for {task.assignee} but profile is {pane_profile}",
                    )
                    return None, None
        except Exception as exc:
            log_line(log_path, f"role guard check failed (non-fatal): {exc}")
        return result

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
