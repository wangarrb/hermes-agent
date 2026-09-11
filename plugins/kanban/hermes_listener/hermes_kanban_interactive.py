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
    zellij_submit_enter,
    tag_injected_text,
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
    "designer ❯",
)

_HERMES_BUSY_MARKERS = (
    "activity: thinking",
    "ruminating",
    "preparing terminal",
    "💻 preparing terminal",
    "preparing process",
    "preparing read_file",
    "preparing write_file",
    "⚙ wait ",
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
    semantic_delivery_required = True

    _COMPOSER_PROMPT_RE = re.compile(
        r"^\s*(?:(?:coordinator|planner|implementer|critic|reviewer|designer)\s+)?"
        r"[›❯](?:\s?(.*))?$"
    )

    def composer_input_text(self, screen: str) -> str | None:
        """Return the current Hermes composer text.

        ``""`` means a supported, visibly empty composer. ``None`` means the
        pane does not expose a recognisable Hermes composer (or is showing a
        live interrupt/status row). Only the last prompt row is considered;
        transcript text containing ``❯`` is never treated as current input.
        """
        if not screen or not screen.strip():
            return None
        lines = screen.splitlines()
        prompt_index: int | None = None
        first_text = ""
        for index in range(len(lines) - 1, -1, -1):
            match = self._COMPOSER_PROMPT_RE.match(lines[index])
            if match:
                prompt_index = index
                first_text = (match.group(1) or "").strip()
                break
        if prompt_index is None:
            return None
        # The interrupt command-hint row is rendered while the agent is live,
        # not as a user-editable composer.
        if "msg=interrupt" in lines[prompt_index].lower():
            return None

        parts = [first_text] if first_text else []
        for line in lines[prompt_index + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if self._DECORATIVE_LINE_RE.match(stripped):
                break
            lowered = stripped.lower()
            if lowered.startswith("⚕") or "context" in lowered and "│" in stripped:
                break
            if stripped.startswith(("└", "┌", "╭", "╰")) and any(
                marker in lowered for marker in ("preparing", "activity", "error")
            ):
                break
            parts.append(stripped)
        return "\n".join(parts).strip()

    def wait_for_stable_composer_input(
        self, *, session: str, pane_id: str, log_path: Path,
        initial_screen: str | None = None,
        screen_reader: object | None = None,
    ) -> bool:
        """Permit automation only at a known-empty Hermes composer boundary."""
        read_screen = screen_reader or self.read_pane_screen
        screen = initial_screen
        if screen is None:
            screen = read_screen(session=session, pane_id=pane_id, log_path=log_path)
        if not screen or not self.pane_is_idle(screen):
            return False
        return self.composer_input_text(screen) == ""

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

    def read_pane_screen(
        self, *, session: str, pane_id: str, log_path: Path,
    ) -> str | None:
        return zellij_dump_screen(
            session=session, pane_id=pane_id, log_path=log_path,
        )

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

        # Use the current live error box only.  A stale API-error sentence in
        # transcript scrollback must not trigger a retry after Hermes is idle.
        tail_lines = _tail_nonempty_lines(screen, limit=20)

        # Idle marker must be the last meaningful line (prompt/status bar at
        # bottom), not just anywhere in tail — › can appear in tool output and
        # scrollback. Hermes may render a decorative border below the prompt.
        # A bare role prompt is the only idle rendering. The
        # "⚕ ❯ msg=interrupt · ..." placeholder is emitted while
        # ``cli._agent_running`` is true and is therefore explicitly busy.
        last_line = self._last_non_decorative_line(screen)
        has_idle = self._is_truly_idle_line(last_line)
        has_busy = self._has_recent_busy_marker(screen)
        if not has_idle or has_busy:
            # Pane is busy or not showing idle prompt — reset retry state
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = None
            self._hermes_retry_key = None
            self._reset_idle_followup()
            return

        prompt_index = None
        for index in range(len(tail_lines) - 1, -1, -1):
            if self._is_truly_idle_line(tail_lines[index]):
                prompt_index = index
                break
        error_kind: str | None = None
        if prompt_index is not None:
            before_prompt = tail_lines[:prompt_index]
            # Simple API failures are rendered immediately above the prompt.
            short_region = before_prompt[-6:]
            for marker in self._HERMES_STRICT_ERROR_MARKERS:
                if any(
                    marker in line.lower()
                    and line.lstrip().startswith(
                        ("⚠", "✗", "error", "api", "╭", "│", "┌", "└")
                    )
                    for line in short_region
                ):
                    error_kind = marker
                    break
            # Structured Hermes error boxes can span many lines, but always
            # include a visible box edge in the same live suffix.
            if error_kind is None:
                wide_region = before_prompt[-20:]
                has_box_edge = any(
                    line.lstrip().startswith(("┌", "╭", "╰", "└", "│", "┃"))
                    for line in wide_region
                )
                if has_box_edge:
                    for marker in self._HERMES_STRICT_ERROR_MARKERS:
                        if any(marker in line.lower() for line in wide_region):
                            error_kind = marker
                            break

        if error_kind is None:
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = None
            self._handle_idle_task_followup(args, conn, task_id, log_path)
            return

        retry_key = (str(task_id), error_kind)
        if getattr(self, "_hermes_retry_key", None) != retry_key:
            self._hermes_retry_key = retry_key
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = error_kind

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

        if not self.wait_for_stable_composer_input(
            session=zellij_session,
            pane_id=zellij_pane_id,
            log_path=log_path,
            initial_screen=screen,
        ):
            return
        self._api_retry_count += 1
        self._api_retry_first_at = None
        log_line(log_path, f"api-error-retry {self._api_retry_count}/{self.API_RETRY_MAX} for task {task_id}: injecting 继续 after {elapsed:.0f}s")
        zellij_inject(
            session=zellij_session,
            pane_id=zellij_pane_id,
            text=tag_injected_text("继续", source_profile="watcher"),
            expected_pane_prefix="hermes-kanban",
            log_path=log_path,
        )
        time.sleep(0.5)
        zellij_submit_enter(
            session=zellij_session, pane_id=zellij_pane_id,
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=f"task:{task_id}:api:{error_kind}", log_path=log_path,
        )

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

    # Only a bare prompt is idle. The command-hint placeholder contains a
    # prompt symbol too, but cli.py renders it only while ``_agent_running`` is
    # true. Arbitrary text after the prompt likewise remains busy.
    # Matches: "❯ ", "planner ❯ "
    # Does NOT match: "❯ some text", "❯/steer 记住…",
    #                 "⚕ ❯ msg=interrupt · ..."
    _IDLE_ONLY_RE = re.compile(
        r'^(?:(?:coordinator|planner|implementer|critic|reviewer|designer)\s*)?'
        r'[›❯]\s*$'
    )

    def _is_truly_idle_line(self, line: str) -> bool:
        """Return True only for a known idle prompt/status rendering."""
        return bool(self._IDLE_ONLY_RE.match(line.strip()))

    def _last_non_decorative_line(self, screen: str) -> str:
        """Return the final meaningful pane line, ignoring Hermes borders."""
        for line in reversed(screen.splitlines()):
            stripped = line.strip()
            if stripped and not self._DECORATIVE_LINE_RE.match(stripped):
                return stripped
        return ""

    def _has_recent_busy_marker(self, screen: str) -> bool:
        """Return whether current activity is visible near the pane bottom."""
        recent_tail = "\n".join(_tail_nonempty_lines(screen, limit=5)).lower()
        return any(marker.lower() in recent_tail for marker in self.busy_markers)

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
            last_line = self._last_non_decorative_line(screen)
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
            has_busy = self._has_recent_busy_marker(screen)
            if has_busy:
                log_line(log_path, f"on_claim_pre_check attempt {attempt+1}/2: busy marker detected, NOT idle")
                return False
            composer = self.composer_input_text(screen)
            if composer is None:
                log_line(log_path, f"on_claim_pre_check attempt {attempt+1}/2: composer unknown")
                return False
            if composer:
                log_line(log_path, "on_claim_pre_check: composer has draft text")
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
        """Close the race between claim pre-check and prompt injection."""
        del task_id
        session = str(getattr(args, "zellij_session", "") or "")
        pane_id = str(getattr(args, "zellij_pane_id", "") or "")
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(session=session, pane_id=pane_id, log_path=log_path)
        if not screen or not self.pane_is_idle(screen):
            return False
        composer = self.composer_input_text(screen)
        if composer != "":
            if composer:
                log_line(log_path, "on_claim_post_confirm: composer has draft text")
            else:
                log_line(log_path, "on_claim_post_confirm: composer unknown")
            return False
        return True

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

    _POST_INJECT_CONFIRM_S = 1.5
    _POST_INJECT_MAX_RETRIES = 3

    def on_post_inject(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
        injected_marker: str | None = None,
        pre_write_composer: str | None = None,
        correlation: str | None = None,
    ) -> str:
        """Confirm that an injected payload left Hermes' current composer."""
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

        def marker_present(composer: str | None) -> bool:
            if composer is None:
                return False
            return " ".join(marker.split()) in " ".join(composer.split())

        def live_busy(screen: str) -> bool:
            lines = _tail_nonempty_lines(screen, limit=5)
            for index, line in enumerate(lines):
                lower = line.lower()
                if not any(item.lower() in lower for item in self.busy_markers):
                    continue
                # Busy rows above a currently editable prompt are stale output.
                if any(self._is_truly_idle_line(tail) for tail in lines[index + 1 :]):
                    continue
                return True
            return False

        for attempt in range(1, self._POST_INJECT_MAX_RETRIES + 1):
            time.sleep(self._POST_INJECT_CONFIRM_S)
            try:
                screen = zellij_dump_screen(
                    session=zellij_session,
                    pane_id=zellij_pane_id,
                    log_path=log_path,
                )
            except Exception:
                return "unknown"
            if not screen:
                return "unknown"
            if live_busy(screen):
                return "confirmed"
            composer = self.composer_input_text(screen)
            if composer is None:
                return "unknown"
            if marker_present(composer):
                saw_marker = True
                zellij_submit_enter(
                    session=zellij_session,
                    pane_id=zellij_pane_id,
                    expected_pane_prefix=self.expected_pane_prefix(),
                    correlation=submit_correlation,
                    log_path=log_path,
                )
                log_line(
                    log_path,
                    f"hermes post-inject: queued prompt remained; sent Enter "
                    f"(attempt {attempt}/{self._POST_INJECT_MAX_RETRIES})",
                )
                continue
            if saw_marker:
                return "confirmed"
            return "unknown"
        return "known_unsubmitted" if saw_marker else "unknown"

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
