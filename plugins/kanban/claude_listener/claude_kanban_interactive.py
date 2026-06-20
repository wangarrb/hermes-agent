#!/usr/bin/env python3
"""ClaudeInteractiveListener — Hermes Kanban listener for Claude Code CLI.

Claude Code (claude) is an interactive TUI agent by Anthropic.
Key characteristics:
  - CLI: `claude --continue --dangerously-skip-permissions --append-system-prompt <file>`
  - Sessions stored in ~/.claude/projects/<encoded-cwd>/
  - Idle markers: TBD (need dump-screen testing on Claude Code TUI)
  - Injection: write prompt to .md file, inject single-line command

Claude Code uses prompt_toolkit, so LF in injected text should be
handled correctly (like Reasonix).  However, to be safe and consistent
with the "no \\n in inject_text" principle, we use single-line text.

NOTE: Idle/busy markers for Claude Code are preliminary and may need
adjustment after real-world testing.  The dump-screen output format
depends on Claude Code's terminal rendering.
"""
from __future__ import annotations

import argparse
import base64
import os
import subprocess
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
    _noop_signal,
    _pane_can_accept_new_kanban_task,
    _tail_nonempty_lines,
    reclaim_orphaned_running_task,
)

# ── Claude-specific imports ──
HERMES_REPO = HERMES_AGENT_ROOT
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402


# ──────────────────────────────────────────────
# Claude-specific constants
# ──────────────────────────────────────────────

# Preliminary idle/busy markers for Claude Code TUI.
# These may need adjustment after real-world testing.
_CLAUDE_IDLE_MARKERS = (
    ">",  # Claude Code prompt prefix
    "claude>",  # Alternative prompt prefix
    "? for shortcuts",  # Help hint in status bar
    "/help",  # Command hint
)
_CLAUDE_BUSY_MARKERS = (
    "thinking",
    "running",
    "tool:",
    "reading",
    "writing",
    "editing",
)
_CLAUDE_QUEUED_INPUT_MARKERS = ()


# ──────────────────────────────────────────────
# Claude subclass
# ──────────────────────────────────────────────

class ClaudeInteractiveListener(BaseInteractiveListener):
    agent_name = "Claude"
    agent_slug = "claude"

    # Idle markers are preliminary — may need adjustment
    idle_markers = _CLAUDE_IDLE_MARKERS
    busy_markers = _CLAUDE_BUSY_MARKERS
    queued_input_markers = _CLAUDE_QUEUED_INPUT_MARKERS

    # ── Build TUI command ──
    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build Claude Code CLI command.

        Claude Code CLI (v2.1.179+):
          claude [--continue] [--dangerously-skip-permissions]
                 [--append-system-prompt <file>] [--model <id>]
                 [--session-id <id>]
        """
        cmd = ["claude"]
        if continue_session:
            cmd.append("--continue")
        # Kanban mode: auto-approve all tool calls
        cmd.append("--dangerously-skip-permissions")
        # Model selection
        if model:
            cmd.extend(["--model", model])
        # Extra args
        cmd.extend(extra_args or [])
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check if Claude Code has saved sessions for this workspace.

        Claude Code stores sessions in ~/.claude/projects/<encoded-cwd>/.
        The encoding is a URL-safe variant of the workspace path.
        """
        claude_dir = Path.home() / ".claude" / "projects"
        if not claude_dir.is_dir():
            return False
        # Try to find a project directory matching this workspace
        # Claude Code encodes the cwd path in the directory name
        try:
            for project_dir in claude_dir.iterdir():
                if not project_dir.is_dir():
                    continue
                # Check if any session files exist
                session_files = list(project_dir.glob("*.json"))
                if session_files:
                    return True
        except Exception:
            pass
        return False

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build single-line injection for Claude Code.

        Claude Code accepts freeform text at its prompt.
        We tell it to read the task prompt file and execute.
        No \\n in injected text (safe for all TUI types).
        """
        return (
            f"请读取 {prompt_path} 中的 Kanban 任务并执行。"
            f" [任务 {task_id}: {title}]"
        )

    def pane_label(self, task_id: str | None = None) -> str:
        if task_id:
            return f"claude-kanban [{task_id}]"
        return "claude-kanban"

    # ── Override: build_launch_env ──
    def build_launch_env(self, args: argparse.Namespace) -> dict[str, str]:
        env = super().build_launch_env(args)
        # Claude Code uses ANTHROPIC_AUTH_TOKEN (not API_KEY) for custom endpoints
        # and BASE_URL for provider routing.
        # These are typically set in ~/.claude/settings.json env block,
        # but we can also pass them via environment.
        claude_settings_env = self._read_claude_settings_env()
        for key, value in claude_settings_env.items():
            if key not in env or not env[key]:
                env[key] = value
        return env

    def _read_claude_settings_env(self) -> dict[str, str]:
        """Read env vars from ~/.claude/settings.json if present."""
        settings_path = Path.home() / ".claude" / "settings.json"
        if not settings_path.is_file():
            return {}
        try:
            import json
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            return data.get("env", {}) or {}
        except Exception:
            return {}

    # ── Override: launcher_main with Claude-specific setup ──
    def launcher_main(self, args: argparse.Namespace) -> int:
        self._init_from_args(args)
        board = self._board
        workspace = self._workspace
        log_path = self._log_path

        if not workspace.exists():
            print(f"错误: workspace 不存在: {workspace}", file=sys.stderr)
            return 2

        zellij_session = getattr(args, "zellij_session", "") or os.environ.get("ZELLIJ_SESSION_NAME")
        zellij_pane_id = getattr(args, "zellij_pane_id", "") or os.environ.get("ZELLIJ_PANE_ID")
        if not zellij_session or not zellij_pane_id:
            print(f"错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 Claude Code TUI。", file=sys.stderr)
            return 2

        args.zellij_session = zellij_session
        args.zellij_pane_id = zellij_pane_id

        # Write system prompt file for --append-system-prompt
        prompt_d = prompt_dir(workspace, board, args.profile, agent_slug="claude")
        prompt_d.mkdir(parents=True, exist_ok=True)
        sys_prompt_path = prompt_d / "kanban-system-prompt.md"
        sys_prompt_path.write_text(
            role_guidance(args.profile) + "\n\n"
            + "Kanban 任务会以文件路径的形式注入到你的输入框。"
            + "请读取指定文件中的任务描述并执行。"
            + "完成后运行 `hermes kanban --board " + board + " complete <task_id> --summary \"...\"`，"
            + "阻塞时运行 `hermes kanban --board " + board + " block <task_id> --reason \"...\"`。\n",
            encoding="utf-8",
        )

        # Build watcher command
        watcher_cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--watch-child",
            "--profile", args.profile,
            "--claim-assignees", ",".join(claim_assignees(args)),
            "--board", board,
            "--workspace", str(workspace),
            "--ttl", str(args.ttl),
            "--zellij-session", zellij_session,
            "--zellij-pane-id", zellij_pane_id,
            "--startup-delay-s", str(getattr(args, "startup_delay_s", 0) or 0),
            "--assist-claim-delay-s", str(getattr(args, "assist_claim_delay_s", 0.0)),
        ]
        if args.poll is not None:
            watcher_cmd.extend(["--poll", str(args.poll)])

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

        print(f"Claude Code interactive kanban mode")
        print(f"  board:     {board}")
        print(f"  profile:   {args.profile}")
        print(f"  claims:    {', '.join(claim_assignees(args))}")
        print(f"  workspace: {workspace}")
        print(f"  pane:      {zellij_session}:{zellij_pane_id}")
        print(f"  log:       {log_path}")
        print("")
        print(f"按 Enter 进入 interactive Claude Code；后台 listener 会按优先级 claim ready 任务并注入到当前 TUI。")
        print(f"claude-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

        watch_only = getattr(args, "watch_only", False)
        if watch_only:
            print("listener-only 模式：不会启动 Claude Code，只运行后台 listener 并向指定 Zellij pane 注入任务。")
            return self.watcher_main(args)

        auto_start = getattr(args, "auto_start", False)
        if not auto_start:
            try:
                input()
            except EOFError:
                pass

        env = self.build_launch_env(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")
        log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")

        import subprocess
        watcher = subprocess.Popen(
            watcher_cmd,
            stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
            text=True, env=env, start_new_session=True,
        )
        print(f"claude-kanban listener started: pid={watcher.pid} profile={args.profile} board={board} poll={poll_label}", flush=True)

        # Build Claude command with system prompt
        continue_session = getattr(args, "continue_session", True) and self.has_saved_sessions(workspace)
        claude_cmd = self.build_tui_cmd(
            workspace, continue_session=continue_session,
            model=getattr(args, "model", None),
            extra_args=(getattr(args, "claude_arg", None) or [])
            + ["--append-system-prompt", str(sys_prompt_path)],
        )

        log_line(log_path, f"launcher starting claude: {' '.join(claude_cmd)}")
        rc = 0
        try:
            rc = subprocess.call(claude_cmd, cwd=str(workspace), env=env)
        finally:
            log_line(log_path, f"claude exited rc={rc}; stopping watcher pid={watcher.pid}")
            try:
                watcher.terminate()
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.kill()
            log_f.close()
        return rc

    # ── Override: _build_parser with Claude-specific args ──
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = super()._build_parser()
        parser.add_argument("--continue", dest="continue_session", action="store_true", default=True, help="Resume most recent session (default: True)")
        parser.add_argument("--no-continue", dest="continue_session", action="store_false", help="Start a new session")
        return parser


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = ClaudeInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
