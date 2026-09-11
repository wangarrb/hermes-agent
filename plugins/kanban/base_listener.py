#!/usr/bin/env python3
"""Base class for all Hermes Kanban interactive listeners.

All agent-backed listeners (Codex, CodeWhale, Claude, Reasonix) share the
same poll/claim/inject lifecycle.  This module provides BaseInteractiveListener
which implements the complete watcher+launcher architecture; each subclass
only provides agent-specific methods:

  - agent_name / agent_slug (identity)
  - build_tui_cmd (how to launch the TUI)
  - has_saved_sessions (session detection)
  - inject_text (what text to inject into the pane)
  - pane_label (pane title formatting)

Optional hooks for further customization:

  - on_claim_pre_check / on_claim_post_confirm
  - on_task_running_monitor / on_watcher_loop_idle
  - build_launch_env / build_watcher_extra_args
  - idle_markers / busy_markers / queued_input_markers

IMPORTANT: inject_text must NEVER contain ``\\n`` (LF, 0x0A).
In PTY raw mode, LF is NOT the same as Enter (CR, 0x0D).
LF inserts a newline in the input buffer but does NOT submit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Source layout: <repo>/plugins/kanban/base_listener.py
HERMES_REPO = Path(__file__).resolve().parents[1]
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402

try:  # Package import in tests and installed Hermes entry points.
    from .role_context import (  # type: ignore[import-not-found]
        DEFAULT_PROFILES_ROOT,
        DEFAULT_SHARED_SKILLS_ROOT,
        render_effective_role_context as _render_effective_role_context,
    )
except ImportError:  # Direct listener scripts put plugins/kanban on sys.path.
    from role_context import (  # type: ignore[no-redef]
        DEFAULT_PROFILES_ROOT,
        DEFAULT_SHARED_SKILLS_ROOT,
        render_effective_role_context as _render_effective_role_context,
    )


# ──────────────────────────────────────────────
# Exceptions / globals
# ──────────────────────────────────────────────

_STOP = False

_REVIEWER_CHECKPOINT_PENDING_RE = re.compile(
    r"\bREVIEWER_CHECKPOINT_PENDING\s+(t_[0-9A-Za-z]+)\b"
)
_REVIEWER_CHECKPOINT_OPEN_STATUSES = {
    "triage", "todo", "scheduled", "ready", "running", "review",
}


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def stop_requested() -> bool:
    """Return the shared watcher stop flag for listener-specific loops."""
    return _STOP


def _result_notifications_enabled() -> bool:
    raw = os.environ.get("HERMES_KANBAN_RESULT_NOTIFICATIONS", "1")
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _has_open_reviewer_checkpoint(conn: Any, origin_task_id: str) -> bool:
    """Return whether the origin goal explicitly waits on an open review card."""
    for comment in reversed(kb.list_comments(conn, origin_task_id)):
        for reviewer_task_id in _REVIEWER_CHECKPOINT_PENDING_RE.findall(comment.body):
            task = kb.get_task(conn, reviewer_task_id)
            if (
                task is not None
                and task.assignee == "reviewer"
                and task.status in _REVIEWER_CHECKPOINT_OPEN_STATUSES
            ):
                return True
    return False


# ──────────────────────────────────────────────
# Shared utility functions (used by all listeners)
# ──────────────────────────────────────────────

def now_s() -> int:
    return int(time.time())


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_line(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_label()}] {msg}\n")


def task_log_path(task_id: str, board: str | None) -> Path:
    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{task_id}.log"


def _noop_signal(_pid: int, _sig: int) -> None:
    pass


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return False


def _task_status(conn: Any, task_id: str) -> tuple[str | None, int | None]:
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if not row:
        return None, None
    return row["status"], row["current_run_id"]


def _task_claim_state(
    conn: Any, task_id: str,
) -> tuple[str | None, int | None, int | None, str | None]:
    row = conn.execute(
        "SELECT status, current_run_id, generation, claim_lock "
        "FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    if not row:
        return None, None, None, None
    return (
        row["status"],
        row["current_run_id"],
        row["generation"],
        row["claim_lock"],
    )


def _workspace_matches(row_workspace: str | None, workspace: Path) -> bool:
    if not row_workspace:
        return False
    return str(workspace) == row_workspace


def _skip_reclaim_signal(pid: int, signum: int) -> None:  # noqa: ARG001
    pass


def _reclaim_task_without_signaling_worker(
    conn: Any,
    task_id: str,
    *,
    reason: str,
    expected_run_id: int | None = None,
    expected_generation: int | None = None,
    expected_claim_lock: str | None = None,
) -> bool:
    """Reclaim a task without sending SIGTERM to its worker process."""
    try:
        return kb.reclaim_task(
            conn,
            task_id,
            reason=reason,
            signal_fn=_skip_reclaim_signal,
            expected_run_id=expected_run_id,
            expected_generation=expected_generation,
            expected_claim_lock=expected_claim_lock,
        )
    except Exception:
        return False


def _cleanup_active_claim(
    *,
    board: str,
    task_id: str | None,
    expected_run_id: int | None = None,
    expected_generation: int | None = None,
    expected_claim_lock: str | None = None,
    run_id: int | None = None,
    log_path: Path,
) -> None:
    """On watcher exit, if a task is still running, reclaim it."""
    if not task_id:
        return
    claim_run_id = expected_run_id if expected_run_id is not None else run_id
    try:
        with kb.connect(board=board) as conn:
            status, _ = _task_status(conn, task_id)
            if status == "running":
                reclaimed = kb.reclaim_task(
                    conn, task_id,
                    reason="watcher exited while task still running",
                    signal_fn=_skip_reclaim_signal,
                    expected_run_id=claim_run_id,
                    expected_generation=expected_generation,
                    expected_claim_lock=expected_claim_lock,
                )
                if reclaimed:
                    log_line(log_path, f"reclaimed {task_id} on watcher exit")
                else:
                    log_line(log_path, f"skipped stale reclaim for {task_id} on watcher exit")
    except Exception as exc:
        log_line(log_path, f"cleanup reclaim failed: {exc}")


# ──────────────────────────────────────────────
# Shared role guidance
# ──────────────────────────────────────────────

def role_guidance(profile: str) -> str:
    """Role-bound guidance shared by all agent backends."""
    p = (profile or "").strip().lower()
    common = "职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用不同 agent 运行，也要按当前角色工作。"
    owner = (
        "你是连续执行 owner：负责方案、计划、实现、证据、任务流和目标闭环。"
        "耗时或可并行工作优先使用自己的后台子 agent，并保留集成责任；implementer 卡不是默认选择，"
        "但后台子 agent 不适用、合同已冻结且 Kanban 持久化收益更高时可以发布。"
        "收到显式跨项目切换请求时，先清空后台工作，再调用 "
        "hermes-kanban-switch-owner-project；普通技术决策自行完成，承重方向交 reviewer。"
    )
    per_role = {
        "coordinator": owner,
        "planner": owner,
        "designer": owner,
        "reviewer": "你是 reviewer：既能独立制定计划，也能审核 planner 的计划，与 planner 多轮协商直到敲定最终计划。你的职责不是找 planner 的纰漏，而是从全局角度把控计划的方向、范围和内容是否合理、完整、有效。具体来说：(1) 方向——计划是否在解决正确的问题？是否与项目目标对齐？有没有偏离核心目标做无关优化？(2) 范围——计划的边界是否清晰？哪些该做哪些不该做？有没有遗漏的关键路径或不需要的过度设计？(3) 内容——方案是否完整覆盖目标？假设是否成立？验收标准是否可测试无歧义？依赖和风险是否充分识别？有无更简单可靠的替代方案？工作流程：(a) 收到 planner 的计划后，先独立思考同一目标你会怎么做——形成自己的计划草案；(b) 从全局视角对比两份计划，找出方向偏差、范围遗漏、内容缺陷；(c) 通过 kanban comment 反馈你的审核意见、独立方案和修改建议；(d) planner 根据你的反馈修改计划后，再次审核——可能需要多轮协商才能达成一致；(e) 双方认可后，最终计划交给 implementer 执行。不要为了结束协商而妥协——真正有分歧的点必须充分讨论清楚。",
        "implementer": (
            "你是 implementer：主要协助 reviewer 完成合同已冻结的确定性 diff/测试/artifact 盘点、复现或小修；"
            "也可接受 owner 的例外委派，但它不是 owner 耗时工作的默认路径。"
            "正式成功率、算法方向、路线重置、审核结论和最终 handback 只由 reviewer 决定。"
            "可写任务必须显式给出绝对 workspace、branch、base SHA、write set 和 commit ownership；"
            "缺失时只做只读证据工作。"
        ),
        "critic": "你是 critic：负责审查、找漏洞和独立验证。不要默认相信 planner/implementer 结论；重点检查证据链、遗漏风险、指标口径和可复现性。",
    }
    return common + "\n" + per_role.get(p, f"你当前角色是 {profile}：按该 assignee 的职责完成任务。")


# ──────────────────────────────────────────────
# Shared prompt builders
# ──────────────────────────────────────────────

def build_interactive_prompt(
    *, agent_name: str, board: str, profile: str, task_id: str,
    task_assignee: str, task_title: str, context: str, workspace: Path,
    run_id: int | None = None, generation: int = 1,
) -> str:
    """Build the full task prompt (written to a file, not injected directly)."""
    assist_note = ""
    if task_assignee != profile:
        assist_note = f"（注意：当前 profile={profile}，但本任务 assignee={task_assignee}，按 {task_assignee} 职责执行）"
    run_fence = f" --run-id {run_id}" if run_id is not None else ""
    generation_fence = f" --generation {int(generation)}"
    task_role_section = ""
    if task_assignee != profile:
        task_role_section = "当前任务角色说明：\n" + role_guidance(task_assignee)
    prompt = textwrap.dedent(
        f"""
        ─── Kanban 任务 {task_id} ───
        标题：{task_title}
        角色：{task_assignee} {assist_note}

        关键规则：
        1. 完成后调用 `hermes kanban --board {board} complete {task_id}{run_fence}{generation_fence} --summary "..."`
        2. 阻塞时调用 `hermes kanban --board {board} block {task_id} "..."{run_fence}{generation_fence}`
        3. 默认中文；路径/命令保留英文

        执行边界：task_id={task_id} run_id={run_id if run_id is not None else 'none'} generation={int(generation)}
        HERMES_KANBAN_TASK={task_id} HERMES_KANBAN_RUN_ID={run_id if run_id is not None else ''} HERMES_KANBAN_GENERATION={int(generation)}

        __TASK_ROLE_SECTION__

        上下文：
        {context}

        ─── 开始执行任务 {task_id} ───
        """
    ).strip()
    return prompt.replace("__TASK_ROLE_SECTION__", task_role_section)


def prompt_dir(workspace: Path, board: str, pane_profile: str, *, agent_slug: str) -> Path:
    safe_board = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in board)
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pane_profile)
    return workspace / f".{agent_slug}-kanban" / safe_board / safe_profile


def write_task_prompt(
    *, agent_name: str, agent_slug: str, board: str, profile: str,
    task_id: str, task_assignee: str, task_title: str, context: str,
    workspace: Path, run_id: int | None = None, generation: int = 1,
) -> Path:
    d = prompt_dir(workspace, board, profile, agent_slug=agent_slug)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"task-{task_id}.md"
    p.write_text(
        build_interactive_prompt(
            agent_name=agent_name, board=board, profile=profile,
            task_id=task_id, task_assignee=task_assignee,
            task_title=task_title, context=context, workspace=workspace,
            run_id=run_id, generation=generation,
        ),
        encoding="utf-8",
    )
    return p


# ──────────────────────────────────────────────
# Shared zellij helpers
# ──────────────────────────────────────────────

_INJECTION_SOURCE_PROFILE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def tag_injected_text(text: str, *, source_profile: str) -> str:
    """Append a visible provenance marker to injected prompt text.

    Raw terminal control bytes and TUI slash commands do not use this helper.
    ``source_profile`` is deliberately generic so future role-authorized pane
    injections can use ``[by planner]``, ``[by reviewer]``, and so on.
    """
    source = str(source_profile or "").strip().lower()
    if not _INJECTION_SOURCE_PROFILE_RE.fullmatch(source):
        raise ValueError("source_profile must contain only letters, digits, '.', '_' or '-'")
    marker = f"[by {source}]"
    payload = str(text).rstrip()
    if payload.endswith(marker):
        return payload
    return f"{payload} {marker}" if payload else marker


def _zellij_cmd_base(session: str) -> list[str]:
    return ["zellij", "--session", session, "action"] if session else ["zellij", "action"]


def _zellij_pane_title_matches(title: str, expected_prefix: str) -> bool:
    """Match a role title without accepting a similarly named foreign pane."""
    title = str(title or "").strip().casefold()
    prefix = str(expected_prefix or "").strip().casefold()
    if not prefix or not title.startswith(prefix):
        return False
    if len(title) == len(prefix):
        return True
    return title[len(prefix)] == "[" or title[len(prefix)].isspace()


def _zellij_validate_pane(
    *, session: str, pane_id: str, expected_pane_prefix: str | None,
    log_path: Path,
) -> bool:
    """Confirm the pane still exists and belongs to the expected role.

    Zellij 0.44.1 exits successfully for unknown pane IDs, so action return
    codes cannot be used as an existence check.  ``list-panes --all --json``
    is authoritative and lets us fail closed before writing into a reassigned
    pane.  Generic callers may omit ``expected_pane_prefix`` and only require
    a live non-plugin pane with the requested ID.
    """
    try:
        result = subprocess.run(
            _zellij_cmd_base(session) + ["list-panes", "--all", "--json"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        if result is None:
            raise ValueError("list-panes returned no result")
        payload = json.loads(getattr(result, "stdout", "") or "[]")
        if not isinstance(payload, list):
            raise ValueError("list-panes JSON must be a list")
        for pane in payload:
            if not isinstance(pane, dict):
                continue
            if str(pane.get("pane_id")) != str(pane_id):
                continue
            if pane.get("is_plugin") is True or pane.get("exited") is True:
                continue
            title_valid = expected_pane_prefix is None or _zellij_pane_title_matches(
                str(pane.get("title", "")), expected_pane_prefix,
            )
            if not title_valid:
                continue
            if expected_pane_prefix is not None:
                command = str(pane.get("terminal_command") or pane.get("pane_command") or "").casefold()
                if command and "<defunct>" not in command and "conda" not in command:
                    prefix_tokens = expected_pane_prefix.casefold().split("-")
                    role = prefix_tokens[0]
                    backend = {"codex": "codex", "hermes": "hermes", "claude": "claude", "codewhale": "codewhale", "deepseek": "deepseek", "reasonix": "reasonix"}.get(role, prefix_tokens[-1])
                    if role == "implementer" and "deepseek" in prefix_tokens:
                        backend = "deepseek"
                    if role == "implementer" and "reasonix" in prefix_tokens:
                        backend = "reasonix"
                    known_roles = {"codex", "hermes", "claude", "codewhale", "deepseek", "reasonix"}
                    tokens = set(re.findall(r"[a-z0-9_-]+", command))
                    foreign = (tokens & known_roles) - {role, backend}
                    backend_tokens = {backend}
                    if backend == "deepseek":
                        backend_tokens.add("codewhale")
                    if foreign or not (backend_tokens & tokens):
                        continue
            return True
        log_line(log_path, f"event=transport_rejected pane={pane_id!r} prefix={expected_pane_prefix!r}")
        return False
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        log_line(log_path, f"event=transport_rejected pane={pane_id!r} prefix={expected_pane_prefix!r} error={str(detail).strip()}")
        return False


def zellij_write_text(
    *, session: str, pane_id: str, text: str, log_path: Path,
    expected_pane_prefix: str | None = None, correlation: str | None = None,
) -> bool:
    """Write one text payload to a Zellij pane; submission is separate."""
    allow_lf = str(expected_pane_prefix or "").casefold().endswith("-reasonix")
    if any((ord(ch) < 0x20 and not (allow_lf and ch == "\n")) or ord(ch) == 0x7F for ch in str(text)):
        log_line(log_path, f"event=transport_rejected pane={pane_id!r} prefix={expected_pane_prefix!r} correlation={correlation!r} reason=control_byte")
        return False
    if not _zellij_validate_pane(
        session=session, pane_id=pane_id,
        expected_pane_prefix=expected_pane_prefix, log_path=log_path,
    ):
        return False
    try:
        subprocess.run(
            _zellij_cmd_base(session) + ["write-chars", "-p", str(pane_id), str(text)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        log_line(log_path, f"event=transport_accepted pane={pane_id!r} prefix={expected_pane_prefix!r} correlation={correlation!r}")
        return True
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        rc = getattr(exc, "returncode", "?")
        log_line(log_path, f"event=transport_failed pane={pane_id!r} prefix={expected_pane_prefix!r} rc={rc} error={str(detail).strip()}")
        return False


def zellij_inject(
    *, session: str, pane_id: str, text: str, log_path: Path,
    expected_pane_prefix: str | None = None, correlation: str | None = None,
) -> bool:
    """Compatibility name for writing one text payload (without Enter).

    IMPORTANT: *text* must NOT contain ``\\n`` (LF) characters.
    In PTY raw mode, LF (0x0A) is NOT the same as Enter (CR, 0x0D).
    LF inserts a newline in the input buffer but does NOT submit the
    prompt, causing the "typed but not sent" bug.
    Use single-line text only; the TUI will word-wrap.
    """
    return zellij_write_text(
        session=session, pane_id=pane_id, text=text, log_path=log_path,
        expected_pane_prefix=expected_pane_prefix,
        correlation=correlation,
    )


def zellij_submit(
    *, session: str, pane_id: str, log_path: Path,
    expected_pane_prefix: str | None = None, correlation: str | None = None,
) -> bool:
    """Send exactly one raw carriage return (byte 13) to a live pane."""
    if not _zellij_validate_pane(
        session=session, pane_id=pane_id,
        expected_pane_prefix=expected_pane_prefix, log_path=log_path,
    ):
        return False
    try:
        subprocess.run(
            _zellij_cmd_base(session) + ["write", "-p", str(pane_id), "13"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        log_line(log_path, f"event=submit_sent pane={pane_id!r} prefix={expected_pane_prefix!r} correlation={correlation!r}")
        return True
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        rc = getattr(exc, "returncode", "?")
        log_line(log_path, f"event=submit_failed pane={pane_id!r} prefix={expected_pane_prefix!r} rc={rc} error={str(detail).strip()}")
        return False


# Descriptive public alias used by listener integrations and tests.
zellij_submit_enter = zellij_submit


def zellij_rename_pane(*, session: str, pane_id: str, name: str, log_path: Path) -> bool:
    try:
        cmd_base = ["zellij", "--session", session, "action"] if session else ["zellij", "action"]
        subprocess.run(
            cmd_base + ["rename-pane", "-p", str(pane_id), name],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=5,
        )
        return True
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        log_line(log_path, f"zellij rename-pane failed rc={getattr(exc, 'returncode', '?')}: {str(detail).strip()}")
        return False


def zellij_dump_screen(*, session: str, pane_id: str, log_path: Path) -> str | None:
    """Dump the current screen content of a Zellij pane."""
    try:
        cmd_base = ["zellij", "--session", session, "action"] if session else ["zellij", "action"]
        result = subprocess.run(
            cmd_base + ["dump-screen", "-p", str(pane_id), "--full"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout or ""
    except Exception as exc:
        log_line(log_path, f"zellij dump-screen failed: {exc}")
        return None


# ──────────────────────────────────────────────
# Shared claim / delay / reset helpers
# ──────────────────────────────────────────────

def claim_assignees(args: argparse.Namespace, *, default_profile: str = "implementer") -> list[str]:
    return worker_runtime.claim_assignees_from_args(args, default_profile=default_profile)


def reset_kanban_claims(
    *, board: str, profile: str, claim_assignees_list: list[str],
    workspace: Path, listener_kind: str, reason: str = "operator reset-kanban",
) -> list[str]:
    reset_ids = worker_runtime.reset_interactive_claims(
        board=board, profile=profile,
        claim_assignees=claim_assignees_list,
        workspace=workspace, listener_kind=listener_kind, reason=reason,
    )
    return list(dict.fromkeys(reset_ids))


def assist_claim_delay_s(args: argparse.Namespace, *, default_profile: str = "implementer") -> float:
    return worker_runtime.claim_policy_from_args(args, default_profile=default_profile).assist_claim_delay_s


def _delay_specs(raw: Any) -> list[str]:
    return worker_runtime.split_csv_values(raw)


def assist_claim_delays(args: argparse.Namespace) -> dict[str, float]:
    return worker_runtime.assist_claim_delays_from_args(args)


def assist_claim_delay_for(args: argparse.Namespace, assignee: str, *, default_profile: str = "implementer") -> float:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.assist_claim_delay_for(policy, assignee)


def _ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    return worker_runtime.ready_since(conn, task_id, fallback_created_at)


def _assist_candidate_ready(conn, args: argparse.Namespace, task: kb.Task,
                             assignee: str, *, default_profile: str = "implementer") -> bool:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee)


def _select_ready_candidate(conn, args: argparse.Namespace, *, default_profile: str = "implementer") -> kb.Task | None:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    for assignee in policy.claim_assignees:
        ready = kb.list_tasks(
            conn,
            assignee=assignee,
            status="ready",
            limit=worker_runtime.listener_policy.READY_TASK_SCAN_LIMIT,
        )
        for task in ready:
            if not kb.role_policy_candidate_allowed(
                conn, task, actor_role=policy.profile,
            ):
                continue
            if worker_runtime.assist_candidate_ready(
                conn, policy=policy, task=task, assignee=assignee,
            ):
                return task
    return None


# ──────────────────────────────────────────────
# Shared result sanitization
# ──────────────────────────────────────────────

def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "done").strip().lower()
    if status not in {"done", "blocked"}:
        status = "done"
    summary = str(result.get("summary") or "").strip()
    if not summary:
        summary = "Agent completed without summary" if status != "blocked" else "Agent blocked without summary"
    details = str(result.get("details") or "").strip()
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    block_reason = str(result.get("block_reason") or "").strip() if status == "blocked" else ""
    return {
        "status": status, "summary": summary, "details": details,
        "metadata": metadata,
        **({"block_reason": block_reason} if block_reason else {}),
    }


def _result_failure_text(result: dict[str, Any]) -> str:
    return str(result.get("block_reason") or result.get("summary") or "unknown failure")


def _is_provider_failure_result(rc: int, result: dict[str, Any]) -> bool:
    if rc == 0:
        return False
    text = _result_failure_text(result).lower()
    keywords = ["rate limit", "api key", "429", "503", "quota", "provider",
                "authentication", "unauthorized", "overloaded"]
    return any(kw in text for kw in keywords)


# ──────────────────────────────────────────────
# Shared tail / idle detection helpers
# ──────────────────────────────────────────────

def _tail_nonempty_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-limit:] if len(lines) > limit else lines


def _looks_like_idle_pane(
    text: str,
    idle_markers: tuple[str, ...],
    busy_markers: tuple[str, ...],
) -> bool:
    """Check if a pane screen looks idle, using configurable markers."""
    if not idle_markers:
        return True  # no markers configured → assume idle (trust heartbeat)
    tail = "\n".join(_tail_nonempty_lines(text, limit=40)).lower()
    has_idle = any(marker in tail for marker in idle_markers)
    has_busy = any(marker in tail for marker in busy_markers)
    if has_busy and not has_idle:
        return False
    if has_idle and not has_busy:
        return True
    # Ghost state: both idle and busy markers visible
    return has_idle


def _pane_can_accept_new_kanban_task(
    text: str,
    idle_markers: tuple[str, ...],
    busy_markers: tuple[str, ...],
    queued_input_markers: tuple[str, ...],
) -> bool:
    """Return True when it is safe to inject a new Kanban prompt.

    Checks that the LAST non-empty line contains an idle marker (prompt).
    This is stricter than checking anywhere in the tail — prompt characters
    like › can appear in tool output, error messages, or scrollback; the
    pane is only truly idle when the prompt sits at the very bottom.
    """
    if not idle_markers:
        return True  # no screen detection → always accept
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    last_line = lines[-1].lower()
    has_idle = any(marker.lower() in last_line for marker in idle_markers)
    has_queued = any(marker.lower() in last_line for marker in queued_input_markers)
    if has_queued:
        return False
    if not has_idle:
        return False
    # Idle marker in last line — also check busy markers in the same scope
    # (entire tail) to catch cases where the pane is switching state
    tail = "\n".join(lines[-40:]).lower()
    has_busy = any(marker.lower() in tail for marker in busy_markers)
    return not has_busy


# ──────────────────────────────────────────────
# Shared reclaim-orphan helper
# ──────────────────────────────────────────────

def reclaim_orphaned_running_task(
    args: argparse.Namespace, *, log_path: Path, conn: Any,
    listener_kind: str = "interactive",
) -> bool:
    """Reclaim tasks whose worker_pid is no longer alive."""
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    reclaimed = False
    try:
        rows = conn.execute(
            "SELECT id, worker_pid, workspace_path, current_run_id, generation, "
            "claim_lock FROM tasks WHERE status='running'"
        ).fetchall()
        for row in rows:
            pid = row["worker_pid"]
            if pid and not _pid_alive(pid):
                ws = row["workspace_path"] if "workspace_path" in row.keys() else None
                if ws and not _workspace_matches(ws, workspace):
                    continue
                reason = f"orphaned running task {row['id']} old_pid={pid}"
                ok = _reclaim_task_without_signaling_worker(
                    conn,
                    row["id"],
                    reason=reason,
                    expected_run_id=row["current_run_id"],
                    expected_generation=row["generation"],
                    expected_claim_lock=row["claim_lock"],
                )
                if ok:
                    reclaimed = True
                    log_line(log_path, reason)
    except Exception as exc:
        log_line(log_path, f"reclaim_orphaned error: {exc}")
    return reclaimed


# ──────────────────────────────────────────────
# BASE CLASS
# ──────────────────────────────────────────────

class BaseInteractiveListener:
    """Base class for all interactive Kanban listeners.

    Subclasses must define:
      - agent_name, agent_slug (str properties)
      - build_tui_cmd(workspace, continue_session, model, sandbox, extra_args) -> list[str]
      - has_saved_sessions(workspace) -> bool
      - inject_text(task_id, title, assignee, profile, prompt_path, board) -> str
      - pane_label(task_id=None) -> str

    Subclasses may override hooks:
      - idle_markers, busy_markers, queued_input_markers
      - on_claim_pre_check, on_claim_post_confirm
      - on_task_running_monitor, on_watcher_loop_idle
      - build_launch_env, build_watcher_extra_args
    """

    # ── Identity (subclass must override) ──
    agent_name: str = ""
    agent_slug: str = ""
    role_context_backend: str = ""

    # ── Idle/busy markers for screen-based detection ──
    # Empty tuple = no screen-based idle detection (always accept).
    idle_markers: tuple[str, ...] = ()
    busy_markers: tuple[str, ...] = ()
    queued_input_markers: tuple[str, ...] = ()
    # Backends can opt into strict semantic acknowledgement.  Legacy
    # backends retain transport-only delivery until they implement the
    # tri-state hook explicitly.
    semantic_delivery_required: bool = False

    # ── Abstract methods (subclass MUST implement) ──

    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        provider: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build the command list to launch the TUI process."""
        raise NotImplementedError

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check whether the agent has saved sessions for this workspace."""
        raise NotImplementedError

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build the single-line text to inject into the pane.

        MUST NOT contain \\n (LF). Use spaces; TUI word-wraps."""
        raise NotImplementedError

    def pane_label(self, task_id: str | None = None) -> str:
        """Return the zellij pane title."""
        raise NotImplementedError

    def expected_pane_prefix(self) -> str:
        return f"{self.agent_slug}-kanban"

    def render_effective_role_context(
        self,
        *,
        board: str,
        workspace: Path,
        pane_profile: str,
        task: Any,
        output_path: Path,
        profiles_root: Path = DEFAULT_PROFILES_ROOT,
        shared_skills_root: Path = DEFAULT_SHARED_SKILLS_ROOT,
        backend: str | None = None,
    ) -> str:
        """Render the common task-assignee role context for this backend."""
        return _render_effective_role_context(
            board=board,
            workspace=workspace,
            pane_profile=pane_profile,
            task=task,
            output_path=output_path,
            profiles_root=profiles_root,
            shared_skills_root=shared_skills_root,
            backend=backend or self.role_context_backend or self.agent_slug,
        )

    # ── Optional hooks (default: no-op / basic) ──

    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        """Return True if the pane is ready to accept a new task."""
        if not self.idle_markers:
            return True
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if not screen or not screen.strip():
            return False
        return _pane_can_accept_new_kanban_task(
            screen, self.idle_markers, self.busy_markers, self.queued_input_markers,
        )

    def pane_is_idle(self, screen: str) -> bool:
        """Return whether the current pane viewport is a safe idle boundary."""
        return _looks_like_idle_pane(
            screen,
            idle_markers=self.idle_markers,
            busy_markers=self.busy_markers,
        )

    def composer_input_text(self, screen: str) -> str | None:  # noqa: ARG002
        """Return non-empty composer text, or ``None`` when empty/unsupported.

        Backends with a persistent composer should override this. The shared
        stability guard then protects every automatic injection path without
        teaching the base listener each TUI's screen layout.
        """
        return None

    def read_pane_screen(
        self, *, session: str, pane_id: str, log_path: Path,
    ) -> str | None:
        """Read a pane through the backend's patchable screen-reader seam."""
        return zellij_dump_screen(
            session=session, pane_id=pane_id, log_path=log_path,
        )

    INPUT_STABILITY_INTERVAL_S: float = 10.0
    INPUT_STABILITY_UNCHANGED_CONFIRMATIONS: int = 3
    INPUT_STABILITY_CACHE_S: float = 5.0

    def wait_for_stable_composer_input(
        self,
        *,
        session: str,
        pane_id: str,
        log_path: Path,
        initial_screen: str | None = None,
        screen_reader: Any | None = None,
    ) -> bool:
        """Allow injection only after non-empty composer text stops changing.

        A non-empty composer needs three unchanged *intervals* of ten seconds,
        i.e. at least thirty seconds after the last observed edit. Any content
        change resets the counter. Empty composers pass immediately; unknown,
        missing, or newly-busy panes fail closed.
        """
        read_screen = screen_reader or self.read_pane_screen
        screen = initial_screen
        if screen is None:
            screen = read_screen(
                session=session, pane_id=pane_id, log_path=log_path,
            )
        if not screen or not screen.strip() or not self.pane_is_idle(screen):
            return False

        current = self.composer_input_text(screen)
        cache_key = (str(session), str(pane_id))
        if not current:
            self._stable_composer_cache.pop(cache_key, None)
            return True

        cached = self._stable_composer_cache.get(cache_key)
        now = time.time()
        if (
            cached is not None
            and cached[0] == current
            and now - cached[1] <= self.INPUT_STABILITY_CACHE_S
        ):
            return True

        unchanged = 0
        while unchanged < self.INPUT_STABILITY_UNCHANGED_CONFIRMATIONS:
            time.sleep(self.INPUT_STABILITY_INTERVAL_S)
            screen = read_screen(
                session=session, pane_id=pane_id, log_path=log_path,
            )
            if not screen or not screen.strip() or not self.pane_is_idle(screen):
                self._stable_composer_cache.pop(cache_key, None)
                return False
            observed = self.composer_input_text(screen)
            if not observed:
                self._stable_composer_cache.pop(cache_key, None)
                return True
            if observed == current:
                unchanged += 1
            else:
                current = observed
                unchanged = 0
                log_line(
                    log_path,
                    "composer input changed; reset 10s stability confirmations",
                )

        self._stable_composer_cache[cache_key] = (current, time.time())
        log_line(
            log_path,
            "composer input unchanged for 3x10s; automatic injection allowed",
        )
        return True

    def on_claim_post_confirm(self, args: argparse.Namespace, log_path: Path) -> bool:
        """After claim, confirm the pane is still idle before injecting.

        Default: no extra confirmation (return True immediately).
        DeepSeek overrides this with a 2-round idle check.
        """
        return True

    def on_post_inject(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
        injected_marker: str | None = None,
        pre_write_composer: str | None = None,
    ) -> str | None:
        """Hook called after zellij_inject succeeds.

        Subclasses can override to perform extra actions after injection,
        e.g. sending an additional Enter for TUIs that queue input.
        Default: no-op.
        """
        return None

    def _post_injection_contract(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
        injected_marker: str, pre_write_composer: str | None,
        correlation: str,
    ) -> str:
        """Return the post-write delivery state.

        A delivery is acknowledged only by an explicit ``confirmed`` result.
        Legacy hooks are still called (without the new keyword arguments when
        necessary), but an absent/invalid result is treated as unknown and the
        caller must reclaim/release its lease.
        """
        try:
            result = self.on_post_inject(
                args,
                zellij_session=zellij_session,
                zellij_pane_id=zellij_pane_id,
                log_path=log_path,
                injected_marker=injected_marker,
                pre_write_composer=pre_write_composer,
            )
        except TypeError as exc:
            # Existing backend listeners may not yet accept the expanded
            # contract.  Calling the old shape preserves their retry behavior,
            # while their implicit ``None`` remains non-confirming.
            if "injected_marker" not in str(exc) and "pre_write_composer" not in str(exc):
                kind, ident = (correlation.split(":", 1) + [""])[:2]
                self._log_delivery_event(log_path, state="unknown", correlation_kind=kind,
                    correlation_id=ident, pane_id=zellij_pane_id)
                return "unknown"
            try:
                result = self.on_post_inject(
                    args,
                    zellij_session=zellij_session,
                    zellij_pane_id=zellij_pane_id,
                    log_path=log_path,
                )
            except Exception as retry_exc:
                kind, ident = (correlation.split(":", 1) + [""])[:2]
                self._log_delivery_event(log_path, state="unknown", correlation_kind=kind,
                    correlation_id=ident, pane_id=zellij_pane_id)
                return "unknown"
        except Exception as exc:
            kind, ident = (correlation.split(":", 1) + [""])[:2]
            self._log_delivery_event(log_path, state="unknown", correlation_kind=kind,
                correlation_id=ident, pane_id=zellij_pane_id)
            return "unknown"

        state = str(result or "").strip().lower()
        if state in {"confirmed", "known_unsubmitted", "unknown"}:
            return state

        if not self.semantic_delivery_required:
            # Explicitly preserve legacy transport-only semantics without
            # presenting this as a semantic confirmation.
            return "transport_accepted"

        # A hook that predates the tri-state API can still be classified from
        # the pane when possible.  Failure to observe a decisive boundary is
        # deliberately fail-closed as ``unknown``.
        try:
            screen = self.read_pane_screen(
                session=zellij_session, pane_id=zellij_pane_id,
                log_path=log_path,
            )
            if not screen or not screen.strip():
                return "unknown"
            if injected_marker and injected_marker in screen:
                return "known_unsubmitted"
            tail = "\n".join(_tail_nonempty_lines(screen, limit=20)).lower()
            if any(marker.lower() in tail for marker in self.busy_markers):
                return "confirmed"
            observed = self.composer_input_text(screen)
            if pre_write_composer is not None and observed != pre_write_composer:
                return "confirmed"
        except Exception:
            return "unknown"
        return "unknown"

    @staticmethod
    def _delivery_marker(text: str, correlation: str) -> str:
        """Return the visible payload marker passed to post-injection observers."""
        del correlation  # correlation is carried separately in logs/transport
        return str(text)

    def _pre_write_composer(
        self, *, session: str, pane_id: str, log_path: Path,
    ) -> str | None:
        try:
            screen = self.read_pane_screen(
                session=session, pane_id=pane_id, log_path=log_path,
            )
            return self.composer_input_text(screen) if screen else None
        except Exception:
            return None

    def _log_delivery_event(
        self, log_path: Path, *, state: str, correlation_kind: str,
        correlation_id: str, task_id: str | None = None,
        run_id: int | None = None, generation: int | None = None,
        pane_id: str = "", reclaimed: bool | None = None,
    ) -> None:
        fields = {
            "event": f"delivery_{state}", "state": state,
            "correlation_kind": correlation_kind,
            "correlation_id": correlation_id, "task_id": task_id or "-",
            "run_id": "-" if run_id is None else run_id,
            "generation": "-" if generation is None else generation,
            "pane_id": pane_id, "pane_prefix": self.expected_pane_prefix(),
        }
        if reclaimed is not None:
            fields["reclaimed"] = reclaimed
        log_line(log_path, " ".join(f"{key}={value}" for key, value in fields.items()))

    def on_task_running_monitor(
        self, args: argparse.Namespace, conn: Any,
        task_id: str, log_path: Path,
    ) -> None:
        """Extra monitoring while a task is running (progress watch, etc).

        Default: check for API failure on idle pane and inject '继续' to retry.
        """
        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not zellij_session or not zellij_pane_id:
            return

        screen = zellij_dump_screen(session=zellij_session, pane_id=zellij_pane_id, log_path=log_path)
        if not screen:
            return

        # Check if pane is idle (not busy)
        if not self.pane_is_idle(screen):
            # Pane is busy (agent recovered) — reset retry state so next error
            # cycle starts fresh
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = None
            self._reset_idle_followup()
            return

        # Pane is idle while task is running — check for API error
        if self.check_api_failure_retry(
            session=zellij_session, pane_id=zellij_pane_id, screen=screen,
            task_id=task_id, log_path=log_path,
        ):
            return
        self._handle_idle_task_followup(args, conn, task_id, log_path)

    def on_watcher_loop_idle(
        self, args: argparse.Namespace, conn: Any, log_path: Path,
    ) -> None:
        """Called each watcher loop tick when no task is active.

        Default: no action. DeepSeek uses this for auto-dismiss steering.
        """
        pass

    def build_launch_env(self, args: argparse.Namespace) -> dict[str, str]:
        """Build environment variables for the TUI process."""
        env = os.environ.copy()
        board = args.board or kb.get_current_board() or "default"
        workspace = Path(args.workspace).expanduser().resolve()
        env.update({
            "HERMES_KANBAN_BOARD": board,
            "HERMES_KANBAN_PROFILE": args.profile,
            "HERMES_KANBAN_CLAIM_ASSIGNEES": ",".join(claim_assignees(args)),
            "HERMES_KANBAN_WORKSPACE": str(workspace),
        })
        return env

    def build_watcher_extra_args(self, args: argparse.Namespace) -> list[str]:
        """Extra args to append to the watcher --watch-child command."""
        return []

    # ── Instance state ──
    def __init__(self):
        self._profile: str = ""
        self._board: str = ""
        self._workspace: Path = Path(".")
        self._log_path: Path = Path("/tmp/kanban.log")
        self._api_retry_count: int = 0       # per-task API failure retry counter
        self._api_retry_first_at: float | None = None  # when first API-idle was seen
        self._api_retry_kind: str | None = None  # generic vs model-capacity failure
        self._active_control_id: int | None = None
        self._active_task_id: str | None = None
        self._active_run_id: int | None = None
        self._active_generation: int | None = None
        self._active_claim_lock: str | None = None
        self._idle_followup_task_id: str | None = None
        self._idle_followup_since: float | None = None
        self._idle_followup_sent: bool = False
        self._goal_completion_last_sent_at: dict[str, float] = {}
        self._goal_result_wait_since: dict[str, float] = {}
        self._goals_waiting_on_results: set[str] = set()
        self._goal_check_stall_counts: dict[str, int] = {}
        self._goal_check_watermark_event: dict[str, int] = {}
        self._goal_check_watermark_comment: dict[str, int] = {}
        self._goal_check_suspended: set[str] = set()
        self._stable_composer_cache: dict[tuple[str, str], tuple[str, float]] = {}

    # An idle prompt is a safe input boundary, but brief idle flashes occur
    # between tool calls.  Require a stable idle interval before injecting.
    IDLE_FOLLOWUP_GRACE_S: float = 15.0
    DAYTIME_GOAL_COMPLETION_INTERVAL_S: float = 2 * 60.0
    OVERNIGHT_GOAL_COMPLETION_INTERVAL_S: float = 30 * 60.0
    RESULT_WAIT_GOAL_INTERVAL_S: float = 120 * 60.0
    OVERNIGHT_GOAL_COMPLETION_START_HOUR: int = 1
    OVERNIGHT_GOAL_COMPLETION_END_HOUR: int = 9

    def _goal_completion_interval_s(self, now: float) -> float:
        hour = time.localtime(now).tm_hour
        if (
            self.OVERNIGHT_GOAL_COMPLETION_START_HOUR
            <= hour
            < self.OVERNIGHT_GOAL_COMPLETION_END_HOUR
        ):
            return self.OVERNIGHT_GOAL_COMPLETION_INTERVAL_S
        return self.DAYTIME_GOAL_COMPLETION_INTERVAL_S

    def _reset_idle_followup(self) -> None:
        self._idle_followup_task_id = None
        self._idle_followup_since = None
        self._idle_followup_sent = False

    # After this many consecutive GOAL_COMPLETION_CHECK injections with no
    # durable progress on the task, stop re-injecting the check.  The goal
    # stays running; the stall is recorded durably and the circuit breaker
    # auto-resets as soon as real progress appears.
    GOAL_COMPLETION_STALL_LIMIT: int = 3

    # Events that count as durable progress.  Heartbeats and comment echoes
    # are excluded — the check loop itself generates both.
    _GOAL_PROGRESS_EVENTS: frozenset[str] = frozenset({
        "created", "claimed", "promoted", "promoted_manual", "spawned",
        "linked", "completed", "blocked", "block_loop_detected",
        "dependency_wait", "returned_for_rework", "invalidated_for_rework",
        "unblocked", "gave_up", "reclaimed",
    })

    def _goal_check_has_progress(
        self, conn: Any, task_id: str, marker: str,
    ) -> bool | None:
        """Whether durable progress happened since the last check injection.

        Progress = (a) any non-heartbeat/non-commented task_event newer than
        the last observed one, or (b) any new comment that is not an echo of
        this same check loop (echo comments start with ``marker`` — the
        agent's ``GOAL_COMPLETION_CHECK #N …`` responses and this loop's own
        suspension note all do).  Returns ``None`` on the first observation
        for a task (baseline established, no judgment possible).
        """
        last_event = self._goal_check_watermark_event.get(task_id, -1)
        last_comment = self._goal_check_watermark_comment.get(task_id, -1)
        row = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        max_event = int(row[0]) if row and row[0] is not None else 0
        row = conn.execute(
            "SELECT MAX(id) FROM task_comments WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        max_comment = int(row[0]) if row and row[0] is not None else 0
        if last_event < 0 and last_comment < 0:
            self._goal_check_watermark_event[task_id] = max_event
            self._goal_check_watermark_comment[task_id] = max_comment
            return None
        progress = False
        if last_event >= 0:
            for r in conn.execute(
                "SELECT DISTINCT kind FROM task_events "
                "WHERE task_id = ? AND id > ?",
                (task_id, last_event),
            ):
                if r[0] in self._GOAL_PROGRESS_EVENTS:
                    progress = True
                    break
        if not progress and last_comment >= 0:
            for r in conn.execute(
                "SELECT body FROM task_comments "
                "WHERE task_id = ? AND id > ? ORDER BY id DESC",
                (task_id, last_comment),
            ):
                body = str(r[0]).lstrip()
                if f"[{marker}]" in body or body.startswith(marker):
                    continue
                progress = True
                break
        self._goal_check_watermark_event[task_id] = max_event
        self._goal_check_watermark_comment[task_id] = max_comment
        return progress

    def _goal_check_circuit_breaker(
        self,
        conn: Any,
        task_id: str,
        marker: str,
        log_path: Path,
    ) -> bool:
        """Evaluate the stall circuit right before a check injection.

        Called only at the moment the watcher is about to inject (or is
        suspended from injecting) a GOAL_COMPLETION_CHECK, so a stall counts
        injections-without-progress, not idle ticks.  Returns True when the
        injection must be suppressed.

        - Suspended + no progress → keep suppressing.
        - Suspended + progress → reset the breaker and allow.
        - Active + progress → clear the stall count.
        - Active + stall reaching GOAL_COMPLETION_STALL_LIMIT → suspend,
          write one durable note, and suppress this round.
        """
        progress = self._goal_check_has_progress(conn, task_id, marker)
        if task_id in self._goal_check_suspended:
            if not progress:
                # Keep suppressing but keep the episode alive so every
                # eligible window re-evaluates (fast progress detection).
                return True
            # Real progress → reset the breaker and let the check through.
            self._goal_check_stall_counts.pop(task_id, None)
            self._goal_check_suspended.discard(task_id)
            return False
        if progress is None:
            # First observation for this task: baseline only, no judgment.
            return False
        stall = 0 if progress else self._goal_check_stall_counts.get(task_id, 0) + 1
        self._goal_check_stall_counts[task_id] = stall
        if stall < self.GOAL_COMPLETION_STALL_LIMIT:
            return False
        self._goal_check_suspended.add(task_id)
        stall_note = (
            f"[{marker}] goal-check circuit breaker: {stall} consecutive "
            "checks produced no durable progress (no new non-echo comments, "
            "no lifecycle events). Suspending further GOAL_COMPLETION_CHECK "
            "injections to stop the no-op loop. The goal stays running; the "
            "watcher resumes checks automatically after any real durable "
            "progress (or an operator note) on this task."
        )
        try:
            kb.add_comment(
                conn, task_id,
                f"{self.agent_slug}-interactive-listener",
                stall_note,
            )
        except Exception as exc:
            log_line(
                log_path,
                f"goal-check stall note failed for {task_id}: {exc}",
            )
        log_line(
            log_path,
            f"goal completion check suspended for {task_id} "
            f"after {stall} stalled injections",
        )
        return True

    def _handle_idle_task_followup(
        self,
        args: argparse.Namespace,
        conn: Any,
        task_id: str,
        log_path: Path,
    ) -> bool:
        """Check goal completion/reviewer lifecycle once per stable idle episode.

        Goal work gets a neutral completion question: normal idle may mean the
        agent believes it is done, so do not presuppose that it must continue.
        API/model failure recovery remains the only path that injects a bare
        continuation.  Reviewer work gets one lifecycle reminder so a written
        analysis cannot leave the card running forever.  Ordinary tasks keep
        the existing behavior.
        """
        try:
            task = kb.get_task(conn, task_id)
        except Exception as exc:
            log_line(log_path, f"idle followup task lookup failed for {task_id}: {exc}")
            return False
        if task is None or task.status != "running":
            self._reset_idle_followup()
            return False

        now = time.time()
        waiting_state: kb.ResultWaitState | None = None
        if task.goal_mode:
            if _result_notifications_enabled():
                waiting_state = kb.result_wait_state(
                    conn,
                    task.assignee or self._profile,
                    exclude_task_id=task_id,
                )
            waiting_on_results = bool(
                waiting_state
                and (waiting_state.watched_tasks or waiting_state.queue_ids)
            )
            if waiting_on_results:
                if task_id not in self._goals_waiting_on_results:
                    self._goals_waiting_on_results.add(task_id)
                    self._goal_result_wait_since[task_id] = now
                watched = ", ".join(
                    f"{watched_id}({status})"
                    for watched_id, status in waiting_state.watched_tasks[:8]
                ) or "none"
                queued = ",".join(
                    str(queue_id) for queue_id in waiting_state.queue_ids[:8]
                ) or "none"
                marker = "WAITING_ON_TASK_RESULTS"
                text = (
                    f"[{marker}] Kanban goal {task_id} is waiting on subscribed "
                    f"task results: tasks={watched}; queue={queued}. This is the "
                    "120-minute insurance check. Do not poll or create a "
                    "continuation/notification card; continue only work that is "
                    "independent of those results and let the watcher deliver them."
                )
            else:
                if task_id in self._goals_waiting_on_results:
                    self._goals_waiting_on_results.discard(task_id)
                    self._goal_result_wait_since.pop(task_id, None)
                    self._goal_completion_last_sent_at.pop(task_id, None)
                    self._goal_check_stall_counts.pop(task_id, None)
                    self._goal_check_suspended.discard(task_id)
                    self._idle_followup_task_id = task_id
                    self._idle_followup_since = now
                    self._idle_followup_sent = False
                if _has_open_reviewer_checkpoint(conn, task_id):
                    self._reset_idle_followup()
                    return True
                marker = "GOAL_COMPLETION_CHECK"
                text = (
                    f"[{marker}] Kanban task {task_id}: 任务都完成了吗？"
                    "请依据任务的北极星目标、durable history 和实际证据检查。"
                    "若已全部完成，更新证据并 complete；若尚未完成，在同一任务内"
                    "推进下一个具体步骤。中间 NO_CLAIM、局部产物或普通阻塞不等于"
                    "完成，除非任务合同明确将其定义为终态。若未达成北极星且不满足"
                    "升级条件，不得向用户列出普通技术选项；选择第一个未满足的承重 gate "
                    "并立即执行。"
                )
        elif task.assignee == "reviewer":
            marker = "REVIEW_LIFECYCLE"
            text = (
                f"[{marker}] Reviewer task {task_id} is still running. "
                "If the evidence is not decisive, continue the review. If it is "
                "decisive, first write the deterministic verdict and handback "
                "to the durable Kanban comment, then explicitly complete the task."
            )
        else:
            self._reset_idle_followup()
            return False

        if self._idle_followup_task_id != task_id:
            self._idle_followup_task_id = task_id
            self._idle_followup_since = now
            self._idle_followup_sent = False
            return True
        if self._idle_followup_since is None:
            self._idle_followup_since = now
            return True
        if self._idle_followup_sent:
            return True
        if now - self._idle_followup_since < self.IDLE_FOLLOWUP_GRACE_S:
            return True
        if task.goal_mode:
            if task_id in self._goals_waiting_on_results:
                wait_since = self._goal_result_wait_since.get(task_id, now)
                if now - wait_since < self.RESULT_WAIT_GOAL_INTERVAL_S:
                    return True
            else:
                last_sent_at = self._goal_completion_last_sent_at.get(task_id)
                interval_s = self._goal_completion_interval_s(now)
                if last_sent_at is not None and now - last_sent_at < interval_s:
                    return True
                if self._goal_check_circuit_breaker(conn, task_id, marker, log_path):
                    return True

        session = getattr(args, "zellij_session", "")
        pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not session or not pane_id:
            return False
        if not self.wait_for_stable_composer_input(
            session=session,
            pane_id=pane_id,
            log_path=log_path,
        ):
            return True
        log_line(log_path, f"idle followup for {task_id}: {marker}")
        ok = zellij_inject(
            session=session,
            pane_id=pane_id,
            text=tag_injected_text(text, source_profile="watcher"),
            expected_pane_prefix=self.expected_pane_prefix(),
            log_path=log_path,
        )
        if ok is False:
            return True
        time.sleep(0.5)
        if not zellij_submit(
            session=session, pane_id=pane_id,
            expected_pane_prefix=self.expected_pane_prefix(), log_path=log_path,
        ):
            return True
        if task.goal_mode:
            if task_id in self._goals_waiting_on_results:
                self._goal_result_wait_since[task_id] = now
            else:
                self._goal_completion_last_sent_at[task_id] = now
        self._idle_followup_sent = True
        return True

    def _remember_active_claim(self, task: kb.Task) -> None:
        self._active_task_id = task.id
        self._active_run_id = task.current_run_id
        self._active_generation = task.generation
        self._active_claim_lock = task.claim_lock

    def _clear_active_claim_identity(self) -> None:
        self._active_task_id = None
        self._active_run_id = None
        self._active_generation = None
        self._active_claim_lock = None
        self._api_retry_count = 0
        self._api_retry_first_at = None
        self._api_retry_kind = None
        self._reset_idle_followup()

    # ── API failure retry on idle ──
    # When agent goes idle mid-task due to API error, inject "继续"
    # up to API_RETRY_MAX times with backoff. After max retries,
    # fall through to existing idle-pane-reclaim logic.
    # Retry intervals: 5min → 10min → 20min (total ~35min window).
    API_RETRY_MAX: int = 3
    API_RETRY_BACKOFF: list[float] = [300.0, 600.0, 1200.0]  # 5min, 10min, 20min
    # Capacity can recover faster than a transport outage, but retries remain
    # bounded and stay in the same Codex session/work item.
    API_CAPACITY_RETRY_BACKOFF: list[float] = [30.0, 90.0, 300.0]
    API_CAPACITY_ERROR_MARKERS: tuple[str, ...] = (
        "selected model is at capacity",
        "model capacity exhausted",
        "model is at capacity",
    )
    API_ERROR_MARKERS: tuple[str, ...] = (
            # ── Generic API errors ──
            "api call failed", "api error", "api request failed",
            "request failed", "request error",
            # ── xunfei-specific ──
            "xunfei request failed", "xunfei request error",
            "notenoughcv", "engineinternalerror", "system is busy",
            "invalid params", "appidnoauth",
            # ── HTTP-level / transport errors (httpx, requests, urllib3) ──
            # Connection-level
            "connectionerror", "connection error",
            "connectionrefused", "connection refused",
            "connectionreset", "connection reset",
            "connection aborted",
            "connection broken",
            "connection closed by remote",
            "connecterror", "connect error",
            "connecttimeout", "connect timeout",
            # Read/Write
            "readtimeout", "read timeout",
            "writetimeout", "write timeout",
            "pooltimeout", "pool timeout",
            # Proxy
            "proxyerror", "proxy error",
            # SSL/TLS
            "sslerror", "ssl error",
            "certificate_verify_failed", "certificate verify failed",
            # Protocol
            "broken pipe", "brokenpipeerror",
            "remote end closed connection",
            "remote protocol error",
            "local protocol error",
            "eof occurred",
            "protocol error", "protocolexception",
            "bad status line",
            "chunked encoding",
            "content length mismatch",
            # Retry / pool
            "maxretryerror", "max retries exceeded",
            "newconnectionerror",
            # Network unreachable
            "network is unreachable",
            "no route to host",
            "cannot connect",
            "failed to connect",
            "name resolution error",
            "dns lookup failed",
            "name or service not known",
            "temporary failure in name resolution",
            # ── Chinese (domestic API providers, OSS proxies) ──
            "连接失败", "连接超时", "连接被拒",
            "网络错误", "网络异常", "网络不可达",
            "请求异常", "请求超时",
            "服务不可达",
            # ── HTTP status codes ──
            "503", "502", "504", "429",
            # ── Timeout ──
            "timeout",
        )

    def check_api_failure_retry(
        self, *, session: str, pane_id: str, screen: str,
        task_id: str, log_path: Path,
    ) -> bool:
        """Check if pane is idle with an API error and inject '继续' to retry.

        Returns True if a retry was injected this tick (caller should skip reclaim).
        Returns False if no retry was needed or max retries exhausted.
        """
        if self._api_retry_count >= self.API_RETRY_MAX:
            return False

        tail = "\n".join(_tail_nonempty_lines(screen)).lower()
        capacity_matches = [
            marker for marker in self.API_CAPACITY_ERROR_MARKERS
            if marker.lower() in tail
        ]
        generic_matches = [
            marker for marker in self.API_ERROR_MARKERS
            if marker.lower() in tail
        ]
        error_kind = "capacity" if capacity_matches else "generic" if generic_matches else None

        if error_kind is None:
            # No API error visible — reset retry state
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = None
            return False

        if self._api_retry_kind != error_kind:
            self._api_retry_count = 0
            self._api_retry_first_at = None
            self._api_retry_kind = error_kind

        # API error detected — record which markers matched
        matched = capacity_matches + generic_matches
        log_line(log_path, f"api-error-matched for task {task_id}: markers={matched} (retry {self._api_retry_count}/{self.API_RETRY_MAX})")

        # check if we should retry now
        now = time.time()
        if self._api_retry_first_at is None:
            self._api_retry_first_at = now
            log_line(log_path, f"api-error-idle observed for task {task_id} (retry {self._api_retry_count}/{self.API_RETRY_MAX})")

        elapsed = now - self._api_retry_first_at
        schedule = (
            self.API_CAPACITY_RETRY_BACKOFF
            if error_kind == "capacity"
            else self.API_RETRY_BACKOFF
        )
        backoff = schedule[self._api_retry_count] if self._api_retry_count < len(schedule) else schedule[-1]

        if elapsed < backoff:
            return True  # still waiting for backoff; skip reclaim this tick

        # Backoff elapsed — inject "继续"
        if not self.wait_for_stable_composer_input(
            session=session,
            pane_id=pane_id,
            log_path=log_path,
            initial_screen=screen,
        ):
            return True
        self._api_retry_count += 1
        self._api_retry_first_at = None  # reset timer; next error detection starts fresh
        log_line(log_path, f"api-error-retry kind={error_kind} {self._api_retry_count}/{self.API_RETRY_MAX} for task {task_id}: injecting 继续 after {elapsed:.0f}s")

        ok = zellij_inject(
            session=session,
            pane_id=pane_id,
            text=tag_injected_text("继续", source_profile="watcher"),
            expected_pane_prefix=self.expected_pane_prefix(),
            log_path=log_path,
        )
        if ok is False:
            return True
        time.sleep(0.5)
        if not zellij_submit(
            session=session, pane_id=pane_id,
            expected_pane_prefix=self.expected_pane_prefix(), log_path=log_path,
        ):
            return True

        return True

    def _init_from_args(self, args: argparse.Namespace) -> None:
        self._profile = args.profile
        self._board = args.board or kb.get_current_board() or "default"
        self._workspace = Path(args.workspace).expanduser().resolve()
        self._log_path = kb.worker_logs_dir(board=self._board) / f"{self.agent_slug}-interactive-{self._profile}.log"

    # ── Claim lock ──
    def _claim_lock(self) -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{self.agent_slug}-interactive"

    def _control_receiver(self, args: argparse.Namespace) -> str:
        """Stable pane identity, preserved when a watcher process restarts."""
        session = str(getattr(args, "zellij_session", "") or "session")
        pane_id = str(getattr(args, "zellij_pane_id", "") or "pane")
        return (
            f"{self.agent_slug}:{self._board}:{self._profile}:"
            f"{session}:{pane_id}"
        )

    def _mark_prompt_superseded(self, control: kb.ControlMessage) -> Path:
        path = prompt_dir(
            self._workspace, self._board, self._profile,
            agent_slug=self.agent_slug,
        ) / f"task-{control.task_id}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        marker = (
            f"\n\nSUPERSEDED by control {control.id}: run {control.run_id}, "
            f"generation {control.generation}, returned via "
            f"{control.return_task_id}. Do not complete this prompt.\n"
        )
        if path.exists():
            existing = path.read_text(encoding="utf-8", errors="replace")
            if f"SUPERSEDED by control {control.id}:" not in existing:
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(marker)
        else:
            path.write_text(marker.lstrip(), encoding="utf-8")
        return path

    def _control_prompt(self, control: kb.ControlMessage) -> str:
        return (
            f"[SYSTEM CONTROL {control.id}] Task {control.task_id} run "
            f"{control.run_id} generation {control.generation} is SUPERSEDED "
            f"by return-for-rework {control.return_task_id}. Stop the old "
            f"contract at this safe input boundary; do not complete or block "
            f"it. Read the durable pause comment on {control.task_id} and "
            f"reviewer reason on {control.return_task_id}, then acknowledge "
            f"with `hermes kanban "
            f"--board {self._board} control-ack {control.id}`."
        )

    def _control_safe_boundary(
        self, args: argparse.Namespace, log_path: Path,
    ) -> bool:
        """Require positive pane-idle evidence before a control injection."""
        if not self.idle_markers:
            return False
        session = str(getattr(args, "zellij_session", "") or "")
        pane_id = str(getattr(args, "zellij_pane_id", "") or "")
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(
            session=session, pane_id=pane_id, log_path=log_path,
        )
        if not screen or not screen.strip():
            return False
        # Reuse each backend's idle semantics (Codex's prompt is not on the
        # last line, while other TUIs use the base implementation). The
        # positive screen probe above prevents their normal claim fail-open
        # behavior from becoming a control-plane interrupt.
        return bool(self.on_claim_pre_check(args, log_path))

    def pump_control_messages(
        self, args: argparse.Namespace, conn: Any, log_path: Path,
    ) -> bool:
        """Deliver or hold one cooperative control before normal claim logic.

        Returns True whenever the pane must not claim a task this tick: a
        control is pending while the pane is busy, being delivered, or already
        delivered and awaiting ACK.
        """
        profiles = claim_assignees(args)
        receiver = self._control_receiver(args)
        control = kb.peek_control_message(
            conn, profiles=profiles, receiver=receiver,
        )
        session = str(getattr(args, "zellij_session", "") or "")
        pane_id = str(getattr(args, "zellij_pane_id", "") or "")
        if control is None:
            if self._active_control_id is not None:
                self._active_control_id = None
                zellij_rename_pane(
                    session=session,
                    pane_id=pane_id,
                    name=self.pane_label(),
                    log_path=log_path,
                )
            return False

        if control.status == "delivered":
            self._active_control_id = control.id
            zellij_rename_pane(
                session=session,
                pane_id=pane_id,
                name=f"{self.agent_slug}-kanban [PAUSE {control.task_id}]",
                log_path=log_path,
            )
            return True

        if not self._control_safe_boundary(args, log_path):
            self._active_control_id = control.id
            zellij_rename_pane(
                session=session,
                pane_id=pane_id,
                name=f"{self.agent_slug}-kanban [PAUSE {control.task_id}]",
                log_path=log_path,
            )
            return True

        leased = kb.lease_control_message(
            conn,
            profiles=profiles,
            receiver=receiver,
        )
        if leased is None:
            return False
        if leased.status == "delivered":
            self._active_control_id = leased.id
            return True

        self._mark_prompt_superseded(leased)
        prompt = tag_injected_text(
            self._control_prompt(leased), source_profile="watcher",
        )
        correlation = f"control:{leased.id}"
        pre_write_composer = self._pre_write_composer(
            session=session, pane_id=pane_id, log_path=log_path,
        )
        ok = zellij_inject(
            session=session,
            pane_id=pane_id,
            text=prompt,
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation,
            log_path=log_path,
        )
        if not ok:
            kb.release_control_lease(conn, leased.id, receiver=receiver)
            log_line(log_path, f"control injection failed; released {leased.id}")
            return True
        if not zellij_submit(
            session=session, pane_id=pane_id,
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation, log_path=log_path,
        ):
            kb.release_control_lease(conn, leased.id, receiver=receiver)
            log_line(log_path, f"control submit failed; released {leased.id}")
            return True

        state = self._post_injection_contract(
            args,
            zellij_session=session,
            zellij_pane_id=pane_id,
            log_path=log_path,
            injected_marker=self._delivery_marker(prompt, correlation),
            pre_write_composer=pre_write_composer,
            correlation=correlation,
        )
        if state not in {"confirmed", "transport_accepted"}:
            released = kb.release_control_lease(conn, leased.id, receiver=receiver)
            event = "delivery_requeued" if state == "known_unsubmitted" else "delivery_unknown"
            self._log_delivery_event(
                log_path, state=event.removeprefix("delivery_"),
                correlation_kind="control", correlation_id=str(leased.id),
                task_id=leased.task_id, run_id=leased.run_id,
                generation=leased.generation, pane_id=pane_id,
                reclaimed=released,
            )
            if not released:
                self._log_delivery_event(
                    log_path, state="reclaim_race", correlation_kind="control",
                    correlation_id=str(leased.id), task_id=leased.task_id,
                    run_id=leased.run_id, generation=leased.generation,
                    pane_id=pane_id,
                )
            return True

        self._log_delivery_event(
            log_path,
            state="confirmed" if state == "confirmed" else "transport_accepted",
            correlation_kind="control", correlation_id=str(leased.id),
            task_id=leased.task_id, run_id=leased.run_id,
            generation=leased.generation, pane_id=pane_id,
        )

        if not kb.mark_control_delivered(conn, leased.id, receiver=receiver):
            log_line(log_path, f"control {leased.id} injected but delivery CAS failed")
        self._active_control_id = leased.id
        zellij_rename_pane(
            session=session,
            pane_id=pane_id,
            name=f"{self.agent_slug}-kanban [PAUSE {leased.task_id}]",
            log_path=log_path,
        )
        log_line(
            log_path,
            f"delivered control {leased.id} for {leased.task_id}; awaiting ACK",
        )
        return True

    def pump_result_notifications(
        self, args: argparse.Namespace, conn: Any, log_path: Path,
    ) -> bool:
        """Deliver one durable FIFO batch, or hold normal injection while pending.

        The task transition that created the queue item never waits for this
        method. A busy or unavailable pane simply leaves the durable head in
        place for a later watcher tick.
        """
        if not _result_notifications_enabled():
            return False
        profile = str(getattr(args, "profile", "") or "").strip()
        if not profile:
            return False
        wait_state = kb.result_wait_state(
            conn, profile, exclude_task_id=self._active_task_id,
        )
        if not wait_state.queue_ids:
            return False

        session = str(getattr(args, "zellij_session", "") or "")
        pane_id = str(getattr(args, "zellij_pane_id", "") or "")
        if not session or not pane_id:
            return True
        if not self.wait_for_stable_composer_input(
            session=session,
            pane_id=pane_id,
            log_path=log_path,
        ):
            return True

        receiver = f"{self._control_receiver(args)}:results"
        try:
            items = kb.lease_result_notifications(
                conn,
                target_profile=profile,
                lease_owner=receiver,
                limit=8,
                lease_seconds=90,
            )
        except (ValueError, sqlite3.DatabaseError) as exc:
            log_line(log_path, f"result queue lease failed: {exc}")
            return True
        if not items:
            return True

        parts: list[str] = []
        for item in items:
            detail = ""
            payload = item.payload or {}
            summary = payload.get("summary") or payload.get("reason")
            if summary:
                compact = " ".join(str(summary).split())[:160]
                detail = f" ({compact})"
            parts.append(
                f"q{item.id}/e{item.event_id}:{item.task_id}={item.event_kind}{detail}"
            )
        prompt = tag_injected_text(
            "[TASK_RESULTS_READY] "
            + "; ".join(parts)
            + ". Read each task's durable summary/comments/events, then continue "
              "the existing objective; repeated queue/event IDs are replays.",
            source_profile="watcher",
        )
        queue_ids = [item.id for item in items]
        correlation = "result:" + ",".join(str(queue_id) for queue_id in queue_ids)
        pre_write_composer = self._pre_write_composer(
            session=session, pane_id=pane_id, log_path=log_path,
        )
        ok = zellij_inject(
            session=session,
            pane_id=pane_id,
            text=prompt,
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation,
            log_path=log_path,
        )
        if not ok:
            kb.release_result_notification_lease(
                conn, queue_ids, lease_owner=receiver,
            )
            log_line(log_path, f"result injection failed; released {queue_ids}")
            return True
        if not zellij_submit(
            session=session, pane_id=pane_id,
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation, log_path=log_path,
        ):
            kb.release_result_notification_lease(
                conn, queue_ids, lease_owner=receiver,
            )
            log_line(log_path, f"result submit failed; released {queue_ids}")
            return True
        state = self._post_injection_contract(
            args,
            zellij_session=session,
            zellij_pane_id=pane_id,
            log_path=log_path,
            injected_marker=self._delivery_marker(prompt, correlation),
            pre_write_composer=pre_write_composer,
            correlation=correlation,
        )
        first_item = items[0]
        first_task = kb.get_task(conn, first_item.task_id)
        event_row = conn.execute(
            "SELECT run_id, payload FROM task_events WHERE id = ?",
            (first_item.event_id,),
        ).fetchone()
        result_run_id = (
            int(event_row["run_id"])
            if event_row is not None and event_row["run_id"] is not None
            else (first_task.current_run_id if first_task is not None else None)
        )
        result_generation = first_task.generation if first_task is not None else None
        if state not in {"confirmed", "transport_accepted"}:
            released = kb.release_result_notification_lease(
                conn, queue_ids, lease_owner=receiver,
            )
            event = "delivery_requeued" if state == "known_unsubmitted" else "delivery_unknown"
            self._log_delivery_event(
                log_path, state=event.removeprefix("delivery_"),
                correlation_kind="result", correlation_id=",".join(map(str, queue_ids)),
                task_id=first_item.task_id, run_id=result_run_id,
                generation=result_generation, pane_id=pane_id, reclaimed=released,
            )
            if not released:
                self._log_delivery_event(
                    log_path, state="reclaim_race", correlation_kind="result",
                    correlation_id=",".join(map(str, queue_ids)),
                    task_id=first_item.task_id, run_id=result_run_id,
                    generation=result_generation, pane_id=pane_id,
                )
            return True
        self._log_delivery_event(
            log_path,
            state="confirmed" if state == "confirmed" else "transport_accepted",
            correlation_kind="result", correlation_id=",".join(map(str, queue_ids)),
            task_id=first_item.task_id, run_id=result_run_id,
            generation=result_generation, pane_id=pane_id,
        )
        if not kb.mark_result_notifications_delivered(
            conn, queue_ids, lease_owner=receiver,
        ):
            log_line(
                log_path,
                f"result queue injected but delivery CAS failed for {queue_ids}",
            )
        else:
            log_line(log_path, f"delivered result queue rows {queue_ids} to {profile}")
        return True

    # ── claim_and_inject_one ──
    def claim_and_inject_one(
        self, args: argparse.Namespace, *, log_path: Path, conn: Any | None = None,
    ) -> tuple[str | None, int | None]:
        board = self._board
        pane_profile = self._profile

        # Every backend reaches this shared path, including subclasses with a
        # custom watcher loop. Controls therefore preempt claims uniformly.
        if conn is not None and self.pump_control_messages(args, conn, log_path):
            return None, None

        # Pre-check: is pane ready?
        if not self.on_claim_pre_check(args, log_path):
            log_line(log_path, f"skip claim: {self.agent_name} pane not ready for injection")
            return None, None

        _owns_conn = conn is None
        if _owns_conn:
            try:
                conn = kb.connect(board=board)
            except Exception as exc:
                log_line(log_path, f"claim DB error (non-fatal): {type(exc).__name__}: {exc}")
                return None, None

        try:
            kb.release_stale_claims(conn)
            kb.recompute_ready(conn)
            candidate = _select_ready_candidate(conn, args)
            if candidate is None:
                return None, None
            claimed = kb.claim_task(conn, candidate.id, ttl_seconds=args.ttl, claimer=self._claim_lock())
            if claimed is None:
                return None, None
            claim_run_id = claimed.current_run_id
            claim_generation = claimed.generation
            claim_lock = claimed.claim_lock
            self._remember_active_claim(claimed)
        except Exception as exc:
            log_line(log_path, f"claim DB error (non-fatal): {type(exc).__name__}: {exc}")
            return None, None

        # ── Role-content mismatch guard ──────────────────────────────
        # After claim, check if task content keywords suggest a different role.
        # If mismatch detected and not authorized via assist-role, add comment
        # and reclaim the task back to ready.
        authorized_assignees = set(claim_assignees(args))
        task_assignee = getattr(claimed, "assignee", None) or pane_profile
        if task_assignee != pane_profile and task_assignee not in authorized_assignees:
            # Not our task and not an authorized assist — task stealing guard
            # (hermes_listener has its own; this covers codex/codewhale/claude)
            log_line(log_path, f"role guard: reclaiming {claimed.id}: assignee={task_assignee} not in claim_assignees={sorted(authorized_assignees)} (task stealing guard)")
            _reclaim_task_without_signaling_worker(
                conn, claimed.id,
                reason=f"{self.agent_slug} task stealing guard: assignee={task_assignee} profile={pane_profile}",
                expected_run_id=claim_run_id,
                expected_generation=claim_generation,
                expected_claim_lock=claim_lock,
            )
            return None, None

        claim_fence = {
            "expected_run_id": claim_run_id,
            "expected_generation": claim_generation,
            "expected_claim_lock": claim_lock,
        }
        try:
            if claim_run_id is None or not claim_lock:
                raise RuntimeError("claimed task has incomplete run identity")
            task_workspace = kb.resolve_workspace(claimed, board=board)
            expected_branch = claimed.branch_name
            if claimed.workspace_kind == "worktree":
                expected_branch = kb._git_current_branch(Path(task_workspace))  # type: ignore[attr-defined]
                if not expected_branch:
                    raise RuntimeError(
                        f"could not resolve actual branch for worktree {task_workspace}"
                    )
            if not kb.set_workspace_path(
                conn, claimed.id, task_workspace, **claim_fence,
            ):
                raise RuntimeError("stale claim while persisting workspace")
            if claimed.workspace_kind == "worktree":
                if not kb.set_branch_name(
                    conn, claimed.id, expected_branch, **claim_fence,
                ):
                    raise RuntimeError("stale claim while persisting branch")
            resolved_claimed = kb.get_task(conn, claimed.id)
            if resolved_claimed is None:
                raise RuntimeError("claimed task disappeared after workspace resolution")
            persisted_workspace = (
                Path(resolved_claimed.workspace_path or "")
                .expanduser()
                .resolve(strict=False)
            )
            expected_workspace = (
                Path(task_workspace).expanduser().resolve(strict=False)
            )
            if persisted_workspace != expected_workspace:
                raise RuntimeError(
                    f"workspace identity mismatch: resolved={expected_workspace} "
                    f"persisted={persisted_workspace}"
                )
            if (
                resolved_claimed.status != "running"
                or resolved_claimed.current_run_id != claim_run_id
                or resolved_claimed.generation != claim_generation
                or resolved_claimed.claim_lock != claim_lock
                or resolved_claimed.branch_name != expected_branch
            ):
                raise RuntimeError(
                    "task run/branch/generation identity changed after workspace resolution"
                )
            claimed = resolved_claimed
            context = kb.build_worker_context(conn, claimed.id)
            if self.role_context_backend:
                role_context_path = (
                    prompt_dir(
                        task_workspace,
                        board,
                        pane_profile,
                        agent_slug=self.agent_slug,
                    )
                    / f"task-{claimed.id}"
                    / "role-context.json"
                )
                role_context = self.render_effective_role_context(
                    board=board,
                    workspace=task_workspace,
                    pane_profile=pane_profile,
                    task=claimed,
                    output_path=role_context_path,
                )
                context = f"{context}\n\n{role_context}"
            if not kb._set_worker_pid(  # type: ignore[attr-defined]
                conn, claimed.id, os.getpid(), **claim_fence,
            ):
                raise RuntimeError("stale claim while persisting listener pid")
            prompt_path = write_task_prompt(
                agent_name=self.agent_name, agent_slug=self.agent_slug,
                board=board, profile=pane_profile,
                task_id=claimed.id,
                task_assignee=getattr(claimed, "assignee", None) or pane_profile,
                task_title=claimed.title,
                context=context, workspace=task_workspace,
                run_id=claim_run_id,
                generation=claim_generation,
            )
            prompt_claim = kb.get_task(conn, claimed.id)
            if (
                prompt_claim is None
                or prompt_claim.status != "running"
                or prompt_claim.current_run_id != claim_run_id
                or prompt_claim.generation != claim_generation
                or prompt_claim.claim_lock != claim_lock
            ):
                raise RuntimeError("stale claim after writing task prompt")
        except Exception as exc:
            # A deterministic workspace-contract failure (bad base_commit,
            # branch mismatch, missing worktree) will NEVER clear on its own:
            # reclaiming would make the watcher claim the same broken task on
            # the next poll and loop forever without the publisher ever
            # learning.  Notify the publisher on the task and block it as
            # needs_input instead of reclaiming into the retry storm.
            from hermes_cli.kanban_workspace_contract import WorkspaceContractError

            if isinstance(exc, WorkspaceContractError):
                contract_note = (
                    f"{self.agent_slug}-listener could not claim {claimed.id}: "
                    f"deterministic workspace-contract failure "
                    f"({type(exc).__name__}: {exc}). The task record needs a "
                    f"publisher fix (base_commit/branch/workspace) before it can "
                    f"run; blocked as needs_input to stop the claim loop. "
                    f"Fix the record then unblock."
                )
                try:
                    kb.add_comment(
                        conn, claimed.id,
                        f"{self.agent_slug}-interactive-listener",
                        contract_note,
                        **claim_fence,
                    )
                    kb.block_task(
                        conn, claimed.id,
                        reason=contract_note[:400],
                        kind="needs_input",
                        expected_run_id=claim_fence["expected_run_id"],
                        expected_generation=claim_fence["expected_generation"],
                    )
                except Exception as comment_exc:
                    log_line(
                        log_path,
                        f"task {claimed.id} contract-failure notify/block failed: "
                        f"{type(comment_exc).__name__}: {comment_exc}",
                    )
                log_line(
                    log_path,
                    f"task {claimed.id} deterministic workspace-contract failure; "
                    f"commented + blocked needs_input (publisher must fix record); "
                    f"{type(exc).__name__}: {exc}",
                )
                self._clear_active_claim_identity()
                return None, None

            reason = (
                f"{self.agent_slug}-interactive workspace resolution/identity failed: "
                f"{type(exc).__name__}: {exc}"
            )
            reclaimed = _reclaim_task_without_signaling_worker(
                conn, claimed.id, reason=reason, **claim_fence,
            )
            log_line(
                log_path,
                f"task {claimed.id} workspace resolution/identity failed; "
                f"reclaimed={reclaimed}; {type(exc).__name__}: {exc}",
            )
            self._clear_active_claim_identity()
            return None, None

        # Post-claim idle confirmation
        if not self.on_claim_post_confirm(args, log_path):
            log_line(log_path, f"idle confirmation failed; aborting injection for {claimed.id}")
            try:
                _reclaim_task_without_signaling_worker(
                    conn, claimed.id,
                    reason=f"{self.agent_slug}-interactive pane not stably idle before injection",
                    **claim_fence,
                )
            except Exception:
                pass
            self._clear_active_claim_identity()
            return None, None

        task_assignee = getattr(claimed, "assignee", None) or pane_profile
        inject_str = self.inject_text(
            task_id=claimed.id, title=claimed.title,
            assignee=task_assignee, profile=pane_profile,
            prompt_path=prompt_path, board=board,
        )
        inject_str = tag_injected_text(
            inject_str, source_profile="watcher",
        )

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")
        correlation = f"task:{claimed.id}:run:{claim_run_id}:generation:{claim_generation}"
        pre_write_composer = self._pre_write_composer(
            session=str(zellij_session), pane_id=str(zellij_pane_id),
            log_path=log_path,
        )

        ok = zellij_inject(
            session=zellij_session, pane_id=str(zellij_pane_id),
            text=inject_str, expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation,
            log_path=log_path,
        )
        if not ok:
            try:
                _reclaim_task_without_signaling_worker(
                    conn, claimed.id,
                    reason=f"{self.agent_slug}-interactive zellij injection failed",
                    **claim_fence,
                )
            except Exception:
                pass
            self._clear_active_claim_identity()
            return None, None

        if not zellij_submit(
            session=zellij_session, pane_id=str(zellij_pane_id),
            expected_pane_prefix=self.expected_pane_prefix(),
            correlation=correlation, log_path=log_path,
        ):
            try:
                _reclaim_task_without_signaling_worker(
                    conn, claimed.id,
                    reason=f"{self.agent_slug}-interactive zellij submit failed",
                    **claim_fence,
                )
            except Exception:
                pass
            self._clear_active_claim_identity()
            return None, None

        # Hook: subclass post-inject actions (e.g. extra Enter for queued-input TUIs)
        state = self._post_injection_contract(
            args,
            zellij_session=str(zellij_session),
            zellij_pane_id=str(zellij_pane_id),
            log_path=log_path,
            injected_marker=self._delivery_marker(inject_str, correlation),
            pre_write_composer=pre_write_composer,
            correlation=correlation,
        )
        if state not in {"confirmed", "transport_accepted"}:
            reclaimed = _reclaim_task_without_signaling_worker(
                conn, claimed.id,
                reason=(
                    f"{self.agent_slug}-interactive delivery {state} after injection"
                ),
                **claim_fence,
            )
            event = "delivery_requeued" if state == "known_unsubmitted" else "delivery_unknown"
            self._log_delivery_event(
                log_path, state=event.removeprefix("delivery_"),
                correlation_kind="task", correlation_id=claimed.id,
                task_id=claimed.id, run_id=claim_run_id,
                generation=claim_generation, pane_id=str(zellij_pane_id),
                reclaimed=reclaimed,
            )
            if not reclaimed:
                self._log_delivery_event(
                    log_path, state="reclaim_race", correlation_kind="task",
                    correlation_id=claimed.id, task_id=claimed.id,
                    run_id=claim_run_id, generation=claim_generation,
                    pane_id=str(zellij_pane_id),
                )
            self._clear_active_claim_identity()
            return None, None

        self._log_delivery_event(
            log_path,
            state="confirmed" if state == "confirmed" else "transport_accepted",
            correlation_kind="task", correlation_id=claimed.id,
            task_id=claimed.id, run_id=claim_run_id,
            generation=claim_generation, pane_id=str(zellij_pane_id),
        )

        # Post-inject DB ops
        try:
            comment_id = kb.add_comment(
                conn, claimed.id,
                f"{self.agent_slug}-interactive-listener",
                f"Injected into Zellij pane {zellij_pane_id}; prompt file: {prompt_path}",
                **claim_fence,
            )
            heartbeat_ok = kb.heartbeat_worker(
                conn, claimed.id,
                note=f"{self.agent_slug}-interactive injected prompt: {prompt_path}",
                **claim_fence,
            )
            if not comment_id or not heartbeat_ok:
                log_line(
                    log_path,
                    f"post-inject metadata skipped for stale claim {claimed.id} "
                    f"run={claim_run_id}",
                )
        except Exception as exc:
            log_line(log_path, f"post-inject DB op failed (non-fatal): {exc}")

        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(task_id=claimed.id),
            log_path=log_path,
        )
        log_line(log_path, f"claimed+injected {claimed.id}: {claimed.title} prompt={prompt_path}")
        return claimed.id, claim_run_id

    # ── watcher_main ──
    def watcher_main(self, args: argparse.Namespace) -> int:
        self._init_from_args(args)
        board = self._board
        log_path = self._log_path
        workspace = self._workspace

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        if not workspace.exists():
            log_line(log_path, f"workspace does not exist: {workspace}")
            return 2

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")
        if not zellij_session or not zellij_pane_id:
            log_line(log_path, f"missing zellij session/pane id; cannot inject into {self.agent_name} TUI")
            return 2

        # ── Watcher identity lock: adopt (reload exec) or acquire (cold start) ──
        # Ensures exactly one functional claim loop per (board, profile,
        # session, pane).  A second watcher with the same identity exits
        # non-zero after logging the owner PID/start-time.
        try:
            from watcher_runtime import WatcherIdentity, WatcherLock
        except ImportError:
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(
                "watcher_runtime",
                Path(__file__).parent / "watcher_runtime.py",
            )
            assert _spec is not None and _spec.loader is not None
            _wr = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_wr)
            WatcherIdentity = _wr.WatcherIdentity
            WatcherLock = _wr.WatcherLock

        _watcher_identity = WatcherIdentity.from_values(
            board=board,
            profile=args.profile,
            session=zellij_session,
            pane=str(zellij_pane_id),
        )

        # Order matters: after a reload self-exec the inherited lock FD is
        # still held by this process.  We MUST check and adopt it BEFORE any
        # acquire(); otherwise acquire() would open a *new* file description
        # for watcher.lock and self-deadlock against our own inherited flock,
        # exiting with a misleading "lock contended" failure.
        _inherited_lock_fd_str = os.environ.get("HERMES_KANBAN_WATCHER_LOCK_FD", "")
        _reload_nonce_env = os.environ.get("HERMES_KANBAN_RELOAD_NONCE", "")
        _reload_identity_env = os.environ.get("HERMES_KANBAN_RELOAD_IDENTITY", "")
        _is_reload_exec = bool(
            _inherited_lock_fd_str
            and _reload_nonce_env
            and _reload_identity_env == _watcher_identity.digest
        )

        if _inherited_lock_fd_str and not _is_reload_exec:
            log_line(
                log_path,
                "incomplete or identity-mismatched inherited reload state; "
                "fail closed without acquiring another watcher lock",
            )
            return 1

        if _is_reload_exec:
            try:
                _inherited_fd = int(_inherited_lock_fd_str)
                _watcher_lock = WatcherLock.adopt_inherited(
                    _watcher_identity, _inherited_fd,
                )
            except (ValueError, RuntimeError) as exc:
                log_line(
                    log_path,
                    f"FAILED to adopt inherited lock FD={_inherited_lock_fd_str}: "
                    f"{exc}; fail closed (no acquire fallback while the "
                    "inherited lock may still be held)",
                )
                return 1
            log_line(log_path, f"adopted inherited lock FD={_inherited_fd} for reload")
        else:
            try:
                _watcher_lock = WatcherLock.acquire(_watcher_identity)
            except SystemExit:
                log_line(log_path, f"watcher lock contended for {_watcher_identity.digest}; exiting")
                return 1
            log_line(log_path, f"watcher lock acquired: identity={_watcher_identity.digest}")

        # Active claim state — declared early so post-exec adoption can restore it
        active_task: str | None = None
        active_run_id: int | None = None
        active_generation: int | None = None
        active_claim_lock: str | None = None

        # ── SIGUSR1 reload handler (flag-only, no I/O in handler) ──
        _reload_flag = False
        # Becomes True once the post-reload ACK has been written (idle or
        # active path) so exactly one ACK is emitted per reload nonce.
        _reload_ack_written = False

        # ── Resolve watcher_runtime module for reload helpers ──
        _wr_module = sys.modules.get("watcher_runtime")
        if _wr_module is None:
            _wr_module = _wr  # type: ignore[name-defined]

        # ── Post-exec handoff restoration ──
        # Only reached on a genuine reload exec (lock already adopted above).
        # Restore the active claim from the handoff without claiming or
        # injecting again; the success ACK is written later once the
        # post-reload health requirement is met (idle: one healthy DB tick;
        # active: DB claim equality + one heartbeat).
        if _is_reload_exec and _reload_nonce_env:
            root = _wr_module.runtime_root()
            handoff = _wr_module.read_reload_handoff(root, _watcher_identity.digest)
            task_state = None
            if handoff and handoff.task_id:
                try:
                    conn_check = kb.connect(board=board)
                    task_state = (
                        status,
                        current_run_id,
                        current_generation,
                        current_claim_lock,
                    ) = _task_claim_state(conn_check, handoff.task_id)
                    conn_check.close()
                except Exception as exc:
                    log_line(log_path, f"reload handoff DB verification failed: {exc}")
            handoff_error = _wr_module.validate_reload_handoff(
                handoff,
                nonce=_reload_nonce_env,
                identity_digest=_watcher_identity.digest,
                current_pid=os.getpid(),
                task_state=task_state,
            )
            if handoff_error:
                _wr_module.write_reload_ack(
                    root, _watcher_identity.digest,
                    _wr_module.ReloadACK(
                        nonce=_reload_nonce_env,
                        pid=os.getpid(),
                        proc_start_time=_wr_module._proc_start_time(os.getpid()),
                        code_revision=_wr_module._code_revision(),
                        ok=False,
                        error=handoff_error,
                        identity_digest=_watcher_identity.digest,
                    ),
                )
                _wr_module.delete_reload_request(root, _watcher_identity.digest)
                _wr_module.delete_reload_handoff(root, _watcher_identity.digest)
                log_line(log_path, f"reload handoff rejected: {handoff_error}")
                return 1
            if handoff and handoff.task_id:
                log_line(log_path, f"reload handoff verified: task={handoff.task_id} run={handoff.run_id}")
                active_task = handoff.task_id
                active_run_id = handoff.run_id
                active_generation = handoff.generation
                active_claim_lock = handoff.claim_lock
            # Clean up reload state (request/handoff/ACK will be rewritten
            # by the new process as needed)
            _wr_module.cleanup_reload_state(root, _watcher_identity.digest)

        def _handle_sigusr1(signum: int, frame: Any) -> None:  # noqa: ARG001
            nonlocal _reload_flag
            _reload_flag = True

        signal.signal(signal.SIGUSR1, _handle_sigusr1)

        def _try_reload_at_safe_boundary() -> bool:
            """Check for a valid reload request and self-exec if present.

            Returns True if reload was initiated (process will be replaced).
            Returns False if no reload needed, or if reload failed and old
            loop should continue.
            """
            nonlocal _reload_flag
            if not _reload_flag:
                return False
            _reload_flag = False

            root = _wr_module.runtime_root()
            req = _wr_module.read_reload_request(root, _watcher_identity.digest)
            if req is None:
                log_line(log_path, "SIGUSR1 received but no valid reload request; ignoring")
                return False

            log_line(log_path, f"reload requested: nonce={req.nonce}")

            # Verify the requesting owner still matches our lock metadata
            # (PID AND proc start-time — PID reuse is not enough).
            meta_path = _watcher_lock.lock_path.parent / "metadata.json"
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                log_line(log_path, "could not read lock metadata for reload verification; ignoring")
                return False
            if (
                meta.get("pid") != req.owner_pid
                or meta.get("proc_start_time") != req.owner_start_time
            ):
                log_line(
                    log_path,
                    f"reload request owner mismatch: meta pid/start="
                    f"{meta.get('pid')}/{meta.get('proc_start_time')} "
                    f"req pid/start={req.owner_pid}/{req.owner_start_time}; "
                    "ignoring (fail closed)",
                )
                # Distinguishable FAILED ACK so the CLI stops immediately
                # instead of waiting for the ACK timeout.
                _wr_module.write_reload_ack(
                    root, _watcher_identity.digest,
                    _wr_module.ReloadACK(
                        nonce=req.nonce,
                        pid=os.getpid(),
                        proc_start_time=_wr_module._proc_start_time(os.getpid()),
                        code_revision=_wr_module._code_revision(),
                        ok=False,
                        error="owner_mismatch",
                        identity_digest=_watcher_identity.digest,
                    ),
                )
                _wr_module.delete_reload_request(root, _watcher_identity.digest)
                return False

            # Always write a handoff.  An explicit empty handoff distinguishes
            # a healthy idle watcher from a missing/corrupt active handoff.
            handoff = _wr_module.ReloadHandoff(
                nonce=req.nonce,
                identity_digest=_watcher_identity.digest,
                task_id=active_task or "",
                run_id=active_run_id or 0,
                generation=active_generation or 0,
                claim_lock=active_claim_lock or "",
                worker_pid=os.getpid(),
                original_pid=os.getpid(),
            )
            _wr_module.write_reload_handoff(root, handoff)
            if active_task is not None:
                log_line(log_path, f"reload handoff written: task={active_task} run={active_run_id}")

            # Preflight: verify the entry point is importable and parseable
            try:
                subclass_file = Path(sys.modules[type(self).__module__].__file__ or __file__).resolve()
                import py_compile
                py_compile.compile(str(subclass_file), doraise=True)
            except Exception as exc:
                log_line(log_path, f"reload preflight failed: {exc}; continuing old loop")
                _wr_module.delete_reload_request(root, _watcher_identity.digest)
                _wr_module.delete_reload_handoff(root, _watcher_identity.digest)
                # Write FAILED ACK (ok=False so the CLI stops rolling)
                ack = _wr_module.ReloadACK(
                    nonce=req.nonce, pid=os.getpid(),
                    proc_start_time=_wr_module._proc_start_time(os.getpid()),
                    code_revision=_wr_module._code_revision(),
                    ok=False,
                    error="preflight",
                    identity_digest=_watcher_identity.digest,
                )
                _wr_module.write_reload_ack(root, _watcher_identity.digest, ack)
                return False

            # Close DB connection and prepare for exec
            nonlocal _conn
            if _conn is not None:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

            # Make lock FD inheritable
            lock_fd = _watcher_lock.make_inheritable()

            # Set env for post-exec adoption
            env = dict(os.environ)
            env["HERMES_KANBAN_WATCHER_LOCK_FD"] = str(lock_fd)
            env["HERMES_KANBAN_RELOAD_NONCE"] = req.nonce
            env["HERMES_KANBAN_RELOAD_IDENTITY"] = _watcher_identity.digest

            # Build argv (same entry point, same args)
            subclass_file = Path(sys.modules[type(self).__module__].__file__ or __file__).resolve()
            argv = [sys.executable, str(subclass_file)] + sys.argv[1:]

            try:
                log_line(log_path, f"execve for reload: {argv[0]}")
                os.execve(argv[0], argv, env)
            except OSError as exc:
                # execve failed — reopen resources and continue old loop
                log_line(log_path, f"execve failed: {exc}; reopening resources and continuing")
                _watcher_lock.make_non_inheritable()
                # Write FAILED ACK (ok=False so the CLI stops rolling)
                ack = _wr_module.ReloadACK(
                    nonce=req.nonce, pid=os.getpid(),
                    proc_start_time=_wr_module._proc_start_time(os.getpid()),
                    code_revision=_wr_module._code_revision(),
                    ok=False,
                    error="execve_failed",
                    identity_digest=_watcher_identity.digest,
                )
                _wr_module.write_reload_ack(root, _watcher_identity.digest, ack)
                _wr_module.delete_reload_request(root, _watcher_identity.digest)
                _wr_module.delete_reload_handoff(root, _watcher_identity.digest)
                return False

            # Unreachable — execve replaces the process
            return True

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        log_line(
            log_path,
            f"interactive watcher started profile={args.profile} "
            f"claim_assignees={','.join(claim_assignees(args))} board={board} "
            f"workspace={workspace} pane={zellij_session}:{zellij_pane_id} "
            f"poll={poll_s:g}s agent={self.agent_name}",
        )
        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(), log_path=log_path,
        )

        startup_delay = getattr(args, "startup_delay_s", 0) or 0
        if startup_delay > 0:
            time.sleep(startup_delay)

        # ── Persistent DB connection ──
        MAX_CONSECUTIVE_DB_ERRORS = 5
        consecutive_db_errors = 0
        _CONN_RECYCLE_S = 60.0
        _conn: Any = None
        _conn_created_at: float = 0.0

        def _ensure_conn() -> Any:
            nonlocal _conn, _conn_created_at, consecutive_db_errors

            if _conn is not None and (time.time() - _conn_created_at) >= _CONN_RECYCLE_S:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

            if _conn is not None:
                try:
                    _conn.execute("SELECT 1")
                    return _conn
                except sqlite3.OperationalError:
                    try:
                        _conn.close()
                    except Exception:
                        pass
                    _conn = None

            for attempt in range(3):
                try:
                    _conn = kb.connect(board=board)
                    _conn_created_at = time.time()
                    consecutive_db_errors = 0
                    return _conn
                except sqlite3.OperationalError as exc:
                    consecutive_db_errors += 1
                    delay = 2.0 * (2 ** attempt)
                    log_line(log_path, f"DB OperationalError (attempt {attempt+1}/3, consecutive={consecutive_db_errors}): {exc}")
                    time.sleep(delay)
                except sqlite3.DatabaseError as exc:
                    msg = str(exc).lower()
                    if "malformed" in msg or "corrupt" in msg:
                        log_line(
                            log_path,
                            f"DB corruption detected: {exc}; refusing unsafe "
                            "in-watcher REINDEX repair (fail-closed)",
                        )
                    consecutive_db_errors += 1
                    time.sleep(4.0)
                except Exception as exc:
                    consecutive_db_errors += 1
                    log_line(log_path, f"DB connect error: {type(exc).__name__}: {exc}")
                    time.sleep(4.0)
            return None

        last_hb = 0.0

        try:
            while not _STOP:
                # ── Safe-boundary reload checkpoint ──
                if _reload_flag:
                    if _try_reload_at_safe_boundary():
                        pass  # execve succeeded — unreachable
                    # If reload failed, _try_reload already cleaned up;
                    # continue old loop normally

                now = time.time()
                conn = _ensure_conn()
                if conn is None:
                    if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                        log_line(log_path, f"too many consecutive DB errors ({consecutive_db_errors}); stopping watcher")
                        break
                    time.sleep(poll_s)
                    continue

                # Idle-watcher reload ACK: after adoption, one healthy DB tick
                # (a successful connect) is the idle health requirement.
                if _reload_nonce_env and not _reload_ack_written and not active_task:
                    root_ack = _wr_module.runtime_root()
                    ack = _wr_module.ReloadACK(
                        nonce=_reload_nonce_env,
                        pid=os.getpid(),
                        proc_start_time=_wr_module._proc_start_time(os.getpid()),
                        code_revision=_wr_module._code_revision(),
                        ok=True,
                        identity_digest=_watcher_identity.digest,
                    )
                    _wr_module.write_reload_ack(root_ack, _watcher_identity.digest, ack)
                    log_line(log_path, f"reload ACK written (idle): nonce={_reload_nonce_env}")
                    _reload_ack_written = True  # Only write once

                if active_task:
                    try:
                        (
                            status,
                            current_run_id,
                            current_generation,
                            current_claim_lock,
                        ) = _task_claim_state(conn, active_task)
                    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                        consecutive_db_errors += 1
                        log_line(log_path, f"DB error checking task status: {exc}; will retry")
                        time.sleep(min(poll_s, 5.0))
                        continue
                    consecutive_db_errors = 0

                    if (
                        status == "running"
                        and current_run_id == active_run_id
                        and current_generation == active_generation
                        and current_claim_lock == active_claim_lock
                    ):
                        # Cooperative controls preempt ordinary callbacks; durable
                        # result callbacks in turn preempt goal/lifecycle nudges.
                        # Heartbeats still run below even while a pane is busy and
                        # an injection remains queued.
                        injection_preempted = self.pump_control_messages(
                            args, conn, log_path,
                        )
                        if not injection_preempted:
                            injection_preempted = self.pump_result_notifications(
                                args, conn, log_path,
                            )
                        if not injection_preempted:
                            # Hook: subclass may do progress watch / idle reclaim / etc
                            self.on_task_running_monitor(args, conn, active_task, log_path)

                        if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                            try:
                                claim_heartbeat_ok = kb.heartbeat_claim(
                                    conn, active_task,
                                    ttl_seconds=args.ttl,
                                    claimer=self._claim_lock(),
                                    expected_run_id=active_run_id,
                                    expected_generation=active_generation,
                                    expected_claim_lock=active_claim_lock,
                                )
                                worker_heartbeat_ok = claim_heartbeat_ok and kb.heartbeat_worker(
                                    conn, active_task,
                                    note=f"{self.agent_slug}-interactive waiting for complete/block from {self.agent_name} TUI",
                                    expected_run_id=active_run_id,
                                    expected_generation=active_generation,
                                    expected_claim_lock=active_claim_lock,
                                )
                            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                                consecutive_db_errors += 1
                                log_line(log_path, f"DB error on heartbeat: {exc}")
                            else:
                                if not claim_heartbeat_ok or not worker_heartbeat_ok:
                                    log_line(
                                        log_path,
                                        f"active claim became stale during heartbeat: "
                                        f"{active_task} run={active_run_id}",
                                    )
                                    active_task = None
                                    active_run_id = None
                                    active_generation = None
                                    active_claim_lock = None
                                    self._clear_active_claim_identity()
                                    last_hb = 0.0
                                    zellij_rename_pane(
                                        session=zellij_session,
                                        pane_id=str(zellij_pane_id),
                                        name=self.pane_label(),
                                        log_path=log_path,
                                    )
                                    continue
                                last_hb = now
                                # If this is the first heartbeat after a reload adoption, write ACK
                                if _reload_nonce_env and not _reload_ack_written:
                                    root_ack = _wr_module.runtime_root()
                                    ack = _wr_module.ReloadACK(
                                        nonce=_reload_nonce_env,
                                        pid=os.getpid(),
                                        proc_start_time=_wr_module._proc_start_time(os.getpid()),
                                        code_revision=_wr_module._code_revision(),
                                        ok=True,
                                        identity_digest=_watcher_identity.digest,
                                        task_id=active_task,
                                        run_id=active_run_id or 0,
                                    )
                                    _wr_module.write_reload_ack(root_ack, _watcher_identity.digest, ack)
                                    log_line(log_path, f"reload ACK written: nonce={_reload_nonce_env}")
                                    _reload_ack_written = True  # Only write once
                        time.sleep(min(poll_s, 5.0))
                        continue

                    log_line(log_path, f"active task left running state: {active_task} status={status} run={current_run_id}")
                    active_task = None
                    active_run_id = None
                    active_generation = None
                    active_claim_lock = None
                    self._clear_active_claim_identity()
                    last_hb = 0.0
                    zellij_rename_pane(
                        session=zellij_session, pane_id=str(zellij_pane_id),
                        name=self.pane_label(), log_path=log_path,
                    )

                # Control and result queues have priority over backend idle hooks
                # and ready-task discovery. Enqueue never waits for this boundary.
                if self.pump_control_messages(args, conn, log_path):
                    time.sleep(poll_s)
                    continue
                if self.pump_result_notifications(args, conn, log_path):
                    time.sleep(poll_s)
                    continue

                # Hook: subclass idle-loop actions (e.g. auto-dismiss steering)
                self.on_watcher_loop_idle(args, conn, log_path)

                reclaim_orphaned_running_task(args, log_path=log_path, conn=conn)
                active_task, active_run_id = self.claim_and_inject_one(args, log_path=log_path, conn=conn)
                if active_task:
                    active_generation = self._active_generation
                    active_claim_lock = self._active_claim_lock
                    consecutive_db_errors = 0
                    last_hb = 0.0
                    if args.once:
                        continue
                else:
                    active_generation = None
                    active_claim_lock = None
                    self._clear_active_claim_identity()
                    if args.once:
                        log_line(log_path, "no ready task; exiting --once")
                        return 0
                    time.sleep(poll_s)
        finally:
            if _conn is not None:
                try:
                    _conn.close()
                except Exception:
                    pass
            _cleanup_active_claim(
                board=board,
                task_id=active_task,
                expected_run_id=active_run_id,
                expected_generation=active_generation,
                expected_claim_lock=active_claim_lock,
                log_path=log_path,
            )
            self._clear_active_claim_identity()
            try:
                _watcher_lock.release()
            except Exception:
                pass
        log_line(log_path, "interactive watcher stopped")
        return 0

    # ── launcher_main ──
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
            print(f"错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 {self.agent_name} TUI。", file=sys.stderr)
            return 2

        # Store zellij info in args for watcher
        args.zellij_session = zellij_session
        args.zellij_pane_id = zellij_pane_id

        # Build watcher command — use the subclass file, not base_listener.py
        subclass_file = Path(sys.modules[type(self).__module__].__file__ or __file__).resolve()
        watcher_cmd = [
            sys.executable, str(subclass_file),
            "--watch-child",
            "--profile", args.profile,
            "--claim-assignees", ",".join(claim_assignees(args)),
            "--board", board,
            "--workspace", str(workspace),
            "--ttl", str(args.ttl),
            "--zellij-session", zellij_session,
            "--zellij-pane-id", zellij_pane_id,
            "--startup-delay-s", str(getattr(args, "startup_delay_s", 0) or 0),
            "--assist-claim-delay-s", str(assist_claim_delay_s(args)),
        ]
        watcher_cmd.extend(self.build_watcher_extra_args(args))
        for spec in _delay_specs(getattr(args, "assist_claim_delay_for", None)):
            watcher_cmd.extend(["--assist-claim-delay-for", spec])
        if args.poll is not None:
            watcher_cmd.extend(["--poll", str(args.poll)])

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

        print(f"{self.agent_name} interactive kanban mode")
        print(f"  board:     {board}")
        print(f"  profile:   {args.profile}")
        print(f"  claims:    {', '.join(claim_assignees(args))}")
        print(f"  workspace: {workspace}")
        print(f"  pane:      {zellij_session}:{zellij_pane_id}")
        print(f"  log:       {log_path}")
        print("")
        print(f"按 Enter 进入 interactive {self.agent_name}；后台 listener 会按优先级 claim ready 任务并注入到当前 TUI。")
        print(f"{self.agent_slug}-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

        watch_only = getattr(args, "watch_only", False)
        if watch_only:
            print("listener-only 模式：不会启动 TUI，只运行后台 listener 并向指定 Zellij pane 注入任务。")
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

        watcher = subprocess.Popen(
            watcher_cmd,
            stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
            text=True, env=env, start_new_session=True,
        )
        print(f"{self.agent_slug}-kanban listener started: pid={watcher.pid} profile={args.profile} board={board} poll={poll_label}", flush=True)

        continue_session = self.has_saved_sessions(workspace)
        tui_cmd = self.build_tui_cmd(
            workspace, continue_session=continue_session,
            model=args.model if hasattr(args, "model") else None,
            sandbox=args.sandbox if hasattr(args, "sandbox") else None,
            provider=args.provider if hasattr(args, "provider") else None,
            extra_args=getattr(args, f"{self.agent_slug}_arg", None) or [],
        )

        log_line(log_path, f"launcher starting {self.agent_name}: {' '.join(tui_cmd)}")
        rc = 0
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env)
            else:
                try:
                    with open("/dev/tty", "rb") as tty_stdin:
                        rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env, stdin=tty_stdin)
                except OSError:
                    rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env, stdin=sys.stdin)
        finally:
            log_line(log_path, f"{self.agent_name} exited rc={rc}; stopping watcher pid={watcher.pid}")
            try:
                watcher.terminate()
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.kill()
        return rc

    # ── main ──
    def main(self, argv: list[str] | None = None) -> int:
        parser = self._build_parser()
        args = parser.parse_args(argv)
        self._init_from_args(args)

        if getattr(args, "reset_kanban", False):
            board = self._board
            workspace = self._workspace
            reset_ids = reset_kanban_claims(
                board=board, profile=args.profile,
                claim_assignees_list=claim_assignees(args),
                workspace=workspace,
                listener_kind=f"{self.agent_slug}-interactive",
            )
            if reset_ids:
                print(f"reset-kanban reclaimed: {', '.join(reset_ids)}")
            else:
                print("reset-kanban: no matching running claim")
            return 0

        if getattr(args, "watch_child", False):
            return self.watcher_main(args)
        return self.launcher_main(args)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=f"Run {self.agent_name} as a Hermes Kanban listener/profile"
        )
        parser.add_argument("--profile", default=os.environ.get("HERMES_PROFILE") or "", help="Kanban assignee/profile")
        parser.add_argument("--claim-assignees", default=os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES") or "", help="Comma-separated assignees this worker may claim")
        parser.add_argument("--board", default=os.environ.get("HERMES_KANBAN_BOARD"), help="Board slug")
        parser.add_argument("--workspace", default=os.environ.get(f"{self.agent_slug.upper()}_KANBAN_WORKSPACE") or os.environ.get("HERMES_KANBAN_WORKSPACE") or ".", help="Workspace directory")
        parser.add_argument("--poll", type=float, default=None, help="Poll interval override (seconds)")
        parser.add_argument("--ttl", type=int, default=listener_policy.LISTENER_HEALTH_CLAIM_TTL_SECONDS, help="Claim TTL (seconds)")
        parser.add_argument("--model", default=None, help="Optional model override")
        parser.add_argument("--sandbox", default=None, help="Optional sandbox override")
        parser.add_argument("--provider", default=None, help="Optional provider override (e.g. xunfei-relay, openai)")
        parser.add_argument("--assist-claim-delay-s", type=float, default=0.0, help="Delay before claiming secondary assignees")
        parser.add_argument("--assist-claim-delay-for", action="append", default=[], help="Per-assignee assist delay")
        parser.add_argument("--previous-worker-delay-s", type=float,
                            default=float(os.environ.get("HERMES_KANBAN_PREVIOUS_WORKER_DELAY_S", "0")),
                            help="Seconds a non-previous-worker must wait before claiming a reworked task")
        parser.add_argument("--startup-delay-s", type=float, default=8.0, help="Delay before first claim (seconds)")
        parser.add_argument("--once", action="store_true", help="Process at most one task then exit")
        parser.add_argument("--watch-child", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--reset-kanban", action="store_true", help="Reclaim running tasks and exit")
        parser.add_argument("--auto-start", action="store_true", help="Skip the Enter prompt")
        parser.add_argument("--watch-only", action="store_true", help="Only run watcher, don't launch TUI")
        parser.add_argument("--zellij-session", default=os.environ.get("ZELLIJ_SESSION_NAME"), help="Target Zellij session for task injection")
        parser.add_argument("--zellij-pane-id", default=os.environ.get("ZELLIJ_PANE_ID"), help="Target Zellij pane id for task injection")
        # Agent-specific args placeholder
        parser.add_argument(f"--{self.agent_slug}-arg", action="append", default=[], help=f"Extra args for {self.agent_name}")
        return parser
