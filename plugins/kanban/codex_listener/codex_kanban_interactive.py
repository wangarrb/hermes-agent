#!/usr/bin/env python3
"""Interactive Codex + Hermes Kanban listener bridge.

This mode keeps Codex as a visible interactive TUI while a small background
watcher claims Hermes Kanban tasks and injects a short instruction into the
same Zellij pane.  Unlike the non-interactive codex_kanban_listener.py, Codex
itself is responsible for completing/blocking the task via `hermes kanban`.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Source layout: <repo>/plugins/kanban/codex_listener/codex_kanban_interactive.py
HERMES_REPO = Path(__file__).resolve().parents[3]
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402

_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def claim_lock() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:codex-interactive"


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_label()}] {msg}\n")


def prompt_dir(workspace: Path, board: str, pane_profile: str) -> Path:
    # Keep the prompt inside the Codex workspace so sandboxed/read-limited Codex
    # can always open it.  The directory is intentionally hidden and small.
    safe_board = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in board)
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pane_profile)
    return workspace / ".codex-kanban" / safe_board / safe_profile


def claim_assignees(args: argparse.Namespace) -> list[str]:
    """Return assignees this single pane may claim, in priority order."""
    return worker_runtime.claim_assignees_from_args(args, default_profile="planner")


def reset_kanban_claims(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    reason: str = "operator reset-kanban",
) -> list[str]:
    reset_ids = worker_runtime.reset_interactive_claims(
        board=board,
        profile=profile,
        claim_assignees=claim_assignees,
        workspace=workspace,
        listener_kind="codex-interactive",
        reason=reason,
    )
    return list(dict.fromkeys(reset_ids))


def assist_claim_delay_s(args: argparse.Namespace) -> float:
    return worker_runtime.claim_policy_from_args(args, default_profile="planner").assist_claim_delay_s


def _delay_specs(raw: Any) -> list[str]:
    return worker_runtime.split_csv_values(raw)


def _split_delay_spec(spec: str) -> tuple[str, float] | None:
    if "=" in spec:
        key, value = spec.rsplit("=", 1)
    elif ":" in spec:
        key, value = spec.rsplit(":", 1)
    else:
        key, value = "implementer", spec
    key = key.strip()
    if not key:
        key = "implementer"
    try:
        delay_s = max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None
    return key, delay_s


def _split_profile_delay_spec(spec: str) -> tuple[str, str, float] | None:
    if "=" in spec:
        key, value = spec.rsplit("=", 1)
        parts = [part.strip() for part in key.split(":")]
    else:
        parts = [part.strip() for part in spec.split(":")]
        value = parts.pop() if parts else ""
    if not parts:
        profile = "implementer"
        assignee = "implementer"
    elif len(parts) == 1:
        profile = parts[0] or "implementer"
        assignee = "implementer"
    else:
        profile = parts[0] or "implementer"
        assignee = parts[1] or "implementer"
    try:
        delay_s = max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None
    return profile, assignee, delay_s


def assist_claim_delays(args: argparse.Namespace) -> dict[str, float]:
    return worker_runtime.assist_claim_delays_from_args(args)


def assist_claim_profile_delays(args: argparse.Namespace) -> dict[tuple[str, str], float]:
    return worker_runtime.assist_claim_profile_delays_from_args(args)


def assist_claim_delay_for(args: argparse.Namespace, assignee: str) -> float:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="planner")
    return worker_runtime.assist_claim_delay_for(policy, assignee)


def _ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    return worker_runtime.ready_since(conn, task_id, fallback_created_at)


def _assist_candidate_ready(conn, args: argparse.Namespace, task: kb.Task, assignee: str) -> bool:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="planner")
    return worker_runtime.assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee)


def _select_ready_candidate(conn, args: argparse.Namespace) -> kb.Task | None:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="planner")
    return worker_runtime.select_ready_candidate(conn, policy=policy)


def role_guidance(profile: str) -> str:
    """Role-bound guidance shared by all agent backends.

    The visible pane can be Codex, DeepSeek-TUI, or Hermes; the job semantics
    must still come from the Kanban role/profile.
    """
    p = (profile or "").strip().lower()
    common = "职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用 Codex 运行，也要按当前角色工作。"
    per_role = {
        "coordinator": "你是 coordinator：和用户对齐目标，拆分任务，维护 Kanban 流转；除非任务明确很小，否则不要替 planner/implementer/critic 做大段执行。",
        "planner": "你是 planner：负责方案设计、实验计划和任务拆分。输出必须具体到文件路径、函数/类名、命令、预期结果和验收标准。",
        "implementer": "你是 implementer：负责落地执行。先读上下文和相关代码，再小步修改；改完运行最小可行验证，并在结果里说明改了什么、如何验证。",
        "critic": "你是 critic：负责审查、找漏洞和独立验证。不要默认相信 planner/implementer 结论；重点检查证据链、遗漏风险、指标口径和可复现性。最终放行只有 PASS/FAIL，没有带病通过；所有任务范围内能解决的问题都解决后才能 PASS，否则必须 FAIL 并列出具体修复/复审要求。NO_CLAIM 只表示指标不支持 claim，不是缺陷豁免。",
    }
    return common + "\n" + per_role.get(p, f"你当前角色是 {profile}：按该 assignee 的职责完成任务，不要因为运行在 Codex 中而改变角色边界。")


def build_interactive_prompt(
    *,
    board: str,
    pane_profile: str,
    task_assignee: str,
    task: kb.Task,
    context: str,
    workspace: Path,
) -> str:
    metadata = {
        "executor": "codex-interactive",
        "listener": "codex_kanban_interactive",
        "board": board,
        "pane_profile": pane_profile,
        "task_assignee": task_assignee,
        "workspace": str(workspace),
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    complete_template = (
        f"hermes kanban --board {board} complete {task.id} "
        "--summary '<一句话摘要>' "
        "--result '<交付细节/关键结论>' "
        f"--metadata '{metadata_json}'"
    )
    block_template = f"hermes kanban --board {board} block {task.id} '<阻塞原因>'"
    role_prompt = textwrap.indent(role_guidance(task_assignee), "        ")
    assist_note = ""
    if pane_profile != task_assignee:
        assist_note = (
            f"虽然当前 TUI pane/profile 是 {pane_profile}，但这个 Kanban 任务的 assignee 是 "
            f"{task_assignee}；本次必须按 {task_assignee} 职责工作。"
        )
    return textwrap.dedent(
        f"""
        # Hermes Kanban task for role: {task_assignee}

        你现在承担 Hermes Kanban 角色：{task_assignee}。
        底层执行器：interactive Codex；职责/提示词按角色绑定，不按 agent 类型绑定。
        Board: {board}
        Pane/profile: {pane_profile}
        Task assignee/role: {task_assignee}
        Task: {task.id} — {task.title}
        Workspace/root: {workspace}
        {assist_note}

        角色提示：
{role_prompt}

        这个任务已经被后台 listener 原子 claim，其他 worker 不会抢同一个任务。
        交互式模式下，你需要自己在完成后流转 Kanban 状态；listener 只负责 claim、注入、heartbeat。

        必须遵守：
        1. 默认中文输出；路径、命令、技术名词保留英文。
        2. 先读任务上下文，再行动。需要查父任务/评论时可运行：
           hermes kanban --board {board} context {task.id}
           hermes kanban --board {board} show {task.id}
        3. 如果要创建后继任务，必须用真实返回 ID，不要编造：
           hermes kanban --board {board} create ... --json
        4. 完成后必须运行 complete 或 block，不能只在对话里说完成。
        5. 如果任务涉及 Egomotion4D 仓库命令，遵守仓库 AGENTS.md：不要裸跑 python/pytest/pip；本地用 conda run -n egomotion4d env PYTHONPATH=src ...；正式 GPU/benchmark 默认去 gpuserver。

        计划/分工要求（当你的角色是 planner 或任务涉及制定计划/拆分子任务时）：
        - 如果当前任务是 Kanban planner/任务拆分/跨 worker 分工，不要把 Codex 内部 sub-agent 当成唯一执行路径；当前系统规则有时不允许 Codex 主动开 sub-agent，这是正常限制，不是阻塞。
        - 如果用户明确要求你在当前 Codex 会话里直接写程序/实现功能，而不是做 Kanban 分工计划，则允许按 Codex 自身能力使用内部 sub-agent；若系统拒绝开启 sub-agent，再退回当前会话内执行或改用 Hermes Kanban 分工。
        - Kanban 分工场景下，如果需要其他角色参与，不要把全部实现改成在当前 Codex 会话里按红绿循环推进。正确做法是：用 Hermes Kanban 创建明确的后继任务，分配给 `implementer` / `critic` 等可见 pane，由对应 worker 执行。
        - 创建任务必须用真实返回 ID，不要编造。例如：`hermes kanban --board {board} create '<标题>' --assignee implementer --body '<无歧义任务说明>' --json`。创建后在当前 planner 任务结果里列出这些真实 task_id。
        - 计划必须让低能力 coding agent（如 DeepSeek-TUI implementer）也能无歧义理解执行。每个步骤要给出：具体文件路径、具体函数/类名、具体命令、预期输出/行为。不要写"修改相关文件"这种模糊描述。
        - 预估每个步骤可能踩的坑，明确写出来让执行者避坑。例如：某函数参数顺序容易搞反、某配置项必须用特定值否则会静默失败、某步骤依赖上一步的特定输出格式。
        - 计划必须可验证：每个步骤完成后，给出验证命令或检查点（如：运行某测试、检查某文件存在且包含某内容、对比某指标值）。执行者做完一步就能自己确认对不对，而不是做完一整串才发现方向错了。

        完成命令模板：
        {complete_template}

        阻塞命令模板：
        {block_template}

        下面是 Hermes Kanban worker context：

        {context}
        """
    ).strip() + "\n"


def write_task_prompt(
    *,
    workspace: Path,
    board: str,
    pane_profile: str,
    task: kb.Task,
    context: str,
) -> Path:
    task_assignee = getattr(task, "assignee", None) or pane_profile
    d = prompt_dir(workspace, board, pane_profile)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task.id}.md"
    p.write_text(
        build_interactive_prompt(
            board=board,
            pane_profile=pane_profile,
            task_assignee=task_assignee,
            task=task,
            context=context,
            workspace=workspace,
        ),
        encoding="utf-8",
    )
    return p


def zellij_inject(*, session: str, pane_id: str, text: str, log_path: Path) -> bool:
    """Type text into a target Zellij pane and press Enter."""
    try:
        subprocess.run(
            ["zellij", "--session", session, "action", "write-chars", "-p", str(pane_id), text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Zellij returns after queuing bytes, while Codex may still be updating
        # its multi-line composer.  A tiny delay makes the following Enter act
        # as submit instead of racing with the pasted text.
        time.sleep(0.3)
        subprocess.run(
            ["zellij", "--session", session, "action", "send-keys", "-p", str(pane_id), "Enter"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij injection failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


def zellij_rename_pane(*, session: str, pane_id: str, name: str, log_path: Path) -> bool:
    """Set a persistent Zellij pane title so listener state is visible."""
    try:
        subprocess.run(
            ["zellij", "--session", session, "action", "rename-pane", "-p", str(pane_id), name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij rename-pane failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


def _task_status(conn, task_id: str) -> tuple[str | None, int | None]:
    row = conn.execute("SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return None, None
    return row["status"], row["current_run_id"]


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _workspace_matches(row_workspace: str | None, workspace: Path) -> bool:
    if not row_workspace:
        return True
    try:
        return Path(row_workspace).expanduser().resolve() == workspace
    except Exception:
        return False


def _skip_reclaim_signal(pid: int, signum: int) -> None:  # noqa: ARG001
    """Avoid killing the interactive watcher while it repairs Kanban state."""
    raise ProcessLookupError(pid)


def _reclaim_task_without_signaling_worker(conn, task_id: str, *, reason: str) -> bool:
    return kb.reclaim_task(conn, task_id, reason=reason, signal_fn=_skip_reclaim_signal)


def reclaim_orphaned_running_task(args: argparse.Namespace, *, log_path: Path, conn: Any = None) -> bool:
    """Requeue this pane's stale running Codex task so it can receive a fresh prompt.

    Interactive Codex cannot safely resume a prompt that was injected into a
    previous Zellij session.  On startup, only repair tasks owned by this pane's
    primary profile whose old watcher PID is dead.
    """
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    host = socket.gethostname()
    if conn is None:
        conn = kb.connect(board=board)
    rows = conn.execute(
        """
        SELECT id, claim_lock, worker_pid, workspace_path
          FROM tasks
         WHERE assignee = ?
           AND status = 'running'
           AND claim_lock IS NOT NULL
         ORDER BY started_at ASC, id ASC
        """,
        (args.profile,),
    ).fetchall()
    for row in rows:
        old_lock = row["claim_lock"] or ""
        if not old_lock.endswith(":codex-interactive"):
            continue
        lock_host = old_lock.split(":", 1)[0] if ":" in old_lock else ""
        if lock_host != host:
            continue
        if not _workspace_matches(row["workspace_path"], workspace):
            continue
        old_pid = int(row["worker_pid"]) if row["worker_pid"] else None
        if _pid_alive(old_pid):
            continue
        task_id = row["id"]
        if _reclaim_task_without_signaling_worker(
            conn,
            task_id,
            reason=(
                "codex-interactive startup found orphaned running task; "
                "requeueing for fresh prompt after watcher/session restart"
            ),
        ):
            log_line(log_path, f"reclaimed orphaned running task {task_id} old_pid={old_pid}")
            return True
    return False


def _cleanup_active_claim(*, board: str, task_id: str | None, run_id: int | None, log_path: Path) -> None:
    if not task_id:
        return
    try:
        with kb.connect(board=board) as conn:
            status, current_run_id = _task_status(conn, task_id)
            if status == "running" and (run_id is None or current_run_id == run_id):
                if _reclaim_task_without_signaling_worker(
                    conn,
                    task_id,
                    reason="codex-interactive listener stopped before completion",
                ):
                    log_line(log_path, f"reclaimed active task on stop: {task_id}")
    except Exception as exc:  # best-effort cleanup only
        log_line(log_path, f"cleanup active claim failed for {task_id}: {type(exc).__name__}: {exc}")


def claim_and_inject_one(args: argparse.Namespace, *, log_path: Path, conn: Any = None) -> tuple[str | None, int | None]:
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    pane_profile = args.profile
    # Reuse caller's connection or open a new one (Plan 2).
    if conn is None:
        conn = kb.connect(board=board)
    kb.release_stale_claims(conn)
    kb.recompute_ready(conn)
    candidate = _select_ready_candidate(conn, args)
    if candidate is None:
        return None, None
    claimed = kb.claim_task(conn, candidate.id, ttl_seconds=args.ttl, claimer=claim_lock())
    if claimed is None:
        return None, None
    # Interactive Codex runs in a stable project root.  Persist that root as
    # the task workspace so humans and downstream tools see the same path.
    kb.set_workspace_path(conn, claimed.id, workspace)
    claimed = kb.get_task(conn, claimed.id) or claimed
    try:
        kb._set_worker_pid(conn, claimed.id, os.getpid())  # type: ignore[attr-defined]
    except Exception:
        pass
    context = kb.build_worker_context(conn, claimed.id)

    prompt_path = write_task_prompt(
        workspace=workspace,
        board=board,
        pane_profile=pane_profile,
        task=claimed,
        context=context,
    )
    task_assignee = getattr(claimed, "assignee", None) or pane_profile
    inject_text = (
        f"Hermes Kanban 已领取任务 {claimed.id}: {claimed.title}\n"
        f"Pane/profile: {pane_profile}; task assignee/role: {task_assignee}\n"
        f"请读取并执行这个任务文件：{prompt_path}\n"
        f"完成后必须运行 `hermes kanban --board {board} complete {claimed.id} ...` "
        f"或 `hermes kanban --board {board} block {claimed.id} ...`。"
    )
    ok = zellij_inject(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        text=inject_text,
        log_path=log_path,
    )
    if not ok:
        _reclaim_task_without_signaling_worker(
            conn,
            claimed.id,
            reason="codex-interactive zellij injection failed",
        )
        return None, None

    # Post-inject DB ops reuse the same connection.
    kb.add_comment(
        conn,
        claimed.id,
        "codex-interactive-listener",
        f"Injected into Zellij pane {args.zellij_pane_id}; prompt file: {prompt_path}",
    )
    kb.heartbeat_worker(
        conn,
        claimed.id,
        note=f"codex-interactive injected prompt: {prompt_path}",
        expected_run_id=claimed.current_run_id,
    )
    zellij_rename_pane(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        name=f"{pane_profile}-codex running {claimed.id}",
        log_path=log_path,
    )
    log_line(log_path, f"claimed+injected {claimed.id}: {claimed.title} prompt={prompt_path}")
    return claimed.id, claimed.current_run_id


def watcher_main(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    board = args.board or kb.get_current_board() or "default"
    log_path = kb.worker_logs_dir(board=board) / f"codex-interactive-{args.profile}.log"
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        log_line(log_path, f"workspace does not exist: {workspace}")
        return 2
    if not args.zellij_session or not args.zellij_pane_id:
        log_line(log_path, "missing zellij session/pane id; cannot inject into Codex TUI")
        return 2

    poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
    log_line(
        log_path,
        f"interactive watcher started profile={args.profile} claim_assignees={','.join(claim_assignees(args))} board={board} workspace={workspace} "
        f"pane={args.zellij_session}:{args.zellij_pane_id} poll={poll_s:g}s",
    )
    zellij_rename_pane(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        name=f"{args.profile}-codex listening",
        log_path=log_path,
    )
    if args.startup_delay_s > 0:
        time.sleep(args.startup_delay_s)

    # ── Persistent connection (Plan 2) ──────────────────────────────
    # Instead of opening a new connection on every loop iteration (which
    # causes repeated PRAGMA setup, WAL lock churn, and FD leaks because
    # ``with kb.connect()`` only commits but never closes), we keep one
    # connection open for the lifetime of the watcher and reconnect
    # periodically or on error.
    MAX_CONSECUTIVE_DB_ERRORS = 5
    consecutive_db_errors = 0
    _CONN_RECYCLE_S = 60.0  # reconnect every 60 s to release WAL locks
    _conn: Any = None
    _conn_created_at: float = 0.0

    def _ensure_conn() -> Any:
        """Return a live DB connection, reconnecting if necessary."""
        nonlocal _conn, _conn_created_at, consecutive_db_errors

        # Recycle stale connection
        if _conn is not None and (time.time() - _conn_created_at) >= _CONN_RECYCLE_S:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

        if _conn is not None:
            # Quick liveness probe
            try:
                _conn.execute("SELECT 1")
                return _conn
            except sqlite3.OperationalError:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

        # Open a new connection with retry
        for attempt in range(3):
            try:
                _conn = kb.connect(board=board)
                _conn_created_at = time.time()
                consecutive_db_errors = 0
                return _conn
            except sqlite3.OperationalError as exc:
                consecutive_db_errors += 1
                delay = 2.0 * (2 ** attempt)
                log_line(
                    log_path,
                    f"DB OperationalError (attempt {attempt+1}/3, "
                    f"consecutive={consecutive_db_errors}): {exc}; "
                    f"retrying in {delay:.0f}s",
                )
                time.sleep(delay)
            except sqlite3.DatabaseError as exc:
                # Index corruption ("database disk image is malformed") is
                # recoverable via REINDEX.  OperationalError is handled above.
                msg = str(exc).lower()
                if "malformed" in msg or "corrupt" in msg:
                    log_line(
                        log_path,
                        f"DB corruption detected: {exc}; attempting REINDEX repair",
                    )
                    try:
                        repair_conn = sqlite3.connect(
                            str(kb.kanban_db_path(board=board)),
                            timeout=120.0,
                        )
                        repair_conn.execute("PRAGMA integrity_check")
                        repair_conn.execute("REINDEX")
                        repair_conn.close()
                        log_line(log_path, "REINDEX repair succeeded; retrying connect")
                    except Exception as repair_exc:
                        log_line(log_path, f"REINDEX repair failed: {repair_exc}")
                consecutive_db_errors += 1
                time.sleep(4.0)
            except Exception as exc:
                consecutive_db_errors += 1
                log_line(log_path, f"DB connect error: {type(exc).__name__}: {exc}")
                time.sleep(4.0)

        # All retries exhausted — return None, caller will skip this tick
        return None

    active_task: str | None = None
    active_run_id: int | None = None
    last_hb = 0.0
    try:
        while not _STOP:
            now = time.time()
            conn = _ensure_conn()
            if conn is None:
                if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                    log_line(log_path, f"too many consecutive DB errors ({consecutive_db_errors}); stopping watcher")
                    break
                time.sleep(poll_s)
                continue
            if active_task:
                try:
                    status, current_run_id = _task_status(conn, active_task)
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                    consecutive_db_errors += 1
                    log_line(log_path, f"DB error checking task status: {exc}; will retry")
                    time.sleep(min(poll_s, 5.0))
                    continue
                consecutive_db_errors = 0
                if status == "running" and (active_run_id is None or current_run_id == active_run_id):
                    if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                        try:
                            kb.heartbeat_claim(conn, active_task, ttl_seconds=args.ttl, claimer=claim_lock())
                            kb.heartbeat_worker(
                                conn,
                                active_task,
                                note="codex-interactive waiting for complete/block from Codex TUI",
                                expected_run_id=active_run_id,
                            )
                        except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                            consecutive_db_errors += 1
                            log_line(log_path, f"DB error on heartbeat: {exc}")
                        else:
                            last_hb = now
                    time.sleep(min(poll_s, 5.0))
                    continue
                log_line(log_path, f"active task left running state: {active_task} status={status} run={current_run_id}")
                active_task = None
                active_run_id = None
                last_hb = 0.0
                zellij_rename_pane(
                    session=args.zellij_session,
                    pane_id=args.zellij_pane_id,
                    name=f"{args.profile}-codex listening",
                    log_path=log_path,
                )

            reclaim_orphaned_running_task(args, log_path=log_path, conn=conn)
            active_task, active_run_id = claim_and_inject_one(args, log_path=log_path, conn=conn)
            if active_task:
                consecutive_db_errors = 0
                last_hb = 0.0
                if args.once:
                    # Keep heartbeating the claimed task until it is completed or blocked.
                    continue
            else:
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
        _cleanup_active_claim(board=board, task_id=active_task, run_id=active_run_id, log_path=log_path)
    log_line(log_path, "interactive watcher stopped")
    return 0


def has_saved_codex_sessions(workspace: Path) -> bool:
    """Check whether Codex has saved interactive sessions for this workspace."""
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return False
    workspace_s = str(workspace.expanduser().resolve())
    try:
        for path in sorted(sessions_root.glob("**/*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    # session_meta is normally the first line and contains payload.cwd.
                    for idx, line in enumerate(f):
                        if idx > 5:
                            break
                        if workspace_s in line:
                            return True
            except Exception:
                continue
    except Exception:
        return False
    return False


def build_codex_cmd(args: argparse.Namespace) -> list[str]:
    workspace = Path(args.workspace).expanduser().resolve()
    workspace_s = str(workspace)
    startup_prompt = getattr(args, "startup_prompt", None)
    if args.continue_session and has_saved_codex_sessions(workspace):
        # resume --last: do NOT pass --cd (not supported by the resume
        # subcommand) and do NOT pass startup_prompt (it would be
        # interpreted as SESSION_ID).  All kanban context is in env vars.
        cmd = [args.codex_bin, "resume", "--last"]
    else:
        cmd = [args.codex_bin, "--cd", workspace_s]
        if startup_prompt:
            cmd.append(str(startup_prompt))  # file path as PROMPT positional arg
    if args.no_alt_screen:
        cmd.append("--no-alt-screen")
    if args.model:
        cmd.extend(["--model", args.model])
    # Auto-approve all tool calls: -a never means never ask for approval
    cmd.extend(["-a", "never"])
    if args.sandbox:
        cmd.extend(["--sandbox", args.sandbox])
    for extra in args.codex_arg or []:
        cmd.append(extra)
    return cmd


def launcher_main(args: argparse.Namespace) -> int:
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        print(f"错误: workspace 不存在: {workspace}", file=sys.stderr)
        return 2
    if not (workspace / ".git").exists():
        print(f"警告: {workspace} 看起来不是 git repo；Codex 交互模式可能拒绝执行。", file=sys.stderr)

    zellij_session = args.zellij_session or os.environ.get("ZELLIJ_SESSION_NAME")
    zellij_pane_id = args.zellij_pane_id or os.environ.get("ZELLIJ_PANE_ID")
    if not zellij_session or not zellij_pane_id:
        print("错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 interactive Codex。", file=sys.stderr)
        print("请在 zellij pane 内运行，或显式传 --zellij-session / --zellij-pane-id。", file=sys.stderr)
        return 2

    log_path = kb.worker_logs_dir(board=board) / f"codex-interactive-{args.profile}.log"
    watcher_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--watch-child",
        "--profile",
        args.profile,
        "--claim-assignees",
        ",".join(claim_assignees(args)),
        "--board",
        board,
        "--workspace",
        str(workspace),
        "--ttl",
        str(args.ttl),
        "--zellij-session",
        zellij_session,
        "--zellij-pane-id",
        zellij_pane_id,
        "--startup-delay-s",
        str(args.startup_delay_s),
        "--assist-claim-delay-s",
        str(args.assist_claim_delay_s),
    ]
    for spec in _delay_specs(getattr(args, "assist_claim_delay_for", None)):
        watcher_cmd.extend(["--assist-claim-delay-for", spec])
    for spec in _delay_specs(getattr(args, "assist_claim_profile_delay", None)):
        watcher_cmd.extend(["--assist-claim-profile-delay", spec])
    if args.poll is not None:
        watcher_cmd.extend(["--poll", str(args.poll)])

    poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
    poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

    print("Codex interactive kanban mode")
    print(f"  board:     {board}")
    print(f"  profile:   {args.profile}")
    print(f"  claims:    {', '.join(claim_assignees(args))}")
    print(f"  workspace: {workspace}")
    print(f"  pane:      {zellij_session}:{zellij_pane_id}")
    print(f"  log:       {log_path}")
    print("")
    print(f"按 Enter 进入 interactive Codex；后台 listener 会按优先级 claim {', '.join(claim_assignees(args))} ready 任务并注入到当前 Codex。")
    print("退出 Codex 后 listener 会一起停止；如果任务未 complete/block，会自动 reclaim。")
    print(f"codex-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

    if args.watch_only:
        print("listener-only 模式：不会启动 Codex TUI，只运行后台 listener 并向指定 Zellij pane 注入任务。")
        return watcher_main(args)

    if not args.auto_start:
        try:
            input()
        except EOFError:
            pass

    env = os.environ.copy()
    env.update(
        {
            "HERMES_KANBAN_BOARD": board,
            "HERMES_KANBAN_PROFILE": args.profile,
            "HERMES_KANBAN_CLAIM_ASSIGNEES": ",".join(claim_assignees(args)),
            "HERMES_KANBAN_WORKSPACE": str(workspace),
        }
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")
    log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")
    watcher = subprocess.Popen(
        watcher_cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    print(f"codex-kanban listener started: pid={watcher.pid} profile={args.profile} board={board} poll={poll_label} pane={zellij_session}:{zellij_pane_id} log={log_path}", flush=True)
    codex_cmd = build_codex_cmd(args)
    log_line(log_path, f"launcher starting codex: {' '.join(codex_cmd)}")
    rc = 0
    try:
        if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
            rc = subprocess.call(codex_cmd, cwd=str(workspace), env=env)
        else:
            try:
                with open("/dev/tty", "rb") as tty_stdin:
                    rc = subprocess.call(codex_cmd, cwd=str(workspace), env=env, stdin=tty_stdin)
            except OSError:
                rc = subprocess.call(codex_cmd, cwd=str(workspace), env=env, stdin=sys.stdin)
    finally:
        log_line(log_path, f"codex exited rc={rc}; stopping watcher pid={watcher.pid}")
        try:
            watcher.terminate()
            watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:
            watcher.kill()
            watcher.wait(timeout=5)
        log_f.close()
    return int(rc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive Codex while listening to Hermes Kanban")
    parser.add_argument("--profile", default=os.environ.get("HERMES_PROFILE") or "planner", help="Kanban assignee/profile to claim (default: planner)")
    parser.add_argument("--claim-assignees", default=os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES"), help="Comma-separated Kanban assignees this pane may claim, in priority order; defaults to --profile")
    parser.add_argument("--board", default=os.environ.get("HERMES_KANBAN_BOARD"), help="Board slug; defaults to Hermes current board")
    parser.add_argument("--workspace", default=os.environ.get("CODEX_KANBAN_WORKSPACE") or os.getcwd(), help="Codex workspace/git repo root")
    parser.add_argument("--poll", type=float, default=None, help="Ready-task poll interval override; default uses shared Hermes listener policy")
    parser.add_argument("--ttl", type=int, default=listener_policy.LISTENER_HEALTH_CLAIM_TTL_SECONDS, help="Claim TTL seconds")
    parser.add_argument("--startup-delay-s", type=float, default=8.0, help="Delay after Codex launch before injecting first task")
    parser.add_argument("--assist-claim-delay-s", type=float, default=float(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAY_S") or 0.0), help="Only claim non-primary assist assignees after their ready task has waited this many seconds. 0 disables the delay.")
    parser.add_argument(
        "--assist-claim-delay-for",
        action="append",
        default=_delay_specs(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAYS")),
        metavar="ASSIGNEE=SECONDS",
        help="Override assist claim delay for a target assignee in this listener, e.g. implementer=180. Repeatable.",
    )
    parser.add_argument(
        "--assist-claim-profile-delay",
        action="append",
        default=_delay_specs(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS")),
        metavar="PROFILE:ASSIGNEE=SECONDS",
        help="Profile-qualified assist delay rule, e.g. backup_immplementer:implementer=300. Repeatable.",
    )
    parser.add_argument("--model", default=None, help="Optional Codex model override")
    parser.add_argument("--no-continue", dest="continue_session", action="store_false", default=True, help="Start a fresh Codex TUI instead of resuming the most recent session for this workspace")
    parser.add_argument("--sandbox", default=os.environ.get("CODEX_KANBAN_SANDBOX") or "danger-full-access", help="Optional Codex sandbox override (default: danger-full-access to avoid bubblewrap failures in Zellij)")
    parser.add_argument("--codex-bin", default="codex", help="Codex executable path/name")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra raw arg passed to interactive codex; repeatable")
    parser.add_argument("--alt-screen", dest="no_alt_screen", action="store_false", default=True, help="Allow Codex alt-screen instead of the default --no-alt-screen")
    parser.add_argument("--zellij-session", default=os.environ.get("ZELLIJ_SESSION_NAME"), help="Target Zellij session for task injection")
    parser.add_argument("--zellij-pane-id", default=os.environ.get("ZELLIJ_PANE_ID"), help="Target Zellij pane id for task injection")
    parser.add_argument("--watch-only", action="store_true", help="Only run the background listener; do not launch interactive Codex TUI")
    parser.add_argument("--auto-start", action="store_true", help="Skip the 'press Enter' prompt and start immediately (for scripted/automated launches)")
    parser.add_argument("--watch-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reset-kanban", action="store_true", help="Reclaim this profile/pane's running interactive Kanban task(s) and exit")
    parser.add_argument("--once", action="store_true", help="Watcher child: process at most one task")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_kanban:
        board = args.board or kb.get_current_board() or "default"
        workspace = Path(args.workspace).expanduser().resolve()
        reset_ids = reset_kanban_claims(
            board=board,
            profile=args.profile,
            claim_assignees=claim_assignees(args),
            workspace=workspace,
        )
        if reset_ids:
            print(f"reset-kanban reclaimed: {', '.join(reset_ids)}")
        else:
            print("reset-kanban: no matching running claim")
        return 0
    if args.watch_child:
        return watcher_main(args)
    return launcher_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
