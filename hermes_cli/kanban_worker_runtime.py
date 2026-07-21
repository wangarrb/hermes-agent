"""Shared runtime helpers for visible interactive Kanban workers.

Codex, DeepSeek-TUI, and Hermes visible listeners all need the same control
plane rules for multi-role claiming: primary role first, optional assist roles
after a configurable delay, and operator reset by board/workspace/listener
kind.  Keep those rules here so each adapter only owns its UI-specific work
such as prompt files, pane safety checks, and process launching.
"""
from __future__ import annotations

import argparse
import os
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_listener_policy as listener_policy

@dataclass(frozen=True)
class ClaimPolicy:
    """Priority-ordered claim policy for one visible worker pane."""

    profile: str
    claim_assignees: list[str] = field(default_factory=list)
    assist_claim_delay_s: float = 0.0
    assist_claim_delays: dict[str, float] = field(default_factory=dict)
    assist_claim_profile_delays: dict[tuple[str, str], float] = field(default_factory=dict)
    previous_worker_delay_s: float = 0.0


def _dedupe_nonempty(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in out:
            out.append(value)
    return out


def split_csv_values(raw: Any) -> list[str]:
    """Return comma-split non-empty values from strings or iterables."""
    values: list[str] = []
    if isinstance(raw, str):
        values.extend(raw.split(","))
    elif raw:
        for item in raw:
            values.extend(str(item).split(","))
    return [item.strip() for item in values if item and item.strip()]


def claim_assignees_from_args(args: argparse.Namespace, *, default_profile: str) -> list[str]:
    """Return assignees this single pane may claim, in priority order."""
    profile = str(getattr(args, "profile", "") or default_profile)
    raw_values = split_csv_values(getattr(args, "claim_assignees", None))
    return _dedupe_nonempty([profile, *raw_values]) or [default_profile]


def _split_delay_spec(spec: str) -> tuple[str, float] | None:
    if "=" in spec:
        key, value = spec.rsplit("=", 1)
    elif ":" in spec:
        key, value = spec.rsplit(":", 1)
    else:
        key, value = "implementer", spec
    key = key.strip() or "implementer"
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


def _float_arg(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value if value is not None else default))
    except (TypeError, ValueError):
        return default


def assist_claim_delays_from_args(args: argparse.Namespace) -> dict[str, float]:
    """Parse per-assignee assist delays from env and adapter args."""
    out: dict[str, float] = {}
    specs = [
        *split_csv_values(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAYS")),
        *split_csv_values(getattr(args, "assist_claim_delay_for", None)),
    ]
    for spec in specs:
        parsed = _split_delay_spec(spec)
        if parsed is None:
            continue
        assignee, delay_s = parsed
        if ":" not in assignee:
            out[assignee] = delay_s
    return out


def assist_claim_profile_delays_from_args(args: argparse.Namespace) -> dict[tuple[str, str], float]:
    """Parse profile-qualified assist delays from env and adapter args."""
    out: dict[tuple[str, str], float] = {}
    specs = [
        *split_csv_values(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS")),
        *split_csv_values(getattr(args, "assist_claim_profile_delay", None)),
    ]
    for spec in specs:
        parsed = _split_profile_delay_spec(spec)
        if parsed is None:
            continue
        profile, assignee, delay_s = parsed
        if profile and assignee:
            out[(profile, assignee)] = delay_s
    return out


def claim_policy_from_args(args: argparse.Namespace, *, default_profile: str) -> ClaimPolicy:
    """Build a normalized claim policy from an adapter argparse namespace."""
    profile = str(getattr(args, "profile", "") or default_profile)
    return ClaimPolicy(
        profile=profile,
        claim_assignees=claim_assignees_from_args(args, default_profile=default_profile),
        assist_claim_delay_s=_float_arg(getattr(args, "assist_claim_delay_s", 0.0)),
        assist_claim_delays=assist_claim_delays_from_args(args),
        assist_claim_profile_delays=assist_claim_profile_delays_from_args(args),
        previous_worker_delay_s=_float_arg(getattr(args, "previous_worker_delay_s", 0.0)),
    )


def assist_claim_delay_for(policy: ClaimPolicy, assignee: str) -> float:
    """Return how long an assist pane must wait before claiming assignee."""
    if assignee == policy.profile:
        return 0.0
    local_delay = policy.assist_claim_delays.get(assignee)
    if local_delay is not None:
        return local_delay
    profile_delay = policy.assist_claim_profile_delays.get((policy.profile, assignee))
    if profile_delay is not None:
        return profile_delay
    return policy.assist_claim_delay_s


def ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    """Return the most recent timestamp where a task became claimable."""
    row = conn.execute(
        "SELECT MAX(created_at) AS ts FROM task_events "
        "WHERE task_id=? AND kind IN ("
        "'created','promoted','assigned','reclaimed','unblocked',"
        "'rework_hold_released')",
        (task_id,),
    ).fetchone()
    try:
        return int(row["ts"] or fallback_created_at or 0)
    except Exception:
        return int(fallback_created_at or 0)


def _previous_worker_profile(conn, task_id: str) -> str | None:
    """Return the profile of the most recent execution attempt for *task_id*.

    Used for the previous-worker priority window: a task returned for rework
    should be claimable first by the worker who last executed it.
    """
    row = conn.execute(
        "SELECT profile FROM task_runs "
        "WHERE task_id=? "
        "ORDER BY started_at DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    if row is None:
        return None
    return str(row["profile"] or "").strip() or None


def assist_candidate_ready(
    conn,
    *,
    policy: ClaimPolicy,
    task: kb.Task,
    assignee: str,
    now_fn=time.time,
) -> bool:
    """Return True when *task* is eligible for the pane's claim policy.

    If ``previous_worker_delay_s > 0`` and the task was previously executed by
    a different profile, this pane must wait *previous_worker_delay_s* seconds
    before it can claim the task.  The previous worker pane has no delay.
    """
    delay_s = assist_claim_delay_for(policy, assignee)
    if delay_s > 0:
        ready_age = now_fn() - ready_since(conn, task.id, int(getattr(task, "created_at", 0) or 0))
        if ready_age < delay_s:
            return False

    prev_delay = getattr(policy, "previous_worker_delay_s", 0.0)
    if prev_delay > 0:
        prev_profile = _previous_worker_profile(conn, task.id)
        if prev_profile is not None and prev_profile != policy.profile:
            ready_age = now_fn() - ready_since(conn, task.id, int(getattr(task, "created_at", 0) or 0))
            if ready_age < prev_delay:
                return False

    return True


def select_ready_candidate(conn, *, policy: ClaimPolicy) -> kb.Task | None:
    """Return the first ready task allowed by *policy*, without claiming it."""
    for assignee in policy.claim_assignees:
        ready = kb.list_tasks(
            conn,
            assignee=assignee,
            status="ready",
            limit=listener_policy.READY_TASK_SCAN_LIMIT,
        )
        for task in ready:
            if assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee):
                return task
    return None


def claim_ready_candidate(
    conn,
    *,
    policy: ClaimPolicy,
    ttl_seconds: int,
    claimer: str | None = None,
) -> kb.Task | None:
    """Claim the first ready task allowed by *policy*.

    The selection and claim are intentionally in one helper so every visible
    worker backend uses the same multi-role fairness rules.  If another worker
    wins the CAS between list and claim, keep scanning instead of returning
    idle spuriously.
    """
    for assignee in policy.claim_assignees:
        ready = kb.list_tasks(
            conn,
            assignee=assignee,
            status="ready",
            limit=listener_policy.READY_TASK_SCAN_LIMIT,
        )
        for task in ready:
            if not assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee):
                continue
            claimed = kb.claim_task(
                conn,
                task.id,
                ttl_seconds=ttl_seconds,
                claimer=claimer,
            )
            if claimed is not None:
                return claimed
    return None


def _workspace_matches(row_workspace: str | None, workspace: Path | str | None) -> bool:
    if workspace is None:
        return True
    if not row_workspace:
        return False
    try:
        return Path(row_workspace).expanduser().resolve() == Path(workspace).expanduser().resolve()
    except Exception:
        return False


def find_current_claim(
    conn,
    *,
    policy: ClaimPolicy,
    workspace: Path,
    listener_kind: str,
    owner: str | None = None,
) -> kb.Task | None:
    """Return an existing running claim for this pane, if any.

    Self-poll workers may call ``kanban next`` repeatedly while still working
    the current task.  Returning the current claim prevents one pane from
    accidentally claiming multiple tasks.
    """
    targets = [a for a in dict.fromkeys(policy.claim_assignees or [policy.profile]) if a]
    if not targets:
        return None
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status='running' "
        f"AND assignee IN ({','.join('?' for _ in targets)}) "
        "ORDER BY started_at ASC, id ASC",
        tuple(targets),
    ).fetchall()
    safe_owner = safe_path_component(owner or policy.profile)
    suffix = f":{safe_owner}:{listener_kind}" if listener_kind else ""
    for row in rows:
        lock = (row["claim_lock"] or "").strip()
        if suffix and not lock.endswith(suffix):
            continue
        if not _workspace_matches(row["workspace_path"], workspace):
            continue
        return kb.Task.from_row(row)
    return None


def heartbeat_current_claim(
    conn,
    *,
    task: kb.Task,
    ttl_seconds: int,
    note: str = "self-poll current",
) -> kb.Task:
    """Extend lease + heartbeat for an already-running self-poll claim."""
    if task.claim_lock:
        kb.heartbeat_claim(
            conn,
            task.id,
            ttl_seconds=ttl_seconds,
            claimer=task.claim_lock,
        )
    kb.heartbeat_worker(
        conn,
        task.id,
        note=note,
        expected_run_id=task.current_run_id,
    )
    return kb.get_task(conn, task.id) or task


def reset_interactive_claims(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    listener_kind: str,
    reason: str = "operator reset-kanban",
) -> list[str]:
    """Reclaim running tasks for one interactive listener kind.

    ``listener_kind`` is the claim-lock suffix without a leading colon, for
    example ``codex-interactive`` or ``deepseek-interactive``.
    """
    from hermes_cli.kanban_listener import reset_kanban_running_claims

    return reset_kanban_running_claims(
        board=board,
        assignees=claim_assignees or [profile],
        workspace=workspace,
        claim_lock_suffix=f":{listener_kind}",
        reason=reason,
    )


def reset_self_poll_claims(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    listener_kind: str,
    owner: str | None = None,
    reason: str = "operator reset-current",
) -> list[str]:
    """Reclaim running self-poll claims for one stable pane owner."""
    from hermes_cli.kanban_listener import reset_kanban_running_claims

    safe_owner = safe_path_component(owner or profile)
    return reset_kanban_running_claims(
        board=board,
        assignees=claim_assignees or [profile],
        workspace=workspace,
        claim_lock_suffix=f":{safe_owner}:{listener_kind}",
        reason=reason,
    )


def safe_path_component(value: str) -> str:
    """Return a filesystem-safe component for board/profile subdirectories."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "default"))


def default_self_poll_owner(
    *,
    profile: str,
    listener_kind: str,
    pane_id: str | None = None,
    pid: int | None = None,
) -> str:
    """Return a stable-enough owner for one self-poll pane.

    The owner is part of the self-poll claim lock suffix.  It must be narrower
    than just ``profile`` because two visible panes may both be allowed to claim
    the same assignee.  Prefer an explicit operator override, then the Zellij
    pane id, and finally this launcher process PID.
    """
    explicit = os.environ.get("HERMES_KANBAN_SELF_POLL_OWNER")
    if explicit:
        return safe_path_component(explicit)
    source = pane_id or os.environ.get("ZELLIJ_PANE_ID") or str(pid or os.getpid())
    return safe_path_component(f"{profile}-{source}")


def build_next_command(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    ttl: int,
    listener_kind: str,
    owner: str | None = None,
) -> str:
    """Return the machine-readable self-poll command for visible workers."""
    parts = ["hermes", "kanban"]
    if board:
        parts.extend(["--board", board])
    parts.extend(
        [
            "next",
            "--profile",
            profile,
            "--claim-assignees",
            ",".join(claim_assignees or [profile]),
            "--workspace",
            str(workspace),
            "--ttl",
            str(int(ttl)),
            "--listener-kind",
            listener_kind,
            "--owner",
            owner or profile,
            "--json",
        ]
    )
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_reset_current_command(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    listener_kind: str,
    owner: str | None = None,
) -> str:
    """Return the command that reclaims this worker's current self-poll task."""
    parts = ["hermes", "kanban"]
    if board:
        parts.extend(["--board", board])
    parts.extend(
        [
            "reset-current",
            "--profile",
            profile,
            "--claim-assignees",
            ",".join(claim_assignees or [profile]),
            "--workspace",
            str(workspace),
            "--listener-kind",
            listener_kind,
            "--owner",
            owner or profile,
            "--json",
        ]
    )
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_self_poll_startup_prompt(
    *,
    agent_label: str,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    ttl: int,
    listener_kind: str,
    owner: str | None = None,
    role_guidance_text: str,
) -> str:
    """Build the one-time startup instruction for self-polling workers."""
    next_command = build_next_command(
        board=board,
        profile=profile,
        claim_assignees=claim_assignees,
        workspace=workspace,
        ttl=ttl,
        listener_kind=listener_kind,
        owner=owner or profile,
    )
    reset_command = build_reset_current_command(
        board=board,
        profile=profile,
        claim_assignees=claim_assignees,
        workspace=workspace,
        listener_kind=listener_kind,
        owner=owner or profile,
    )
    claims = ",".join(claim_assignees or [profile])
    return (
        "# Hermes Kanban self-poll worker\n\n"
        f"Executor: {agent_label}\n"
        f"Board: {board}\n"
        f"Pane/profile: {profile}\n"
        f"Claim assignees: {claims}\n"
        f"Self-poll owner: {owner or profile}\n"
        f"Workspace: {workspace}\n\n"
        "你是一个 Hermes Kanban self-poll worker。启动后不要等待外部逐任务注入；"
        "空闲时自己调用下面的机器接口领取任务：\n\n"
        f"```bash\n{next_command}\n```\n\n"
        "领取规则：\n"
        "1. 每次只领取一个任务；拿到 `status=claimed` 后先读取 JSON 里的 `context_path`。\n"
        "2. 按 context 文件里的 `Task assignee/role` 工作；即使当前 pane/profile 是辅助角色，也以任务 assignee 为准。\n"
        "3. 完成或阻塞后必须运行 JSON 里的 complete/block command，再回到第 1 步领取下一个任务。\n"
        "4. 如果重复调用时返回 `status=current`，继续完成该 current 任务，不要 claim 第二个任务。\n"
        "5. 如果返回 `status=idle`，等待一小段时间后再试，不要编造任务。\n"
        "6. 如需放弃当前卡死 claim，运行：\n\n"
        f"```bash\n{reset_command}\n```\n\n"
        "角色基线：\n"
        f"{role_guidance_text}\n"
    )
