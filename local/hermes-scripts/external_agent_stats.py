"""
外部 CLI Agent 调用统计采集模块 — daily_stats.py 扩展。

采集 Codex (~/.codex/state_5.sqlite) 和 DeepSeek-TUI (~/.deepseek/sessions/*.json)
的模型调用数据。

核心设计：快照差值模式
- Codex state_5.threads.tokens_used 是会话生命周期累计量
- DeepSeek session total_tokens 也是累计量
- 用上次快照值做差值，得到当日增量
- 跨天 session 在多天日报里出现是正常的，每天只报增量部分
- 快照只在 morning cron (08:30) 推进，手动 rerun 不覆盖

容错原则：
- 每个采集函数独立 try/except，任何错误只标记不崩溃
- 数据源缺失/格式变化时静默降级
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEEPSEEK_SESSIONS_DIR = Path.home() / ".deepseek" / "sessions"
SNAPSHOT_FILE = Path(__file__).parent / ".external-agent-snapshot.json"
CODEX_DIR = Path.home() / ".codex"

REASONIX_DIR = Path.home() / ".reasonix"
REASONIX_USAGE_FILE = REASONIX_DIR / "usage.jsonl"
REASONIX_SESSIONS_DIRS = [
    Path.home() / ".config" / "reasonix" / "sessions",  # v1.3.0+
    Path.home() / ".reasonix" / "sessions",              # v1.2.0 / Codex layout
]

TZ = timezone(timedelta(hours=8), "CST")


def find_codex_state_dbs() -> list[Path]:
    """Find Codex state DBs by globbing state_*.sqlite.

    Codex hardcodes the version number (currently state_5.sqlite),
    but upgrades may bump it to state_6 or later.  Migrate creates
    the new DB but may leave the old file around, so we read the
    two highest versions and deduplicate by thread.id later.

    Returns list of paths sorted by version number (highest first), max 2.
    """
    candidates = list(CODEX_DIR.glob("state_*.sqlite"))
    if not candidates:
        return []
    sorted_dbs = sorted(
        candidates,
        key=lambda p: int(m.group(1))
        if (m := re.match(r"state_(\d+)\.sqlite", p.name)) else 0,
        reverse=True,
    )
    # At most 2: current version + previous version if migration left it around
    return sorted_dbs[:2]


# ── Snapshot I/O ─────────────────────────────────────────────

def load_snapshot() -> dict[str, Any]:
    if SNAPSHOT_FILE.exists():
        try:
            data = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_snapshot(snapshot: dict[str, Any]) -> None:
    payload = dict(snapshot)
    payload["_snapshot_at"] = datetime.now(tz=TZ).isoformat()
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def should_save_snapshot(now: datetime) -> bool:
    """Only scheduled morning runs should advance the snapshot baseline."""
    local = now.astimezone(TZ)
    return local.hour == 8 or (local.hour == 9 and local.minute <= 30)


# ── Codex (state_5.threads) ─────────────────────────────────

def _count_codex_llm_calls(start_ts: float, end_ts: float) -> dict[str, int]:
    """Count Codex LLM API calls from logs_2.sqlite response.completed events.

    Returns {model: call_count} for the time window.
    Uses codex_otel.log_only target (deduped — same event appears in trace_safe too).
    Only counts events with input_token_count (actual completions, not errors/duration-only).
    """
    try:
        log_db = CODEX_DIR / "logs_2.sqlite"
        if not log_db.exists():
            return {}
        con = sqlite3.connect(str(log_db))
        rows = con.execute(
            "SELECT feedback_log_body FROM logs "
            "WHERE target = 'codex_otel.log_only' "
            "AND feedback_log_body LIKE '%response.completed%' "
            "AND feedback_log_body LIKE '%input_token_count%' "
            "AND ts >= ? AND ts < ?",
            (int(start_ts), int(end_ts)),
        ).fetchall()
        con.close()

        counts: dict[str, int] = {}
        for body, in rows:
            m = re.search(r"model=(\S+)", body)
            if m:
                model = m.group(1)
                counts[model] = counts.get(model, 0) + 1
        return counts
    except Exception:
        return {}


def _extract_last_token_usage(rollout_path: str, *, tail_lines: int = 5000) -> dict[str, int] | None:
    """Extract the last total_token_usage from a Codex session jsonl.

    Codex writes event_msg entries with info.total_token_usage containing
    {input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens, total_tokens}.
    These are cumulative per-thread, so the last entry gives the true totals.

    Only reads the last `tail_lines` lines of the file for efficiency.
    Returns None if no usage data found.
    """
    import subprocess as sp
    path = Path(rollout_path)
    if not path.exists():
        return None
    try:
        # Use tail for efficiency on large files (some are 166M+)
        r = sp.run(
            ["tail", "-n", str(tail_lines), str(path)],
            capture_output=True, text=True, timeout=30,
        )
        last_usage = None
        for line in r.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") != "event_msg":
                continue
            info = d.get("payload", {}).get("info", {})
            if not isinstance(info, dict):
                continue
            tu = info.get("total_token_usage")
            if isinstance(tu, dict) and "input_tokens" in tu:
                last_usage = tu
        if last_usage:
            return {
                "input_tokens": int(last_usage.get("input_tokens", 0)),
                "cached_input_tokens": int(last_usage.get("cached_input_tokens", 0)),
                "output_tokens": int(last_usage.get("output_tokens", 0)),
                "reasoning_output_tokens": int(last_usage.get("reasoning_output_tokens", 0)),
                "total_tokens": int(last_usage.get("total_tokens", 0)),
            }
    except Exception:
        pass
    return None


def collect_codex_usage(
    start_ts: float,
    end_ts: float,
    prev_snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    """Collect Codex token usage via state_5.threads + session jsonl with snapshot deltas.

    Token breakdown (input/cached/output/reasoning) comes from session jsonl
    event_msg.info.total_token_usage (cumulative per-thread).
    Falls back to state_5.tokens_used (total only, no breakdown) if jsonl unavailable.

    Returns (rows, error_or_none, new_snapshot_data).
    Each row: profile, model, calls, input_tokens, output_tokens, cache_read_tokens, ...
    """
    try:
        db_paths = find_codex_state_dbs()
        if not db_paths:
            return [], None, {}

        # Read threads from ALL state DBs, dedup by thread id
        all_threads: dict[str, tuple] = {}
        for db_path in db_paths:
            try:
                con = sqlite3.connect(str(db_path))
                rows = con.execute(
                    "SELECT id, model, tokens_used, created_at, updated_at, "
                    "first_user_message, agent_role, cwd, rollout_path "
                    "FROM threads WHERE tokens_used > 0"
                ).fetchall()
                con.close()
                for tid, model, tokens_used, created_at, updated_at, first_msg, agent_role, cwd, rollout_path in rows:
                    if tid not in all_threads:
                        all_threads[tid] = (model, tokens_used, created_at, updated_at, first_msg, agent_role, cwd, rollout_path)
            except Exception:
                continue

        if not all_threads:
            return [], None, {}

        # Get LLM call counts from logs_2
        llm_calls = _count_codex_llm_calls(start_ts, end_ts)

        prev_codex = prev_snapshot.get("codex_threads", {}) or {}
        new_codex: dict[str, Any] = {}
        results: dict[str, dict[str, Any]] = {}

        for tid, (model, tokens_used, created_at, updated_at, first_msg, agent_role, cwd, rollout_path) in all_threads.items():
            model = str(model or "unknown")
            tokens_used = int(tokens_used or 0)

            # Infer kanban role for Codex
            codex_role = ""
            if first_msg and "Hermes Kanban" in str(first_msg):
                codex_role = "planner"

            # Extract detailed token usage from session jsonl
            jsonl_usage = _extract_last_token_usage(rollout_path) if rollout_path else None

            # Save current state for next snapshot (with breakdown)
            snap_entry: dict[str, Any] = {"tokens_used": tokens_used, "model": model, "codex_role": codex_role}
            if jsonl_usage:
                snap_entry["input_tokens"] = jsonl_usage["input_tokens"]
                snap_entry["cached_input_tokens"] = jsonl_usage["cached_input_tokens"]
                snap_entry["output_tokens"] = jsonl_usage["output_tokens"]
                snap_entry["reasoning_output_tokens"] = jsonl_usage["reasoning_output_tokens"]
            new_codex[tid] = snap_entry

            # Compute deltas
            prev_data = prev_codex.get(tid)
            prev_tokens = int(prev_data.get("tokens_used", 0)) if prev_data else 0
            delta = tokens_used - prev_tokens

            if delta <= 0:
                continue

            # Try breakdown deltas from jsonl data
            delta_input = 0
            delta_cached = 0
            delta_output = 0
            has_breakdown = False

            if jsonl_usage and prev_data:
                prev_input = int(prev_data.get("input_tokens", 0))
                prev_cached = int(prev_data.get("cached_input_tokens", 0))
                prev_output = int(prev_data.get("output_tokens", 0))
                cur_input = jsonl_usage["input_tokens"]
                cur_cached = jsonl_usage["cached_input_tokens"]
                cur_output = jsonl_usage["output_tokens"]
                if cur_input >= prev_input and cur_output >= prev_output:
                    delta_input = cur_input - prev_input
                    delta_cached = cur_cached - prev_cached
                    delta_output = cur_output - prev_output
                    has_breakdown = True

            # Infer role from current thread data or previous snapshot
            effective_role = codex_role or str((prev_data or {}).get("codex_role", "") or "")

            # Aggregate by role/model
            profile_label = effective_role if effective_role else "codex"
            key = f"{profile_label}/{model}"
            row = results.setdefault(key, {
                "profile": profile_label,
                "model": model,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "_thread_count": 0,
            })

            if has_breakdown:
                row["input_tokens"] += delta_input
                row["cache_read_tokens"] += delta_cached
                row["output_tokens"] += delta_output
            else:
                # Fallback: no breakdown, report total as input_tokens
                row["input_tokens"] += delta

            row["_thread_count"] += 1

        # Finalize: set calls and notes
        for row in results.values():
            model = row["model"]
            tc = row.pop("_thread_count", 0)
            logged_calls = llm_calls.get(model, 0)
            row["calls"] = tc
            note_parts = [f"{tc} thread(s) with token delta"]
            if logged_calls:
                note_parts.append(f"logs_2 response.completed={logged_calls}")
            note_parts.append("(sub-calls not logged; real LLM call count is much higher)")
            row["_note"] = "; ".join(note_parts)

        return list(results.values()), None, {"codex_threads": new_codex}

    except Exception as exc:
        return [], f"codex detection: {type(exc).__name__}: {exc}", {}


# ── DeepSeek-TUI ──────────────────────────────────────────────

def collect_deepseek_usage(
    start_ts: float,
    end_ts: float,
    prev_snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    """Collect DeepSeek-TUI session usage with snapshot deltas.

    Sessions active in the time window (by mtime) are included.
    Delta = current total_tokens - snapshot total_tokens.

    Returns (rows, error_or_none, new_snapshot_data).
    """
    try:
        if not DEEPSEEK_SESSIONS_DIR.is_dir():
            return [], None, {}

        prev_ds = prev_snapshot.get("deepseek_sessions", {}) or {}
        new_ds: dict[str, Any] = {}
        results: list[dict[str, Any]] = []
        total_errors = 0

        for sf in sorted(os.listdir(str(DEEPSEEK_SESSIONS_DIR))):
            if not sf.endswith(".json"):
                continue
            filepath = DEEPSEEK_SESSIONS_DIR / sf
            if not filepath.is_file():
                continue

            try:
                # Filter by mtime: only include sessions with activity in window
                mtime = filepath.stat().st_mtime
                if not (start_ts <= mtime <= end_ts):
                    continue

                with filepath.open("r", encoding="utf-8") as f:
                    sess = json.load(f)

                meta = sess.get("metadata") or {}
                session_id = sf.replace(".json", "")
                total_tokens = int(meta.get("total_tokens", 0) or 0)
                msg_count = int(meta.get("message_count", 0) or 0)

                # Skip empty sessions
                if msg_count == 0 and total_tokens == 0:
                    new_ds[session_id] = {"total_tokens": 0, "model": str(meta.get("model", "unknown"))}
                    continue

                model = str(meta.get("model", "unknown"))
                cost = meta.get("cost") or {}
                cost_usd = float(cost.get("session_cost_usd", 0) or 0) if isinstance(cost, dict) else 0
                cost_cny = float(cost.get("session_cost_cny", 0) or 0) if isinstance(cost, dict) else 0
                subagent_usd = float(cost.get("subagent_cost_usd", 0) or 0) if isinstance(cost, dict) else 0
                subagent_cny = float(cost.get("subagent_cost_cny", 0) or 0) if isinstance(cost, dict) else 0

                # Save current state for next snapshot (include workspace for role inference)
                new_ds[session_id] = {
                    "total_tokens": total_tokens,
                    "message_count": msg_count,
                    "model": model,
                    "workspace": str(meta.get("workspace", "") or ""),
                    "cost_usd": cost_usd,
                    "cost_cny": cost_cny,
                    "subagent_cost_usd": subagent_usd,
                    "subagent_cost_cny": subagent_cny,
                }

                # Compute delta
                prev_data = prev_ds.get(session_id)
                prev_tokens = int(prev_data.get("total_tokens", 0)) if prev_data else 0
                prev_msg_count = int(prev_data.get("message_count", 0)) if prev_data else 0
                delta_tokens = total_tokens - prev_tokens
                delta_msgs = msg_count - prev_msg_count

                prev_cost_usd = float(prev_data.get("cost_usd", 0)) if prev_data else 0
                prev_cost_cny = float(prev_data.get("cost_cny", 0)) if prev_data else 0
                prev_subagent_usd = float(prev_data.get("subagent_cost_usd", 0)) if prev_data else 0
                prev_subagent_cny = float(prev_data.get("subagent_cost_cny", 0)) if prev_data else 0

                delta_cost_usd = cost_usd - prev_cost_usd
                delta_cost_cny = cost_cny - prev_cost_cny
                delta_subagent_usd = subagent_usd - prev_subagent_usd
                delta_subagent_cny = subagent_cny - prev_subagent_cny

                # Skip if no delta
                if delta_tokens <= 0 and delta_cost_cny <= 0:
                    continue

                # Infer kanban role from workspace path (e.g. .ds-sessions/implementer)
                # Use current metadata.workspace, fall back to snapshot workspace
                workspace = str(meta.get("workspace", "") or "") or str((prev_data or {}).get("workspace", "") or "")
                ds_role = ""
                for segment in reversed(workspace.split("/")):
                    if segment in ("implementer", "planner", "critic", "coordinator"):
                        ds_role = segment
                        break

                results.append({
                    "profile": ds_role if ds_role else "deepseek-tui",
                    "model": model,
                    "calls": max(delta_msgs, 0),
                    "input_tokens": max(delta_tokens, 0),
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "_cost_usd": max(delta_cost_usd, 0),
                    "_cost_cny": max(delta_cost_cny, 0),
                    "_subagent_cost_usd": max(delta_subagent_usd, 0),
                    "_subagent_cost_cny": max(delta_subagent_cny, 0),
                    "_session_id": session_id,
                    "_note": "delta from snapshot; total_tokens has no input/output breakdown",
                })

            except Exception:
                total_errors += 1
                if total_errors > 3:
                    break
                continue

        err = f"deepseek session parse errors: {total_errors}" if total_errors else None
        return results, err, {"deepseek_sessions": new_ds}

    except Exception as exc:
        return [], f"deepseek detection: {type(exc).__name__}: {exc}", {}


# ── Reasonix ──────────────────────────────────────────────────

# DeepSeek API pricing (per 1M tokens, USD) — used for cost estimation
# when usage.jsonl is unavailable (Reasonix v1.3.0+).
_DS_PRICING = {
    "deepseek-v4-flash": {"input": 1.0, "cache_hit": 0.02, "output": 2.0},
    "deepseek-v4-pro": {"input": 3.0, "cache_hit": 0.025, "output": 6.0},
}

# Typical DeepSeek cache hit ratio observed in usage.jsonl
_DS_TYPICAL_CACHE_RATIO = 0.995  # 99.5% of prompt tokens are cache hits


def _parse_reasonix_session_filename(filename: str) -> tuple[str, str]:
    """Extract session ID and primary model from a Reasonix v1.3.0+ session filename.

    Examples:
        "20260608-020028.214210696-deepseek-v4-flash + planner deepseek-v4-pro.jsonl"
        → ("20260608-020028.214210696-deepseek-v4-flash + planner deepseek-v4-pro",
           "deepseek-v4-flash")

        "20260606-023923.744395247-deepseek-v4-flash.jsonl"
        → ("20260606-023923.744395247-deepseek-v4-flash", "deepseek-v4-flash")

    Returns (session_id, primary_model).
    """
    name = filename
    if name.endswith(".jsonl"):
        name = name[:-6]
    elif name.endswith(".jsonl.meta"):
        name = name[:-11]

    # Model is after the last hyphen-dot or hyphen-plus pattern
    # Format: <timestamp>-<model> or <timestamp>-<model> + planner <planner_model>
    models_in_name = []
    # Split on " + " to get multiple models
    for part in name.split(" + "):
        part = part.strip()
        # Extract model from part like "planner deepseek-v4-pro" or "deepseek-v4-flash"
        if part.startswith("planner "):
            models_in_name.append(part[8:])
        else:
            # The model is the last segment after the final hyphen-dot boundary
            # e.g. "20260608-020028.214210696-deepseek-v4-flash"
            segments = part.split("-")
            # Try to find a known model name
            for seg in segments:
                if any(m in seg for m in ["deepseek", "gpt", "claude", "o1", "o3"]):
                    # Reconstruct the full model name from remaining segments
                    idx = segments.index(seg)
                    model_parts = segments[idx:]
                    models_in_name.append("-".join(model_parts))
                    break

    primary_model = models_in_name[0] if models_in_name else "deepseek-v4-flash"
    return name, primary_model


def _estimate_session_usage_from_chatlog(
    session_path: Path,
    filename: str,
) -> dict[str, Any] | None:
    """Estimate Reasonix session usage by parsing the chat log.

    Reasonix v1.3.0 stopped writing usage.jsonl, so we estimate:
    - Number of LLM calls = count of assistant messages
    - Model name from filename
    - Token estimates from message content length (~4 chars/token for CJK, ~4.5 for English)
    - Cache hit ratio from typical DeepSeek API behavior (~99.5%)

    Returns dict with estimated usage, or None if session has no assistant messages.
    """
    session_id, primary_model = _parse_reasonix_session_filename(filename)

    calls = 0
    total_input_chars = 0
    total_output_chars = 0
    planner_calls = 0

    try:
        with session_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue

                role = d.get("role", "")
                if role == "assistant":
                    calls += 1
                    # Output tokens from assistant content
                    content = d.get("content", "") or ""
                    reasoning = d.get("reasoning_content", "") or ""
                    total_output_chars += len(content) + len(reasoning)
                elif role == "user":
                    # Input tokens from user content
                    content = d.get("content", "") or ""
                    total_input_chars += len(content)
                elif role == "tool":
                    # Tool responses also contribute to input context
                    content = d.get("content", "") or ""
                    total_input_chars += len(content)
    except Exception:
        return None

    if calls == 0:
        return None

    # Estimate tokens (conservative: ~4 chars/token for mixed content)
    chars_per_token = 4.0
    estimated_output_tokens = int(total_output_chars / chars_per_token)
    estimated_real_input_tokens = int(total_input_chars / chars_per_token)

    # With DeepSeek API, each call re-sends the full conversation context.
    # The total prompt per call grows linearly with turns.
    # Estimate: average prompt size ≈ (first_turn_input + last_turn_input) / 2 * calls
    # Simplification: total prompt ≈ real_input * (1 + cache_ratio) where cache_ratio ≈ 99.5%
    estimated_total_prompt = int(estimated_real_input_tokens / (1 - _DS_TYPICAL_CACHE_RATIO))
    estimated_cache = estimated_total_prompt - estimated_real_input_tokens

    # Ensure non-negative
    estimated_cache = max(estimated_cache, 0)
    estimated_total_prompt = max(estimated_total_prompt, estimated_real_input_tokens)

    # Estimate cost
    pricing = _DS_PRICING.get(primary_model, _DS_PRICING.get("deepseek-v4-flash"))
    estimated_cost = (
        estimated_real_input_tokens * pricing["input"] / 1_000_000
        + estimated_cache * pricing["cache_hit"] / 1_000_000
        + estimated_output_tokens * pricing["output"] / 1_000_000
    )

    # Check if there's a planner model in the filename
    has_planner = "planner" in filename.lower() or "+ planner" in filename
    if has_planner:
        planner_calls = max(1, calls // 10)  # Rough estimate: ~10% planner calls
        # Get planner model name
        if "+ planner" in filename:
            planner_model = filename.split("+ planner")[-1].split(".")[0].strip()
        else:
            planner_model = "deepseek-v4-pro"
    else:
        planner_model = ""

    return {
        "session_id": session_id,
        "primary_model": primary_model,
        "planner_model": planner_model,
        "planner_calls": planner_calls,
        "calls": calls,
        "promptTokens": estimated_total_prompt,
        "completionTokens": estimated_output_tokens,
        "cacheHitTokens": estimated_cache,
        "cacheMissTokens": estimated_real_input_tokens,
        "costUsd": estimated_cost,
        "models": {
            primary_model: {
                "promptTokens": estimated_total_prompt,
                "completionTokens": estimated_output_tokens,
                "cacheHitTokens": estimated_cache,
                "costUsd": estimated_cost,
                "calls": calls,
            }
        },
        "estimated": True,
    }


def collect_reasonix_usage(
    start_ts: float,
    end_ts: float,
    prev_snapshot: dict[str, Any],
) -> tuple[list[dict[str, Any]], str | None, dict[str, Any]]:
    """Collect Reasonix token usage with snapshot deltas.

    Data sources (in priority order):
    1. ~/.reasonix/usage.jsonl — per-call precise data (Reasonix ≤ v1.2.0)
    2. ~/.reasonix/sessions/*.meta.json — per-session precise totals (≤ v1.2.0)
    3. ~/.config/reasonix/sessions/*.jsonl — chat log estimation (v1.3.0+, no usage.jsonl)

    Reasonix v1.3.0 stopped writing usage.jsonl, so sessions after the upgrade
    rely on chat log estimation (marked with estimated=True).

    Snapshot strategy: save per-session cumulative totals so that incremental
    runs only report the delta.  First run (no snapshot) counts everything in
    the time window as new.

    Returns (rows, error_or_none, new_snapshot_data).
    """
    try:
        prev_rx = prev_snapshot.get("reasonix_sessions", {}) or {}
        new_rx: dict[str, Any] = {}
        results: dict[str, dict[str, Any]] = {}
        has_estimated = False
        notes: list[str] = []

        # ── Source 1: usage.jsonl (precise per-call data) ──
        session_totals: dict[str, dict[str, int | float]] = {}
        if REASONIX_USAGE_FILE.exists():
            with REASONIX_USAGE_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue

                    ts = d.get("ts", 0)
                    session = d.get("session", "")
                    if not session:
                        continue

                    if session not in session_totals:
                        session_totals[session] = {
                            "promptTokens": 0,
                            "completionTokens": 0,
                            "cacheHitTokens": 0,
                            "cacheMissTokens": 0,
                            "costUsd": 0.0,
                            "calls": 0,
                            "models": {},
                            "first_ts": ts,
                            "last_ts": ts,
                        }

                    st = session_totals[session]
                    st["promptTokens"] += d.get("promptTokens", 0)
                    st["completionTokens"] += d.get("completionTokens", 0)
                    st["cacheHitTokens"] += d.get("cacheHitTokens", 0)
                    st["cacheMissTokens"] += d.get("cacheMissTokens", 0)
                    st["costUsd"] += d.get("costUsd", 0)
                    st["calls"] += 1
                    st["last_ts"] = max(st["last_ts"], ts)
                    if ts < st["first_ts"]:
                        st["first_ts"] = ts

                    model = d.get("model", "unknown")
                    if model not in st["models"]:
                        st["models"][model] = {
                            "promptTokens": 0,
                            "completionTokens": 0,
                            "cacheHitTokens": 0,
                            "costUsd": 0.0,
                            "calls": 0,
                        }
                    mm = st["models"][model]
                    mm["promptTokens"] += d.get("promptTokens", 0)
                    mm["completionTokens"] += d.get("completionTokens", 0)
                    mm["cacheHitTokens"] += d.get("cacheHitTokens", 0)
                    mm["costUsd"] += d.get("costUsd", 0)
                    mm["calls"] += 1

        # ── Source 2: v1.2.0 meta.json files (precise per-session totals) ──
        session_meta: dict[str, dict] = {}
        for sessions_dir in REASONIX_SESSIONS_DIRS:
            if not sessions_dir.is_dir():
                continue
            for meta_file in sessions_dir.glob("*.meta.json"):
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    sid = meta_file.stem.replace(".meta", "")
                    session_meta[sid] = meta
                    # If this session isn't in usage.jsonl but has meta with token data,
                    # add it to session_totals
                    if sid not in session_totals and meta.get("totalCompletionTokens", 0) > 0:
                        prompt = meta.get("lastPromptTokens", 0)
                        cache = meta.get("cacheHitTokens", 0)
                        miss = meta.get("cacheMissTokens", 0)
                        # lastPromptTokens is the last call's prompt, not cumulative
                        # For cumulative, estimate from cacheHit + cacheMiss
                        total_prompt = cache + miss
                        session_totals[sid] = {
                            "promptTokens": total_prompt,
                            "completionTokens": meta.get("totalCompletionTokens", 0),
                            "cacheHitTokens": cache,
                            "cacheMissTokens": miss,
                            "costUsd": meta.get("totalCostUsd", 0),
                            "calls": 0,  # unknown from meta alone
                            "models": {},
                            "first_ts": 0,
                            "last_ts": 0,
                        }
                except Exception:
                    pass

        # ── Source 3: v1.3.0+ session chat logs (estimation fallback) ──
        # Scan for sessions not already covered by usage.jsonl or meta.json
        covered_sessions = set(session_totals.keys())
        for sessions_dir in REASONIX_SESSIONS_DIRS:
            if not sessions_dir.is_dir():
                continue
            for jsonl_file in sessions_dir.glob("*.jsonl"):
                # Skip archive files and very small files
                if "__archive_" in jsonl_file.name:
                    continue
                if jsonl_file.stat().st_size < 100:
                    continue

                session_id, _ = _parse_reasonix_session_filename(jsonl_file.name)

                # Only process sessions not already covered by precise data
                if session_id in covered_sessions:
                    continue

                # Check if this session was active in the time window
                mtime = jsonl_file.stat().st_mtime
                # Also check .meta for created_at
                meta_path = jsonl_file.parent / (jsonl_file.name + ".meta")
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        created = meta.get("created_at", "")
                        if created:
                            from datetime import datetime as _dt
                            ct = _dt.fromisoformat(created).timestamp()
                            if ct > end_ts:
                                continue  # created after window
                    except Exception:
                        pass

                est = _estimate_session_usage_from_chatlog(jsonl_file, jsonl_file.name)
                if est is None:
                    continue

                has_estimated = True
                session_id = est["session_id"]
                covered_sessions.add(session_id)

                # Add to session_totals using estimated data
                session_totals[session_id] = {
                    "promptTokens": est["promptTokens"],
                    "completionTokens": est["completionTokens"],
                    "cacheHitTokens": est["cacheHitTokens"],
                    "cacheMissTokens": est["cacheMissTokens"],
                    "costUsd": est["costUsd"],
                    "calls": est["calls"],
                    "models": est["models"],
                    "first_ts": mtime * 1000,  # approximate
                    "last_ts": mtime * 1000,
                }

        if not session_totals:
            return [], None, {}

        # ── Compute deltas and aggregate by model ──
        for session, st in session_totals.items():
            if st["calls"] == 0 and st["promptTokens"] == 0:
                continue

            is_estimated = session not in prev_rx and st.get("calls", 0) > 0 and st.get("promptTokens", 0) == 0
            # Check if this was from estimation
            for sid_test in covered_sessions:
                if sid_test == session:
                    # If it wasn't from usage.jsonl, it's estimated
                    break

            new_rx[session] = {
                "promptTokens": st["promptTokens"],
                "completionTokens": st["completionTokens"],
                "cacheHitTokens": st["cacheHitTokens"],
                "costUsd": st["costUsd"],
                "models": st["models"],
                "workspace": str((session_meta.get(session) or {}).get("workspace", "")),
            }

            prev_data = prev_rx.get(session)
            if prev_data:
                prev_prompt = int(prev_data.get("promptTokens", 0))
                prev_comp = int(prev_data.get("completionTokens", 0))
                prev_cache = int(prev_data.get("cacheHitTokens", 0))
                prev_cost = float(prev_data.get("costUsd", 0))
                prev_models = prev_data.get("models", {})
            else:
                prev_prompt = 0
                prev_comp = 0
                prev_cache = 0
                prev_cost = 0
                prev_models = {}

            for model, mm in st["models"].items():
                prev_mm = prev_models.get(model, {}) if prev_models else {}
                prev_m_prompt = int(prev_mm.get("promptTokens", 0))
                prev_m_comp = int(prev_mm.get("completionTokens", 0))
                prev_m_cache = int(prev_mm.get("cacheHitTokens", 0))
                prev_m_cost = float(prev_mm.get("costUsd", 0))

                delta_prompt = mm["promptTokens"] - prev_m_prompt
                delta_comp = mm["completionTokens"] - prev_m_comp
                delta_cache = mm["cacheHitTokens"] - prev_m_cache
                delta_cost = mm["costUsd"] - prev_m_cost
                delta_calls = mm["calls"] - int(prev_mm.get("calls", 0))

                if delta_prompt <= 0 and delta_comp <= 0:
                    continue

                workspace = str((session_meta.get(session) or {}).get("workspace", "")) \
                    or str((prev_data or {}).get("workspace", ""))
                rx_role = ""
                for segment in reversed(workspace.split("/")):
                    if segment in ("implementer", "planner", "critic", "coordinator"):
                        rx_role = segment
                        break

                # Also infer role from session name (e.g. kanban pane sessions)
                if not rx_role:
                    # Check if session name matches kanban patterns
                    for role in ("implementer", "planner", "critic", "coordinator"):
                        if role in session.lower():
                            rx_role = role
                            break

                project = ""
                if session.startswith("code-"):
                    rest = session[5:]
                    import re as _re
                    m = _re.match(r"(.+?)-\d{10,}", rest)
                    if m:
                        project = m.group(1)
                    else:
                        project = rest.split("-")[0] if "-" in rest else rest

                profile_label = rx_role if rx_role else "reasonix"
                key = f"{profile_label}/{model}"
                row = results.setdefault(key, {
                    "profile": profile_label,
                    "model": model,
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_write_tokens": 0,
                    "_cost_usd": 0.0,
                    "_project": project,
                    "_session_count": 0,
                })

                row["calls"] += max(delta_calls, 0)
                row["input_tokens"] += max(delta_prompt, 0)
                row["output_tokens"] += max(delta_comp, 0)
                row["cache_read_tokens"] += max(delta_cache, 0)
                row["_cost_usd"] += max(delta_cost, 0)
                row["_session_count"] += 1

        # Finalize
        for row in results.values():
            sc = row.pop("_session_count", 0)
            if has_estimated:
                row["_note"] = f"{sc} session(s); v1.3.0 usage estimated from chat logs (usage.jsonl unavailable)"
            else:
                row["_note"] = f"{sc} session(s); usage.jsonl per-call data with full token breakdown"

        error_note = None
        if has_estimated:
            error_note = "Reasonix v1.3.0 stopped writing usage.jsonl; token counts estimated from chat logs (cache ratio assumed 99.5%)"

        return list(results.values()), error_note, {"reasonix_sessions": new_rx}

    except Exception as exc:
        return [], f"reasonix detection: {type(exc).__name__}: {exc}", {}


# ── Unified collector ─────────────────────────────────────────

def collect_external_agent_usage(
    start_ts: float,
    end_ts: float,
    prev_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect usage from all external CLI agents with snapshot deltas.

    Returns dict:
        rows: list[dict] — all usage rows in Hermes-compatible format
        errors: list[str] — non-fatal detection errors
        note: str — summary for report body
        new_snapshot_data: dict — data to merge into snapshot (if saving)
    """
    if prev_snapshot is None:
        prev_snapshot = load_snapshot()

    all_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    new_snap: dict[str, Any] = {}

    codex_rows, codex_err, codex_snap = collect_codex_usage(
        start_ts, end_ts, prev_snapshot
    )
    if codex_rows:
        all_rows.extend(codex_rows)
    if codex_err:
        errors.append(codex_err)
    if codex_snap:
        new_snap.update(codex_snap)

    deepseek_rows, deepseek_err, deepseek_snap = collect_deepseek_usage(
        start_ts, end_ts, prev_snapshot
    )
    if deepseek_rows:
        all_rows.extend(deepseek_rows)
    if deepseek_err:
        errors.append(deepseek_err)
    if deepseek_snap:
        new_snap.update(deepseek_snap)

    reasonix_rows, reasonix_err, reasonix_snap = collect_reasonix_usage(
        start_ts, end_ts, prev_snapshot
    )
    if reasonix_rows:
        all_rows.extend(reasonix_rows)
    if reasonix_err:
        errors.append(reasonix_err)
    if reasonix_snap:
        new_snap.update(reasonix_snap)

    total_calls = sum(r.get("calls", 0) for r in all_rows)
    total_input = sum(r.get("input_tokens", 0) for r in all_rows)
    note_parts = []
    if all_rows:
        note_parts.append(f"外部agent调用共{total_calls}次, 增量tokens={total_input:,}")
        for r in all_rows:
            prof = r.get("profile", "")
            if prof == "codex":
                pfx = "Codex"
            elif prof == "reasonix":
                pfx = "Reasonix"
            else:
                pfx = "DeepSeek"
            model = r.get("model", "?")
            calls = r.get("calls", 0)
            inp = r.get("input_tokens", 0)
            note_parts.append(f"  {pfx}/{model}: {calls}调用 delta_tokens={inp:,}")
        if errors:
            note_parts.append(f"  采集异常: {'; '.join(errors)}")
    elif errors:
        note_parts.append("外部agent调用检测异常")
        note_parts.extend(f"  {e}" for e in errors)
        note_parts.append("（不影响主统计）")
    else:
        note_parts.append("未检测到外部agent调用记录")

    return {
        "rows": all_rows,
        "errors": errors,
        "note": "\n".join(note_parts),
        "new_snapshot_data": new_snap,
    }
