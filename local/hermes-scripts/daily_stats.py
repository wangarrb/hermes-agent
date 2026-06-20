#!/usr/bin/env python3
"""
Hermes 日报统计脚本 — no-agent 模式。

直接采集：
1. Hermes 主 profile + 所有 ~/.hermes/profiles/*/state.db 模型用量；
2. Hindsight API 当前状态；
3. Hindsight PostgreSQL 工作量新增/更新/总数净变化；
4. Hindsight docker logs 中可见的 LLM 调用/token 精确值；
5. 外部 CLI Agent（Codex、DeepSeek-TUI）的模型调用统计。

输出：
- 写入 ~/wiki/auto-maintenance/daily/YYYY-MM-DD.md
- stdout 精简摘要供 no-agent cron 投递

不调用 LLM。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

# 外部 CLI Agent 模型调用统计（快照差值模式）
from external_agent_stats import (
    collect_external_agent_usage,
    load_snapshot as load_ext_snapshot,
    save_snapshot as save_ext_snapshot,
    should_save_snapshot as should_save_ext_snapshot,
)

# Use fixed path for wiki (HOME may be redirected in profile sessions)
REAL_HOME = Path("/home/wyr")
HERMES_HOME = REAL_HOME / ".hermes"
STATE_DB = HERMES_HOME / "state.db"
PROFILE_DIR = HERMES_HOME / "profiles"
WIKI_DIR = REAL_HOME / "wiki" / "auto-maintenance" / "daily"
SNAPSHOT_FILE = WIKI_DIR / ".snapshot.json"
HINDSIGHT_API = os.environ.get("HINDSIGHT_API", "http://127.0.0.1:8888").rstrip("/")
HINDSIGHT_BANK = os.environ.get("HINDSIGHT_BANK", "hermes")
TZ = timezone(timedelta(hours=8), "CST")

PSQL_CANDIDATES = [
    os.environ.get("HINDSIGHT_PSQL"),
    "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql",
    "/home/wyr/.pg0/installation/18.1.0/bin/psql",
    shutil.which("psql"),
]


# ── formatting ────────────────────────────────────────────────


def fmt_num(n: int | float | None) -> str:
    if n is None:
        return "0"
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def fmt_tok(n: int | float | None) -> str:
    if n is None:
        return "0"
    try:
        n = int(n)
    except Exception:
        return str(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def make_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(headers) + " |"
    bar = "|" + "|".join(["---"] * len(headers)) + "|"
    lines = [sep, bar]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ── Hermes state.db stats ─────────────────────────────────────


def iter_state_dbs() -> list[tuple[str, Path]]:
    dbs: list[tuple[str, Path]] = []
    if STATE_DB.exists():
        dbs.append(("default", STATE_DB))
    if PROFILE_DIR.exists():
        for db in sorted(PROFILE_DIR.glob("*/state.db")):
            dbs.append((db.parent.name, db))
    return dbs


def hermes_model_usage(start: float, end: float) -> list[dict[str, Any]]:
    """Query default + profile state.db files for sessions active in window.

    Important: session.started_at alone misses long sessions that started before
    the window but consumed tokens today.  Prefer messages.timestamp to select
    active sessions, then aggregate the corresponding session counters.
    """
    rows_out: list[dict[str, Any]] = []
    for profile, db_path in iter_state_dbs():
        con = sqlite3.connect(str(db_path))
        try:
            try:
                rows = con.execute(
                    """
                    WITH active AS (
                        SELECT DISTINCT session_id
                        FROM messages
                        WHERE timestamp >= ? AND timestamp < ?
                    )
                    SELECT COALESCE(s.model, '(unknown)') AS model,
                           COUNT(DISTINCT s.id) AS sessions,
                           COALESCE(SUM(s.message_count), 0) AS turns,
                           COALESCE(SUM(s.api_call_count), 0) AS calls,
                           COALESCE(SUM(s.input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(s.cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(s.cache_write_tokens), 0) AS cache_write_tokens,
                           COALESCE(SUM(s.output_tokens), 0) AS output_tokens
                    FROM sessions s
                    JOIN active a ON a.session_id = s.id
                    GROUP BY s.model
                    ORDER BY calls DESC, input_tokens DESC
                    """,
                    (start, end),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = con.execute(
                    """
                    SELECT COALESCE(model, '(unknown)') AS model,
                           COUNT(*) AS sessions,
                           COALESCE(SUM(message_count), 0) AS turns,
                           COALESCE(SUM(api_call_count), 0) AS calls,
                           COALESCE(SUM(input_tokens), 0) AS input_tokens,
                           COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens,
                           COALESCE(SUM(cache_write_tokens), 0) AS cache_write_tokens,
                           COALESCE(SUM(output_tokens), 0) AS output_tokens
                    FROM sessions
                    WHERE started_at >= ? AND started_at < ?
                    GROUP BY model
                    ORDER BY calls DESC, input_tokens DESC
                    """,
                    (start, end),
                ).fetchall()
        finally:
            con.close()
        for r in rows:
            rows_out.append(
                {
                    "profile": profile,
                    "model": r[0],
                    "sessions": int(r[1] or 0),
                    "turns": int(r[2] or 0),
                    "calls": int(r[3] or 0),
                    "input_tokens": int(r[4] or 0),
                    "cache_read_tokens": int(r[5] or 0),
                    "cache_write_tokens": int(r[6] or 0),
                    "output_tokens": int(r[7] or 0),
                }
            )
    return sorted(rows_out, key=lambda r: (int(r["calls"]), int(r["input_tokens"])), reverse=True)


# ── Hindsight API / DB / logs ─────────────────────────────────


def hindsight_api(path: str) -> dict[str, Any] | list[Any] | None:
    url = f"{HINDSIGHT_API}{path}"
    try:
        req = Request(url)
        with urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        return {"_error": str(exc)}


def hindsight_stats() -> dict[str, Any] | None:
    data = hindsight_api(f"/v1/default/banks/{quote(HINDSIGHT_BANK, safe='')}/stats")
    return data if isinstance(data, dict) else None


def hindsight_config() -> dict[str, Any] | None:
    data = hindsight_api(f"/v1/default/banks/{quote(HINDSIGHT_BANK, safe='')}/config")
    return data if isinstance(data, dict) else None


def hindsight_failed_ops(limit: int = 5) -> list[dict[str, Any]]:
    data = hindsight_api(
        f"/v1/default/banks/{quote(HINDSIGHT_BANK, safe='')}/operations?status=failed&exclude_parents=true&limit={limit}"
    )
    if isinstance(data, dict):
        return data.get("operations", []) or []
    return []


def find_psql() -> str | None:
    for cand in PSQL_CANDIDATES:
        if cand and Path(cand).exists():
            return cand
    return None


def psql(sql: str, timeout: int = 60) -> list[list[str]]:
    binary = find_psql()
    if not binary:
        raise FileNotFoundError("psql not found in known locations")
    cmd = [
        binary,
        "-h",
        "/tmp",
        "-p",
        "5432",
        "-U",
        "hindsight",
        "-d",
        "hindsight",
        "-q",
        "-t",
        "-A",
        "-F",
        "\t",
        "-c",
        sql,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return [line.split("\t") for line in proc.stdout.splitlines() if line.strip()]


def sql_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("'", "''")


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def snapshot_aligned(prev_snapshot: dict[str, Any], start_dt: datetime, *, tolerance_minutes: int = 20) -> bool:
    """True when the previous snapshot timestamp matches the query-window start.

    Net total changes are always computed against the previous saved snapshot
    when available.  This flag only tells the report whether that snapshot is
    aligned with the 24h query window; manual reruns may show a valid
    "since-last-snapshot" change that is not exactly a 24h change.
    """
    raw = prev_snapshot.get("_snapshot_at")
    if not raw:
        return False
    try:
        snap_dt = datetime.fromisoformat(str(raw))
        if snap_dt.tzinfo is None:
            snap_dt = snap_dt.replace(tzinfo=TZ)
    except Exception:
        return False
    return abs((snap_dt.astimezone(TZ) - start_dt.astimezone(TZ)).total_seconds()) <= tolerance_minutes * 60


def total_change(prev_total: Any, current_total: Any) -> int | None:
    if prev_total is None:
        return None
    return _safe_int(current_total) - _safe_int(prev_total)


def fmt_maybe_delta(n: Any, base: Any = None) -> str:
    """Format a delta with optional percentage. base=previous total for pct calc."""
    if n is None:
        return "n/a"
    try:
        val = int(n)
        if base is not None and _safe_int(base) > 0:
            pct = val / _safe_int(base) * 100
            return f"{val:+,} ({pct:+.1f}%)"
        return f"{val:+,}"
    except Exception:
        return str(n)


def hindsight_db_stats(start_dt: datetime, end_dt: datetime, api_stats: dict[str, Any] | None, prev_snapshot: dict[str, Any]) -> dict[str, Any]:
    start_s = sql_ts(start_dt)
    end_s = sql_ts(end_dt)
    bank = HINDSIGHT_BANK.replace("'", "''")
    stats: dict[str, Any] = {"errors": []}
    try:
        stats["ops"] = psql(
            f"""
            SELECT operation_type, status, count(*)
            FROM async_operations
            WHERE bank_id='{bank}'
              AND created_at >= '{start_s}'::timestamptz
              AND created_at < '{end_s}'::timestamptz
            GROUP BY operation_type, status
            ORDER BY operation_type, status;
            """
        )
        stats["queue"] = psql(
            f"""
            SELECT status, operation_type, count(*)
            FROM async_operations
            WHERE bank_id='{bank}'
            GROUP BY status, operation_type
            ORDER BY status, operation_type;
            """
        )
        stats["docs"] = psql(
            f"""
            SELECT
              count(*) filter(where created_at >= '{start_s}'::timestamptz and created_at < '{end_s}'::timestamptz) AS created,
              count(*) filter(where updated_at >= '{start_s}'::timestamptz and updated_at < '{end_s}'::timestamptz and created_at < '{start_s}'::timestamptz) AS updated_existing,
              count(*) AS total
            FROM documents
            WHERE bank_id='{bank}';
            """
        )
        stats["doc_sources"] = psql(
            f"""
            SELECT CASE
                     WHEN id LIKE 'external-openclaw::%' THEN 'external-openclaw'
                     WHEN id LIKE 'external-chatmemo::%' THEN 'external-chatmemo'
                     WHEN id LIKE 'hermes-session::%' THEN 'hermes-session'
                     WHEN id LIKE 'hermes-offline-consolidation::%' THEN 'hermes-offline-consolidation'
                     ELSE coalesce(metadata->>'source_kind', metadata->>'source', '(none)')
                   END AS source,
                   count(*)
            FROM documents
            WHERE bank_id='{bank}'
              AND created_at >= '{start_s}'::timestamptz
              AND created_at < '{end_s}'::timestamptz
            GROUP BY source
            ORDER BY count(*) DESC, source
            LIMIT 8;
            """
        )
        stats["units_by_type"] = psql(
            f"""
            SELECT fact_type,
                   count(*) filter(where created_at >= '{start_s}'::timestamptz and created_at < '{end_s}'::timestamptz) AS created,
                   count(*) filter(where updated_at >= '{start_s}'::timestamptz and updated_at < '{end_s}'::timestamptz and created_at < '{start_s}'::timestamptz) AS updated_existing,
                   count(*) AS total
            FROM memory_units
            WHERE bank_id='{bank}'
            GROUP BY fact_type
            ORDER BY fact_type;
            """
        )
        stats["consolidation"] = psql(
            f"""
            SELECT
              count(*) filter(where fact_type in ('world','experience') and consolidated_at >= '{start_s}'::timestamptz and consolidated_at < '{end_s}'::timestamptz) AS base_consolidated,
              count(*) filter(where fact_type in ('world','experience') and consolidated_at is null and consolidation_failed_at is null) AS unconsolidated_base,
              count(*) filter(where fact_type in ('world','experience') and consolidation_failed_at is not null) AS failed_base,
              count(*) filter(where fact_type='observation') AS total_observations,
              count(*) filter(where fact_type='observation' and created_at >= '{start_s}'::timestamptz and created_at < '{end_s}'::timestamptz) AS observations_created,
              count(*) filter(where fact_type='observation' and updated_at >= '{start_s}'::timestamptz and updated_at < '{end_s}'::timestamptz and created_at < '{start_s}'::timestamptz) AS observations_updated_existing
            FROM memory_units
            WHERE bank_id='{bank}';
            """
        )
    except Exception as exc:
        stats["errors"].append(str(exc))

    # Add snapshot-based net total changes when DB queries succeeded.
    current = api_stats or {}
    snapshot_ok = snapshot_aligned(prev_snapshot, start_dt)
    docs = (stats.get("docs") or [["0", "0", str(current.get("total_documents", 0))]])[0]
    stats["docs_total_change"] = total_change(prev_snapshot.get("total_documents"), _safe_int(docs[2]))

    prev_by_type = prev_snapshot.get("nodes_by_fact_type") or {}
    changes_by_type: dict[str, int | None] = {}
    for row in stats.get("units_by_type") or []:
        fact_type = row[0]
        changes_by_type[fact_type] = total_change(prev_by_type.get(fact_type), _safe_int(row[3]))
    stats["total_change_window_aligned"] = snapshot_ok
    stats["units_total_change_by_type"] = changes_by_type
    return stats


def run_shell(cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 124, "", repr(exc)


def container_created_at(container: str = "hindsight") -> datetime | None:
    """Return the container's Created time as an aware datetime (CST), or None."""
    code, out, err = run_shell(f"docker inspect {shlex.quote(container)} --format '{{{{.Created}}}}'", timeout=10)
    if code != 0:
        return None
    try:
        created_utc = datetime.fromisoformat(out.strip().rstrip("Z")).replace(tzinfo=timezone.utc)
        return created_utc.astimezone(TZ)
    except Exception:
        return None


def docker_logs_since(start_dt: datetime) -> str:
    # If the container was recreated after start_dt, docker logs --since
    # cannot see logs from before the (re)creation time.  Clamp to the
    # later of start_dt and container_created_at to avoid requesting
    # data that no longer exists.
    created = container_created_at()
    effective_since = max(start_dt, created) if created else start_dt
    since = effective_since.astimezone(timezone.utc).isoformat()
    docker_cmd = f"docker logs --since {shlex.quote(since)} hindsight 2>&1"
    cmd = "sg docker -c " + shlex.quote(docker_cmd)
    code, out, err = run_shell(cmd, timeout=90)
    if code != 0:
        return out + "\n" + err
    return out


def parse_hindsight_llm_logs(log_text: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    llm_re = re.compile(
        r"scope=(?P<scope>[\w.-]+), model=(?P<model>[^,]+), "
        r"input_tokens=(?P<input>\d+), output_tokens=(?P<output>\d+), "
        r"total_tokens=(?P<total>\d+), time=(?P<time>[\d.]+)s"
    )
    usage: dict[tuple[str, str], dict[str, Any]] = {}
    for m in llm_re.finditer(log_text or ""):
        key = (m.group("model"), m.group("scope"))
        row = usage.setdefault(
            key,
            {"model": key[0], "scope": key[1], "calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "seconds": 0.0},
        )
        row["calls"] += 1
        row["input_tokens"] += int(m.group("input"))
        row["output_tokens"] += int(m.group("output"))
        row["total_tokens"] += int(m.group("total"))
        row["seconds"] += float(m.group("time"))

    batch_re = re.compile(
        r"\[CONSOLIDATION\].*?llm_batch #\d+ \((?P<mem>\d+) memories, (?P<calls>\d+) llm calls\)"
        r".*?created=(?P<created>\d+) updated=(?P<updated>\d+) skipped=(?P<skipped>\d+)"
        r"(?: failed=(?P<failed>\d+))?"
    )
    batch_stats = {"batches": 0, "memories": 0, "llm_calls": 0, "created": 0, "updated": 0, "skipped": 0, "failed": 0}
    for m in batch_re.finditer(log_text or ""):
        batch_stats["batches"] += 1
        batch_stats["memories"] += int(m.group("mem"))
        batch_stats["llm_calls"] += int(m.group("calls"))
        batch_stats["created"] += int(m.group("created"))
        batch_stats["updated"] += int(m.group("updated"))
        batch_stats["skipped"] += int(m.group("skipped"))
        batch_stats["failed"] += int(m.group("failed") or 0)
    return sorted(usage.values(), key=lambda r: (str(r["scope"]), str(r["model"]))), batch_stats


# ── Prometheus /metrics LLM usage ───────────────────────────────


def parse_hindsight_llm_prometheus() -> list[dict[str, Any]]:
    """Parse LLM token usage from Hindsight Prometheus /metrics endpoint.

    Returns list of dicts matching parse_hindsight_llm_logs format:
    [{model, scope, calls, input_tokens, output_tokens, total_tokens, seconds}]
    """
    import urllib.request
    # Bypass HTTP proxy for localhost
    _np = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
    if "127.0.0.1" not in _np and "localhost" not in _np:
        os.environ["no_proxy"] = f"127.0.0.1,localhost,{_np}".rstrip(",")
        os.environ["NO_PROXY"] = os.environ["no_proxy"]
    try:
        req = urllib.request.Request("http://127.0.0.1:8888/metrics")
        resp = urllib.request.urlopen(req, timeout=10)
        text = resp.read().decode()
    except Exception:
        return []

    input_tokens: dict[tuple[str, str], int] = defaultdict(int)
    output_tokens: dict[tuple[str, str], int] = defaultdict(int)
    calls: dict[tuple[str, str], int] = defaultdict(int)
    duration: dict[tuple[str, str], float] = defaultdict(float)

    for line in text.split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'(\w+)(?:\{([^}]+)\})?\s+([\d.e+-]+)', line)
        if not m:
            continue
        name, labels_str, value_str = m.group(1), m.group(2) or "", m.group(3)
        val = float(value_str)
        labels: dict[str, str] = {}
        for part in labels_str.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip('"')
        scope = labels.get("scope", "")
        model = labels.get("model", "")
        if not scope or not model:
            continue
        key = (scope, model)
        if name == "hindsight_llm_calls_total":
            calls[key] += int(val)
        elif name == "hindsight_llm_tokens_input_tokens_total":
            input_tokens[key] += int(val)
        elif name == "hindsight_llm_tokens_output_tokens_total":
            output_tokens[key] += int(val)
        elif name == "hindsight_llm_duration_seconds_sum":
            duration[key] += val

    result: list[dict[str, Any]] = []
    for key in sorted(calls.keys()):
        scope, model = key
        inp = input_tokens.get(key, 0)
        out = output_tokens.get(key, 0)
        result.append({
            "model": model,
            "scope": scope,
            "calls": calls[key],
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "seconds": duration.get(key, 0.0),
        })
    return result


# ── snapshot for deltas ───────────────────────────────────────


def load_snapshot() -> dict[str, Any]:
    if SNAPSHOT_FILE.exists():
        try:
            data = json.loads(SNAPSHOT_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_snapshot(stats: dict[str, Any]) -> None:
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    payload["_snapshot_at"] = datetime.now(tz=TZ).isoformat()
    SNAPSHOT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def should_save_snapshot(now: datetime) -> bool:
    """Only scheduled morning runs should advance the delta baseline.

    Manual reruns later in the day are useful for live correction, but if they
    overwrite `.snapshot.json`, the next 08:30 cron no longer has a baseline
    aligned with its 24h window.  Keep the default cron window generous for
    scheduler jitter, but avoid arbitrary manual reruns corrupting tomorrow's
    net-change baseline.
    """
    local = now.astimezone(TZ)
    return local.hour == 8 or (local.hour == 9 and local.minute <= 30)


# ── main ──────────────────────────────────────────────────────


def main() -> int:
    try:
        return _main()
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ daily_stats.py 异常: {e}\n{tb}", file=sys.stderr)
        # 尽量写一个错误日报文件，避免无文件
        try:
            now = datetime.now(tz=TZ)
            date_str = now.strftime("%Y-%m-%d")
            WIKI_DIR.mkdir(parents=True, exist_ok=True)
            err_path = WIKI_DIR / f"{date_str}.md"
            err_path.write_text(f"# Hermes 日报 {date_str}\n\n⚠️ 生成失败: {e}\n\n```\n{tb}\n```\n", encoding="utf-8")
            print(f"已存档(错误): {err_path}", file=sys.stderr)
        except Exception:
            pass
        # stdout 也要有内容，避免 cron 认为 empty = silent
        print(f"⚠️ 日报生成失败: {e}")
        return 1


def _main() -> int:
    now = datetime.now(tz=TZ)
    end_ts = now.timestamp()
    start_dt = now - timedelta(days=1)
    start_ts = start_dt.timestamp()
    date_str = now.strftime("%Y-%m-%d")
    today_str = now.strftime("%Y-%m-%d %H:%M")

    usage = hermes_model_usage(start_ts, end_ts)
    api_stats = hindsight_stats()
    hs_ok = bool(api_stats and "_error" not in api_stats)
    config = hindsight_config() if hs_ok else None
    prev = load_snapshot()
    failed_ops = hindsight_failed_ops(5) if hs_ok else []
    failed_count = _safe_int(api_stats.get("failed_operations")) if hs_ok else 0
    db = hindsight_db_stats(start_dt, now, api_stats if hs_ok else None, prev)
    llm_usage, batch_logs = parse_hindsight_llm_logs(docker_logs_since(start_dt))
    prev_ext = load_ext_snapshot()
    external = collect_external_agent_usage(start_ts, end_ts, prev_ext)

    db_docs_row = (db.get("docs") or [["0", "0", str((api_stats or {}).get("total_documents", 0))]])[0]
    db_cons_row = (db.get("consolidation") or [["0", "0", "0", str((api_stats or {}).get("total_observations", 0)), "0", "0"]])[0]
    display_total_documents = _safe_int(db_docs_row[2]) if hs_ok else 0
    display_total_observations = _safe_int(db_cons_row[3]) if hs_ok else 0

    if hs_ok and prev:
        delta_docs = display_total_documents - _safe_int(prev.get("total_documents"))
        delta_obs = display_total_observations - _safe_int(prev.get("total_observations"))
        delta_nodes = _safe_int(api_stats.get("total_nodes")) - _safe_int(prev.get("total_nodes"))
        delta_links = _safe_int(api_stats.get("total_links")) - _safe_int(prev.get("total_links"))
    else:
        delta_docs = delta_obs = delta_nodes = delta_links = 0

    lines: list[str] = []
    lines.append(f"# Hermes 日报 {date_str}")
    lines.append(f"生成时间: {today_str} {now.tzname()}")
    lines.append(f"统计窗口: {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {today_str} {now.tzname()}")
    lines.append("")

    # Helper for Δ with percentage
    def fmt_delta_pct(val: int, prev: int) -> str:
        if prev > 0:
            pct = val / prev * 100
            return f"{val:+,} ({pct:+.1f}%)"
        return f"{val:+,}"

    lines.append("## 概要")
    if hs_ok:
        lines.append(
            f"Documents: {fmt_num(display_total_documents)} (Δ{fmt_delta_pct(delta_docs, _safe_int(prev.get('total_documents', 0)))}), "
            f"Observations: {fmt_num(display_total_observations)} (Δ{fmt_delta_pct(delta_obs, _safe_int(prev.get('total_observations', 0)))})"
        )
        lines.append(
            f"Nodes: {fmt_num(api_stats.get('total_nodes', 0))} (Δ{fmt_delta_pct(delta_nodes, _safe_int(prev.get('total_nodes', 0)))}), "
            f"Links: {fmt_num(api_stats.get('total_links', 0))} (Δ{fmt_delta_pct(delta_links, _safe_int(prev.get('total_links', 0)))})"
        )
        last_cons = api_stats.get("last_consolidated_at", "")
        if last_cons:
            lines.append(f"Last consolidation: {last_cons}")
        # Annotate large swings (>10% change) in the summary
        notable = []
        if abs(delta_obs) > _safe_int(prev.get("total_observations", 0)) * 0.1:
            notable.append(f"observations {fmt_delta_pct(delta_obs, _safe_int(prev.get('total_observations', 0)))}")
        if abs(delta_nodes) > _safe_int(prev.get("total_nodes", 0)) * 0.1:
            notable.append(f"nodes {fmt_delta_pct(delta_nodes, _safe_int(prev.get('total_nodes', 0)))}")
        if notable:
            lines.append(f"⚠️ 大幅变动: {', '.join(notable)} — 可能因 v2 rebuild/conflict 清理或批量导入")
    else:
        lines.append(f"Hindsight API 不可用: {(api_stats or {}).get('_error', '?')}")
    lines.append("")

    # Merge external agent rows into Hermes usage table
    # External agents (codex, deepseek-tui) now carry kanban role in profile field
    ext_in_hermes = []
    for r in external.get("rows", []):
        prof = str(r.get("profile", ""))
        # Only merge if profile is a known kanban role (not generic "deepseek-tui" or "codex")
        if prof in ("implementer", "planner", "critic", "coordinator"):
            ext_in_hermes.append({
                "profile": prof,
                "model": str(r.get("model", "?")),
                "sessions": 0,  # external agents don't track sessions
                "turns": 0,
                "calls": int(r.get("calls", 0)),
                "input_tokens": int(r.get("input_tokens", 0)),
                "cache_read_tokens": int(r.get("cache_read_tokens", 0)),
                "cache_write_tokens": 0,
                "output_tokens": int(r.get("output_tokens", 0)),
                "_source": "external",
            })

    lines.append("## Hermes / Profiles 模型用量（24h）")
    if usage or ext_in_hermes:
        all_usage = list(usage) + ext_in_hermes
        total = defaultdict(int)
        table_rows = []
        for r in all_usage:
            for k in ("sessions", "turns", "calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
                total[k] += int(r[k])
            real_in = max(int(r["input_tokens"]) - int(r["cache_read_tokens"]), 0)
            table_rows.append(
                [
                    str(r["profile"]) + (" *" if r.get("_source") == "external" else ""),
                    str(r["model"]),
                    fmt_num(r["sessions"]),
                    fmt_num(r["turns"]),
                    fmt_num(r["calls"]),
                    fmt_tok(real_in),
                    fmt_tok(r["cache_read_tokens"]),
                    fmt_tok(r["output_tokens"]),
                ]
            )
        total_real_in = max(int(total["input_tokens"]) - int(total["cache_read_tokens"]), 0)
        table_rows.append(
            [
                "**合计**",
                "",
                fmt_num(total["sessions"]),
                fmt_num(total["turns"]),
                fmt_num(total["calls"]),
                fmt_tok(total_real_in),
                fmt_tok(total["cache_read_tokens"]),
                fmt_tok(total["output_tokens"]),
            ]
        )
        lines.append(make_table(["Profile", "模型", "会话", "轮数", "调用", "输入(增量)", "Cache读", "输出"], table_rows))

        model_totals: dict[str, defaultdict[str, int]] = {}
        for r in all_usage:
            model = str(r["model"])
            row = model_totals.setdefault(model, defaultdict(int))
            for k in ("sessions", "turns", "calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
                row[k] += int(r[k])
        model_rows = []
        for model, row in sorted(model_totals.items(), key=lambda item: (item[1]["calls"], item[1]["input_tokens"]), reverse=True):
            m_real_in = max(int(row["input_tokens"]) - int(row["cache_read_tokens"]), 0)
            model_rows.append(
                [
                    model,
                    fmt_num(row["sessions"]),
                    fmt_num(row["turns"]),
                    fmt_num(row["calls"]),
                    fmt_tok(m_real_in),
                    fmt_tok(row["cache_read_tokens"]),
                    fmt_tok(row["output_tokens"]),
                ]
            )
        model_rows.append(
            [
                "**合计**",
                fmt_num(total["sessions"]),
                fmt_num(total["turns"]),
                fmt_num(total["calls"]),
                fmt_tok(total_real_in),
                fmt_tok(total["cache_read_tokens"]),
                fmt_tok(total["output_tokens"]),
            ]
        )
        lines.append("")
        lines.append("按模型汇总:")
        lines.append(make_table(["模型", "会话", "轮数", "调用", "输入(增量)", "Cache读", "输出"], model_rows))
    else:
        lines.append("无 Hermes / profile 会话记录。")
    lines.append("")

    lines.append("## Hindsight LLM 用量")
    if llm_usage:
        lines.append("数据源: docker logs（精确值）")
        lines.append(
            make_table(
                ["模型", "scope", "调用", "输入", "输出", "总tokens", "耗时"],
                [
                    [
                        str(r["model"]),
                        str(r["scope"]),
                        fmt_num(r["calls"]),
                        fmt_tok(r["input_tokens"]),
                        fmt_tok(r["output_tokens"]),
                        fmt_tok(r["total_tokens"]),
                        f"{float(r['seconds']) / 60:.1f}min",
                    ]
                    for r in llm_usage
                ],
            )
        )
    else:
        # Fallback: try Prometheus /metrics
        prom_llm = parse_hindsight_llm_prometheus()
        if prom_llm:
            lines.append("数据源: Prometheus /metrics（容器重启后 counters 归零，仅反映当前容器生命周期）")
            lines.append(
                make_table(
                    ["模型", "scope", "调用", "输入", "输出", "总tokens", "耗时"],
                    [
                        [
                            str(r["model"]),
                            str(r["scope"]),
                            fmt_num(r["calls"]),
                            fmt_tok(r["input_tokens"]),
                            fmt_tok(r["output_tokens"]),
                            fmt_tok(r["total_tokens"]),
                            f"{float(r['seconds']) / 60:.1f}min",
                        ]
                        for r in prom_llm
                    ],
                )
            )
        else:
            lines.append("未从当前容器日志捕获 Hindsight LLM token 记录；容器重启/日志轮转前的用量可能不可回溯。")
    if batch_logs.get("batches"):
        lines.append(
            f"consolidation batch 日志: {batch_logs['batches']}批 / {batch_logs['memories']} memories / "
            f"{batch_logs['llm_calls']} LLM calls / created={batch_logs['created']} updated={batch_logs['updated']} "
            f"skipped={batch_logs['skipped']} failed={batch_logs['failed']}"
        )
    lines.append("")

    lines.append("## 外部 CLI Agent 调用统计（增量模式，较快照）")
    lines.append("注：输入(增量) = 真实新输入 = prompt - cache；估算行的 Cache读 标 - ，输入含 cache。")
    if external["rows"]:
        ext_rows = []
        for r in external["rows"]:
            inp = r.get("input_tokens", 0)
            cache = r.get("cache_read_tokens", 0)
            is_estimated = "estimated" in str(r.get("_note", ""))
            if is_estimated:
                # 估算行：cache 是假设的 99.5%，不可靠 → 只显示总输入，Cache读 标 -
                display_in = fmt_tok(inp)
                display_cache = "-"
            else:
                # 精确行：有真实 cache 分拆 → 显示增量输入 + 精确 Cache读
                display_in = fmt_tok(max(inp - cache, 0) if cache > 0 else inp)
                display_cache = fmt_tok(cache)
            ext_rows.append([
                str(r.get("profile", "?")),
                str(r.get("model", "?")),
                fmt_num(r.get("calls", 0)),
                display_in,
                display_cache,
                fmt_tok(r.get("output_tokens", 0)),
            ])
        lines.append(make_table(
            ["来源", "模型", "调用/消息", "输入", "Cache读", "输出"],
            ext_rows,
        ))
        # Show cost if available (DeepSeek + Reasonix) — now shows delta cost
        ds_cost_rows = []
        rx_cost_rows = []
        for r in external["rows"]:
            prof = r.get("profile", "")
            if prof == "deepseek-tui":
                cost_usd = r.get("_cost_usd", 0)
                cost_cny = r.get("_cost_cny", 0)
                subagent_usd = r.get("_subagent_cost_usd", 0)
                subagent_cny = r.get("_subagent_cost_cny", 0)
                if cost_usd or cost_cny or subagent_usd or subagent_cny:
                    ds_cost_rows.append([
                        prof,
                        r.get("model", "?"),
                        f"${cost_usd:.4f}",
                        f"¥{cost_cny:.2f}",
                        f"${subagent_usd:.4f}",
                        f"¥{subagent_cny:.2f}",
                    ])
            elif prof == "reasonix" or prof in ("implementer", "planner", "critic", "coordinator"):
                # Reasonix rows also carry _cost_usd (from usage.jsonl)
                cost_usd = r.get("_cost_usd", 0)
                if cost_usd:
                    rx_cost_rows.append([
                        prof,
                        r.get("model", "?"),
                        f"${cost_usd:.4f}",
                    ])
        if ds_cost_rows:
            lines.append("")
            lines.append("DeepSeek 费用（增量）:")
            lines.append(make_table(["来源", "模型", "主任务USD", "主任务CNY", "子agentUSD", "子agentCNY"], ds_cost_rows))
        if rx_cost_rows:
            lines.append("")
            lines.append("Reasonix 费用（增量）:")
            lines.append(make_table(["来源", "模型", "费用USD"], rx_cost_rows))
    else:
        lines.append("未采集到外部 CLI Agent 调用记录。")
    if external["errors"]:
        lines.append("")
        for e in external["errors"]:
            lines.append(f"- ⚠️ 外部 Agent 采集异常: {e}")
    lines.append("")

    lines.append("## Hindsight 工作量 / 数据修改")
    if hs_ok:
        cfg = (config or {}).get("config", {}) if isinstance(config, dict) else {}
        _unconsol = _safe_int(api_stats.get("pending_consolidation", 0))
        _consol_total = _safe_int(api_stats.get("total_nodes", 0)) - _safe_int(api_stats.get("total_observations", 0))
        _unconsol_pct = f"{_unconsol / max(1, _consol_total) * 100:.1f}%" if _consol_total > 0 else "?"
        lines.append(
            f"状态: healthy；enable_observations={cfg.get('enable_observations', '?')}；"
            f"待整合={_unconsol} ({_unconsol_pct})；"
            f"failed={api_stats.get('failed_consolidation', 0)}"
        )
        ops_rows = db.get("ops") or []
        if ops_rows:
            lines.append("操作队列（窗口内创建）:")
            lines.append(make_table(["operation", "status", "数量"], [[a, b, fmt_num(c)] for a, b, c in ops_rows]))

        docs = (db.get("docs") or [["0", "0", str(api_stats.get("total_documents", 0))]])[0]
        cons = (db.get("consolidation") or [["0", "0", "0", str(api_stats.get("total_observations", 0)), "0", "0"]])[0]
        unit_rows = db.get("units_by_type") or []

        lines.append("核心修改统计（窗口内 DB 修改）:")
        lines.append(
            make_table(
                ["对象", "新增", "更新(既有)", "净变化"],
                [
                    ["documents", fmt_num(docs[0]), fmt_num(docs[1]), fmt_maybe_delta(int(docs[0]) - int(docs[1]), base=docs[2])],
                    ["observations", fmt_num(cons[4]), fmt_num(cons[5]), fmt_maybe_delta(int(cons[4]) - int(cons[5]), base=cons[3])],
                ],
            )
        )
        if unit_rows:
            lines.append("memory_units 修改统计（窗口内）:")
            lines.append(
                make_table(
                    ["fact_type", "新增", "更新(既有)"],
                    [[r[0], fmt_num(r[1]), fmt_num(r[2])] for r in unit_rows],
                )
            )

        element_rows: list[list[str]] = []

        def add_element(category: str, name: str, prev_value: Any, current_value: Any) -> None:
            element_rows.append(
                [
                    category,
                    name,
                    fmt_num(prev_value) if prev_value is not None else "n/a",
                    fmt_num(current_value),
                    fmt_maybe_delta(total_change(prev_value, current_value), base=prev_value),
                ]
            )

        prev_nodes_by_type = prev.get("nodes_by_fact_type") or {}
        prev_links_by_type = prev.get("links_by_link_type") or {}
        prev_links_by_fact_type = prev.get("links_by_fact_type") or {}
        current_nodes_total = sum(_safe_int(r[3]) for r in unit_rows) or api_stats.get("total_nodes", 0)

        add_element("top", "documents", prev.get("total_documents"), docs[2])
        add_element("top", "memory_units/nodes", prev.get("total_nodes"), current_nodes_total)
        add_element("top", "observations", prev.get("total_observations"), cons[3])
        add_element("top", "links", prev.get("total_links"), api_stats.get("total_links", 0))
        for r in unit_rows:
            add_element("memory_units", str(r[0]), prev_nodes_by_type.get(r[0]), r[3])
        for link_type, current_value in sorted((api_stats.get("links_by_link_type") or {}).items()):
            add_element("links_by_link_type", str(link_type), prev_links_by_type.get(link_type), current_value)
        for fact_type, current_value in sorted((api_stats.get("links_by_fact_type") or {}).items()):
            add_element("links_by_fact_type", str(fact_type), prev_links_by_fact_type.get(fact_type), current_value)
        if element_rows:
            lines.append("Hindsight 元素总数变化（较快照）:")
            lines.append(make_table(["类别", "元素", "上次快照", "当前总数", "变化"], element_rows))

        lines.append(
            f"base units 已 consolidation: {fmt_num(cons[0])}；unconsolidated_base: {fmt_num(cons[1])}；failed_base: {fmt_num(cons[2])}"
        )
        if not db.get("total_change_window_aligned"):
            lines.append("注：总数变化=当前总数-上次日报快照总数；本次快照与24h统计窗口不完全对齐，因此它表示较上次快照的净变化，不等同于窗口内新增。")
        sources = db.get("doc_sources") or []
        if sources:
            lines.append("documents 新增来源(top): " + "; ".join(f"{s}:{c}" for s, c in sources))
        if db.get("errors"):
            lines.append("DB 统计异常: " + "; ".join(str(e) for e in db["errors"]))
    else:
        lines.append("Hindsight API 不可用。")
    lines.append("")

    lines.append("## Hindsight 当前队列")
    if hs_ok:
        ops_by_status = api_stats.get("operations_by_status", {}) or {}
        queue_rows = db.get("queue") or []
        lines.append(
            make_table(
                ["指标", "当前值"],
                [
                    ["Pending operations", fmt_num(api_stats.get("pending_operations", 0))],
                    ["Failed operations", fmt_num(api_stats.get("failed_operations", 0))],
                    ["Completed operations", fmt_num(ops_by_status.get("completed", 0))],
                    ["待整合 base units", f"{fmt_num(api_stats.get('pending_consolidation', 0))} ({_unconsol_pct})"],
                ],
            )
        )
        if queue_rows:
            lines.append("队列按 status/type:")
            lines.append(make_table(["status", "operation", "数量"], [[a, b, fmt_num(c)] for a, b, c in queue_rows]))
    else:
        lines.append("未读取到队列统计。")
    lines.append("")

    lines.append("## 异常")
    anomalies: list[str] = []
    if failed_count > 0:
        anomalies.append(f"- ⚠️ Hindsight failed_operations = {failed_count}")
        for op in failed_ops:
            err = str(op.get("error_message", "?"))[:120]
            anomalies.append(f"  - [{op.get('task_type','?')}] {str(op.get('id','?'))[:12]}: {err}")
    if db.get("errors"):
        anomalies.append("- ⚠️ Hindsight DB 统计异常: " + "; ".join(str(e) for e in db["errors"]))
    if not anomalies:
        anomalies.append("无异常。")
    lines.extend(anomalies)
    lines.append("")

    lines.append("---")
    lines.append("_自动生成，无 LLM 调用；Hindsight token 为容器日志可见值；总数变化为当前总数减上次日报快照总数。_")
    lines.append("")

    report = "\n".join(lines)
    WIKI_DIR.mkdir(parents=True, exist_ok=True)
    wiki_path = WIKI_DIR / f"{date_str}.md"
    wiki_path.write_text(report, encoding="utf-8")
    print(f"已存档: {wiki_path}", file=sys.stderr)

    if hs_ok and should_save_snapshot(now):
        save_snapshot(api_stats)

    # Save external agent snapshot (Codex/DeepSeek thread-level cumulative values)
    # Merge: keep previous snapshot entries, update with new data
    if should_save_ext_snapshot(now) and external.get("new_snapshot_data"):
        merged = dict(prev_ext)
        for key, new_val in external["new_snapshot_data"].items():
            if key in merged and isinstance(merged[key], dict) and isinstance(new_val, dict):
                merged[key] = {**merged[key], **new_val}
            else:
                merged[key] = new_val
        save_ext_snapshot(merged)

    # stdout → WeChat (concise)
    print(f"📊 Hermes 日报 {date_str}")
    print()
    if usage:
        print("【模型用量 24h】")
        for r in usage[:6]:
            label = f"{r['profile']}/{r['model']}"
            real_in = max(int(r['input_tokens']) - int(r['cache_read_tokens']), 0)
            print(
                f"  {str(label)[:30]:30s} "
                f"{r['sessions']:2d}会话 {r['turns']:4d}轮 {r['calls']:4d}调用 "
                f"{fmt_tok(real_in):>8s}输入(增量) "
                f"cache {fmt_tok(r['cache_read_tokens'])} "
                f"{fmt_tok(r['output_tokens']):>8s} out"
            )
        total = defaultdict(int)
        for r in usage:
            for k in ("sessions", "turns", "calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
                total[k] += int(r[k])
        total_real_in = max(int(total['input_tokens']) - int(total['cache_read_tokens']), 0)
        print(
            f"  合计: {total['sessions']}会话 {total['turns']}轮 {total['calls']}调用, "
            f"{fmt_tok(total_real_in)}输入(增量) / cache {fmt_tok(total['cache_read_tokens'])} / {fmt_tok(total['output_tokens'])} out"
        )
        print()

    if llm_usage:
        print("【Hindsight LLM】")
        for r in llm_usage[:4]:
            print(f"  {r['scope']} {r['model']}: {r['calls']}调用 {fmt_tok(r['input_tokens'])} in / {fmt_tok(r['output_tokens'])} out")
        print()

    if hs_ok:
        docs = (db.get("docs") or [["0", "0", str(api_stats.get("total_documents", 0))]])[0]
        cons = (db.get("consolidation") or [["0", "0", "0", str(api_stats.get("total_observations", 0)), "0", "0"]])[0]
        print("【Hindsight】")
        print(f"  Documents: {fmt_num(display_total_documents)} (较快照 {fmt_maybe_delta(db.get('docs_total_change'))}; 新增{docs[0]} / 更新{docs[1]})")
        print(f"  Observations: {fmt_num(display_total_observations)} (较快照 {fmt_maybe_delta((db.get('units_total_change_by_type') or {}).get('observation'))}; 新增{cons[4]} / 更新{cons[5]})")
        print(f"  Nodes: {fmt_num(api_stats.get('total_nodes',0))} (Δ{fmt_delta_pct(delta_nodes, _safe_int(prev.get('total_nodes', 0)))})  Links: {fmt_num(api_stats.get('total_links',0))} (Δ{fmt_delta_pct(delta_links, _safe_int(prev.get('total_links', 0)))})")
        if failed_count:
            print(f"  ⚠️ Failed ops: {failed_count}")
        else:
            print("  ✅ 无 failed ops")
        print()

    if external["rows"]:
        print("【外部 Agent 调用】")
        for r in external["rows"]:
            # Determine agent prefix: Codex threads may have kanban roles (planner/implementer/critic)
            # instead of "codex". Use model name as fallback: gpt-5.*, o3.*, o4.* → Codex.
            profile = r.get("profile", "")
            model = r.get("model", "?")
            if profile == "codex":
                pfx = "Codex"
            elif profile == "reasonix":
                pfx = "Reasonix"
            elif profile in ("planner", "implementer", "critic", "coordinator") and re.match(r"(gpt-5|o[34]|codex)", model):
                pfx = "Codex"
            elif profile in ("planner", "implementer", "critic", "coordinator"):
                pfx = "Codex"  # Kanban role always means Codex regardless of model
            else:
                pfx = "DeepSeek"
            calls = r.get("calls", 0)
            inp = r.get("input_tokens", 0) or 0
            cache = r.get("cache_read_tokens", 0) or 0
            outp = r.get("output_tokens", 0) or 0
            cost = r.get("_cost_cny", 0)
            cost_str = f" ¥{cost:.2f}" if cost else ""
            real_in = inp - cache if cache > 0 else inp
            is_codex = pfx == "Codex"
            has_cache_breakdown = cache > 0

            if is_codex and has_cache_breakdown:
                # Codex with jsonl breakdown: real_in is actual new input, cache is cached context
                print(f"  {pfx}/{model}: {calls}线程 输入(增量)={fmt_tok(real_in)} cache={fmt_tok(cache)} out={fmt_tok(outp)}{cost_str}")
            elif is_codex and inp > 0:
                # Fallback: no breakdown, input includes cached
                print(f"  {pfx}/{model}: {calls}线程 输入={fmt_tok(inp)}(含cache) out={fmt_tok(outp)}{cost_str}")
            elif pfx == "Reasonix":
                is_estimated = "estimated" in str(r.get("_note", ""))
                if is_estimated:
                    # 估算行：cache 是假设的 99.5%，不可靠 → 只显示总输入
                    print(f"  {pfx}/{model}: {calls}调用 输入={fmt_tok(inp)}(含cache) out={fmt_tok(outp)}")
                else:
                    # 精确行：usage.jsonl 有真实 cache 分拆
                    print(f"  {pfx}/{model}: {calls}调用 输入(增量)={fmt_tok(real_in)} cache={fmt_tok(cache)} out={fmt_tok(outp)}")
            else:
                # DeepSeek
                if has_cache_breakdown:
                    print(f"  {pfx}/{model}: {calls}调用 输入(增量)={fmt_tok(real_in)} cache={fmt_tok(cache)} out={fmt_tok(outp)}{cost_str}")
                else:
                    print(f"  {pfx}/{model}: {calls}调用 输入={fmt_tok(inp)} out={fmt_tok(outp)}{cost_str}")
        if external["errors"]:
            for e in external["errors"]:
                print(f"  ⚠️ {e}")
        print()

    if failed_ops:
        print("⚠️ 异常:")
        for op in failed_ops[:3]:
            err = str(op.get("error_message", "?"))[:80]
            print(f"  [{op.get('task_type','?')}] {err}")
        print()

    print(f"📄 日报已存档 wiki/auto-maintenance/daily/{date_str}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
