#!/usr/bin/env python3
"""Hermes 每日日报数据注入脚本。

输出给 cron agent 直接采用的文字/表格统计，不包含凭证。
统计窗口默认最近 24 小时；重点补充 Hindsight 模型调用与 retain/reflect/consolidation/observations 工作量。
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib import request as urlrequest

# Disable proxy for localhost connections (cron environment may have stale proxy env vars)
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

HERMES_HOME = Path("/home/wyr/.hermes")
STATE_DB = HERMES_HOME / "state.db"
PROFILE_DIR = HERMES_HOME / "profiles"
PSQL = os.environ.get("HINDSIGHT_PSQL", "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql")
HINDSIGHT_API = os.environ.get("HINDSIGHT_API", "http://127.0.0.1:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "hermes")
MENTAL_MODEL_ROOT = HERMES_HOME / "mental-models" / "egomotion4d"
MENTAL_MODEL_REGISTRY = MENTAL_MODEL_ROOT / "registry.json"
PROJECT_MAINTENANCE_ROOT = Path(
    "/home/wyr/wiki/auto-maintenance/project/egomotion4d"
)
RESEARCH_DIGEST_DIR = PROJECT_MAINTENANCE_ROOT / "research-digests"


def fmt_num(num: int | float | None) -> str:
    if num is None:
        return "0"
    try:
        n = int(num)
    except Exception:
        return str(num)
    return f"{n:,}"


def fmt_tok(num: int | float | None) -> str:
    if num is None:
        return "0"
    try:
        n = int(num)
    except Exception:
        return str(num)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def run(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 124, "", repr(exc)


def psql(sql: str, timeout: int = 60) -> list[list[str]]:
    if not Path(PSQL).exists():
        raise FileNotFoundError(f"psql not found: {PSQL}")
    cmd = [
        PSQL,
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
    return dt.astimezone().isoformat()


def iter_state_dbs() -> list[tuple[str, Path]]:
    """Return default + profile state DBs.

    Hermes profiles keep independent state.db files.  Kanban visible-agent
    profiles (coordinator/planner/implementer/critic) therefore do not appear
    in ~/.hermes/state.db and must be aggregated explicitly.
    """
    dbs: list[tuple[str, Path]] = []
    if STATE_DB.exists():
        dbs.append(("default", STATE_DB))
    if PROFILE_DIR.exists():
        for db in sorted(PROFILE_DIR.glob("*/state.db")):
            profile = db.parent.name
            dbs.append((profile, db))
    return dbs


WIKI_DAILY_DIR = Path("/home/wyr/wiki/auto-maintenance/daily")


def parse_prev_report(date_str: str) -> dict | None:
    """Parse previous day's report to get baseline values for delta."""
    prev_path = WIKI_DAILY_DIR / f"{date_str}.md"
    if not prev_path.exists():
        return None
    stats = {}
    try:
        text = prev_path.read_text()
        # Extract: documents 总数 | X (may contain commas: 27,207)
        m = re.search(r'documents 总数[|\s]*([\d,]+)', text)
        if not m:
            m = re.search(r'Documents=([\d,]+)', text)
        if m:
            stats['documents'] = int(m.group(1).replace(',', ''))
        m = re.search(r'observations 总数[|\s]*([\d,]+)', text)
        if not m:
            m = re.search(r'Observations=([\d,]+)', text)
        if m:
            stats['observations'] = int(m.group(1).replace(',', ''))
        m = re.search(r'base units 已 consolidation[|\s]*(\d+)', text)
        if not m:
            m = re.search(r'Consolidation:\s*base_done=([\d,]+)', text)
        if m:
            stats['consolidated'] = int(m.group(1).replace(',', ''))
        m = re.search(r'unconsolidated_base 剩余[|\s]*(\d+)', text)
        if not m:
            m = re.search(r'unconsolidated=([\d,]+)', text)
        if m:
            stats['unconsolidated'] = int(m.group(1).replace(',', ''))
    except Exception:
        return None
    return stats if stats else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def mental_model_rows(
    start: datetime,
    end: datetime,
    *,
    registry_path: Path = MENTAL_MODEL_REGISTRY,
) -> list[dict[str, str]]:
    """Return concise, registry-derived model state for the report window."""
    if not registry_path.exists():
        return []
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    rows: list[dict[str, str]] = []
    for logical_id, model in sorted(registry.get("models", {}).items()):
        accepted = model.get("accepted_revision")
        if not isinstance(accepted, dict):
            rows.append(
                {
                    "logical_id": logical_id,
                    "state": str(model.get("last_verdict") or "INITIAL"),
                    "change": "NO_ACCEPTED_REVISION",
                    "quality": "-",
                    "revision": "-",
                }
            )
            continue

        exact = (
            model.get("last_verdict") == "PASS_PUBLISH"
            and accepted.get("slot") == model.get("active_slot")
            and accepted.get("source_evidence_sha")
            == model.get("source_evidence_sha")
            and bool(accepted.get("content_sha"))
        )
        accepted_at = _parse_timestamp(accepted.get("accepted_at"))
        changed = bool(
            accepted_at
            and start.timestamp() <= accepted_at.timestamp() < end.timestamp()
        )
        quality_match = re.search(r"quality=(\d+)", str(model.get("verdict_detail", "")))
        rows.append(
            {
                "logical_id": logical_id,
                "state": "READY" if exact else "STALE_OR_BLOCKED",
                "change": "PUBLISHED" if changed else "UNCHANGED",
                "quality": quality_match.group(1) if quality_match else "-",
                "revision": str(accepted.get("content_sha", ""))[:12] or "-",
            }
        )
    return rows


def window_operation_summary(rows: list[list[str]]) -> str:
    """Summarize only operations created inside the report window."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for operation, status, count in rows:
        grouped[operation][status] += int(count)
    parts = []
    for operation in sorted(grouped):
        states = grouped[operation]
        details = []
        if states.get("completed"):
            details.append(f"{states['completed']} ok")
        if states.get("failed"):
            details.append(f"{states['failed']} failed")
        if states.get("pending"):
            details.append(f"{states['pending']} pending")
        if states.get("processing"):
            details.append(f"{states['processing']} processing")
        if details:
            parts.append(f"{operation}=" + "/".join(details))
    return "; ".join(parts) if parts else "none"


def current_operation_alerts(
    _window_rows: list[list[str]], all_time_queue: list[list[str]]
) -> list[str]:
    """Report currently active work; window failures remain in the data table."""
    return [
        f"active queue: {operation} {status} x{count}"
        for status, operation, count in all_time_queue
        if status in {"pending", "processing"} and int(count) > 0
    ]


def load_research_digest(report_date: str) -> str | None:
    path = RESEARCH_DIGEST_DIR / f"{report_date}.md"
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8")
        return re.sub(r"\A\s*<!--.*?-->\s*", "", content, flags=re.DOTALL).strip()
    except OSError:
        return None


def pitfall_summary(start: datetime, end: datetime) -> dict[str, object]:
    path = MENTAL_MODEL_ROOT / "pitfall_index.json"
    result: dict[str, object] = {"counts": {}, "changes": []}
    if not path.exists():
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return result
    counts: dict[str, int] = defaultdict(int)
    changes = []
    for entry in data.get("entries", []):
        status = str(entry.get("status", "unknown"))
        counts[status] += 1
        lifecycle_times = [
            timestamp
            for timestamp in (
                _parse_timestamp(entry.get("created_at")),
                _parse_timestamp(entry.get("updated_at")),
            )
            if timestamp is not None
        ]
        if any(start <= timestamp < end for timestamp in lifecycle_times):
            changes.append(
                {
                    "p_id": str(entry.get("p_id", "?")),
                    "status": status,
                    "title": str(entry.get("title", "")),
                }
            )
    result["counts"] = dict(counts)
    result["changes"] = changes
    return result


def hermes_model_usage(start: datetime, end: datetime) -> list[dict[str, int | str]]:
    """Extract actual per-call model usage from agent.log files.

    Uses the 'API call #N: model=xxx ... in=yyy out=zzz' log lines to get
    the actual model used per API call, not the session-startup model.
    Falls back to state.db sessions.model if agent.log is unavailable.
    """
    rows_out: list[dict[str, int | str]] = []

    # Regex to parse: "API call #N: model=xxx provider=yyy in=A out=B total=C latency=Ds cache=E/F (P%)"
    _call_re = re.compile(
        r'API call #\d+:\s+model=(\S+)\s+provider=\S+\s+'
        r'in=(\d+)\s+out=(\d+)\s+total=\d+\s+latency=[\d.]+s\s+'
        r'cache=(\d+)/(\d+)'
    )

    # Map profile → agent.log path
    log_paths: list[tuple[str, Path]] = []
    default_log = HERMES_HOME / "logs" / "agent.log"
    if default_log.exists():
        log_paths.append(("default", default_log))
    if PROFILE_DIR.exists():
        for p in sorted(PROFILE_DIR.iterdir()):
            agent_log = p / "logs" / "agent.log"
            if agent_log.exists():
                log_paths.append((p.name, agent_log))

    start_ts_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_ts_str = end.strftime("%Y-%m-%d %H:%M:%S")

    for profile, log_path in log_paths:
        # Per-model aggregation
        agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "input": 0, "output": 0, "cache_read": 0, "sessions": set()})
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    # Quick filter by date prefix
                    if len(line) < 20 or line[0] != '2':
                        continue
                    ts = line[:19]  # "2026-07-16 10:19:57"
                    if ts < start_ts_str or ts >= end_ts_str:
                        continue
                    m = _call_re.search(line)
                    if not m:
                        continue
                    model = m.group(1)
                    inp = int(m.group(2))
                    outp = int(m.group(3))
                    cache_r = int(m.group(4))
                    # Extract session_id from [session_id] in the log line
                    sess_m = re.search(r'\[([0-9a-f_]+)\]', line)
                    if sess_m:
                        agg[model]["sessions"].add(sess_m.group(1))
                    agg[model]["calls"] += 1
                    agg[model]["input"] += inp
                    agg[model]["output"] += outp
                    agg[model]["cache_read"] += cache_r
        except OSError:
            continue

        for model, d in agg.items():
            rows_out.append({
                "profile": profile,
                "model": model,
                "sessions": len(d["sessions"]),
                "turns": 0,  # turns not available per-call from log
                "calls": d["calls"],
                "input_tokens": d["input"],
                "cache_read_tokens": d["cache_read"],
                "cache_write_tokens": 0,  # not in log
                "output_tokens": d["output"],
            })

    # Fallback: if no agent.log data, use state.db sessions.model
    if not rows_out:
        for profile, db_path in iter_state_dbs():
            con = sqlite3.connect(db_path)
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
                        (start.timestamp(), end.timestamp()),
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
                        (start.timestamp(), end.timestamp()),
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


def hindsight_health() -> str:
    try:
        with urlrequest.urlopen(f"{HINDSIGHT_API}/health", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            status = data.get("status", "unknown")
            db = data.get("database")
            return f"{status}" + (f" / db={db}" if db else "")
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def parse_codex_usage(target_date: str) -> dict | None:
    """Parse aggregate Codex/CodeWhale rollout usage (cumulative, need delta).

    token_count events carry per-session cumulative totals. Different sessions
    have independent counters starting from 0, so we must compute deltas per-session
    first, then sum across sessions for the target date.

    Returns a dict with per-source breakdown:
      {
        'sources': {
            'Codex': {'input': N, 'cached': N, 'output': N},
            'CodeWhale': {'input': N, 'cached': N, 'output': N},
            'Codex(kanban)': {'input': N, 'cached': N, 'output': N},
        },
        'total': {'input': N, 'cached': N, 'output': N},
      }
    """
    codex_base = Path('/home/wyr/.codex/sessions')
    cw_base = Path('/home/wyr/.codewhale/sessions')
    codex_kanban_base = Path('/home/wyr/.codex-kanban')

    # Map each scan dir to a source label
    source_dirs: list[tuple[str, Path]] = [
        ('Codex', codex_base),
        ('CodeWhale', cw_base),
    ]
    if codex_kanban_base.exists():
        for role_dir in sorted(codex_kanban_base.iterdir()):
            role_sessions = role_dir / 'sessions'
            if role_sessions.is_dir():
                source_dirs.append(('Codex(kanban)', role_sessions))

    # Collect per-session first/last token_count per date, per source
    # key = (source, session_file, date)
    session_date_entries: dict[tuple, dict] = {}

    for source_label, base in source_dirs:
        if not base.exists():
            continue
        for year_dir in sorted(base.glob('2026/*')):
            if not year_dir.is_dir():
                continue
            for day_dir in sorted(year_dir.glob('*')):
                if not day_dir.is_dir():
                    continue
                for session_file in day_dir.glob('rollout-*.jsonl'):
                    try:
                        with open(session_file) as f:
                            for line in f:
                                try:
                                    d = json.loads(line)
                                    ts = d.get('timestamp', '')
                                    if not ts or ts[:10] < '2026-05-01':
                                        continue
                                    if d.get('type') == 'event_msg':
                                        payload = d.get('payload', {})
                                        if isinstance(payload, dict) and payload.get('type') == 'token_count':
                                            info = payload.get('info', {})
                                            total = info.get('total_token_usage', {})
                                            date = ts[:10]
                                            entry = {
                                                'input': total.get('input_tokens', 0),
                                                'cached': total.get('cached_input_tokens', 0),
                                                'output': total.get('output_tokens', 0),
                                            }
                                            key = (source_label, str(session_file), date)
                                            if key not in session_date_entries:
                                                session_date_entries[key] = {'first': entry, 'last': entry}
                                            else:
                                                session_date_entries[key]['last'] = entry
                                except Exception:
                                    pass
                    except Exception:
                        pass

    from datetime import datetime, timedelta
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    prev_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')

    sources: dict[str, dict[str, int]] = {}
    found_any = False

    for (source_label, _sf, date), entries in session_date_entries.items():
        if date not in (target_date, prev_date):
            continue
        found_any = True
        first = entries['first']
        last = entries['last']
        d_input = last['input'] - first['input']
        d_cached = last['cached'] - first['cached']
        d_output = last['output'] - first['output']
        if source_label not in sources:
            sources[source_label] = {'input': 0, 'cached': 0, 'output': 0}
        sources[source_label]['input'] += d_input
        sources[source_label]['cached'] += d_cached
        sources[source_label]['output'] += d_output

    if not found_any:
        return None

    total = {'input': 0, 'cached': 0, 'output': 0}
    for s in sources.values():
        total['input'] += s['input']
        total['cached'] += s['cached']
        total['output'] += s['output']
    return {'sources': sources, 'total': total}


HINDSIGHT_ENV_FILE = Path("/home/wyr/.hindsight-docker/hindsight-native.env")
_HINDSIGHT_ENV_FILTER = re.compile(
    r"HINDSIGHT_API_(LLM|RETAIN_LLM|REFLECT_LLM|CONSOLIDATION_LLM|"
    r"ENABLE_OBSERVATIONS|WORKER_CONSOLIDATION_MAX_SLOTS|WORKER_MAX_SLOTS|RETAIN_MAX_CONCURRENT)"
)
# docker 时代环境变量名（无 HINDSIGHT_API_ 前缀）→ native env 文件中的带前缀名
_HINDSIGHT_ENV_ALIASES = {
    "WORKER_MAX_SLOTS": "HINDSIGHT_API_WORKER_MAX_SLOTS",
    "WORKER_CONSOLIDATION_MAX_SLOTS": "HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS",
    "RETAIN_MAX_CONCURRENT": "HINDSIGHT_API_RETAIN_MAX_CONCURRENT",
}


def hindsight_env() -> dict[str, str]:
    """读取 Hindsight 运行配置。

    优先解析 systemd EnvironmentFile（hindsight.service，native pip 部署）；
    docker 容器仍在运行时用 docker exec 兜底（旧部署）。
    """
    env: dict[str, str] = {}
    if HINDSIGHT_ENV_FILE.is_file():
        for line in HINDSIGHT_ENV_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            if not _HINDSIGHT_ENV_FILTER.match(k):
                continue
            if any(secret in k.upper() for secret in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
                continue
            env[k] = v.strip().strip('"').strip("'")
    if env:
        # 别名归一化：docker 时代的无前缀变量名 → 带前缀值
        for dst, src in _HINDSIGHT_ENV_ALIASES.items():
            if src in env and dst not in env:
                env[dst] = env[src]
        return env
    # docker 兜底（旧部署，容器仍在运行时）
    cmd = "docker exec hindsight sh -lc 'env | grep -E \"HINDSIGHT_API_(LLM|RETAIN_LLM|REFLECT_LLM|CONSOLIDATION_LLM|ENABLE_OBSERVATIONS|WORKER_CONSOLIDATION_MAX_SLOTS|WORKER_MAX_SLOTS|RETAIN_MAX_CONCURRENT).*\" | sort'"
    code, out, _ = run(cmd, timeout=20)
    if code != 0:
        return {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if any(secret in k.upper() for secret in ("API_KEY", "TOKEN", "SECRET", "PASSWORD")):
            continue
        env[k] = v
    return env


def hindsight_model_config(env: dict[str, str]) -> list[dict[str, str]]:
    scopes = {
        "retain": "HINDSIGHT_API_RETAIN_LLM_",
        "reflect": "HINDSIGHT_API_REFLECT_LLM_",
        "consolidation": "HINDSIGHT_API_CONSOLIDATION_LLM_",
        "default": "HINDSIGHT_API_LLM_",
    }
    rows = []
    for scope, prefix in scopes.items():
        model = env.get(prefix + "MODEL")
        provider = env.get(prefix + "PROVIDER")
        base_url = env.get(prefix + "BASE_URL")
        if model or provider or base_url:
            rows.append(
                {
                    "scope": scope,
                    "provider": provider or "?",
                    "model": model or "?",
                    "base_url": base_url or "?",
                    "location": "systemd:hindsight",
                }
            )
    return rows


def hindsight_logs_since(start: datetime, end: datetime) -> str:
    """读取 Hindsight 日志文本（窗口 start~end）。

    native pip 部署下从 systemd journald 读取（hindsight.service，日志持久化，
    服务重启不影响回溯）；迁移期补齐：journal 首条日志晚于窗口起点时，拼接
    docker 容器段日志（docker 容器已停止也可读）。journal 完全无日志时 docker 兜底。
    """
    since = start.strftime("%Y-%m-%d %H:%M:%S")
    until = end.strftime("%Y-%m-%d %H:%M:%S")
    # 1) journald 主数据源
    code, out, err = run(
        f"journalctl -u hindsight.service --since {shlex.quote(since)} --until {shlex.quote(until)} -o cat 2>&1",
        timeout=90,
    )
    if code != 0 and not out.strip():
        return ""  # journalctl 不可用且无输出（无权限等），放弃
    journal_text = out
    # 2) 迁移期补齐：journal 首条日志时间 vs 窗口起点
    try:
        code2, first_line, _ = run(
            f"journalctl -u hindsight.service --since {shlex.quote(since)} -o short-iso 2>&1 | head -1",
            timeout=30,
        )
        m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", first_line)
        if m:
            first_ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=start.tzinfo)
            if first_ts.timestamp() > start.timestamp():
                d_since = start.astimezone(timezone.utc).isoformat()
                d_until = first_ts.astimezone(timezone.utc).isoformat()
                code3, dout, _ = run(
                    f"docker logs --since {shlex.quote(d_since)} --until {shlex.quote(d_until)} hindsight 2>&1",
                    timeout=90,
                )
                if code3 == 0 and dout.strip():
                    journal_text = dout + journal_text
    except Exception:
        pass
    return journal_text


def parse_hindsight_llm_logs(log_text: str) -> tuple[list[dict[str, int | float | str]], dict[str, int]]:
    # Example:
    # slow llm call: scope=consolidation, model=openai/glm-5, input_tokens=50877, output_tokens=14188, total_tokens=65065, time=295.017s
    llm_re = re.compile(
        r"scope=(?P<scope>[\w.-]+), model=(?P<model>[^,]+), "
        r"input_tokens=(?P<input>\d+), output_tokens=(?P<output>\d+), "
        r"total_tokens=(?P<total>\d+), time=(?P<time>[\d.]+)s"
    )
    usage: dict[tuple[str, str], dict[str, int | float | str]] = {}
    for m in llm_re.finditer(log_text):
        key = (m.group("model"), m.group("scope"))
        row = usage.setdefault(
            key,
            {
                "model": key[0],
                "scope": key[1],
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "seconds": 0.0,
            },
        )
        row["calls"] = int(row["calls"]) + 1
        row["input_tokens"] = int(row["input_tokens"]) + int(m.group("input"))
        row["output_tokens"] = int(row["output_tokens"]) + int(m.group("output"))
        row["total_tokens"] = int(row["total_tokens"]) + int(m.group("total"))
        row["seconds"] = float(row["seconds"]) + float(m.group("time"))

    batch_re = re.compile(
        r"\[CONSOLIDATION\].*?llm_batch #\d+ \((?P<mem>\d+) memories, (?P<calls>\d+) llm calls\)"
        r".*?created=(?P<created>\d+) updated=(?P<updated>\d+) skipped=(?P<skipped>\d+)"
        r"(?: failed=(?P<failed>\d+))?"
    )
    batch_stats = {"batches": 0, "memories": 0, "llm_calls": 0, "created": 0, "updated": 0, "skipped": 0, "failed": 0}
    for m in batch_re.finditer(log_text):
        batch_stats["batches"] += 1
        batch_stats["memories"] += int(m.group("mem"))
        batch_stats["llm_calls"] += int(m.group("calls"))
        batch_stats["created"] += int(m.group("created"))
        batch_stats["updated"] += int(m.group("updated"))
        batch_stats["skipped"] += int(m.group("skipped"))
        batch_stats["failed"] += int(m.group("failed") or 0)

    return sorted(usage.values(), key=lambda x: (str(x["scope"]), str(x["model"]))), batch_stats


def hindsight_db_stats(start: datetime, end: datetime) -> dict:
    start_s = sql_ts(start).replace("'", "''")
    end_s = sql_ts(end).replace("'", "''")
    stats: dict[str, object] = {"errors": []}
    try:
        stats["ops"] = psql(
            f"""
            SELECT operation_type, status, count(*)
            FROM async_operations
            WHERE bank_id='{BANK}'
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
            WHERE bank_id='{BANK}'
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
            WHERE bank_id='{BANK}';
            """
        )
        stats["doc_sources"] = psql(
            f"""
            SELECT coalesce(metadata->>'source','(none)') AS source, count(*)
            FROM documents
            WHERE bank_id='{BANK}'
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
            WHERE bank_id='{BANK}'
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
            WHERE bank_id='{BANK}';
            """
        )
    except Exception as exc:
        stats["errors"].append(str(exc))
    return stats


def offline_output_counts(now: datetime) -> dict[str, str | int]:
    v2_dir = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
    rebuild_dir = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_rebuild"
    v2_manifest = v2_dir / "manifest.json"
    v2_latest = rebuild_dir / "latest.json"

    # 读取 v2 manifest 统计
    card_count = 0
    obs_index_count = 0
    if v2_manifest.exists():
        try:
            mf = json.loads(v2_manifest.read_text(encoding="utf-8"))
            card_count = mf.get("card_count", 0)
            obs_index_count = mf.get("observation_index_count", 0)
        except Exception:
            pass

    return {
        "v2_card_count": card_count,
        "v2_observation_count": obs_index_count,
        "v2_latest_mtime": datetime.fromtimestamp(v2_latest.stat().st_mtime).strftime("%m-%d %H:%M") if v2_latest.exists() else "-",
    }


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Hermes 日报生成")
    parser.add_argument("--date", type=str, default=None,
                        help="报告日期 YYYY-MM-DD，默认当前日期")
    parser.add_argument("--write", action="store_true",
                        help="补跑 --date 时也原子写入对应日报")
    args = parser.parse_args()

    now = datetime.now().astimezone()
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
        end = target_date + timedelta(hours=8, minutes=30)
        start = target_date - timedelta(hours=15, minutes=30)
        report_date_str = args.date
        now_for_print = target_date
    else:
        end = now
        start = now - timedelta(days=1)
        report_date_str = now.strftime("%Y-%m-%d")
        now_for_print = now

    hermes_rows = hermes_model_usage(start, end)
    env = hindsight_env()
    model_cfg = hindsight_model_config(env)
    logs = hindsight_logs_since(start, end)
    llm_usage, consolidation_batch_logs = parse_hindsight_llm_logs(logs)
    db = hindsight_db_stats(start, end)
    offline = offline_output_counts(now_for_print)
    model_rows = mental_model_rows(start, end)
    digest_date = (now_for_print - timedelta(days=1)).strftime("%Y-%m-%d")
    research_digest = load_research_digest(digest_date)
    pitfalls = pitfall_summary(start, end)

    print(f"📊 Hermes日报 {report_date_str}")
    print(f"统计窗口: {start.strftime('%Y-%m-%d %H:%M')} ~ {end.strftime('%Y-%m-%d %H:%M')} ({now_for_print.tzname()})")
    print()

    # Summary line with delta
    docs_meta = (db.get("docs") or [["0", "0", "0"]])[0]
    cons_meta = (db.get("consolidation") or [["0", "0", "0", "0", "0", "0"]])[0]
    total_docs = int(docs_meta[2])
    total_obs = int(cons_meta[3])
    uncons = int(cons_meta[1])
    failed = int(cons_meta[2])

    prev = None
    d_doc = d_obs = 0
    if not args.date:
        prev_date = (datetime.now().astimezone() - timedelta(days=2)).strftime("%Y-%m-%d")
        prev = parse_prev_report(prev_date)
    if prev:
        d_doc = total_docs - prev.get("documents", total_docs)
        d_obs = total_obs - prev.get("observations", total_obs)
    print(f"概要: Documents={fmt_num(total_docs)}{f' (Δ+{d_doc})' if prev and 'documents' in prev and d_doc >= 0 else f' (Δ{d_doc})' if prev and 'documents' in prev else ''}  "
          f"Observations={fmt_num(total_obs)}{f' (Δ+{d_obs})' if prev and 'observations' in prev and d_obs >= 0 else f' (Δ{d_obs})' if prev and 'observations' in prev else ''}  "
          f"unconsolidated={fmt_num(uncons)}  failed_base={fmt_num(failed)}")
    print(f"Consolidation: base_done={fmt_num(int(cons_meta[0]))}  "
          f"remaining={fmt_num(int(cons_meta[1]))}  "
          f"failed={fmt_num(int(cons_meta[2]))}")
    print()

    ops_rows = db.get("ops") or []
    queue_rows = db.get("queue") or []
    alerts = current_operation_alerts(ops_rows, queue_rows)
    ready_models = sum(row["state"] == "READY" for row in model_rows)
    published_models = sum(row["change"] == "PUBLISHED" for row in model_rows)

    print("【今日结论】")
    print(f"• Hindsight: {hindsight_health()}；consolidation backlog={fmt_num(uncons)}，failed_base={fmt_num(failed)}。")
    print(f"• Mental Models: {ready_models}/{len(model_rows)} READY；本窗口发布 {published_models} 个。")
    print(f"• 需关注事项: {len(alerts) + int(failed > 0)}；只统计窗口内失败与当前 active queue，不重复展示历史终态。")
    print()

    # ── Research Digest (full) ──
    print(f"【研发进展（{digest_date}）】")
    if research_digest:
        print(research_digest)
    else:
        print("无 compact research digest")
    print()

    # ── Mental Models (full table) ──
    print("【Mental Models】")
    if model_rows:
        print_table(
            ["模型", "窗口变化", "当前状态", "质量", "Revision"],
            [
                [
                    row["logical_id"].removeprefix("egomotion4d-"),
                    row["change"],
                    row["state"],
                    row["quality"],
                    row["revision"],
                ]
                for row in model_rows
            ],
        )
    else:
        print("无 mental models")
    print()

    # ── Hermes / Profiles 模型用量 (full table) ──
    print("【Hermes / Profiles 模型用量】")
    if hermes_rows:
        # Aggregate by profile+model
        print_table(
            ["Profile", "模型", "会话", "轮数", "调用", "输入", "Cache读", "Cache写", "输出", "命中率"],
            [
                [
                    str(r["profile"]),
                    str(r["model"]),
                    str(r["sessions"]),
                    str(r["turns"]),
                    str(r["calls"]),
                    fmt_tok(int(r["input_tokens"])),
                    fmt_tok(int(r["cache_read_tokens"])),
                    fmt_tok(int(r["cache_write_tokens"])),
                    fmt_tok(int(r["output_tokens"])),
                    f"{int(r['cache_read_tokens']) / int(r['input_tokens']) * 100:.1f}%" if int(r["input_tokens"]) else "-",
                ]
                for r in hermes_rows
            ],
        )
        # Totals row
        total_calls = sum(int(r["calls"]) for r in hermes_rows)
        total_sessions = sum(int(r["sessions"]) for r in hermes_rows)
        total_turns = sum(int(r["turns"]) for r in hermes_rows)
        total_inp = sum(int(r["input_tokens"]) for r in hermes_rows)
        total_cr = sum(int(r["cache_read_tokens"]) for r in hermes_rows)
        total_cw = sum(int(r["cache_write_tokens"]) for r in hermes_rows)
        total_out = sum(int(r["output_tokens"]) for r in hermes_rows)
        hit = f"{total_cr / total_inp * 100:.1f}%" if total_inp else "-"
        print_table(
            ["Profile", "模型", "会话", "轮数", "调用", "输入", "Cache读", "Cache写", "输出", "命中率"],
            [["合计", "", str(total_sessions), str(total_turns), str(total_calls),
              fmt_tok(total_inp), fmt_tok(total_cr), fmt_tok(total_cw), fmt_tok(total_out), hit]],
        )
    else:
        print("无记录")
    print()

    # ── 外部 CLI Agent 聚合用量 (full table) ──
    print("【外部 CLI Agent 聚合用量】")
    codex_usage = parse_codex_usage(report_date_str)
    if codex_usage:
        print_table(
            ["来源", "输入", "Cache读", "输出", "命中率"],
            [
                [
                    src,
                    fmt_tok(stats["input"]),
                    fmt_tok(stats["cached"]),
                    fmt_tok(stats["output"]),
                    f"{stats['cached'] / stats['input'] * 100:.1f}%" if stats["input"] else "-",
                ]
                for src, stats in sorted(codex_usage["sources"].items())
            ],
        )
        t = codex_usage["total"]
        t_hit = f"{t['cached'] / t['input'] * 100:.1f}%" if t["input"] else "-"
        print_table(
            ["来源", "输入", "Cache读", "输出", "命中率"],
            [["合计", fmt_tok(t["input"]), fmt_tok(t["cached"]), fmt_tok(t["output"]), t_hit]],
        )
    else:
        print("未采集到外部 CLI Agent 调用记录")
    print()

    # ── Hindsight 运行模型配置 (full table) ──
    print("【Hindsight 运行模型配置】")
    print(f"健康状态: {hindsight_health()}")
    if model_cfg:
        print_table(
            ["用途", "provider", "model", "base_url", "location"],
            [
                [row["scope"], row["provider"], row["model"], row["base_url"], row["location"]]
                for row in model_cfg
            ],
        )
    enable_obs = env.get("HINDSIGHT_API_ENABLE_OBSERVATIONS", "?")
    worker_slots = env.get("WORKER_MAX_SLOTS", "?")
    cons_slots = env.get("WORKER_CONSOLIDATION_MAX_SLOTS", "?")
    retain_conc = env.get("RETAIN_MAX_CONCURRENT", "?")
    print(f"enable_observations={enable_obs}, worker_slots={worker_slots}, consolidation_slots={cons_slots}, retain_max_concurrent={retain_conc}")
    print()

    # ── Hindsight LLM 用量 (full table) ──
    print("【Hindsight LLM 用量（journald 日志精确值）】")
    # journald 持久化下 --since 过滤准确，服务重启不影响；仅当窗口起点前无任何
    # hindsight 日志（服务尚未启动或 journal 被清理）时提示缺失。
    # 注意：journal 首条晚于窗口起点时，docker 段已在 hindsight_logs_since() 补齐，
    # 此处用 docker 容器启动时间判断补齐覆盖，只提示真正缺失的段。
    log_gap_note = None
    try:
        cmd = f"journalctl -u hindsight.service --since {shlex.quote(start.strftime('%Y-%m-%d %H:%M:%S'))} -o short-iso 2>&1"
        code, jout, _ = run(cmd, timeout=30)
        first_line = (jout.strip().splitlines() or [""])[0]
        m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", first_line)
        if m:
            first_ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=start.tzinfo)
            if first_ts.timestamp() > start.timestamp():
                cover_start = first_ts
                try:
                    code2, cs, _ = run("docker inspect hindsight --format '{{.State.StartedAt}}'", timeout=10)
                    if code2 == 0 and cs.strip():
                        container_start = _parse_timestamp(cs.strip())
                        if container_start and container_start.timestamp() < cover_start.timestamp():
                            cover_start = container_start
                except Exception:
                    pass
                if cover_start.timestamp() > start.timestamp() + 300:  # 缺 <5min 视为正常
                    gap = int(cover_start.timestamp() - start.timestamp())
                    log_gap_note = (
                        f"⚠️ 日志最早从 {cover_start.astimezone(start.tzinfo).strftime('%H:%M')} 起（journal+docker 段），"
                        f"窗口起点 {start.strftime('%H:%M')} 前缺 {gap // 3600}h{(gap % 3600) // 60}m（LLM 用量偏少）。"
                    )
        elif not jout.strip():
            log_gap_note = "⚠️ journal 中无该窗口的 hindsight 日志（服务未运行或 journal 被清理），LLM 用量可能缺失。"
    except Exception:
        pass
    if log_gap_note:
        print(log_gap_note)

    if llm_usage:
        print_table(
            ["模型", "scope", "调用", "输入", "输出", "总tokens", "耗时"],
            [
                [
                    str(r["model"]),
                    str(r["scope"]),
                    str(r["calls"]),
                    fmt_tok(int(r["input_tokens"])),
                    fmt_tok(int(r["output_tokens"])),
                    fmt_tok(int(r["total_tokens"])),
                    f"{float(r['seconds']) / 60:.1f}min",
                ]
                for r in llm_usage
            ],
        )
    else:
        print("无 LLM 调用记录")
    print()

    # ── Hindsight 数据变化 (full table) ──
    print("【Hindsight 数据变化】")
    print_table(
        ["领域", "窗口变化", "当前状态"],
        [
            [
                "Documents",
                f"+{int(docs_meta[0])} / 更新 {int(docs_meta[1])}",
                fmt_num(total_docs),
            ],
        ],
    )
    # Memory units by type: [fact_type, created, updated_existing, total]
    units_rows = db.get("units_by_type") or []
    for ur in units_rows:
        if len(ur) >= 4:
            fact_type = ur[0]
            created = int(ur[1])
            updated_existing = int(ur[2])
            total = int(ur[3])
            delta_parts = [f"+{created}"]
            if updated_existing:
                delta_parts.append(f"更新 {updated_existing}")
            print(f"| {fact_type} | {' '.join(delta_parts)} | {fact_type} {fmt_num(total)} |")
    print(f"| Consolidation | 完成 {fmt_num(int(cons_meta[0]))} | 待处理 {fmt_num(uncons)} / 失败 {fmt_num(failed)} |")

    # Operations summary
    ops_summary = window_operation_summary(ops_rows)
    print(f"| Operations | {ops_summary} | 仅窗口内创建 |")

    # V2 cards
    v2_count = offline.get("v2_card_count", 0)
    v2_obs = offline.get("v2_observation_count", 0)
    v2_mtime = offline.get("v2_latest_mtime", "-")
    print(f"| V2 cards | rebuild {v2_mtime} | {v2_count} cards / {fmt_num(v2_obs)} observations |")

    # Doc sources
    doc_sources = db.get("doc_sources") or []
    if doc_sources:
        src_parts = [f"{s}:{c}" for s, c in doc_sources[:8]]
        print(f"新增来源: {'; '.join(src_parts)}")

    # Consolidation batches
    if consolidation_batch_logs and consolidation_batch_logs.get("batches", 0) > 0:
        b = consolidation_batch_logs
        print(f"Consolidation batches: {b['batches']} 批 / {b['memories']} memories / created {b['created']} / updated {b['updated']} / failed {b['failed']}")
    print()

    # ── 算法坑 ──
    print("【算法坑】")
    pitfall_counts = pitfalls.get("counts", {})
    print("当前: " + ", ".join(f"{s}={c}" for s, c in sorted(pitfall_counts.items())))
    changed_pitfalls = pitfalls.get("changes", [])
    if changed_pitfalls:
        for entry in changed_pitfalls:
            print(f"• {entry['p_id']} [{entry['status']}]: {entry['title']}")
    else:
        print("• 本窗口无可归因的 Pitfall lifecycle 变更。")
        print("• 研发 digest 不分配 P-id；唯一 writer 为 pitfall_writer.py。")
    print()

    # ── 异常 ──
    print("【异常】")
    errors = db.get("errors") or []
    failed_base = int(cons_meta[2]) if cons_meta and len(cons_meta) > 2 else 0
    if not errors and not alerts and failed_base == 0:
        print("• 无未解决 active queue 或 failed_base；窗口内终态失败已在数据表中展示。")
    else:
        for e in errors:
            print(f"• DB异常: {e}")
        for alert in alerts:
            print(f"• {alert}")
        if failed_base:
            print(f"• failed_base={failed_base}")
    print()

    # ── 索引 ──
    print("【索引】")
    print("• 项目知识: /home/wyr/wiki/auto-maintenance/project/egomotion4d/knowledge/")
    print("• Mental Models: /home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/")
    print("• 算法坑: /home/wyr/wiki/auto-maintenance/project/egomotion4d/mental-models/pitfall-catalog.md")
    print("• Hindsight 状态: /home/wyr/wiki/auto-maintenance/reports/")
    return 0


if __name__ == "__main__":
    import sys as _sys
    import io as _io
    _buf = _io.StringIO()
    _old_stdout = _sys.stdout
    _sys.stdout = _buf
    try:
        _ret = main()
    finally:
        _sys.stdout = _old_stdout
    _output = _buf.getvalue()
    print(_output, end="")
    # Write the canonical daily entry. Backfills require explicit --write.
    if "--date" not in _sys.argv or "--write" in _sys.argv:
        _wiki_dir = Path("/home/wyr/wiki/auto-maintenance/daily")
        _wiki_dir.mkdir(parents=True, exist_ok=True)
        if "--date" in _sys.argv:
            _date_str = _sys.argv[_sys.argv.index("--date") + 1]
        else:
            _date_str = datetime.now().strftime("%Y-%m-%d")
        _wiki_path = _wiki_dir / f"{_date_str}.md"
        _tmp_path = _wiki_dir / f".{_date_str}.tmp"
        _tmp_path.write_text(_output, encoding="utf-8")
        os.replace(_tmp_path, _wiki_path)
    raise SystemExit(_ret)
