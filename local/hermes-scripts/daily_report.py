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
HINDSIGHT_REFLECT = HERMES_HOME / "hindsight" / "offline_reflect"
PSQL = os.environ.get("HINDSIGHT_PSQL", "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql")
HINDSIGHT_API = os.environ.get("HINDSIGHT_API", "http://127.0.0.1:8888")
BANK = os.environ.get("HINDSIGHT_BANK", "hermes")


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
        if m:
            stats['documents'] = int(m.group(1).replace(',', ''))
        m = re.search(r'observations 总数[|\s]*([\d,]+)', text)
        if m:
            stats['observations'] = int(m.group(1).replace(',', ''))
        m = re.search(r'base units 已 consolidation[|\s]*(\d+)', text)
        if m:
            stats['consolidated'] = int(m.group(1))
        m = re.search(r'unconsolidated_base 剩余[|\s]*(\d+)', text)
        if m:
            stats['unconsolidated'] = int(m.group(1))
    except Exception:
        return None
    return stats if stats else None


def hermes_model_usage(start: datetime, end: datetime) -> list[dict[str, int | str]]:
    rows_out: list[dict[str, int | str]] = []
    for profile, db_path in iter_state_dbs():
        con = sqlite3.connect(db_path)
        try:
            # Use messages.timestamp to find sessions active in the window.
            # Filtering only by sessions.started_at misses long-running sessions
            # that started yesterday but consumed tokens today.
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
    """Parse Codex rollout jsonl for token usage (cumulative, need delta).

    token_count events carry per-session cumulative totals. Different sessions
    have independent counters starting from 0, so we must compute deltas per-session
    first, then sum across sessions for the target date.
    """
    codex_base = Path('/home/wyr/.codex/sessions')
    cw_base = Path('/home/wyr/.codewhale/sessions')
    codex_kanban_base = Path('/home/wyr/.codex-kanban')
    all_session_files = []

    # Collect from Codex global sessions, CodeWhale sessions, and
    # per-role Codex kanban sessions (start-kanban.sh uses per-role CODEX_HOME)
    scan_dirs = [codex_base, cw_base]
    if codex_kanban_base.exists():
        for role_dir in sorted(codex_kanban_base.iterdir()):
            role_sessions = role_dir / 'sessions'
            if role_sessions.is_dir():
                scan_dirs.append(role_sessions)

    for base in scan_dirs:
        if not base.exists():
            continue
        # Scan all available year/month/day dirs
        for year_dir in sorted(base.glob('2026/*')):
            if not year_dir.is_dir():
                continue
            for day_dir in sorted(year_dir.glob('*')):
                if not day_dir.is_dir():
                    continue
                for session_file in day_dir.glob('rollout-*.jsonl'):
                    all_session_files.append(session_file)

    # Collect per-session first/last token_count per date
    session_date_entries: dict[tuple, dict] = {}

    for session_file in all_session_files:
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
                                key = (str(session_file), date)
                                if key not in session_date_entries:
                                    session_date_entries[key] = {'first': entry, 'last': entry}
                                else:
                                    session_date_entries[key]['last'] = entry
                    except Exception:
                        pass
        except Exception:
            pass

    total_input = 0
    total_cached = 0
    total_output = 0
    found_any = False

    from datetime import datetime, timedelta
    target_dt = datetime.strptime(target_date, '%Y-%m-%d')
    prev_date = (target_dt - timedelta(days=1)).strftime('%Y-%m-%d')
    for (sf, date), entries in session_date_entries.items():
        if date not in (target_date, prev_date):
            continue
        found_any = True
        first = entries['first']
        last = entries['last']
        total_input += last['input'] - first['input']
        total_cached += last['cached'] - first['cached']
        total_output += last['output'] - first['output']

    if not found_any:
        return None

    return {
        'input': total_input - total_cached,
        'cached': total_cached,
        'output': total_output,
    }


def docker_env() -> dict[str, str]:
    cmd = "docker exec hindsight sh -lc 'env | grep -E \"HINDSIGHT_API_(LLM|RETAIN_LLM|REFLECT_LLM|CONSOLIDATION_LLM|ENABLE_OBSERVATIONS|WORKER_CONSOLIDATION_MAX_SLOTS|WORKER_MAX_SLOTS|RETAIN_MAX_CONCURRENT).*\" | sort'"
    code, out, _ = run(cmd, timeout=20)
    if code != 0:
        return {}
    env: dict[str, str] = {}
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
                    "location": "docker:hindsight",
                }
            )
    return rows


def docker_logs_since(start: datetime) -> str:
    since = start.astimezone(timezone.utc).isoformat()
    cmd = f"docker logs --since {shlex.quote(since)} hindsight 2>&1"
    code, out, err = run(cmd, timeout=90)
    if code != 0:
        return out + "\n" + err
    return out


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
        stats["offline_docs"] = psql(
            f"""
            WITH od AS (
              SELECT d.id, coalesce(count(m.id),0) units
              FROM documents d
              LEFT JOIN memory_units m ON m.bank_id=d.bank_id AND m.document_id=d.id
              WHERE d.bank_id='{BANK}' AND d.id LIKE 'hermes-offline-consolidation::%'
              GROUP BY d.id
            )
            SELECT count(*) total,
                   count(*) filter(where units=0) zero,
                   count(*) filter(where units>0) with_units,
                   coalesce(sum(units),0) units
            FROM od;
            """
        )
    except Exception as exc:
        stats["errors"].append(str(exc))
    return stats


def offline_output_counts(now: datetime) -> dict[str, str | int]:
    v2_dir = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
    rebuild_dir = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_rebuild"
    v2_global_md = v2_dir / "global" / "global.md"
    v2_global_json = v2_dir / "global" / "global.json"
    v2_manifest = v2_dir / "manifest.json"
    v2_obs_index = v2_dir / "observations_index.jsonl"
    v2_topics = v2_dir / "topics"
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

    # 统计 v2 目录下 topic cards 个数（取 manifest 的 card_count 更准）
    return {
        "v2_global_md": str(v2_global_md) if v2_global_md.exists() else "不存在",
        "v2_global_md_mtime": datetime.fromtimestamp(v2_global_md.stat().st_mtime).strftime("%m-%d %H:%M") if v2_global_md.exists() else "-",
        "v2_global_json": str(v2_global_json) if v2_global_json.exists() else "不存在",
        "v2_manifest": str(v2_manifest) if v2_manifest.exists() else "不存在",
        "v2_obs_index": str(v2_obs_index) if v2_obs_index.exists() else "不存在",
        "v2_obs_index_size": v2_obs_index.stat().st_size if v2_obs_index.exists() else 0,
        "v2_topics": str(v2_topics) if v2_topics.exists() else "不存在",
        "v2_topic_dirs": len(list(v2_topics.iterdir())) if v2_topics.exists() else 0,
        "v2_card_count": card_count,
        "v2_observation_count": obs_index_count,
        "v2_latest": str(v2_latest) if v2_latest.exists() else "不存在",
        "v2_latest_mtime": datetime.fromtimestamp(v2_latest.stat().st_mtime).strftime("%m-%d %H:%M") if v2_latest.exists() else "-",
        # 旧目录（v0.5.x 遗留）
        "legacy_daily": "不存在（v0.6.x 改用 V2 cards 体系，无日度目录产物）",
        "legacy_weekly": "不存在（v0.6.x 改用 V2 cards 体系，无周度目录产物）",
    }


def topic_progress(now: datetime) -> list[str]:
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    daily_dir = HINDSIGHT_REFLECT / "daily" / yesterday_str
    points: list[str] = []
    if not daily_dir.exists():
        return points
    for f in sorted(daily_dir.glob("*.md"))[:10]:
        topic = f.name.split("__")[0] if "__" in f.name else f.stem
        try:
            content = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = re.search(r"## Executive Summary\n(.*?)(?:\n##|\Z)", content, re.DOTALL)
        if not match:
            continue
        for line in match.group(1).splitlines():
            line = line.strip()
            if line.startswith("-") and len(line) > 10:
                points.append(f"• {topic}: {line[1:].strip()[:80]}")
                break
    return points[:6]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Hermes 日报生成")
    parser.add_argument("--date", type=str, default=None,
                        help="目标日期 YYYY-MM-DD，默认昨天")
    args = parser.parse_args()

    now = datetime.now().astimezone()
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").replace(tzinfo=now.tzinfo)
        # 统计窗口：目标日期 08:30 ~ 次日 08:30
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
    env = docker_env()
    model_cfg = hindsight_model_config(env)
    logs = docker_logs_since(start)
    llm_usage, consolidation_batch_logs = parse_hindsight_llm_logs(logs)
    db = hindsight_db_stats(start, end)
    offline = offline_output_counts(now_for_print)
    progress = topic_progress(now_for_print)

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
        pct_doc = f"({d_doc:+.0f})" if prev.get("documents") else ""
        pct_obs = f"({d_obs:+.0f})" if prev.get("observations") else ""
    print(f"概要: Documents={fmt_num(total_docs)}{f' (Δ{d_doc:+d})' if prev and 'documents' in prev else ''}  "
          f"Observations={fmt_num(total_obs)}{f' (Δ{d_obs:+d})' if prev and 'observations' in prev else ''}  "
          f"unconsolidated={fmt_num(uncons)}  failed_base={fmt_num(failed)}")
    last_cons = (db.get("consolidation") or [["0"] * 6])[0]
    # Try to extract last_consolidated_at from stats via direct API or fallback to db
    print(f"Consolidation: base_done={fmt_num(int(cons_meta[0]))}  "
          f"remaining={fmt_num(int(cons_meta[1]))}  "
          f"failed={fmt_num(int(cons_meta[2]))}")
    print()

    print("【工作概要】")
    if progress:
        for line in progress:
            print(line)
    else:
        print("• 今日重点以 Hermes/Hindsight 使用统计、离线管线状态和异常检查为主。")
    print()

    print("【Hermes / Profiles 模型用量】")
    if hermes_rows:
        table = []
        total = defaultdict(int)
        for r in hermes_rows:
            for k in ("sessions", "turns", "calls", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens"):
                total[k] += int(r[k])
            table.append(
                [
                    str(r.get("profile", "default")),
                    str(r["model"]),
                    fmt_num(int(r["sessions"])),
                    fmt_num(int(r["turns"])),
                    fmt_num(int(r["calls"])),
                    fmt_tok(int(r["input_tokens"])),
                    fmt_tok(int(r["cache_read_tokens"])),
                    fmt_tok(int(r["cache_write_tokens"])),
                    fmt_tok(int(r["output_tokens"])),
                ]
            )
        table.append(
            [
                "合计",
                "",
                fmt_num(total["sessions"]),
                fmt_num(total["turns"]),
                fmt_num(total["calls"]),
                fmt_tok(total["input_tokens"]),
                fmt_tok(total["cache_read_tokens"]),
                fmt_tok(total["cache_write_tokens"]),
                fmt_tok(total["output_tokens"]),
            ]
        )
        print_table(["Profile", "模型", "会话", "轮数", "调用", "输入", "Cache读", "Cache写", "输出"], table)
    else:
        print("无 Hermes / profile 会话记录。")
    print()

    print("【外部 CLI Agent 调用统计（Codex/CodeWhale 增量提取）】")
    codex_usage = parse_codex_usage(report_date_str)
    if codex_usage:
        print("注：输入(增量) = 真实新输入 = prompt - cache；Cache读为累计 cached_input_tokens。")
        print_table(
            ["来源", "输入", "Cache读", "输出"],
            [
                ["codex", fmt_tok(codex_usage['input']), fmt_tok(codex_usage['cached']), fmt_tok(codex_usage['output'])],
            ],
        )
    else:
        print("未采集到外部 CLI Agent 调用记录。")
    print()

    print("【Hindsight 运行模型配置】")
    print(f"健康状态: {hindsight_health()}")
    if model_cfg:
        print_table(
            ["用途", "provider", "model", "base_url", "location"],
            [[r["scope"], r["provider"], r["model"], r["base_url"], r["location"]] for r in model_cfg],
        )
    else:
        print("未能读取 hindsight 容器模型环境。")
    print(
        f"enable_observations={env.get('HINDSIGHT_API_ENABLE_OBSERVATIONS', '?')}, "
        f"worker_slots={env.get('HINDSIGHT_API_WORKER_MAX_SLOTS', '?')}, "
        f"consolidation_slots={env.get('HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS', '?')}, "
        f"retain_max_concurrent={env.get('HINDSIGHT_API_RETAIN_MAX_CONCURRENT', '?')}"
    )
    print()

    print("【Hindsight LLM 用量（日志可见精确值；容器重建前日志可能无法回溯）】")
    if llm_usage:
        print_table(
            ["模型", "scope", "调用", "输入", "输出", "总tokens", "耗时"],
            [
                [
                    str(r["model"]),
                    str(r["scope"]),
                    fmt_num(int(r["calls"])),
                    fmt_tok(int(r["input_tokens"])),
                    fmt_tok(int(r["output_tokens"])),
                    fmt_tok(int(r["total_tokens"])),
                    f"{float(r['seconds'])/60:.1f}min",
                ]
                for r in llm_usage
            ],
        )
    else:
        print("未从当前容器日志捕获 Hindsight LLM token 记录；保留 DB 工作量统计。")
    print()

    print("【Hindsight 工作量 / 数据修改】")
    ops_rows = db.get("ops") or []
    if ops_rows:
        print("操作队列（窗口内创建）:")
        print_table(["operation", "status", "数量"], [[a, b, fmt_num(int(c))] for a, b, c in ops_rows])
    docs = (db.get("docs") or [["0", "0", "0"]])[0]
    cons = (db.get("consolidation") or [["0", "0", "0", "0", "0", "0"]])[0]
    off = (db.get("offline_docs") or [["0", "0", "0", "0"]])[0]
    print("核心修改统计:")
    print_table(
        ["指标", "数量"],
        [
            ["documents 新增", fmt_num(int(docs[0]))],
            ["documents 更新(既有)", fmt_num(int(docs[1]))],
            ["documents 总数", fmt_num(int(docs[2]))],
            ["base units 已 consolidation", fmt_num(int(cons[0]))],
            ["unconsolidated_base 剩余", fmt_num(int(cons[1]))],
            ["failed_base", fmt_num(int(cons[2]))],
            ["observations 总数", fmt_num(int(cons[3]))],
            ["observations 新增", fmt_num(int(cons[4]))],
            ["observations 更新(既有)", fmt_num(int(cons[5]))],
            ["offline docs total/with_units/zero/units", f"{off[0]}/{off[2]}/{off[1]}/{off[3]}"],
        ],
    )
    units_rows = db.get("units_by_type") or []
    if units_rows:
        print("memory_units 按类型:")
        print_table(["fact_type", "新增", "更新(既有)", "总数"], [[r[0], fmt_num(int(r[1])), fmt_num(int(r[2])), fmt_num(int(r[3]))] for r in units_rows])
    sources = db.get("doc_sources") or []
    if sources:
        print("documents 新增来源(top): " + "; ".join(f"{s}:{c}" for s, c in sources))
    if consolidation_batch_logs.get("batches"):
        print(
            "consolidation batch 日志: "
            f"{consolidation_batch_logs['batches']}批 / {consolidation_batch_logs['memories']} memories / "
            f"{consolidation_batch_logs['llm_calls']} LLM calls / "
            f"created={consolidation_batch_logs['created']} updated={consolidation_batch_logs['updated']} skipped={consolidation_batch_logs['skipped']} failed={consolidation_batch_logs['failed']}"
        )
    print()

    print("【Hindsight 当前队列】")
    queue_rows = db.get("queue") or []
    if queue_rows:
        print_table(["status", "operation", "数量"], [[a, b, fmt_num(int(c))] for a, b, c in queue_rows])
    else:
        print("未读取到队列统计。")
    print()

    print("【Consolidation / V2 Cards 体系输出】")
    print(f"V2 global.md: {offline['v2_global_md']} ({offline['v2_global_md_mtime']})")
    print(f"V2 global.json: {offline['v2_global_json']}")
    print(f"V2 manifest: {offline['v2_manifest']} — {offline['v2_card_count']} cards, {offline['v2_observation_count']} observations")
    print(f"V2 obs index: {offline['v2_obs_index']} ({fmt_tok(offline['v2_obs_index_size'])})")
    print(f"V2 topics: {offline['v2_topics']} ({offline['v2_topic_dirs']} topic dirs)")
    print(f"V2 rebuild: {offline['v2_latest']} ({offline['v2_latest_mtime']})")
    print(f"Legacy daily: {offline['legacy_daily']}")
    print(f"Legacy weekly: {offline['legacy_weekly']}")
    print()

    print("【异常】")
    errors = db.get("errors") or []
    failed_ops = [r for r in queue_rows if r[0] == "failed"]
    failed_base = int(cons[2]) if cons and len(cons) > 2 else 0
    if not errors and not failed_ops and failed_base == 0:
        print("• 未发现 Hindsight failed_operations / failed_base。")
    else:
        for e in errors:
            print(f"• Hindsight DB 统计异常: {e}")
        for r in failed_ops:
            print(f"• failed operation: {r[1]} x {r[2]}")
        if failed_base:
            print(f"• failed_base={failed_base}")
    print()

    print("【日报输出偏好】")
    print("• 正文直接用文字/表格展示；不要只发送 md 文件或 MEDIA 附件。")
    print("• 如需归档，可另写 /home/wyr/wiki/auto-maintenance/daily/YYYY-MM-DD.md，但用户可见回复必须包含正文表格。")
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
    # 写 wiki 归档（仅 cron 自动运行时，--date 补跑不写）
    import argparse as _argparse
    if "--date" not in _sys.argv:
        _wiki_dir = Path("/home/wyr/wiki/auto-maintenance/daily")
        _wiki_dir.mkdir(parents=True, exist_ok=True)
        _date_str = datetime.now().strftime("%Y-%m-%d")
        _wiki_path = _wiki_dir / f"{_date_str}.md"
        _wiki_path.write_text(_output)
    raise SystemExit(_ret)
