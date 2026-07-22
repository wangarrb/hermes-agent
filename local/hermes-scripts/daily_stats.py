#!/usr/bin/env python3
"""Hermes Daily Stats Reporter — generates /wiki/auto-maintenance/daily/YYYY-MM-DD.md

Called by hermes cron (hermes-daily-stats, schedule: 30 8 * * *).

Reads:
  - Hindsight API (stats, operations)
  - agent.log + rotated logs (for model usage)
  - Previous day's .md (for delta calculation)
  - Hindsight DB directly (for memory_units changes)

Writes:
  - /home/wyr/wiki/auto-maintenance/daily/YYYY-MM-DD.md (data stats report)
"""

import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# Disable proxy for localhost connections (cron environment may have stale proxy env vars)
for k in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(k, None)
os.environ['no_proxy'] = 'localhost,127.0.0.1'

WIKI_DAILY_DIR = Path("/home/wyr/wiki/auto-maintenance/daily")
HINDSIGHT_API = os.environ.get("HINDSIGHT_API_URL", "http://127.0.0.1:8888")
AGENT_LOG = Path("/home/wyr/.hermes/logs/agent.log")
BANK = "hermes"


def api_get(path):
    url = f"{HINDSIGHT_API}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def fmt_delta(current, previous):
    if previous is None:
        return f"{current:,} (新)"
    delta = current - previous
    pct = (delta / previous * 100) if previous > 0 else 0.0
    sign = "+" if delta >= 0 else ""
    return f"{current:,} (Δ{sign}{delta:,} ({sign}{pct:.1f}%))"


def parse_prev_stats(md_path):
    if not md_path.exists():
        return None
    stats = {}
    try:
        text = md_path.read_text()
        # Fields may appear on the same line (e.g. "Documents: X, Observations: Y")
        # so scan the full text with regex instead of line-by-line startswith
        for field, key in [
            ('Documents', 'documents'),
            ('Observations', 'observations'),
            ('Nodes', 'nodes'),
            ('Links', 'links'),
        ]:
            m = re.search(rf'{field}:\s*([\d,]+)', text)
            if m:
                stats[key] = int(m.group(1).replace(',', ''))
    except Exception:
        pass
    return stats if stats else None


def parse_model_usage(target_date):
    """Parse agent.log and rotated logs for API call model usage."""
    by_model = defaultdict(lambda: {'calls': 0, 'input': 0, 'output': 0, 'sessions': set()})

    log_files = [str(AGENT_LOG)]
    for i in range(1, 5):
        rotated = AGENT_LOG.parent / f"agent.log.{i}"
        if rotated.exists():
            log_files.append(str(rotated))

    for logfile in log_files:
        try:
            with open(logfile) as f:
                for line in f:
                    # Match: 2026-06-22 11:31:55 ... [session_id] ... API call #N: model=X provider=Y in=Z out=W
                    m = re.match(
                        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*\[(\S+)\].*API call #\d+: model=(\S+) provider=\S+ in=(\d+) out=(\d+)',
                        line
                    )
                    if m:
                        dt = datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')
                        if dt.strftime('%Y-%m-%d') != target_date:
                            continue
                        session = m.group(2)
                        model = m.group(3)
                        inp = int(m.group(4))
                        out = int(m.group(5))
                        by_model[model]['calls'] += 1
                        by_model[model]['input'] += inp
                        by_model[model]['output'] += out
                        by_model[model]['sessions'].add(session)
        except Exception:
            pass

    return by_model


def parse_codex_usage(target_date):
    """Parse Codex/CodeWhale rollout jsonl for token usage (cumulative, need delta).

    token_count events carry per-session cumulative totals. Different sessions
    have independent counters starting from 0, so we must compute deltas per-session
    first, then sum across sessions for the target date.
    """
    codex_base = Path('/home/wyr/.codex/sessions')
    cw_base = Path('/home/wyr/.codewhale/sessions')
    all_session_files = []

    # Collect from both Codex and CodeWhale session dirs
    for base in [codex_base, cw_base]:
        if not base.exists():
            continue
        for month_dir in base.glob('2026/06/*'):
            if not month_dir.is_dir():
                continue
            for session_file in month_dir.glob('rollout-*.jsonl'):
                all_session_files.append(session_file)
        for month_dir in base.glob('2026/05/*'):
            if not month_dir.is_dir():
                continue
            for session_file in month_dir.glob('rollout-*.jsonl'):
                all_session_files.append(session_file)

    # Collect per-session first/last token_count per date
    # Key: (session_file, date) -> {first, last}
    session_date_entries = {}

    for session_file in all_session_files:
        try:
            with open(session_file) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        ts = d.get('timestamp', '')
                        if not ts or ts[:10] < '2026-06-01':
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

    # For each session, compute delta for target_date
    # Delta = last - first for that session on that date (per-session cumulative is monotonic)
    total_input = 0
    total_cached = 0
    total_output = 0
    found_any = False

    for (sf, date), entries in session_date_entries.items():
        if date != target_date:
            continue
        found_any = True
        first = entries['first']
        last = entries['last']
        total_input += last['input'] - first['input']
        total_cached += last['cached'] - first['cached']
        total_output += last['output'] - first['output']

    if not found_any:
        return None

    real_input = total_input - total_cached

    return {
        'model': 'gpt-5.5',
        'input': real_input,
        'cached': total_cached,
        'output': total_output,
        'source': 'codex',
    }


def fmt_tokens(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif n >= 1000:
        return f"{n/1000:.1f}K"
    return str(n)


def generate_report():
    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    yesterday = now - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")

    window_start = f"{yesterday_str} 08:30"
    window_end = f"{today_str} 08:30"

    # Get current Hindsight stats
    stats = api_get(f"/v1/default/banks/{BANK}/stats")
    if "error" in stats:
        print(f"ERROR: Cannot connect to Hindsight API: {stats['error']}", file=sys.stderr)
        sys.exit(1)

    # Get previous day's snapshot (day-before-yesterday's report, since our report IS yesterday's)
    day_before_yesterday = now - timedelta(days=2)
    prev_path = WIKI_DAILY_DIR / f"{day_before_yesterday.strftime('%Y-%m-%d')}.md"
    prev = parse_prev_stats(prev_path)

    # Get model usage
    model_usage = parse_model_usage(yesterday_str)

    # Build report
    lines = []
    lines.append(f"# Hermes 日报 {yesterday_str}")
    lines.append(f"生成时间: {now.strftime('%Y-%m-%d %H:%M')} CST")
    lines.append(f"统计窗口: {window_start} ~ {window_end} CST")
    lines.append("")

    # Summary
    curr_docs = stats.get("total_documents", 0)
    curr_obs = stats.get("total_observations", 0)
    curr_nodes = stats.get("total_nodes", 0)
    curr_links = stats.get("total_links", 0)

    prev_docs = prev.get("documents") if prev else None
    prev_obs = prev.get("observations") if prev else None
    prev_nodes = prev.get("nodes") if prev else None
    prev_links = prev.get("links") if prev else None

    lines.append("## 概要")
    lines.append(f"Documents: {fmt_delta(curr_docs, prev_docs)}, Observations: {fmt_delta(curr_obs, prev_obs)}")
    lines.append(f"Nodes: {fmt_delta(curr_nodes, prev_nodes)}, Links: {fmt_delta(curr_links, prev_links)}")

    last_consol = stats.get("last_consolidated_at", "unknown")
    lines.append(f"Last consolidation: {last_consol}")
    lines.append("")

    # Model usage
    lines.append("## Hermes / Profiles 模型用量（24h）")
    if model_usage:
        lines.append("| Profile | 模型 | 会话 | 调用 | 输入 | Cache读 | 输出 |")
        lines.append("|---|---|---|---|---|---|---|")
        total_calls = 0
        total_input = 0
        total_output = 0
        total_sessions = set()
        for model, s in sorted(model_usage.items()):
            n_sessions = len(s['sessions'])
            lines.append(f"| coordinator | {model} | {n_sessions} | {s['calls']:,} | {fmt_tokens(s['input'])} | - | {fmt_tokens(s['output'])} |")
            total_calls += s['calls']
            total_input += s['input']
            total_output += s['output']
            total_sessions.update(s['sessions'])
        lines.append(f"| **合计** | | {len(total_sessions)} | {total_calls:,} | {fmt_tokens(total_input)} | - | {fmt_tokens(total_output)} |")
    else:
        lines.append("无 Hermes / profile 会话记录。")
    lines.append("")

    # Hindsight LLM usage
    lines.append("## Hindsight LLM 用量")
    lines.append("未从当前容器日志捕获 Hindsight LLM token 记录。")
    lines.append("")

    # External CLI Agent usage (Codex)
    lines.append("## 外部 CLI Agent 调用统计（增量模式，较快照）")
    codex_usage = parse_codex_usage(yesterday_str)
    if codex_usage:
        lines.append("注：输入(增量) = 真实新输入 = prompt - cache；估算行的 Cache读 标 - ，输入含 cache。")
        lines.append("| 来源 | 模型 | 输入 | Cache读 | 输出 |")
        lines.append("|---|---|---|---|---|")
        lines.append(f"| {codex_usage['source']} | {codex_usage['model']} | {fmt_tokens(codex_usage['input'])} | {fmt_tokens(codex_usage['cached'])} | {fmt_tokens(codex_usage['output'])} |")
    else:
        lines.append("无外部 Agent 调用记录。")
    lines.append("")

    # Operations queue
    ops = stats.get("operations_by_status", {})
    lines.append("## Hindsight 当前队列")
    lines.append("| 指标 | 当前值 |")
    lines.append("|---|---|")
    lines.append(f"| Pending operations | {ops.get('pending', 0)} |")
    lines.append(f"| Processing operations | {ops.get('processing', 0)} |")
    lines.append(f"| Failed operations | {ops.get('failed', 0)} |")
    lines.append(f"| Completed operations | {ops.get('completed', 0):,} |")
    lines.append("")

    # Nodes by fact type
    nodes_by_type = stats.get("nodes_by_fact_type", {})
    lines.append("## Hindsight 元素分布")
    lines.append("| 类别 | 数量 |")
    lines.append("|---|---|")
    for ft, cnt in sorted(nodes_by_type.items()):
        lines.append(f"| {ft} | {cnt:,} |")
    lines.append("")

    # Anomalies
    lines.append("## 异常")
    failed = ops.get("failed", 0)
    pending = ops.get("pending", 0)
    anomalies = []
    if failed > 0:
        anomalies.append(f"- Failed operations: {failed}")
    if pending > 5:
        anomalies.append(f"- Pending operations 堆积: {pending}")
    obs_delta = (curr_obs - prev_obs) if prev_obs is not None else None
    if obs_delta is not None and obs_delta == 0:
        anomalies.append("- Observations 零增长：consolidation 可能未产出 observations")
    if not anomalies:
        lines.append("无异常。")
    else:
        lines.extend(anomalies)
    lines.append("")

    lines.append("---")
    lines.append(f"_自动生成于 {now.strftime('%Y-%m-%d %H:%M')} CST；无 LLM 调用。_")

    # Write
    WIKI_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = WIKI_DAILY_DIR / f"{yesterday_str}.md"
    out_path.write_text("\n".join(lines))
    print(f"Written: {out_path}")
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"OK: {path}")
