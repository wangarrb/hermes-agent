#!/usr/bin/env python3
"""Generate a Hindsight pipeline status report in ~/wiki/auto-maintenance/.

Called by daily/wiki cron runners. Output: reports/hindsight-status-YYYYMMDD-HHMMSS.md
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

WIKI_DIR = Path.home() / "wiki" / "auto-maintenance" / "reports"
OFFLINE_ROOT = Path.home() / ".hermes" / "hindsight" / "offline_reflect"
V2_ROOT = OFFLINE_ROOT / "v2_rebuild"
WEEKLY_ROOT = OFFLINE_ROOT / "weekly"


def generate_report(output_dir: Path | None = None) -> Path:
    out = output_dir or WIKI_DIR
    out.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    ts = now.strftime("%Y%m%d-%H%M%S")
    report_path = out / f"hindsight-status-{ts}.md"

    lines = [f"# Hindsight Pipeline Status - {now.isoformat()}", ""]

    # API health
    try:
        with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=3) as resp:
            health = resp.read().decode().strip()
    except Exception:
        health = "unreachable"
    lines.append(f"## API Health\n- {health}\n")

    # V2 rebuild
    v2_latest = V2_ROOT / "latest.json"
    if v2_latest.exists():
        with open(v2_latest) as f:
            v2 = json.load(f)
        lines.append("## V2 Rebuild")
        lines.append(f"- mode: {v2.get('mode', '?')}")
        lines.append(f"- decision: {v2.get('decision', '?')}")
        lines.append(f"- published: {v2.get('published', '?')}")
        lines.append(f"- started: {v2.get('started_at', '?')}")
        lines.append(f"- finished: {v2.get('finished_at', '?')}")
        steps = v2.get('steps', {})
        for k in ['reduce', 'conflict_audit', 'eval', 'gate', 'publish']:
            s = steps.get(k, {})
            if isinstance(s, dict):
                lines.append(f"  - {k}: {s.get('decision', s.get('status', '?'))}")
        lines.append("")

    # Progress cache
    progress_file = OFFLINE_ROOT / "offline_reflect_progress.json"
    if progress_file.exists():
        with open(progress_file) as f:
            d = json.load(f)
        v2c = d.get("processed_units_v2", {})
        lines.append("## Progress Cache")
        lines.append(f"- total_cached: {len(v2c)}")
        lines.append(f"- weekly_cached: {sum(1 for k in v2c if k.startswith('weekly::'))}")
        lines.append(f"- daily_cached: {sum(1 for k in v2c if k.startswith('daily::'))}")
        lines.append("")

    # Weekly output stats
    if WEEKLY_ROOT.exists():
        lines.append("## Weekly Output by Period")
        lines.append("| Period | Total | Has Content | Empty | Size |")
        lines.append("|--------|-------|-------------|-------|------|")
        for period_dir in sorted(WEEKLY_ROOT.iterdir()):
            if period_dir.is_dir():
                mds = list(period_dir.glob("*.md"))
                total_size = sum(f.stat().st_size for f in mds)
                empty = sum(1 for f in mds if f.stat().st_size < 500)
                has_content = len(mds) - empty
                size_kb = total_size / 1024
                lines.append(f"| {period_dir.name} | {len(mds)} | {has_content} | {empty} | {size_kb:.0f}k |")
        lines.append("")

    # Conflict audit
    conflict_json = V2_ROOT / "conflict_audit" / "latest.json"
    if conflict_json.exists():
        with open(conflict_json) as f:
            ca = json.load(f)
        summary = ca.get("summary", {})
        lines.append("## Conflict Audit")
        lines.append(f"- case_count: {summary.get('case_count', '?')}")
        lines.append(f"- blocking_cases: {summary.get('blocking_cases', '?')}")
        lines.append(f"- by_severity: {summary.get('by_severity', {})}")
        lines.append(f"- by_type: {summary.get('by_type', {})}")
        lines.append("")

    # Config
    lines.append("## Pipeline Config")
    lines.append("- LLM profile: opencode-go-deepseek-v4-flash")
    lines.append("- max_tokens: 65536")
    lines.append("- budget: 200 units / 10M chars")
    lines.append("")

    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    return report_path


if __name__ == "__main__":
    path = generate_report()
    print(path)
