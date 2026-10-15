"""Read an existing local App archive for the previous UTC+8 calendar day.

This is a report reader, not a collector: no App API, network, model, ledger or
archive writes. A scheduled collection mechanism must populate the archive.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

from hindsight_app_usage import day_window, render_app_daily


def render_daily_sidecar(report_date: str | date | datetime, *, archive_root: Path) -> str:
    """Appendable report text; absent or broken collection is never reported as zero."""
    report_start, _ = day_window(report_date)
    covered_day = (report_start - timedelta(days=1)).date()
    archive_root = Path(archive_root)
    directory = archive_root / "sessions"
    heading = "【App 本地归档补充】\n"
    boundary = "仅覆盖现有 codex-app / chatgpt-app 本地快照，不代表全账号或归档全集覆盖。\n"
    if not directory.is_dir():
        return heading + boundary + f"{covered_day}（UTC+8）：归档未接入，消息数量及 LLM 调用覆盖未知。\n"
    try:
        sessions = []
        for path in sorted(directory.glob("session_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("invalid archive snapshot")
            sessions.append(data)
        status_path = archive_root / "collection-status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        if not isinstance(status, dict):
            raise ValueError("invalid collection status")
        rendered = render_app_daily(sessions, covered_day)
    except (OSError, ValueError, TypeError, KeyError):
        return heading + boundary + "本地归档或采集状态无法核对，本节覆盖未知；未调用采集/模型，其他日报统计不受影响。\n"
    collected_at = status.get("last_success_at")
    if collected_at:
        try:
            collected, _ = day_window(datetime.fromisoformat(str(collected_at).replace("Z", "+00:00")))
        except ValueError:
            collected = None
        label = "采集时间已到达本日结束边界（不证明完整覆盖）" if collected is not None and collected >= report_start else "未证明覆盖本日结束边界"
        # Only emit normalized time metadata, never arbitrary status file text.
        freshness = f"采集新鲜度：{label}。\n"
    else:
        freshness = "采集新鲜度：未知（无 last_success_at）；现有快照不证明本日已完整采集。\n"
    coverage = status.get("coverage")
    limits = ("采集范围限制：未知（未提供 coverage）；仅可核对已有快照。\n" if coverage is None else
              "采集范围限制（采集器声明）：\n```json\n" + json.dumps(coverage, ensure_ascii=False, sort_keys=True) + "\n```\n")
    return heading + boundary + freshness + limits + rendered


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--report-date", required=True, help="日报日期；本节统计其前一 UTC+8 自然日")
    args = parser.parse_args(argv)
    print(render_daily_sidecar(args.report_date, archive_root=args.archive_root), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
