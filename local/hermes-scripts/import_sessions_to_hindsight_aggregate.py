#!/usr/bin/env python3
"""Aggregate Hermes session history into a Hindsight bank with fewer LLM calls.

用途：
- 把已经在旧 hermes-sessions 导入进度里确认成功的 Hermes session 作为源；
- 本地按主题/时间合并成较大 bundle；
- 每个 bundle 作为 1 个 Hindsight memory item 写入目标 bank，降低按次计费请求数；
- 支持 dry-run、submit、中断续跑。

默认目标 bank 固定为 hermes。不要再把历史迁移写入 hermes-sessions。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import requests

HINDSIGHT_API = "http://localhost:8888"
DEFAULT_BANK_ID = "hermes"
DEFAULT_SOURCE_BANK = "hermes-sessions"
DEFAULT_SESSIONS_DIR = Path.home() / ".hermes" / "sessions"
DEFAULT_SOURCE_PROGRESS = Path.home() / ".hermes" / "hindsight" / "import_progress.json"
DEFAULT_PROGRESS_FILE = Path.home() / ".hermes" / "hindsight" / "aggregate_merge_into_hermes.json"
DEFAULT_MANIFEST_FILE = Path.home() / ".hermes" / "hindsight" / "aggregate_merge_into_hermes_manifest.jsonl"
EXTRACTOR_PATH = Path.home() / ".hermes" / "scripts" / "import_sessions_to_hindsight.py"

MIN_CONTENT_CHARS = 30
REQUEST_TIMEOUT = 120
MAX_RETRIES = 4
BACKOFF_SECONDS = [5, 15, 30, 60]
RATE_LIMIT_BACKOFF_SECONDS = 300

TOPIC_KEYWORDS = {
    "egomotion4d": [
        "egomotion4d", "trackingworld", "any4d", "dage", "pi3x", "roma2",
        "gtsam", "ate", "rpe", "unidepth", "metricdepth", "trajectory",
        "轨迹", "尺度", "位姿", "前端", "bundle adjustment",
    ],
    "hermes": [
        "hermes", "hindsight", "cch", "gateway", "provider", "memory",
        "mycompress", "/mycompress", "minimax", "openrouter", "claude code",
        "codex", "skills", "toolsets",
    ],
    "openclaw": ["openclaw", "clawhub"],
    "paper": ["paper", "arxiv", "论文", "pdf", "semanticscholar", "scholar"],
}


@dataclass
class SessionRecord:
    file: str
    session_id: str
    session_start: str
    last_updated: str
    day: str
    week: str
    month: str
    topic: str
    platform: str
    model: str
    chars: int
    content: str


@dataclass
class Bundle:
    index: int
    group_key: str
    topic: str
    start: str
    end: str
    records: list[SessionRecord]
    content: str
    document_id: str


def load_extractor() -> Callable[[dict[str, Any]], str]:
    if EXTRACTOR_PATH.exists():
        spec = importlib.util.spec_from_file_location("hindsight_session_importer", EXTRACTOR_PATH)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            if hasattr(module, "extract_conversation"):
                return module.extract_conversation
    return fallback_extract_conversation


def sanitize_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\x00", "")


def fallback_extract_text(msg: dict[str, Any]) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return sanitize_text(content).strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                cleaned = sanitize_text(item).strip()
                if cleaned:
                    parts.append(cleaned)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    cleaned = sanitize_text(text).strip()
                    if cleaned:
                        parts.append(cleaned)
        return sanitize_text("\n".join(parts)).strip()
    return ""


def fallback_extract_conversation(session_data: dict[str, Any]) -> str:
    conversation = []
    for msg in session_data.get("messages", []):
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = fallback_extract_text(msg)
        if not text:
            continue
        prefix = "User" if role == "user" else "Assistant"
        conversation.append(f"{prefix}: {text}")
    return sanitize_text("\n\n".join(conversation)).strip()


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:19])
    except Exception:
        return None


def day_of(value: str) -> str:
    dt = parse_dt(value)
    return dt.date().isoformat() if dt else (value[:10] if value else "unknown")


def month_of(value: str) -> str:
    d = day_of(value)
    return d[:7] if d != "unknown" else "unknown"


def week_of(value: str) -> str:
    dt = parse_dt(value)
    if not dt:
        return "unknown"
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def classify_topic(text: str) -> str:
    low = text.lower()
    scores = {}
    for topic, keys in TOPIC_KEYWORDS.items():
        scores[topic] = sum(low.count(k.lower()) for k in keys)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def load_source_files(source_progress: Path, sessions_dir: Path, source: str) -> list[str]:
    if source == "all":
        return [p.name for p in sorted(sessions_dir.glob("session_*.json"))]
    if not source_progress.exists():
        raise FileNotFoundError(f"source progress not found: {source_progress}")
    data = json.loads(source_progress.read_text(encoding="utf-8"))
    files = data.get("processed", [])
    return list(dict.fromkeys(str(x) for x in files))


def load_records(source_files: list[str], sessions_dir: Path, extractor: Callable[[dict[str, Any]], str]) -> tuple[list[SessionRecord], dict[str, Any]]:
    records: list[SessionRecord] = []
    skipped = {"missing": 0, "bad_json": 0, "too_short": 0}
    for fn in source_files:
        path = sessions_dir / fn
        if not path.exists():
            skipped["missing"] += 1
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            skipped["bad_json"] += 1
            continue
        content = sanitize_text(extractor(data)).strip()
        if len(content) < MIN_CONTENT_CHARS:
            skipped["too_short"] += 1
            continue
        ts = data.get("session_start") or data.get("last_updated") or ""
        records.append(SessionRecord(
            file=fn,
            session_id=str(data.get("session_id") or Path(fn).stem),
            session_start=str(data.get("session_start") or ""),
            last_updated=str(data.get("last_updated") or ""),
            day=day_of(ts),
            week=week_of(ts),
            month=month_of(ts),
            topic=classify_topic(content),
            platform=str(data.get("platform") or "unknown"),
            model=str(data.get("model") or "unknown"),
            chars=len(content),
            content=content,
        ))
    records.sort(key=lambda r: (r.session_start or r.last_updated, r.file))
    return records, skipped


def group_value(record: SessionRecord, group_by: str) -> str:
    if group_by == "all":
        return "all"
    if group_by == "topic":
        return record.topic
    if group_by == "day":
        return record.day
    if group_by == "week":
        return record.week
    if group_by == "month":
        return record.month
    if group_by == "day-topic":
        return f"{record.day}__{record.topic}"
    if group_by == "week-topic":
        return f"{record.week}__{record.topic}"
    if group_by == "month-topic":
        return f"{record.month}__{record.topic}"
    raise ValueError(f"unsupported group_by: {group_by}")


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def render_session_block(record: SessionRecord, part_index: int | None = None, part_total: int | None = None, text: str | None = None) -> str:
    part = ""
    if part_index is not None and part_total is not None:
        part = f" part={part_index + 1}/{part_total}"
    body = text if text is not None else record.content
    return (
        f"\n\n===== SESSION file={record.file} session_id={record.session_id}{part} "
        f"start={record.session_start or 'unknown'} topic={record.topic} "
        f"platform={record.platform} model={record.model} chars={len(body)} =====\n"
        f"{body.strip()}"
    )


def split_large_record(record: SessionRecord, max_payload_chars: int) -> list[str]:
    text = record.content
    if len(text) <= max_payload_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_payload_chars)
        if end < len(text):
            # 尽量在段落边界切，减少割裂。
            cut = text.rfind("\n\n", start, end)
            if cut > start + max_payload_chars * 0.55:
                end = cut
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]


def render_bundle_content(records: list[SessionRecord], group_key: str, group_by: str, max_bundle_chars: int, bundle_index: int, part_note: str = "") -> str:
    start_values = [r.session_start or r.last_updated for r in records if (r.session_start or r.last_updated)]
    date_start = min(start_values) if start_values else "unknown"
    date_end = max(start_values) if start_values else "unknown"
    topics = sorted(set(r.topic for r in records))
    header_lines = [
        "# Hermes 历史会话合并包",
        "",
        "这是一份为了降低 Hindsight 按次 LLM 调用成本而生成的合并记忆输入。",
        "请从这些历史对话中抽取稳定、有复用价值的事实、偏好、项目决策、工具经验、路径约定和排障结论；",
        "忽略一次性寒暄、明显过时的临时状态、重复系统提示和纯执行噪声。",
        "",
        f"target_bank: hermes",
        f"source_bank: {DEFAULT_SOURCE_BANK}",
        f"aggregate_version: v1",
        f"group_by: {group_by}",
        f"group_key: {group_key}",
        f"bundle_index: {bundle_index}",
        f"max_bundle_chars: {max_bundle_chars}",
        f"date_range: {date_start} .. {date_end}",
        f"topics: {', '.join(topics)}",
        f"source_session_count: {len(records)}",
    ]
    if part_note:
        header_lines.append(f"part_note: {part_note}")
    header_lines.extend(["", "source_sessions:"])
    for r in records[:80]:
        header_lines.append(f"- file={r.file} session_id={r.session_id} start={r.session_start or 'unknown'} topic={r.topic} chars={r.chars}")
    if len(records) > 80:
        header_lines.append(f"- ... omitted {len(records) - 80} more session headers")
    header_lines.extend(["", "--- conversations ---"])
    return "\n".join(header_lines)


def build_bundles(records: list[SessionRecord], group_by: str, max_bundle_chars: int) -> list[Bundle]:
    groups: dict[str, list[SessionRecord]] = {}
    for r in records:
        groups.setdefault(group_value(r, group_by), []).append(r)

    bundles: list[Bundle] = []
    global_idx = 0
    for gkey in sorted(groups):
        group_records = sorted(groups[gkey], key=lambda r: (r.session_start or r.last_updated, r.file))
        current: list[SessionRecord] = []
        current_blocks: list[str] = []
        current_len = 0

        def flush(part_note: str = "") -> None:
            nonlocal global_idx, current, current_blocks, current_len
            if not current:
                return
            header = render_bundle_content(current, gkey, group_by, max_bundle_chars, global_idx, part_note=part_note)
            content = header + "".join(current_blocks)
            starts = [r.session_start or r.last_updated for r in current if (r.session_start or r.last_updated)]
            start = min(starts) if starts else "unknown"
            end = max(starts) if starts else "unknown"
            topics = sorted(set(r.topic for r in current))
            hash_input = f"{group_by}|{max_bundle_chars}|{gkey}|{global_idx}|" + "|".join(r.session_id for r in current) + f"|{len(content)}"
            doc_id = f"hermes-aggregate::{group_by}::{gkey}::{global_idx:04d}::{short_hash(hash_input)}"
            bundles.append(Bundle(
                index=global_idx,
                group_key=gkey,
                topic=topics[0] if len(topics) == 1 else "mixed",
                start=start,
                end=end,
                records=list(current),
                content=content,
                document_id=doc_id,
            ))
            global_idx += 1
            current = []
            current_blocks = []
            current_len = 0

        for record in group_records:
            # header 约占 2k-10k；给单条超长 session 预留头部空间。
            max_payload = max(10000, max_bundle_chars - 12000)
            parts = split_large_record(record, max_payload)
            for pi, part_text in enumerate(parts):
                block = render_session_block(record, pi if len(parts) > 1 else None, len(parts) if len(parts) > 1 else None, text=part_text)
                block_len = len(block) + 1000
                if block_len > max_bundle_chars:
                    if current:
                        flush()
                    # 极端情况下硬切，确保单 bundle 不明显超过上限。
                    hard_payload = max(5000, max_bundle_chars - 12000)
                    hard_parts = [part_text[i:i + hard_payload] for i in range(0, len(part_text), hard_payload)]
                    for hi, hp in enumerate(hard_parts):
                        pseudo_block = render_session_block(record, hi, len(hard_parts), text=hp)
                        current = [record]
                        current_blocks = [pseudo_block]
                        current_len = len(pseudo_block)
                        flush(part_note="single oversized session hard split")
                    continue
                if current and current_len + block_len > max_bundle_chars:
                    flush()
                current.append(record)
                current_blocks.append(block)
                current_len += block_len
        flush()
    return bundles


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return int(s[int(k)])
    return int(s[f] * (c - k) + s[c] * (k - f))


def summarize_plan(records: list[SessionRecord], bundles: list[Bundle], skipped: dict[str, Any], group_by: str, max_bundle_chars: int) -> dict[str, Any]:
    chars = [r.chars for r in records]
    bundle_chars = [len(b.content) for b in bundles]
    topic_counts: dict[str, int] = {}
    for r in records:
        topic_counts[r.topic] = topic_counts.get(r.topic, 0) + 1
    sessions_per_bundle = [len(b.records) for b in bundles]
    return {
        "target_bank": DEFAULT_BANK_ID,
        "source_bank": DEFAULT_SOURCE_BANK,
        "group_by": group_by,
        "max_bundle_chars": max_bundle_chars,
        "source_sessions_with_content": len(records),
        "skipped": skipped,
        "estimated_old_chunk_requests_at_12000_chars": sum(max(1, math.ceil(r.chars / 12000)) for r in records),
        "aggregate_bundle_requests": len(bundles),
        "reduction_vs_old_chunk_requests": round(1 - (len(bundles) / max(1, sum(max(1, math.ceil(r.chars / 12000)) for r in records))), 4),
        "total_content_chars": sum(chars),
        "session_chars": {"p50": percentile(chars, 0.5), "p90": percentile(chars, 0.9), "max": max(chars) if chars else 0},
        "bundle_chars": {"p50": percentile(bundle_chars, 0.5), "p90": percentile(bundle_chars, 0.9), "max": max(bundle_chars) if bundle_chars else 0},
        "sessions_per_bundle": {"p50": percentile(sessions_per_bundle, 0.5), "p90": percentile(sessions_per_bundle, 0.9), "max": max(sessions_per_bundle) if sessions_per_bundle else 0},
        "topic_counts": dict(sorted(topic_counts.items())),
    }


def load_progress(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"processed": [], "failed": [], "last_run": None, "history": []}


def save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")


def write_manifest(path: Path, bundles: list[Bundle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for b in bundles:
            rec = {
                "document_id": b.document_id,
                "bundle_index": b.index,
                "group_key": b.group_key,
                "topic": b.topic,
                "date_range": [b.start, b.end],
                "source_session_count": len(b.records),
                "chars": len(b.content),
                "source_files": [r.file for r in b.records],
                "source_session_ids": [r.session_id for r in b.records],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def should_retry(error_text: str) -> bool:
    s = (error_text or "").lower()
    return any(x in s for x in [
        "429", "throttling", "concurrency allocated quota exceeded", "read timed out",
        "timeout", "temporarily unavailable", "connection reset", "current request",
        "rate limit",
    ])


def get_retry_delay(error_text: str, attempt: int) -> int:
    s = (error_text or "").lower()
    if "429" in s or "throttling" in s:
        return RATE_LIMIT_BACKOFF_SECONDS
    return BACKOFF_SECONDS[min(attempt, len(BACKOFF_SECONDS) - 1)]


def post_bundle(api: str, bank: str, bundle: Bundle, group_by: str, max_bundle_chars: int, source_progress: Path) -> tuple[bool, str | None]:
    first_ids = [r.session_id for r in bundle.records[:60]]
    first_files = [r.file for r in bundle.records[:60]]
    item = {
        "content": bundle.content,
        "document_id": bundle.document_id,
        "context": "hermes_conversation_aggregate",
        "timestamp": bundle.start if bundle.start != "unknown" else None,
        "metadata": {
            "source": "hermes_session_aggregate",
            "aggregate_version": "v1",
            "source_bank": DEFAULT_SOURCE_BANK,
            "target_bank": bank,
            "source_progress_file": str(source_progress),
            "group_by": group_by,
            "group_key": bundle.group_key,
            "topic": bundle.topic,
            "date_range_start": bundle.start,
            "date_range_end": bundle.end,
            "source_session_count": str(len(bundle.records)),
            "source_session_ids_first60": ",".join(first_ids),
            "source_files_first60": ",".join(first_files),
            "bundle_index": str(bundle.index),
            "max_bundle_chars": str(max_bundle_chars),
            "content_chars": str(len(bundle.content)),
        },
        "tags": ["hermes", "session-history", "aggregate", bundle.topic, group_by],
    }
    last_err = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{api}/v1/default/banks/{bank}/memories",
                json={"async": True, "items": [item]},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code in (200, 201, 202):
                return True, None
            err = f"API error: {resp.status_code} {resp.text[:500]}"
        except Exception as e:
            err = str(e)
        last_err = err
        if attempt < MAX_RETRIES and should_retry(err):
            delay = get_retry_delay(err, attempt)
            print(f"retrying bundle {bundle.document_id} after {delay}s due to: {err[:160]}", flush=True)
            time.sleep(delay)
            continue
        return False, err
    return False, last_err or "unknown error"


def check_bank_safety(bank: str, allow_hermes_sessions_target: bool) -> None:
    if bank == "hermes-sessions" and not allow_hermes_sessions_target:
        raise SystemExit("Refusing to write to hermes-sessions. Use --allow-hermes-sessions-target only if you really intend that.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Hermes sessions into Hindsight with fewer LLM calls")
    parser.add_argument("--mode", choices=["dry-run", "submit"], default="dry-run")
    parser.add_argument("--api", default=HINDSIGHT_API)
    parser.add_argument("--bank", default=DEFAULT_BANK_ID)
    parser.add_argument("--sessions-dir", type=Path, default=DEFAULT_SESSIONS_DIR)
    parser.add_argument("--source-progress", type=Path, default=DEFAULT_SOURCE_PROGRESS)
    parser.add_argument("--source", choices=["processed", "all"], default="processed")
    parser.add_argument("--progress-file", type=Path, default=DEFAULT_PROGRESS_FILE)
    parser.add_argument("--manifest-file", type=Path, default=DEFAULT_MANIFEST_FILE)
    parser.add_argument("--group-by", choices=["all", "topic", "day", "week", "month", "day-topic", "week-topic", "month-topic"], default="week-topic")
    parser.add_argument("--max-bundle-chars", type=int, default=120000)
    parser.add_argument("--limit-bundles", type=int, default=None)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--no-manifest", action="store_true")
    parser.add_argument("--allow-hermes-sessions-target", action="store_true")
    args = parser.parse_args()

    check_bank_safety(args.bank, args.allow_hermes_sessions_target)

    extractor = load_extractor()
    source_files = load_source_files(args.source_progress, args.sessions_dir, args.source)
    records, skipped = load_records(source_files, args.sessions_dir, extractor)
    bundles = build_bundles(records, args.group_by, args.max_bundle_chars)
    summary = summarize_plan(records, bundles, skipped, args.group_by, args.max_bundle_chars)
    summary["bank"] = args.bank
    summary["source_progress"] = str(args.source_progress)
    summary["progress_file"] = str(args.progress_file)
    summary["manifest_file"] = str(args.manifest_file)

    if args.limit_bundles is not None:
        bundles = bundles[:args.limit_bundles]
        summary["limited_to_bundles"] = args.limit_bundles

    if not args.no_manifest:
        write_manifest(args.manifest_file, bundles)
        summary["manifest_written"] = str(args.manifest_file)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

    if args.mode == "dry-run":
        print("DRY_RUN_ONLY: no Hindsight operations submitted", flush=True)
        return

    progress = load_progress(args.progress_file)
    done = set(progress.get("processed", []))
    failed_latest = {x.get("document_id"): x for x in progress.get("failed", []) if isinstance(x, dict) and x.get("document_id")}
    progress.setdefault("history", []).append({"time": datetime.now().isoformat(timespec="seconds"), "summary": summary})
    save_progress(args.progress_file, progress)

    submitted = 0
    skipped_done = 0
    for b in bundles:
        if b.document_id in done:
            skipped_done += 1
            continue
        ok, err = post_bundle(args.api, args.bank, b, args.group_by, args.max_bundle_chars, args.source_progress)
        progress["last_run"] = datetime.now().isoformat(timespec="seconds")
        if ok:
            progress.setdefault("processed", []).append(b.document_id)
            done.add(b.document_id)
            failed_latest.pop(b.document_id, None)
            submitted += 1
            print(f"submitted {submitted}/{len(bundles)} doc={b.document_id} sessions={len(b.records)} chars={len(b.content)}", flush=True)
        else:
            failed_latest[b.document_id] = {
                "document_id": b.document_id,
                "group_key": b.group_key,
                "bundle_index": b.index,
                "source_session_count": len(b.records),
                "chars": len(b.content),
                "error": err,
            }
            print(f"FAILED doc={b.document_id}: {err}", flush=True)
        progress["failed"] = list(failed_latest.values())
        if submitted % 10 == 0 or err:
            save_progress(args.progress_file, progress)
        if args.delay > 0:
            time.sleep(args.delay)

    save_progress(args.progress_file, progress)
    print(json.dumps({
        "done": True,
        "submitted_now": submitted,
        "skipped_already_done": skipped_done,
        "processed_total": len(progress.get("processed", [])),
        "failed_total": len(progress.get("failed", [])),
        "progress_file": str(args.progress_file),
    }, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
