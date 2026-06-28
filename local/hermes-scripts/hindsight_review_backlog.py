#!/usr/bin/env python3
"""Build a recoverable Hindsight review backlog sidecar from session manifests.

This script is read-only with respect to Hindsight: it does not call retain, does
not call an LLM, and does not mutate production.  It joins manifest records with
optional quality reports / read-only DB unit counts so later jobs can decide what
to re-score, re-retain, or repair.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hindsight-review-backlog-v1"

CREDENTIAL_RE = re.compile(
    r"(?i)((api[_ -]?key|secret|password|passwd|token)\s*[:=]\s*[A-Za-z0-9._\-/+]{12,}|bearer\s+[a-z0-9._\-]{16,}|sk-[a-z0-9_\-]{12,}|AKIA[0-9A-Z]{12,})"
)
REASONING_RE = re.compile(r"(?is)<think>|</think>|\breasoning(_content)?\b|chain[- ]of[- ]thought|思考过程|思维链")
CONTEXT_COMPACTION_RE = re.compile(r"(?i)CONTEXT COMPACTION|Active Task|Current task|handoff summary|memory-context")
INTERNAL_PLANNING_RE = re.compile(
    r"(?im)^\s*(Assistant:\s*)?(We need to|Need to|Need implement|Let's inspect|Run tests\.?|Now patch|TODO:).{0,120}$"
)
TOOL_LOG_RE = re.compile(r"(?i)\b(exit_code|stderr|stdout|Traceback|process session|terminal output|tool call|session_id=proc_)\b|\[tool:")
LOW_SIGNAL_RE = re.compile(r"(?i)^(hi|hello|ok|继续|好的|嗯|收到|thanks|thank you)[。.!！\s]*$")
DATEISH_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:  # pragma: no cover - defensive CLI error path
                raise SystemExit(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(obj, dict):
                records.append(obj)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def parent_document_id(document_id: str | None, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    if metadata.get("parent_document_id"):
        return str(metadata["parent_document_id"])
    doc = str(document_id or "")
    if "::" not in doc:
        return doc
    parts = doc.split("::")
    if len(parts) >= 2 and parts[0] == "hermes-session":
        return "::".join(parts[:2])
    return doc


def event_date_for_record(rec: dict[str, Any]) -> str | None:
    meta = rec.get("metadata") or {}
    for value in [
        rec.get("event_date"),
        meta.get("event_date"),
        meta.get("started_at"),
        meta.get("session_start"),
        meta.get("created_at"),
        meta.get("session_last_updated"),
        meta.get("last_updated"),
    ]:
        if value:
            return str(value)
    return None


def topic_key(tags: list[str]) -> str:
    semantic = [t for t in tags if isinstance(t, str) and (t.startswith("domain:") or t.startswith("topic:") or t.startswith("project:"))]
    return "+".join(sorted(semantic[:6])) or "<untagged>"


def infer_value_classes(tags: list[str], content: str, hardening: dict[str, Any] | None = None) -> list[str]:
    if hardening and hardening.get("primary_value_classes"):
        return list(hardening.get("primary_value_classes") or [])
    text = " ".join(tags) + "\n" + content[:4000]
    checks = [
        ("experiment_result", r"(?i)experiment|metric|ATE|RPE|结果|指标|对比|ablation|benchmark"),
        ("durable_decision", r"(?i)decision|决定|结论|默认|采用|不再|弃用"),
        ("error_root_cause", r"(?i)root cause|根因|bug|修复|故障|失败|报错|regression"),
        ("tool_lesson", r"(?i)tool|脚本|命令|CLI|Docker|systemd|Hermes|Hindsight|OpenClaw"),
        ("environment_fact", r"(?i)env|provider|port|路径|配置|版本|installed|部署|容器|PostgreSQL"),
        ("project_state", r"(?i)project|项目|进度|状态|TODO|计划|Egomotion4D"),
        ("user_preference", r"(?i)用户.*(偏好|要求|希望|不喜欢|prefer)|老王.*(希望|要求)"),
        ("open_question", r"(?i)open question|待确认|问题|是否|不确定|可能"),
    ]
    out = [name for name, pat in checks if re.search(pat, text)]
    return out or (["low_signal"] if len(content.strip()) < 300 else ["project_state"])


def deterministic_anomalies(rec: dict[str, Any], content: str, event_date: str | None) -> list[str]:
    meta = rec.get("metadata") or {}
    tags = rec.get("tags") or []
    anomalies: list[str] = []
    if not event_date:
        anomalies.append("date_missing")
    elif not DATEISH_RE.search(str(event_date)):
        anomalies.append("date_non_iso_or_ambiguous")
    if CREDENTIAL_RE.search(content):
        anomalies.append("credential_like")
    if REASONING_RE.search(content):
        anomalies.append("explicit_reasoning_or_thinking")
    if CONTEXT_COMPACTION_RE.search(content):
        anomalies.append("context_compaction_or_recall_context")
    if INTERNAL_PLANNING_RE.search(content):
        anomalies.append("internal_planning_like")
    if TOOL_LOG_RE.search(content):
        anomalies.append("tool_log_heavy")
    if int(rec.get("content_chars") or len(content)) > 12000:
        anomalies.append("overlong")
    semantic_tags = [t for t in tags if isinstance(t, str) and (t.startswith("domain:") or t.startswith("topic:") or t.startswith("project:"))]
    if len(set(semantic_tags)) >= 4:
        anomalies.append("multi_scope_or_overbroad_tags")
    if rec.get("action") == "manual_review":
        anomalies.append("manifest_manual_review")
    if rec.get("action") == "skip" or LOW_SIGNAL_RE.search(content.strip()):
        anomalies.append("low_signal_or_skip")
    if meta.get("part_count") and int(meta.get("part_count") or 1) > 1:
        anomalies.append("split_session_part")
    # stable order
    return list(dict.fromkeys(anomalies))


def load_hardening_overlays(paths: list[str]) -> dict[str, dict[str, Any]]:
    overlays: dict[str, dict[str, Any]] = {}
    for path in paths:
        data = load_json(path)
        z = data.get("zero_unit_report") or {}
        source = str(path)
        for section in ["samples", "high_value_retry_candidates", "zero_unit_documents"]:
            items = z.get(section) or []
            if isinstance(items, dict):
                items = list(items.values())
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                doc_id = str(item.get("document_id") or "")
                if not doc_id:
                    continue
                target = overlays.setdefault(doc_id, {"sources": []})
                target["sources"].append(source)
                for key in [
                    "zero_unit_class",
                    "recommended_route",
                    "primary_value_classes",
                    "semantic_score",
                    "noise_ratio",
                    "dropped_ratio",
                    "chars",
                    "model",
                    "tags",
                ]:
                    if key in item and item.get(key) is not None:
                        target[key] = item.get(key)
                target["current_retain_status"] = "zero_units"
    return overlays


def load_retry_gate_overlays(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        data = load_json(path)
        bank = data.get("bank")
        parent_counts = data.get("parent_unit_counts") or {}
        parent_zero = set(data.get("parent_zero") or [])
        counts = data.get("counts") or {}
        variant_stats = data.get("variant_stats") or {}
        for parent, unit_count in parent_counts.items():
            by_parent[str(parent)].append({
                "source": str(path),
                "bank": bank,
                "parent_unit_count": unit_count,
                "parent_status": "zero_units" if str(parent) in parent_zero else "has_units",
                "parent_coverage_ratio": counts.get("parent_coverage_ratio"),
                "artifact_counts": data.get("artifact_counts") or {},
                "variant_stats": variant_stats,
            })
    return by_parent


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def query_bank_unit_counts(*, psql_bin: str, host: str, port: str, user: str, dbname: str, bank: str, timeout: int = 120) -> dict[str, dict[str, Any]]:
    sql = f"""
SELECT d.id,
       COUNT(mu.id) AS unit_count,
       COALESCE(MIN(mu.event_date)::text, '') AS min_unit_event_date,
       COALESCE(MAX(mu.event_date)::text, '') AS max_unit_event_date
FROM documents d
LEFT JOIN memory_units mu ON mu.bank_id = d.bank_id AND mu.document_id = d.id
WHERE d.bank_id = {sql_literal(bank)}
GROUP BY d.id
ORDER BY d.id;
"""
    cmd = [psql_bin, "-h", host, "-p", str(port), "-U", user, "-d", dbname, "-At", "-F", "\t", "-c", sql]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"psql failed with exit {proc.returncode}: {proc.stderr.strip()[:500]}")
    out: dict[str, dict[str, Any]] = {}
    for row in csv.reader(proc.stdout.splitlines(), delimiter="\t"):
        if not row or len(row) < 2:
            continue
        doc_id, unit_count = row[0], row[1]
        out[doc_id] = {
            "memory_unit_count": int(unit_count or 0),
            "min_unit_event_date": row[2] or None if len(row) > 2 else None,
            "max_unit_event_date": row[3] or None if len(row) > 3 else None,
        }
    return out


def review_route_for(record: dict[str, Any], unit_info: dict[str, Any] | None, hardening: dict[str, Any] | None, anomalies: list[str]) -> str:
    action = record.get("action")
    if "credential_like" in anomalies:
        return "manual_review"
    if action == "manual_review" or "manifest_manual_review" in anomalies:
        return "manual_review"
    if action == "skip" or "low_signal_or_skip" in anomalies:
        return "wait"
    if unit_info and unit_info.get("memory_unit_count", 0) == 0:
        if hardening and hardening.get("recommended_route"):
            rr = str(hardening.get("recommended_route"))
            if rr in {"production_windowed", "retry_less_aggressive_cleaning", "retry_custom_mission"}:
                return "cluster_revisit"
        return "cluster_revisit"
    if unit_info and unit_info.get("memory_unit_count", 0) > 0 and anomalies:
        return "quality_review"
    if not unit_info:
        return "raw_only"
    return "monitor"


def build_backlog(
    *,
    manifest_path: str | Path,
    hardening_paths: list[str],
    retry_gate_paths: list[str],
    unit_counts: dict[str, dict[str, Any]] | None = None,
    bank: str | None = None,
    preview_chars: int = 0,
    rehydrate_content: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_records = load_jsonl(manifest_path)
    if rehydrate_content:
        script_dir = Path(__file__).resolve().parent
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        import hindsight_session_retain_runner as retain_runner  # local helper, read-only rehydrate
        hydrated: list[dict[str, Any]] = []
        for rec in manifest_records:
            if rec.get("content"):
                hydrated.append(rec)
                continue
            try:
                hydrated.append(retain_runner.rehydrate_record(rec))
            except Exception:
                hydrated.append(rec)
        manifest_records = hydrated
    hardening = load_hardening_overlays(hardening_paths)
    retry_overlays = load_retry_gate_overlays(retry_gate_paths)
    unit_counts = unit_counts or {}

    generated_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    counters: dict[str, Counter] = {
        "review_route": Counter(),
        "retain_status": Counter(),
        "anomalies": Counter(),
        "topic_key": Counter(),
        "manifest_action": Counter(),
        "value_class": Counter(),
    }

    for rec in manifest_records:
        doc_id = str(rec.get("document_id") or "")
        meta = rec.get("metadata") or {}
        content = rec.get("content") or ""
        content_sha = meta.get("content_sha256") or rec.get("content_sha256") or (sha256_text(content) if content else None)
        event_date = event_date_for_record(rec)
        parent_id = parent_document_id(doc_id, meta)
        tags = list(rec.get("tags") or [])
        overlay = hardening.get(doc_id) or hardening.get(parent_id)
        anomalies = deterministic_anomalies(rec, content, event_date)
        unit_info = unit_counts.get(doc_id) or unit_counts.get(parent_id)
        if unit_info:
            retain_status = "has_units" if int(unit_info.get("memory_unit_count") or 0) > 0 else "zero_units"
            retain_source = "postgres"
        elif overlay and overlay.get("current_retain_status"):
            retain_status = overlay.get("current_retain_status")
            retain_source = "hardening_json"
        elif rec.get("action") in {"skip", "manual_review"}:
            retain_status = "not_submitted_by_manifest_action"
            retain_source = "manifest"
        else:
            retain_status = "unknown_not_checked"
            retain_source = "none"
        value_classes = infer_value_classes(tags, content, overlay)
        route = review_route_for(rec, unit_info, overlay, anomalies)
        if retain_status == "zero_units" and route == "monitor":
            route = "cluster_revisit"
        entry = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at,
            "document_id": doc_id,
            "parent_document_id": parent_id,
            "source": {
                "manifest": str(manifest_path),
                "json_path": meta.get("json_path"),
                "session_id": meta.get("session_id"),
                "model": meta.get("model"),
                "platform": meta.get("platform"),
                "message_count": meta.get("message_count"),
                "part_index": meta.get("part_index"),
                "part_count": meta.get("part_count"),
                "source_file_sha256": meta.get("source_file_sha256"),
            },
            "event_date": event_date,
            "content_sha256": content_sha,
            "full_content_sha256": meta.get("full_content_sha256"),
            "content_chars": rec.get("content_chars") or len(content),
            "estimated_retain_chunks": rec.get("estimated_retain_chunks"),
            "manifest_decision": {
                "action": rec.get("action"),
                "reason": rec.get("reason"),
                "bank_target": rec.get("bank_target"),
                "context": rec.get("context"),
                "update_mode": rec.get("update_mode"),
            },
            "tags": tags,
            "observation_scopes": rec.get("observation_scopes") or [],
            "topic_key": topic_key(tags),
            "deterministic_anomalies": anomalies,
            "value_class_guess": value_classes,
            "current_retain_outcome": {
                "bank": bank,
                "status": retain_status,
                "source": retain_source,
                "memory_unit_count": unit_info.get("memory_unit_count") if unit_info else None,
                "min_unit_event_date": unit_info.get("min_unit_event_date") if unit_info else None,
                "max_unit_event_date": unit_info.get("max_unit_event_date") if unit_info else None,
            },
            "hardening_overlay": overlay or {},
            "retry_evidence": retry_overlays.get(parent_id, []),
            "review": {
                "state": "pending_review",
                "recommended_route": route,
                "llm_score_status": "not_scored",
                "delete_allowed": False,
                "production_mutation_allowed": False,
            },
        }
        if preview_chars > 0 and "credential_like" not in anomalies:
            entry["content_preview"] = content[:preview_chars]
        records.append(entry)

        counters["review_route"][route] += 1
        counters["retain_status"][retain_status] += 1
        counters["manifest_action"][str(rec.get("action"))] += 1
        counters["topic_key"][entry["topic_key"]] += 1
        for a in anomalies:
            counters["anomalies"][a] += 1
        for vc in value_classes:
            counters["value_class"][vc] += 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "manifest": str(manifest_path),
        "bank": bank,
        "record_count": len(records),
        "with_event_date": sum(1 for r in records if r.get("event_date")),
        "missing_event_date": sum(1 for r in records if not r.get("event_date")),
        "with_source_path": sum(1 for r in records if (r.get("source") or {}).get("json_path")),
        "by_review_route": dict(counters["review_route"].most_common()),
        "by_retain_status": dict(counters["retain_status"].most_common()),
        "by_manifest_action": dict(counters["manifest_action"].most_common()),
        "by_anomaly": dict(counters["anomalies"].most_common()),
        "by_value_class_guess": dict(counters["value_class"].most_common()),
        "top_topic_keys": dict(counters["topic_key"].most_common(20)),
        "hardening_sources": hardening_paths,
        "retry_gate_sources": retry_gate_paths,
        "rehydrate_content": rehydrate_content,
    }
    return records, summary


def write_summary_md(path: str | Path, summary: dict[str, Any]) -> None:
    lines = ["# Hindsight review backlog summary", ""]
    for key in ["generated_at", "manifest", "bank", "record_count", "with_event_date", "missing_event_date", "with_source_path"]:
        lines.append(f"- {key}: `{summary.get(key)}`")
    for title, key in [
        ("By review route", "by_review_route"),
        ("By retain status", "by_retain_status"),
        ("By manifest action", "by_manifest_action"),
        ("By anomaly", "by_anomaly"),
        ("By value class guess", "by_value_class_guess"),
        ("Top topic keys", "top_topic_keys"),
    ]:
        lines.extend(["", f"## {title}", "```json", json.dumps(summary.get(key) or {}, ensure_ascii=False, indent=2, sort_keys=True), "```"])
    if summary.get("hardening_sources"):
        lines.extend(["", "## Hardening sources"])
        lines.extend([f"- `{p}`" for p in summary.get("hardening_sources") or []])
    if summary.get("retry_gate_sources"):
        lines.extend(["", "## Retry gate sources"])
        lines.extend([f"- `{p}`" for p in summary.get("retry_gate_sources") or []])
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="Session manifest JSONL")
    ap.add_argument("--output", required=True, help="Output review_backlog JSONL")
    ap.add_argument("--summary-json", help="Output summary JSON")
    ap.add_argument("--summary-md", help="Output summary Markdown")
    ap.add_argument("--hardening-json", action="append", default=[], help="Optional session quality hardening JSON; repeatable")
    ap.add_argument("--retry-gate-json", action="append", default=[], help="Optional fact-quality gate JSON from retry/temp runs; repeatable")
    ap.add_argument("--bank", default=None, help="Production bank to read per-document unit counts from")
    ap.add_argument("--psql-bin", default="/home/wyr/.pg0/installation/18.1.0/bin/psql")
    ap.add_argument("--db-host", default="/tmp")
    ap.add_argument("--db-port", default="5432")
    ap.add_argument("--db-user", default="hindsight")
    ap.add_argument("--db-name", default="hindsight")
    ap.add_argument("--no-db", action="store_true", help="Do not query PostgreSQL even if --bank is set")
    ap.add_argument("--preview-chars", type=int, default=0, help="Include redacted local preview chars; default 0 keeps backlog content-free")
    ap.add_argument("--rehydrate-content", action="store_true", help="Read source JSON files to rebuild cleaned content for hashes/anomaly detection; output still omits content unless --preview-chars > 0")
    ap.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    unit_counts: dict[str, dict[str, Any]] = {}
    db_error = None
    if args.bank and not args.no_db:
        try:
            unit_counts = query_bank_unit_counts(
                psql_bin=args.psql_bin,
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                dbname=args.db_name,
                bank=args.bank,
            )
        except Exception as exc:
            db_error = str(exc)
    records, summary = build_backlog(
        manifest_path=args.manifest,
        hardening_paths=args.hardening_json,
        retry_gate_paths=args.retry_gate_json,
        unit_counts=unit_counts,
        bank=args.bank,
        preview_chars=max(0, args.preview_chars),
        rehydrate_content=args.rehydrate_content,
    )
    if db_error:
        summary["db_error"] = db_error
    summary["db_unit_count_records"] = len(unit_counts)
    write_jsonl(args.output, records)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.summary_md:
        write_summary_md(args.summary_md, summary)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"wrote {len(records)} records to {args.output}")
        if args.summary_json:
            print(f"summary_json: {args.summary_json}")
        if args.summary_md:
            print(f"summary_md: {args.summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
