#!/usr/bin/env python3
"""Repair Hindsight offline-consolidation documents that were written with zero memory units.

Read-only discovery uses PostgreSQL because the public list API can omit lineage
fields. Writes use the official Hindsight retain endpoint. The intended recovery
mode is Hindsight retain_extraction_mode=verbatim with a proven paid LLM, so each
repaired document produces at least one linked memory unit while preserving the
already generated offline summary as source text.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

DEFAULT_API = "http://127.0.0.1:8888"
DEFAULT_BANK = "hermes"
DEFAULT_PSQL = "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
REPORT_DIR = Path.home() / ".hermes" / "hindsight" / "reports"
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_.-]{8,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;|]+"),
]
EMPTY_MARKERS = {"", "(none)", "none", "n/a", "null", "[]", "{}", "(empty)", "- (none)", "- (empty)"}
SECTION_KEYS = [
    "executive_summary",
    "knowledge_points",
    "user_preferences",
    "project_decisions",
    "tooling_lessons",
    "risks",
    "open_questions",
]


def redact(text: str) -> str:
    out = text or ""
    for pat in SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def psql_json(sql: str, *, psql: str, db: str, user: str, host: str, port: int) -> Any:
    cmd = [psql, "-h", host, "-p", str(port), "-U", user, "-d", db, "-q", "-t", "-A", "-c", sql]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(redact(proc.stderr.strip() or proc.stdout.strip()))
    raw = proc.stdout.strip()
    return json.loads(raw) if raw else None


def discover_zero_unit_docs(args: argparse.Namespace) -> list[dict[str, Any]]:
    bank_lit = sql_literal(args.bank)
    prefix_lit = sql_literal(args.prefix + "%")
    period_clause = ""
    if args.period_like:
        period_clause = f" AND d.id LIKE {sql_literal('%::' + args.period_like + '::%')}"
    id_clause = ""
    if getattr(args, "document_id", None):
        ids = ",".join(sql_literal(x) for x in args.document_id)
        id_clause = f" AND d.id IN ({ids})"
    sql = f"""
WITH doc_units AS (
  SELECT
    d.id,
    d.original_text,
    d.tags,
    d.retain_params,
    d.created_at,
    COALESCE(count(m.id),0) AS unit_count
  FROM documents d
  LEFT JOIN memory_units m ON m.bank_id=d.bank_id AND m.document_id=d.id
  WHERE d.bank_id={bank_lit}
    AND d.id LIKE {prefix_lit}
    {period_clause}
    {id_clause}
  GROUP BY d.id,d.original_text,d.tags,d.retain_params,d.created_at
)
SELECT COALESCE(jsonb_agg(to_jsonb(doc_units) ORDER BY created_at, id), '[]'::jsonb)::text
FROM doc_units
WHERE unit_count = 0;
"""
    rows = psql_json(sql, psql=args.psql, db=args.db, user=args.user, host=args.host, port=args.port) or []
    if args.limit:
        rows = rows[: args.limit]
    return rows


def parse_doc_id(doc_id: str) -> dict[str, str]:
    parts = doc_id.split("::")
    out = {"prefix": parts[0] if parts else ""}
    if len(parts) >= 6:
        out.update({"scope": parts[1], "period": parts[2], "topic": parts[3], "index": parts[4], "digest": parts[5]})
    return out


def clean_line(value: Any) -> str:
    if isinstance(value, dict):
        title = value.get("title") or value.get("conclusion") or value.get("insight") or value.get("summary")
        extras = []
        for k in ["status", "evidence", "evidence_ids", "confidence", "scope", "tags", "rationale", "risk", "decision"]:
            v = value.get(k)
            if v in (None, "", [], {}):
                continue
            if isinstance(v, list):
                v = ", ".join(map(str, v[:12]))
            extras.append(f"{k}={v}")
        base = str(title or json.dumps(value, ensure_ascii=False))
        return base + (" | " + "; ".join(extras) if extras else "")
    if isinstance(value, list):
        return "; ".join(clean_line(x) for x in value if clean_line(x))
    return str(value or "").strip()


def is_substantive(line: str) -> bool:
    s = re.sub(r"\s+", " ", line or "").strip()
    if s.lower() in EMPTY_MARKERS:
        return False
    if s in {"-", "—"}:
        return False
    return len(s) >= 8


def load_llm_json(retain_params: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    meta = (retain_params or {}).get("metadata") or {}
    path = meta.get("output_json")
    if not path:
        return None, None
    p = Path(str(path))
    if not p.exists():
        return None, str(p)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data, str(p)
    except Exception:
        return None, str(p)


def build_repair_content(doc: dict[str, Any], *, min_lines: int = 1) -> tuple[str, dict[str, Any]]:
    doc_id = str(doc.get("id") or "")
    parsed_id = parse_doc_id(doc_id)
    retain_params = doc.get("retain_params") or {}
    if isinstance(retain_params, str):
        try:
            retain_params = json.loads(retain_params)
        except Exception:
            retain_params = {}
    llm_data, json_path = load_llm_json(retain_params)
    llm_obj = (llm_data or {}).get("llm_json") if isinstance(llm_data, dict) else None
    unit = (llm_data or {}).get("unit") if isinstance(llm_data, dict) else None
    unit = unit if isinstance(unit, dict) else {}
    meta = retain_params.get("metadata") if isinstance(retain_params, dict) else {}
    meta = meta if isinstance(meta, dict) else {}
    scope = str(meta.get("scope") or unit.get("scope") or parsed_id.get("scope") or "offline")
    period = str(meta.get("period") or unit.get("period") or parsed_id.get("period") or "unknown")
    topic = str(meta.get("topic") or unit.get("topic") or parsed_id.get("topic") or "general")
    date_start = str(meta.get("date_range_start") or unit.get("date_range_start") or "")
    date_end = str(meta.get("date_range_end") or unit.get("date_range_end") or "")
    source_ids = unit.get("source_ids") or []

    lines: list[str] = []
    lines.append(f"# Hindsight offline consolidation repaired unit source")
    lines.append(f"document_id: {doc_id}")
    lines.append(f"scope: {scope}")
    lines.append(f"period: {period}")
    lines.append(f"topic: {topic}")
    if date_start or date_end:
        lines.append(f"date_range: {date_start} .. {date_end}")
    if json_path:
        lines.append(f"local_json: {json_path}")
    lines.append("")

    substantive = 0
    if isinstance(llm_obj, dict):
        for key in SECTION_KEYS:
            value = llm_obj.get(key)
            section_lines: list[str] = []
            if isinstance(value, list):
                section_lines = [clean_line(x) for x in value]
            elif value not in (None, "", [], {}):
                section_lines = [clean_line(value)]
            section_lines = [x for x in section_lines if is_substantive(x)]
            if not section_lines:
                continue
            lines.append(f"## {key}")
            for item in section_lines:
                lines.append(f"- {item}")
                substantive += 1
            lines.append("")

        observations = llm_obj.get("canonical_observations") or []
        obs_lines = [clean_line(x) for x in observations if clean_line(x)] if isinstance(observations, list) else []
        obs_lines = [x for x in obs_lines if is_substantive(x)]
        if obs_lines:
            lines.append("## canonical_observations")
            for item in obs_lines:
                lines.append(f"- {item}")
                substantive += 1
            lines.append("")

    if substantive < min_lines:
        # Fallback to original text only if it has meaningful sections beyond boilerplate.
        original = str(doc.get("original_text") or "")
        bullets = []
        for raw in original.splitlines():
            s = raw.strip()
            if s.startswith("- ") and is_substantive(s[2:]):
                bullets.append(s[2:])
        if bullets:
            lines.append("## extracted_from_markdown")
            for b in bullets[:80]:
                lines.append(f"- {b}")
                substantive += 1
            lines.append("")

    if source_ids:
        lines.append("## source_refs")
        for sid in source_ids[:120]:
            lines.append(f"- {sid}")
        if len(source_ids) > 120:
            lines.append(f"- ... {len(source_ids)-120} more")
        lines.append("")

    report = {
        "document_id": doc_id,
        "scope": scope,
        "period": period,
        "topic": topic,
        "json_path": json_path,
        "substantive_lines": substantive,
        "source_refs": len(source_ids),
        "content_chars": len("\n".join(lines)),
    }
    return redact("\n".join(lines).strip() + "\n"), report


def observation_scopes(scope: str, topic: str) -> list[list[str]]:
    scopes: list[list[str]] = [["offline-consolidation"], [f"scope:{scope}"], [f"topic:{topic}"]]
    if topic == "hermes":
        scopes.append(["domain:hindsight"])
    if topic == "egomotion4d":
        scopes.append(["project:egomotion4d"])
    if topic == "openclaw":
        scopes.append(["project:openclaw"])
    if topic == "paper":
        scopes.append(["domain:paper"])
    return scopes


def post_json(url: str, payload: dict[str, Any], *, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw) if raw.strip() else {}


def submit_items(args: argparse.Namespace, items: list[dict[str, Any]]) -> list[str]:
    op_ids: list[str] = []
    for start in range(0, len(items), args.batch_size):
        batch = items[start : start + args.batch_size]
        payload = {"async": not bool(getattr(args, "sync", False)), "items": batch}
        resp = post_json(f"{args.api.rstrip('/')}/v1/default/banks/{args.bank}/memories", payload, timeout=600)
        ids = resp.get("operation_ids") or ([resp.get("operation_id")] if resp.get("operation_id") else [])
        op_ids.extend([str(x) for x in ids if x])
        print(f"submitted batch {start//args.batch_size+1}: items={len(batch)} async={payload['async']} operation_ids={ids} response={redact(json.dumps(resp, ensure_ascii=False))[:500]}")
        time.sleep(args.submit_delay)
    return op_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Repair zero-unit Hindsight offline-consolidation documents via official retain API")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--prefix", default="hermes-offline-consolidation::")
    ap.add_argument("--period-like", help="Optional period substring, e.g. history-through-2026-W18 or 2026-04-14")
    ap.add_argument("--document-id", action="append", help="Repair only this exact document id; may be repeated")
    ap.add_argument("--mode", choices=["dry-run", "submit"], default="dry-run")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--sync", action="store_true", help="Use synchronous retain request for small smoke tests; submit is async by default")
    ap.add_argument("--submit-delay", type=float, default=1.0)
    ap.add_argument("--min-lines", type=int, default=1)
    ap.add_argument("--psql", default=os.environ.get("HINDSIGHT_PSQL") or DEFAULT_PSQL)
    ap.add_argument("--host", default="/tmp")
    ap.add_argument("--port", type=int, default=5432)
    ap.add_argument("--db", default="hindsight")
    ap.add_argument("--user", default="hindsight")
    ap.add_argument("--report-stem", default=None)
    args = ap.parse_args(argv)

    docs = discover_zero_unit_docs(args)
    reports: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for doc in docs:
        content, rep = build_repair_content(doc, min_lines=args.min_lines)
        reports.append(rep)
        if rep["substantive_lines"] < args.min_lines:
            rep = {**rep, "skip_reason": "no_substantive_lines"}
            skipped.append(rep)
            continue
        retain_params = doc.get("retain_params") or {}
        if isinstance(retain_params, str):
            try:
                retain_params = json.loads(retain_params)
            except Exception:
                retain_params = {}
        meta = retain_params.get("metadata") if isinstance(retain_params, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        tags = list(doc.get("tags") or [])
        for t in ["repair:zero-unit", "repair:verbatim", "offline-consolidation"]:
            if t not in tags:
                tags.append(t)
        timestamp = retain_params.get("event_date") or meta.get("date_range_end") or None
        item = {
            "content": content,
            "document_id": rep["document_id"],
            "context": f"hindsight_offline_zero_unit_repair_{rep['scope']}",
            "metadata": {
                "source": "hindsight_offline_zero_unit_repair",
                "repair_version": "v1",
                "scope": rep["scope"],
                "period": rep["period"],
                "topic": rep["topic"],
                "original_output_json": str(rep.get("json_path") or ""),
                "original_document_id": rep["document_id"],
            },
            "tags": tags,
            "observation_scopes": observation_scopes(rep["scope"], rep["topic"]),
            "update_mode": "replace",
        }
        if timestamp:
            item["timestamp"] = timestamp
        items.append(item)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "bank": args.bank,
        "prefix": args.prefix,
        "period_like": args.period_like,
        "discovered_zero_docs": len(docs),
        "submit_candidates": len(items),
        "skipped": len(skipped),
        "candidate_substantive_lines": sum(r.get("substantive_lines", 0) for r in reports),
        "reports": reports,
        "skipped_docs": skipped,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stem = args.report_stem or f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-zero-unit-repair-{args.mode}"
    report_path = REPORT_DIR / f"{stem}.json"

    if args.mode == "submit" and items:
        op_ids = submit_items(args, items)
        summary["operation_ids"] = op_ids
    else:
        summary["operation_ids"] = []

    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report={report_path}")
    print(json.dumps({k: summary[k] for k in ["mode", "discovered_zero_docs", "submit_candidates", "skipped", "candidate_substantive_lines", "operation_ids"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
