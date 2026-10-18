#!/usr/bin/env python3
"""Read-only Hindsight bank quality audit.

This script is intentionally non-mutating. It audits a bank via the native
Hindsight API client where possible and writes JSON/Markdown reports.

Checks:
- runtime status/config/stats
- document and memory type composition
- broad/system tag pollution
- cross-domain contamination regex counts
- observation source/proof distribution
- missing source ids
- document/memory lineage gaps visible from API data
- recall smoke for fixed queries
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DEFAULT_API, HindsightNativeClient  # noqa: E402

DEFAULT_BANK = "hermes"
DEFAULT_REPORT_DIR = Path.home() / ".hermes" / "hindsight" / "reports"
BROAD_SYSTEM_TAGS = {"hermes", "sqlite", "incremental", "daily", "canonical", "source:tmp", "offline-v2", "observation", "topic"}

RECALL_QUERIES: dict[str, str] = {
    "hindsight_arch": "Hindsight session json native consolidation discard quarantine observation_scopes",
    "egomotion4d": "Egomotion4D VGGT DAGE ATE_metric trajectory scale window",
    "patent": "专利 OA1 审查意见 权利要求 意见陈述书",
    "openclaw": "OpenClaw ClawHub approval gateway probe No session found",
    "user_pref": "用户偏好 简洁 质疑精神 技术排障 直接执行",
    "cch": "CCH gpt-5.5 provider context_length Responses API identity",
}

SECRET_VALUE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_.-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_.-]{2,}\.\.\.[A-Za-z0-9_.-]{3,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;|]+"),
]


def redact_secrets(value: Any) -> Any:
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_VALUE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(v) for v in value)
    if isinstance(value, dict):
        return {k: redact_secrets(v) for k, v in value.items()}
    return value

CONTAMINATION_PATTERNS: dict[str, tuple[str, re.Pattern[str]]] = {
    "egomotion_tag_patent_terms": ("egomotion4d", re.compile(r"patent|OA1|office action|专利|审查意见|权利要求", re.I)),
    "egomotion_tag_openclaw_terms": ("egomotion4d", re.compile(r"OpenClaw|ClawHub|approval|gateway probe|No session found", re.I)),
    "openclaw_tag_patent_terms": ("openclaw", re.compile(r"patent|OA1|office action|专利|审查意见|权利要求", re.I)),
    "openclaw_tag_egomotion_terms": ("openclaw", re.compile(r"Egomotion|VGGT|DAGE|ATE|TrackingWorld|trajectory", re.I)),
    "hindsight_tag_autodrive_terms": ("hindsight", re.compile(r"Egomotion|VGGT|DAGE|ATE|OpenClaw|patent|OA1|专利", re.I)),
}


def item_type(item: dict[str, Any]) -> str:
    return str(item.get("type") or item.get("fact_type") or "<unknown>")


def item_text(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("content") or "")


def tags_of(item: dict[str, Any]) -> list[str]:
    tags = item.get("tags") or []
    return [str(t) for t in tags if t is not None]


def doc_prefix(document_id: Any) -> str:
    if not document_id:
        return "<null>"
    return str(document_id).split("::", 1)[0]


def source_ids(item: dict[str, Any]) -> list[str]:
    ids = item.get("source_memory_ids") or item.get("source_fact_ids") or []
    if isinstance(ids, str):
        return [ids]
    return [str(x) for x in ids if x]


def audit_collections(*, memories: list[dict[str, Any]], documents: list[dict[str, Any]], operations: list[dict[str, Any]]) -> dict[str, Any]:
    memory_ids = {str(m.get("id")) for m in memories if m.get("id")}
    document_ids = {str(d.get("id") or d.get("document_id")) for d in documents if d.get("id") or d.get("document_id")}

    by_type = Counter(item_type(m) for m in memories)
    doc_prefixes = Counter(doc_prefix(d.get("id") or d.get("document_id")) for d in documents)
    memory_doc_prefixes = Counter(doc_prefix(m.get("document_id")) for m in memories)
    tag_counts = Counter(t for m in memories for t in tags_of(m))
    broad_counts = {tag: tag_counts[tag] for tag in sorted(BROAD_SYSTEM_TAGS) if tag_counts[tag]}

    contamination = {name: 0 for name in CONTAMINATION_PATTERNS}
    contamination_samples: list[dict[str, Any]] = []
    for m in memories:
        ts = set(tags_of(m))
        text = item_text(m)
        for name, (tag, pat) in CONTAMINATION_PATTERNS.items():
            if tag in ts and pat.search(text):
                contamination[name] += 1
                if len(contamination_samples) < 20:
                    contamination_samples.append({
                        "id": m.get("id"),
                        "type": item_type(m),
                        "document_id": m.get("document_id"),
                        "tags": tags_of(m),
                        "metric": name,
                        "text": text[:300],
                    })

    observations = [m for m in memories if item_type(m) == "observation"]
    source_count_dist = Counter(str(len(source_ids(m))) for m in observations)
    proof_dist = Counter(str(m.get("proof_count", "<null>")) for m in observations)
    obs_doc_prefixes = Counter(doc_prefix(m.get("document_id")) for m in observations)
    missing_source_refs = 0
    observations_with_missing_source = 0
    for m in observations:
        missing = [sid for sid in source_ids(m) if sid not in memory_ids]
        missing_source_refs += len(missing)
        if missing:
            observations_with_missing_source += 1

    docs_with_units = {str(m.get("document_id")) for m in memories if m.get("document_id")}
    docs_without_units = len([d for d in document_ids if d not in docs_with_units])
    units_missing_document = len([m for m in memories if m.get("document_id") and str(m.get("document_id")) not in document_ids])

    null_scopes = sum(1 for m in memories if m.get("observation_scopes") in (None, [], ""))
    non_null_scopes = len(memories) - null_scopes

    operations_by_status = Counter(str(op.get("status") or "<unknown>") for op in operations)
    operations_by_type_status = Counter((str(op.get("operation_type") or op.get("task_type") or "<unknown>"), str(op.get("status") or "<unknown>")) for op in operations)

    duplicate_text_groups = 0
    text_counts = Counter(item_text(m) for m in memories if item_text(m))
    duplicate_text_groups = sum(1 for _, n in text_counts.items() if n > 1)

    return {
        "counts": {
            "documents": len(documents),
            "memory_units": len(memories),
            "observations": len(observations),
            "operations": len(operations),
        },
        "composition": {
            "memory_units_by_type": dict(by_type.most_common()),
            "documents_by_prefix": dict(doc_prefixes.most_common()),
            "memory_units_by_doc_prefix": dict(memory_doc_prefixes.most_common()),
        },
        "tag_quality": {
            "top_tags": tag_counts.most_common(50),
            "broad_system_tag_counts": broad_counts,
            "broad_system_tag_total": sum(broad_counts.values()),
        },
        "contamination_counts": contamination,
        "contamination_samples": contamination_samples,
        "observation_quality": {
            "source_count_distribution": dict(sorted(source_count_dist.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)),
            "proof_count_distribution": dict(sorted(proof_dist.items())),
            "observation_doc_prefixes": dict(obs_doc_prefixes.most_common()),
        },
        "lineage": {
            "source_refs": sum(len(source_ids(m)) for m in observations),
            "missing_source_refs": missing_source_refs,
            "observations_with_missing_source": observations_with_missing_source,
            "docs_without_units": docs_without_units,
            "units_missing_document": units_missing_document,
        },
        "scope_quality": {
            "null_observation_scopes": null_scopes,
            "non_null_observation_scopes": non_null_scopes,
        },
        "dedup_quality": {
            "duplicate_exact_text_groups": duplicate_text_groups,
        },
        "operations": {
            "by_status": dict(operations_by_status.most_common()),
            "by_type_status": [[typ, status, n] for (typ, status), n in sorted(operations_by_type_status.items())],
        },
    }


def api_lineage_looks_sparse(audit: dict[str, Any], memories: list[dict[str, Any]], documents: list[dict[str, Any]]) -> bool:
    if not memories or not documents:
        return False
    mem_with_doc = sum(1 for m in memories if m.get("document_id"))
    obs_with_sources = sum(1 for m in memories if item_type(m) == "observation" and source_ids(m))
    docs_without_units = int(audit.get("lineage", {}).get("docs_without_units") or 0)
    doc_count = len({str(d.get("id") or d.get("document_id")) for d in documents if d.get("id") or d.get("document_id")})
    # Current Hindsight list API can omit or sparsely expose document_id/source ids.
    # A common false positive is: memory rows have document_id, but API document ids
    # are not the durable document_id values used by memory_units, so every document
    # appears detached. In that case use the read-only PostgreSQL view before
    # reporting lineage gaps.
    api_every_doc_looks_detached = doc_count > 0 and docs_without_units == doc_count and mem_with_doc > 0
    return mem_with_doc == 0 or api_every_doc_looks_detached or (audit.get("counts", {}).get("observations", 0) and obs_with_sources == 0)


def _parse_psql_json_array(stdout: str) -> list[dict[str, Any]]:
    stdout = stdout.strip()
    if not stdout:
        return []
    data = json.loads(stdout)
    return data if isinstance(data, list) else []


def sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def load_db_collections(bank: str, *, psql: str | None = None, host: str = "/tmp", port: int = 5432, db: str = "hindsight", user: str = "hindsight") -> dict[str, Any]:
    """Read-only PostgreSQL fallback for fields not exposed by list API."""
    psql = psql or os.environ.get("HINDSIGHT_PSQL") or "/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
    bank_lit = sql_literal(bank)
    sql = rf'''
WITH docs AS (
  SELECT COALESCE(jsonb_agg(to_jsonb(d) ORDER BY d.created_at), '[]'::jsonb) AS arr
  FROM (
    SELECT id, bank_id, tags, metadata, created_at, updated_at
    FROM documents
    WHERE bank_id = {bank_lit}
  ) d
), mems AS (
  SELECT COALESCE(jsonb_agg(to_jsonb(m) ORDER BY m.created_at), '[]'::jsonb) AS arr
  FROM (
    SELECT id, bank_id, document_id, text, fact_type, tags, proof_count, source_memory_ids, observation_scopes, created_at, updated_at
    FROM memory_units
    WHERE bank_id = {bank_lit}
  ) m
), ops AS (
  SELECT COALESCE(jsonb_agg(to_jsonb(o) ORDER BY o.created_at), '[]'::jsonb) AS arr
  FROM (
    SELECT operation_id, bank_id, operation_type, status, created_at, updated_at, task_payload IS NULL AS payload_null
    FROM async_operations
    WHERE bank_id = {bank_lit}
  ) o
)
SELECT jsonb_build_object('documents',(SELECT arr FROM docs),'memories',(SELECT arr FROM mems),'operations',(SELECT arr FROM ops))::text;
'''
    cmd = [psql, "-h", host, "-p", str(port), "-U", user, "-d", db, "-q", "-t", "-A", "-c", sql]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(proc.stdout.strip())
    return {
        "source": "postgresql",
        "documents": data.get("documents") or [],
        "memories": data.get("memories") or [],
        "operations": data.get("operations") or [],
    }


def recall_smoke(client: Any, *, limit: int = 8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, query in RECALL_QUERIES.items():
        try:
            response = client.recall(query, types=["world", "experience", "observation"], limit=limit, budget="mid", max_tokens=2048)
            results = response.get("results") or response.get("memories") or response.get("items") or []
            rows = []
            for r in results[:limit]:
                rows.append({
                    "type": item_type(r),
                    "doc_prefix": doc_prefix(r.get("document_id")),
                    "tags": tags_of(r)[:10],
                    "text": item_text(r)[:260],
                })
            out[qid] = {"query": query, "count": len(results), "rows": rows}
        except Exception as e:
            out[qid] = {"query": query, "error": repr(e)}
    return out


def run_audit(*, client: Any | None = None, api: str = DEFAULT_API, bank: str = DEFAULT_BANK, max_memories: int = 200000, recall_smoke: bool = False, db_fallback: str = "auto", db_loader: Any | None = None) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank, timeout=60)
    runtime: dict[str, Any] = {}
    warnings: list[str] = []
    for name, fn in [("health", client.health), ("stats", client.stats), ("config", client.get_config)]:
        try:
            runtime[name] = fn()
        except Exception as e:
            runtime[name] = {"error": repr(e)}
    documents = list(client.list_all_documents(max_items=100000))
    memories = list(client.iter_memories(types=["world", "experience", "observation"], max_items=max_memories))
    operations = list(client.iter_operations(max_items=20000))
    data_source = "api"
    collection_audit = audit_collections(memories=memories, documents=documents, operations=operations)
    if db_fallback in {"auto", "always"} and (db_fallback == "always" or api_lineage_looks_sparse(collection_audit, memories, documents)):
        try:
            db_data = (db_loader or load_db_collections)(bank)
            documents = db_data.get("documents") or []
            memories = db_data.get("memories") or []
            operations = db_data.get("operations") or []
            collection_audit = audit_collections(memories=memories, documents=documents, operations=operations)
            data_source = "postgresql_fallback"
            warnings.append("Used read-only PostgreSQL fallback because API list results lacked lineage fields needed for quality audit.")
        except Exception as e:
            warnings.append(f"PostgreSQL fallback unavailable: {type(e).__name__}: {e}")
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bank": bank,
        "api": api,
        "data_source": data_source,
        "warnings": warnings,
        "runtime": runtime,
        "audit": collection_audit,
    }
    if recall_smoke:
        result["recall_smoke"] = globals()["recall_smoke"](client)
    return result


def write_markdown(result: dict[str, Any]) -> str:
    audit = result.get("audit", {})
    runtime = result.get("runtime", {})
    lines: list[str] = []
    lines.append("# Hindsight bank quality audit")
    lines.append("")
    lines.append(f"- generated_at: `{result.get('generated_at')}`")
    lines.append(f"- bank: `{result.get('bank')}`")
    lines.append(f"- api: `{result.get('api')}`")
    stats = runtime.get("stats") or {}
    if stats:
        lines.append(f"- stats: pending={stats.get('pending_operations')} processing={stats.get('processing_operations')} failed={stats.get('failed_operations')} documents={stats.get('total_documents')} nodes={stats.get('total_nodes')} observations={stats.get('total_observations')}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit.get("counts", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Tag quality")
    lines.append("")
    tagq = audit.get("tag_quality", {})
    lines.append("Broad/system tag counts:")
    lines.append("```json")
    lines.append(json.dumps(tagq.get("broad_system_tag_counts", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Contamination counts")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit.get("contamination_counts", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Observation quality")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit.get("observation_quality", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Lineage")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(audit.get("lineage", {}), ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    if result.get("recall_smoke"):
        lines.append("")
        lines.append("## Recall smoke")
        for qid, data in result["recall_smoke"].items():
            lines.append(f"### {qid}")
            lines.append(f"- count: {data.get('count')}")
            if data.get("error"):
                lines.append(f"- error: `{data.get('error')}`")
            for row in (data.get("rows") or [])[:5]:
                lines.append(f"- `{row.get('type')}` `{row.get('doc_prefix')}` tags={row.get('tags')} — {row.get('text')}")
    lines.append("")
    return "\n".join(lines)


def write_reports(result: dict[str, Any], output_dir: str | Path = DEFAULT_REPORT_DIR, *, stem: str | None = None) -> dict[str, str]:
    result = redact_secrets(result)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        stem = f"{stamp}-hindsight-bank-quality-audit-{result.get('bank','bank')}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(write_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Read-only Hindsight bank quality audit")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--max-memories", type=int, default=200000)
    ap.add_argument("--recall-smoke", action="store_true")
    ap.add_argument("--db-fallback", choices=["auto", "always", "never"], default="auto", help="Use read-only PostgreSQL fallback for lineage fields omitted by API list endpoints.")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--stem")
    ap.add_argument("--json", action="store_true", help="Print JSON result to stdout")
    args = ap.parse_args(argv)
    result = run_audit(api=args.api, bank=args.bank, max_memories=args.max_memories, recall_smoke=args.recall_smoke, db_fallback=args.db_fallback)
    result = redact_secrets(result)
    paths = write_reports(result, args.output_dir, stem=args.stem)
    result["paths"] = paths
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"json={paths['json']}")
        print(f"md={paths['md']}")
        counts = result.get("audit", {}).get("counts", {})
        lineage = result.get("audit", {}).get("lineage", {})
        print(f"counts={counts}")
        print(f"lineage={lineage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
