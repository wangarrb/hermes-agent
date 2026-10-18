#!/usr/bin/env python3
"""Conflict-driven audit for Offline Hindsight v2.

Entry points:
- automatic: scan high-level canonical observations for contamination, missing
  lineage, dangling sources, numeric/polarity conflict candidates.
- manual: accept a user-specified conflict claim and put it into the same case
  schema as automatic conflicts.

Read-only: no LLM calls, no Hindsight writes, no DB writes. Source scans use
official Hindsight APIs by default; direct DB fallback is intentionally opt-in
for forensic gaps only.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hindsight_conflict_core as core
from hindsight_native_client import DEFAULT_API, HindsightNativeClient

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_CARDS_ROOT = HERMES_HOME / "hindsight" / "offline_reflect" / "v2_cards"
DEFAULT_OFFLINE_ROOT = HERMES_HOME / "hindsight" / "offline_reflect"
DEFAULT_OUTPUT_DIR = DEFAULT_OFFLINE_ROOT / "conflict_audit"
API_SCAN_ERRORS: list[str] = []


def source_scan_unavailable_cases() -> list[dict[str, Any]]:
    errors = API_SCAN_ERRORS
    if not errors:
        return []
    target = {"id": "source_scan", "preview": "Hindsight source scan unavailable"}
    evidence = {"errors": [core.redact(e, 500) for e in errors[:10]], "layer": "hindsight_api"}
    return [core._case(  # type: ignore[attr-defined]
        "source_scan_unavailable",
        "P1",
        "Conflict audit could not verify Hindsight source documents/memory units through official APIs",
        target,
        evidence,
        repair_class="infrastructure_required_before_publish",
    )]


def db_unavailable_cases() -> list[dict[str, Any]]:
    """Backward-compatible alias for older tests/reports."""
    return source_scan_unavailable_cases()


def _client(bank: str, api: str = DEFAULT_API, client: HindsightNativeClient | None = None) -> HindsightNativeClient:
    return client or HindsightNativeClient(api=api, bank=bank)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_observations(cards_root: Path) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    index_path = cards_root / "observations_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("insight"):
                observations.append(obj)
    if observations:
        return observations
    # Fallback: compact card observations only.
    for p in sorted(list((cards_root / "topics").glob("*.json")) + list((cards_root / "global").glob("*.json"))):
        data = load_json(p, {}) or {}
        for obs in data.get("canonical_observations") or []:
            if isinstance(obs, dict) and obs.get("insight"):
                obj = dict(obs)
                obj.setdefault("topic", data.get("topic"))
                observations.append(obj)
    return observations


def _expand_document_aliases(ids: set[str]) -> set[str]:
    expanded = {str(x) for x in ids if str(x)}
    for doc_id in list(expanded):
        alias = core.offline_consolidation_doc_alias(doc_id)
        if alias:
            expanded.add(alias)
    return expanded


def known_document_ids(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None, max_items: int = 100000) -> set[str]:
    try:
        ids = {str(r.get("id") or "") for r in _client(bank, api, client).list_all_documents(max_items=max_items) if r.get("id")}
        return _expand_document_aliases(ids)
    except Exception as e:
        API_SCAN_ERRORS.append(f"documents API scan failed: {core.redact(repr(e), 500)}")
        return set()


def known_memory_ids(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None, max_items: int = 200000) -> set[str]:
    try:
        return {str(r.get("id") or "").lower() for r in _client(bank, api, client).iter_memories(types=["world", "experience", "observation"], max_items=max_items) if r.get("id")}
    except Exception as e:
        API_SCAN_ERRORS.append(f"memories API scan failed: {core.redact(repr(e), 500)}")
        return set()


def known_files(offline_root: Path, *, limit: int = 20000) -> set[str]:
    out: set[str] = set()
    for i, p in enumerate(offline_root.glob("**/*")):
        if i >= limit:
            break
        if p.is_file():
            out.add(str(p))
    return out


def _collect_local_lineage(obj: Any, document_ids: set[str], memory_ids: set[str]) -> None:
    if isinstance(obj, dict):
        doc_id = obj.get("document_id")
        if isinstance(doc_id, str) and doc_id.startswith("hermes-offline-consolidation::"):
            document_ids.add(doc_id)
        for key in ("source_ids", "evidence_ids"):
            values = obj.get(key)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        for uid in core.UUID_RE.findall(value):
                            memory_ids.add(uid.lower())
        for value in obj.values():
            _collect_local_lineage(value, document_ids, memory_ids)
    elif isinstance(obj, list):
        for value in obj:
            _collect_local_lineage(value, document_ids, memory_ids)


def known_local_lineage_ids(offline_root: Path, *, limit: int = 20000) -> tuple[set[str], set[str]]:
    """Collect traceable local offline source document IDs and evidence UUIDs.

    The v2 cards can point at synthetic `hermes-offline-consolidation::...`
    documents whose content lives as local daily/weekly JSON files rather than as
    current native Hindsight document rows. These are valid local lineage sources
    for the offline quality gate and should not be reported as dangling.
    """
    document_ids: set[str] = set()
    memory_ids: set[str] = set()
    scanned = 0
    for subdir in ("daily", "weekly"):
        root = offline_root / subdir
        if not root.exists():
            continue
        for p in root.glob("**/*.json"):
            if scanned >= limit:
                break
            scanned += 1
            try:
                data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            _collect_local_lineage(data, document_ids, memory_ids)
    return _expand_document_aliases(document_ids), memory_ids


def db_contamination_cases(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None, max_cases: int = 50) -> list[dict[str, Any]]:
    """Find obvious contamination already retained in source facts/docs via public API."""
    cases: list[dict[str, Any]] = []
    try:
        rows = _client(bank, api, client).iter_memories(types=["world", "experience", "observation"], max_items=200000)
        for row in rows:
            text = str(row.get("text") or "")
            hits = core.detect_contamination(text)
            if not hits:
                continue
            severity = core.max_severity(hits, "P2")
            if not core.severity_at_least(severity, "P2"):
                continue
            target = {
                "id": str(row.get("id") or ""),
                "document_id": str(row.get("document_id") or ""),
                "type": str(row.get("type") or row.get("fact_type") or ""),
                "preview": core.redact(text, 260),
            }
            evidence = {"contamination_hits": hits, "layer": "memory_units", "source": "official_api_memories_list"}
            cases.append(core._case(  # type: ignore[attr-defined]
                "source_fact_contamination",
                severity,
                "Retained source fact contains obvious raw/tool/log contamination",
                target,
                evidence,
                repair_class="official_delete_or_retain_reprocess",
            ))
            if len(cases) >= max_cases:
                break
    except Exception as e:
        API_SCAN_ERRORS.append(f"contamination API scan failed: {core.redact(repr(e), 500)}")
    return cases


def fact_outlier_cases(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None, threshold: int = 1000, max_cases: int = 20) -> list[dict[str, Any]]:
    try:
        rows = list(_client(bank, api, client).iter_memories(types=["world", "experience", "observation"], max_items=200000))
    except Exception as e:
        API_SCAN_ERRORS.append(f"fact outlier API scan failed: {core.redact(repr(e), 500)}")
        return []
    counts: Counter[str] = Counter()
    pollutants: Counter[str] = Counter()
    for row in rows:
        doc_id = str(row.get("document_id") or "")
        if not doc_id:
            continue
        counts[doc_id] += 1
        if core.detect_contamination(str(row.get("text") or "")):
            pollutants[doc_id] += 1
    candidates = [
        (doc_id, facts, pollutants.get(doc_id, 0))
        for doc_id, facts in counts.items()
        if facts >= threshold or pollutants.get(doc_id, 0) > 0
    ]
    candidates.sort(key=lambda x: (-x[2], -x[1], x[0]))
    cases: list[dict[str, Any]] = []
    for doc_id, facts, pollutant_count in candidates[:max_cases]:
        severity = "P1" if pollutant_count > 0 else "P2"
        cases.append(core._case(  # type: ignore[attr-defined]
            "source_document_fact_outlier",
            severity,
            "Source document has abnormal fact count or pollutant facts",
            {"id": doc_id, "document_id": doc_id, "preview": doc_id},
            {"facts": facts, "pollutant_facts": pollutant_count, "threshold": threshold, "source": "official_api_memories_list"},
            repair_class="source_document_audit_required",
        ))
    return cases


def dedupe_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for c in cases:
        cid = str(c.get("case_id") or "")
        if cid in seen:
            continue
        seen.add(cid)
        out.append(c)
    return out


def summarize(cases: list[dict[str, Any]], block_severity: str) -> dict[str, Any]:
    by_sev: dict[str, int] = {}
    by_type: dict[str, int] = {}
    blocking = []
    for c in cases:
        sev = str(c.get("severity") or "OK").upper()
        typ = str(c.get("type") or "unknown")
        by_sev[sev] = by_sev.get(sev, 0) + 1
        by_type[typ] = by_type.get(typ, 0) + 1
        if core.severity_at_least(sev, block_severity):
            blocking.append(c)
    return {
        "case_count": len(cases),
        "blocking_cases": len(blocking),
        "block_severity": block_severity,
        "by_severity": dict(sorted(by_sev.items(), key=lambda kv: (-core.severity_value(kv[0]), kv[0]))),
        "by_type": dict(sorted(by_type.items())),
        "blocking_examples": [
            {"case_id": c.get("case_id"), "type": c.get("type"), "severity": c.get("severity"), "target": c.get("target"), "title": c.get("title")}
            for c in blocking[:20]
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Hindsight Conflict / Raw Lineage Audit", ""]
    lines.append(f"generated_at: {report.get('generated_at')}")
    lines.append(f"decision: {report.get('decision')}")
    s = report.get("summary") or {}
    lines.append(f"case_count: {s.get('case_count')}")
    lines.append(f"blocking_cases: {s.get('blocking_cases')} >= {s.get('block_severity')}")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"by_severity: {s.get('by_severity')}")
    lines.append(f"by_type: {s.get('by_type')}")
    lines.append("")
    lines.append("## Blocking Examples")
    for c in s.get("blocking_examples") or []:
        target = c.get("target") or {}
        lines.append(f"- [{c.get('severity')}] {c.get('type')} {c.get('case_id')}: {c.get('title')}")
        lines.append(f"  target: {target.get('id') or target.get('document_id') or target.get('preview')}")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only: no DB writes, no Hindsight writes, no LLM calls.")
    lines.append("- Destructive repairs require explicit user confirmation.")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Conflict-driven Hindsight audit: automatic and manual conflict intake")
    ap.add_argument("--cards-root", default=str(DEFAULT_CARDS_ROOT))
    ap.add_argument("--offline-root", default=str(DEFAULT_OFFLINE_ROOT))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--bank", default="hermes")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--block-severity", default="P1", choices=["P0", "P1", "P2", "P3"])
    ap.add_argument("--manual-claim", help="User-specified conflict/quality concern to enter into the same workflow")
    ap.add_argument("--manual-only", action="store_true", help="Create only the manual conflict case; skip automatic scan for focused handling")
    ap.add_argument("--target-id", help="Target observation/fact/document id for --manual-claim")
    ap.add_argument("--target-text", help="Target text preview for --manual-claim")
    ap.add_argument("--skip-source-scan", action="store_true", help="Skip official Hindsight API source fact/document scan")
    ap.add_argument("--skip-db-scan", action="store_true", help="Deprecated alias for --skip-source-scan")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cards_root = Path(args.cards_root).expanduser()
    offline_root = Path(args.offline_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    API_SCAN_ERRORS.clear()
    client = HindsightNativeClient(api=args.api, bank=args.bank)

    observations = load_observations(cards_root)
    api_docs = known_document_ids(args.bank, api=args.api, client=client) if not args.manual_only else set()
    api_mems = known_memory_ids(args.bank, api=args.api, client=client) if not args.manual_only else set()
    local_docs, local_mems = known_local_lineage_ids(offline_root) if not args.manual_only else (set(), set())
    docs = api_docs | local_docs
    mems = api_mems | local_mems
    files = known_files(offline_root) if not args.manual_only else set()
    manual_cases = []
    if args.manual_claim:
        manual_cases.append(core.manual_conflict_case(claim=args.manual_claim, target_id=args.target_id, target_text=args.target_text, severity="P1"))
    cases: list[dict[str, Any]] = []
    if not args.manual_only:
        cases.extend(core.build_conflict_cases(
            observations,
            known_document_ids=docs,
            known_memory_ids=mems,
            known_file_paths=files,
            max_cases=500,
        ))
        if not (args.skip_source_scan or args.skip_db_scan):
            if not API_SCAN_ERRORS:
                cases.extend(db_contamination_cases(args.bank, api=args.api, client=client))
                cases.extend(fact_outlier_cases(args.bank, api=args.api, client=client))
            cases.extend(source_scan_unavailable_cases())
    cases.extend(manual_cases)
    cases = dedupe_cases(cases)
    summary = summarize(cases, args.block_severity)
    decision = "pass" if int(summary.get("blocking_cases") or 0) == 0 else "blocked_conflict_review_required"
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "decision": decision,
        "cards_root": str(cards_root),
        "offline_root": str(offline_root),
        "bank": args.bank,
        "api": args.api,
        "observation_count": len(observations),
        "manual_case_count": len(manual_cases),
        "manual_only": bool(args.manual_only),
        "known_counts": {
            "documents": len(docs),
            "memory_units": len(mems),
            "api_documents": len(api_docs),
            "api_memory_units": len(api_mems),
            "local_lineage_documents": len(local_docs),
            "local_lineage_memory_ids": len(local_mems),
            "files": len(files),
        },
        "source_scan_errors": [core.redact(e, 500) for e in API_SCAN_ERRORS],
        "db_scan_errors": [],
        "summary": summary,
        "cases": cases,
        "safety": ["read_only", "official_api_source_scan", "no_llm_calls", "no_hindsight_writes", "no_db_writes", "destructive_actions_require_user_confirmation"],
    }
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"conflict-audit-{ts}.json"
    md_path = output_dir / f"conflict-audit-{ts}.md"
    report["json_path"] = str(json_path)
    report["markdown_path"] = str(md_path)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    (output_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "latest.md").write_text(render_markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(report))
        print(f"saved_json: {json_path}")
        print(f"saved_markdown: {md_path}")


if __name__ == "__main__":
    main()
