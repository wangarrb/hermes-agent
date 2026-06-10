#!/usr/bin/env python3
"""Read-only audit for Offline Hindsight v2 readiness.

Checks:
- Hindsight health/stats/queue safety
- Hindsight official API document/fact layering
- local offline JSON outputs vs Hindsight documents
- observation/canonical_observation gap
- language stability of offline outputs
- auto-discovered entity alias fragmentation
- recall source mix for generic and auto-discovered high-level probes

No LLM calls. No Hindsight writes.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hindsight_native_client import DEFAULT_API as HINDSIGHT_DEFAULT_API, HindsightNativeClient

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_API = HINDSIGHT_DEFAULT_API
DEFAULT_BANK = "hermes"
DEFAULT_OFFLINE_ROOT = HERMES_HOME / "hindsight" / "offline_reflect"

CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

GENERIC_RECALL_PROBES = [
    {"id": "user_preferences", "query": "用户偏好 user preferences 规则 rules 沟通风格 communication style 反复纠正 recurring corrections"},
    {"id": "project_decisions", "query": "项目决策 project decisions 架构选择 architecture choices 技术取舍 technical tradeoffs 当前推荐方案"},
    {"id": "tooling_lessons", "query": "工具经验 tooling lessons 环境配置 environment configuration 命令 commands 调试流程 debugging workflow"},
    {"id": "risks_open_questions", "query": "风险 blockers 未解决问题 open questions 禁忌 do not do 注意事项"},
    {"id": "offline_pipeline", "query": "离线记忆 offline memory consolidation pipeline raw facts daily weekly global observations"},
]

GENERIC_ENTITY_PREFIXES = {
    "raw", "old", "new", "backup", "local", "global", "canonical", "final",
    "draft", "temp", "tmp", "latest", "current", "source", "summary", "report",
    "note", "notes", "doc", "docs", "v1", "v2", "v3", "version", "ver", "rev",
}

GENERIC_ENTITY_SUFFIXES = {
    "frontend", "backend", "baseline", "native", "prior", "selector", "provider", "adapter",
    "config", "yaml", "yml", "json", "py", "md", "txt", "api", "key", "model", "service",
    "server", "client", "script", "wrapper", "tool", "pipeline", "optimizer", "factor", "graph",
    "report", "audit", "test", "benchmark", "experiment", "mode", "version", "v1", "v2", "v3",
}

GENERIC_ENTITY_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "user", "assistant", "agent",
    "project", "file", "path", "data", "result", "results", "system", "memory", "offline",
}


def clean_json_text(text: str) -> str:
    return CONTROL_CHARS.sub("", text or "")



def classify_doc(document_id: str) -> str:
    if document_id.startswith("hermes-offline-canonical::"):
        return "canonical"
    if document_id.startswith("hermes-offline-consolidation::weekly::"):
        return "offline_weekly"
    if document_id.startswith("hermes-offline-consolidation::daily::"):
        return "offline_daily"
    if document_id.startswith("hermes-sqlite::"):
        return "sqlite_import"
    return "other"


def audit_db(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None) -> dict[str, Any]:
    """Audit Hindsight store through official APIs.

    Historical name `audit_db` is kept for report compatibility; implementation is
    API-first to avoid coupling to PostgreSQL internals.
    """
    client = client or HindsightNativeClient(api=api, bank=bank)
    errors: list[str] = []
    try:
        docs = client.list_all_documents(max_items=100000)
    except Exception as e:
        docs = []
        errors.append(f"documents API scan failed: {repr(e)[:500]}")
    try:
        memories = list(client.iter_memories(types=["world", "experience", "observation"], max_items=200000))
    except Exception as e:
        memories = []
        errors.append(f"memories API scan failed: {repr(e)[:500]}")
    try:
        operations = list(client.iter_operations(max_items=10000))
    except Exception as e:
        operations = []
        errors.append(f"operations API scan failed: {repr(e)[:500]}")

    docs_by_kind = Counter(classify_doc(str(d.get("id") or "")) for d in docs)
    facts_counter: Counter[tuple[str, str]] = Counter()
    for m in memories:
        kind = classify_doc(str(m.get("document_id") or ""))
        fact_type = str(m.get("type") or m.get("fact_type") or "null")
        facts_counter[(kind, fact_type)] += 1
    operation_counter: Counter[tuple[str, str]] = Counter()
    for op in operations:
        operation_counter[(str(op.get("task_type") or op.get("operation_type") or "unknown"), str(op.get("status") or "unknown"))] += 1
    weekly_periods: Counter[str] = Counter()
    daily_periods: Counter[str] = Counter()
    for d in docs:
        doc_id = str(d.get("id") or "")
        m_week = re.search(r"::weekly::([^:]+)::", doc_id)
        if m_week:
            weekly_periods[m_week.group(1)] += 1
        m_day = re.search(r"::daily::(\d{4}-\d{2}-\d{2})", doc_id)
        if m_day:
            daily_periods[m_day.group(1)] += 1
    return {
        "source": "official_api",
        "errors": errors,
        "documents_by_kind": dict(sorted(docs_by_kind.items())),
        "facts_by_kind_type": [[k, t, str(n)] for (k, t), n in sorted(facts_counter.items())],
        "async_operations": [[typ, status, str(n), "n/a"] for (typ, status), n in sorted(operation_counter.items())],
        "weekly_periods": dict(sorted(weekly_periods.items())),
        "daily_periods": dict(sorted(daily_periods.items())),
    }


def audit_local_outputs(root: Path) -> dict[str, Any]:
    active_files = []
    backup_files = []
    for f in glob.glob(str(root / "**" / "*.json"), recursive=True):
        if "_bad-output-backup" in f:
            backup_files.append(Path(f))
        else:
            active_files.append(Path(f))
    counts = Counter()
    language = Counter()
    empty_sections = Counter()
    topic_counter = Counter()
    doc_ids: dict[str, str] = {}
    sample_problems: list[dict[str, Any]] = []
    canonical_count = 0
    llm_canonical_count = 0
    v2_card_files = 0
    v2_card_observation_count = 0
    v2_card_topics: Counter[str] = Counter()
    for path in active_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            counts[("parse_error", "json")] += 1
            sample_problems.append({"path": str(path), "error": repr(e)})
            continue
        if str(data.get("schema_version") or "").startswith("offline-hindsight-v2-card") or str(data.get("card_id") or "").startswith("offline-v2-card::"):
            v2_card_files += 1
            obs_list = data.get("canonical_observations") or []
            v2_card_observation_count += len(obs_list)
            canonical_count += len(obs_list)
            topic = str(data.get("topic") or "")
            if topic:
                v2_card_topics[topic] += len(obs_list)
            continue
        doc_id = data.get("document_id")
        if doc_id:
            doc_ids[doc_id] = str(path)
        unit = data.get("unit") or {}
        scope = unit.get("scope") or "unknown"
        topic = unit.get("topic") or unit.get("group") or ""
        if topic:
            topic_counter[str(topic)] += 1
        obj = data.get("llm_json") or {}
        counts[(scope, "files")] += 1
        if not obj:
            counts[(scope, "no_llm_json")] += 1
            continue
        if obj.get("canonical_observations"):
            llm_canonical_count += len(obj.get("canonical_observations") or [])
            canonical_count += len(obj.get("canonical_observations") or [])
        drop_notes = obj.get("drop_notes") or []
        for note in drop_notes:
            text = str(note)
            if "single knowledge-point" in text:
                counts[(scope, "single_wrapped")] += 1
            if "LLM returned a list" in text:
                counts[(scope, "list_wrapped")] += 1
        combined = " ".join(map(str, obj.get("executive_summary") or []))
        for kp in obj.get("knowledge_points") or []:
            if isinstance(kp, dict):
                combined += " " + str(kp.get("conclusion") or kp.get("title") or "")
        zh = sum("\u4e00" <= c <= "\u9fff" for c in combined)
        en = sum("a" <= c.lower() <= "z" for c in combined)
        if zh == 0 and en > 50:
            language[(scope, "english_only")] += 1
        elif zh > 0 and en > 0:
            language[(scope, "mixed")] += 1
        elif zh > 0:
            language[(scope, "chinese")] += 1
        else:
            language[(scope, "empty")] += 1
        for sec in ["executive_summary", "knowledge_points", "user_preferences", "project_decisions", "tooling_lessons", "risks", "open_questions", "canonical_observations"]:
            if not obj.get(sec):
                empty_sections[(scope, sec)] += 1
    v2_observation_index_count = 0
    index_path = root / "v2_cards" / "observations_index.jsonl"
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("insight"):
                v2_observation_index_count += 1
    canonical_total = llm_canonical_count + max(v2_card_observation_count, v2_observation_index_count)
    return {
        "active_json_files": len(active_files),
        "backup_json_files": len(backup_files),
        "counts": {"|".join(k): v for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))},
        "language": {"|".join(k): v for k, v in sorted(language.items(), key=lambda kv: str(kv[0]))},
        "empty_sections": {"|".join(k): v for k, v in sorted(empty_sections.items(), key=lambda kv: str(kv[0]))},
        "canonical_observations_in_local_json": canonical_total,
        "llm_canonical_observation_count": llm_canonical_count,
        "v2_card_files": v2_card_files,
        "v2_card_observation_count": v2_card_observation_count,
        "v2_observation_index_count": v2_observation_index_count,
        "v2_card_topics": dict(v2_card_topics.most_common(50)),
        "doc_ids": doc_ids,
        "topics": [t for t, _ in topic_counter.most_common(50)],
        "sample_problems": sample_problems[:10],
    }


def db_offline_doc_ids(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None) -> set[str]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    try:
        docs = client.list_all_documents(q="hermes-offline-consolidation::", max_items=100000)
    except Exception:
        return set()
    return {str(d.get("id") or "") for d in docs if str(d.get("id") or "").startswith("hermes-offline-consolidation::")}


def canonical_entity_key(name: str) -> str:
    """Best-effort generic alias key.

    This intentionally does not know about any user's projects. It groups obvious
    variants such as "Foo", "FooFrontend", "Foo baseline", "foo.yaml", or
    "/path/to/Foo" by lightweight normalization only. It also strips generic
    wrappers like raw/local/global/version prefixes so fragment clusters are
    discovered automatically instead of being hand-coded per project.
    """
    s = (name or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/")
    s = s.rsplit("/", 1)[-1]
    s = re.sub(r"\.(py|md|txt|json|yaml|yml|toml|ini|cfg|sh)$", "", s, flags=re.I)
    s = re.sub(r"[_\-]+", " ", s)
    # Split CamelCase before lowercasing: FooFrontend -> Foo Frontend.
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9]*|[\u4e00-\u9fff]+", s)]
    while words and (words[0] in GENERIC_ENTITY_PREFIXES or re.fullmatch(r"v\d+", words[0])):
        words.pop(0)
    while words and words[-1] in GENERIC_ENTITY_SUFFIXES:
        words.pop()
    words = [w for w in words if w not in GENERIC_ENTITY_STOPWORDS]
    if not words:
        return ""
    key = " ".join(words[:4])
    return key if len(key) >= 3 else ""


def audit_entities(bank: str, *, api: str = DEFAULT_API, client: HindsightNativeClient | None = None, min_alias_count: int = 4, limit_groups: int = 20) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    try:
        names = [str(r.get("canonical_name") or "") for r in client.list_all_entities(max_items=200000) if r.get("canonical_name")]
        source = "official_api"
        errors: list[str] = []
    except Exception as e:
        names = []
        source = "official_api"
        errors = [repr(e)[:500]]
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        key = canonical_entity_key(name)
        if key:
            grouped[key].append(name)
    alias_groups = []
    for key, vals in grouped.items():
        unique = sorted(set(vals), key=lambda x: (len(x), x.lower()))
        if len(unique) >= min_alias_count:
            alias_groups.append({"key": key, "count": len(unique), "examples": unique[:25]})
    alias_groups.sort(key=lambda x: (-int(x["count"]), str(x["key"])))
    return {
        "total_entities": len(names),
        "alias_groups": alias_groups[:limit_groups],
        "alias_group_count": len(alias_groups),
        "method": "generic_normalization_prefix_suffix_no_project_presets",
        "source": source,
        "errors": errors,
    }


def probe_id_for_topic(topic: str) -> str:
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")[:40]
    if ascii_slug:
        return "topic_" + ascii_slug
    digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()[:10]
    return "topic_zh_" + digest


def slugify_id(text: str, *, prefix: str = "item") -> str:
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")[:40]
    if ascii_slug:
        return ascii_slug
    digest = hashlib.sha1((text or prefix).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_zh_{digest}"


def extract_topic_terms_from_local(local: dict[str, Any], *, limit: int = 10) -> list[dict[str, str]]:
    topic_counter: Counter[str] = Counter()
    for topic in local.get("topics", []):
        if not topic:
            continue
        t = str(topic).strip()
        if not t or t.lower() in {"unknown", "general", "global"}:
            continue
        topic_counter[t] += 1
    return [
        {"id": probe_id_for_topic(topic), "query": f"{topic} 当前结论 current conclusions 决策 decisions 风险 risks 经验 lessons 证据 evidence"}
        for topic, _ in topic_counter.most_common(limit)
    ]


def build_recall_probes(local: dict[str, Any], *, auto_probe_limit: int = 5) -> list[dict[str, str]]:
    probes = list(GENERIC_RECALL_PROBES)
    seen = {p["id"] for p in probes}
    for p in extract_topic_terms_from_local(local, limit=auto_probe_limit):
        if p["id"] not in seen:
            probes.append(p)
            seen.add(p["id"])
    return probes


def recall(api: str, bank: str, query: str, limit: int, *, client: HindsightNativeClient | None = None) -> list[dict[str, Any]]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    try:
        data = client.recall(query, limit=limit, types=["observation", "world", "experience"])
        return data.get("results") or []
    except Exception as e:
        return [{"error": repr(e), "query": query}]


def audit_recall_mix(api: str, bank: str, probes: list[dict[str, str]], limit: int = 8, *, client: HindsightNativeClient | None = None) -> dict[str, Any]:
    client = client or HindsightNativeClient(api=api, bank=bank)
    out: dict[str, Any] = {}
    for probe in probes:
        results = recall(api, bank, probe["query"], limit, client=client)
        mix = Counter()
        rows = []
        for rank, r in enumerate(results[:limit], 1):
            if "error" in r:
                rows.append({"rank": rank, "error": r["error"]})
                continue
            doc = r.get("document_id") or ""
            layer = classify_doc(doc)
            mix[layer] += 1
            rows.append({
                "rank": rank,
                "layer": layer,
                "type": r.get("type"),
                "document_id": doc,
                "text_preview": (r.get("text") or "").replace("\n", " ")[:180],
            })
        out[probe["id"]] = {"query": probe["query"], "mix": dict(mix), "top": rows}
    return out


def build_findings(stats: dict[str, Any], db: dict[str, Any], local: dict[str, Any], reconciliation: dict[str, Any], entities: dict[str, Any], recall_mix: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    pending = stats.get("pending_operations", 0) or 0
    processing = stats.get("processing_operations", 0) or 0
    failed = stats.get("failed_operations", 0) or 0
    if pending or processing or failed:
        findings.append({"severity": "P0", "title": "queue_not_clean", "detail": f"pending={pending} processing={processing} failed={failed}"})
    else:
        findings.append({"severity": "OK", "title": "queue_clean", "detail": "pending/processing/failed all zero"})

    observation_count = (stats.get("nodes_by_fact_type") or {}).get("observation", stats.get("total_observations", 0)) or 0
    total_nodes = stats.get("total_nodes", 0) or 0
    canonical_observation_units = 0
    for row in db.get("facts_by_kind_type", []) or []:
        if len(row) >= 3 and row[0] == "canonical" and row[1] == "observation":
            try:
                canonical_observation_units += int(row[2])
            except Exception:
                pass
    if total_nodes and observation_count / max(total_nodes, 1) < 0.02 and canonical_observation_units == 0:
        findings.append({"severity": "P0", "title": "observation_layer_missing", "detail": f"observations={observation_count}, total_nodes={total_nodes}"})
    elif canonical_observation_units:
        findings.append({"severity": "OK", "title": "canonical_observation_layer_present", "detail": f"canonical observation units={canonical_observation_units}"})
    if local.get("canonical_observations_in_local_json", 0) == 0 and canonical_observation_units == 0:
        findings.append({"severity": "P0", "title": "canonical_observations_not_emitted", "detail": "active offline JSON outputs contain no canonical_observations and no canonical DB layer exists"})
    if reconciliation.get("local_not_in_db_count", 0):
        findings.append({"severity": "P1", "title": "local_outputs_not_in_db", "detail": str(reconciliation.get("local_not_in_db_examples", [])[:3])})
    if reconciliation.get("db_not_local_count", 0):
        findings.append({"severity": "P1", "title": "db_offline_docs_missing_local_json", "detail": str(reconciliation.get("db_not_local_examples", [])[:3])})

    weekly_single = local.get("counts", {}).get("weekly|single_wrapped", 0)
    if weekly_single and canonical_observation_units == 0:
        findings.append({"severity": "P1", "title": "weekly_map_reduce_too_fragmented", "detail": f"weekly single_wrapped={weekly_single}; current weekly all-history is map chunks, not true global reduce"})
    elif weekly_single:
        findings.append({"severity": "P2", "title": "legacy_weekly_outputs_fragmented", "detail": f"legacy weekly single_wrapped={weekly_single}; V2 canonical DB layer is present, future weekly default is topic reduce"})

    english_only = sum(v for k, v in local.get("language", {}).items() if k.endswith("english_only"))
    if english_only:
        findings.append({"severity": "P2", "title": "language_inconsistency", "detail": f"english_only offline outputs={english_only}"})

    alias_group_count = int(entities.get("alias_group_count") or 0)
    if alias_group_count >= 20:
        top = entities.get("alias_groups") or []
        severity = "P2" if canonical_observation_units else "P1"
        findings.append({"severity": severity, "title": "entity_alias_fragmentation_auto_discovered", "detail": f"{alias_group_count} generic alias groups; top examples={[g.get('key') for g in top[:5]]}; V2 layer uses local alias normalization"})
    for info in entities.get("alias_groups") or []:
        key = str(info.get("key") or "unknown")
        if info.get("count", 0) >= 8:
            slug = slugify_id(key, prefix="entity")
            severity = "P2" if canonical_observation_units else "P1"
            findings.append({"severity": severity, "title": f"entity_alias_fragmentation_{slug}", "detail": f"{info.get('count')} aliases/examples: {info.get('examples')[:8]}"})

    for probe_id, info in recall_mix.items():
        mix = info.get("mix") or {}
        high = mix.get("local_canonical", 0) + mix.get("canonical", 0) + mix.get("offline_weekly", 0) + mix.get("offline_daily", 0)
        if high < 2:
            findings.append({"severity": "P1", "title": f"recall_high_level_weak_{probe_id}", "detail": f"top source mix={mix}"})
    return findings


def render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Hindsight Offline v2 Audit")
    lines.append(f"generated_at: {report['generated_at']}")
    lines.append("")
    lines.append("== Health / Stats ==")
    lines.append(json.dumps(report.get("health"), ensure_ascii=False))
    stats = report.get("stats") or {}
    lines.append(f"documents={stats.get('total_documents')} nodes={stats.get('total_nodes')} observations={(stats.get('nodes_by_fact_type') or {}).get('observation', stats.get('total_observations'))} pending={stats.get('pending_operations')} processing={stats.get('processing_operations')} failed={stats.get('failed_operations')}")
    lines.append("")
    lines.append("== Hindsight Store Layers (official API) ==")
    lines.append(f"documents_by_kind={report['db'].get('documents_by_kind')}")
    lines.append("facts_by_kind_type:")
    for row in report["db"].get("facts_by_kind_type", []):
        lines.append("  " + "\t".join(row))
    lines.append("")
    lines.append("== Local Offline Outputs ==")
    local = report.get("local_outputs") or {}
    lines.append(f"active_json={local.get('active_json_files')} backup_json={local.get('backup_json_files')} canonical_observations={local.get('canonical_observations_in_local_json')} v2_card_files={local.get('v2_card_files')} v2_card_observations={local.get('v2_card_observation_count')} v2_index_observations={local.get('v2_observation_index_count')}")
    lines.append(f"counts={local.get('counts')}")
    lines.append(f"language={local.get('language')}")
    lines.append("")
    lines.append("== Local/DB Reconciliation ==")
    rec = report.get("reconciliation") or {}
    lines.append(f"local_not_in_db={rec.get('local_not_in_db_count')} db_not_local={rec.get('db_not_local_count')}")
    lines.append("")
    lines.append("== Entity Fragmentation (auto-discovered) ==")
    ent = report.get("entities") or {}
    lines.append(f"method={ent.get('method')} total_entities={ent.get('total_entities')} alias_group_count={ent.get('alias_group_count')}")
    for info in ent.get("alias_groups", []):
        lines.append(f"{info.get('key')}: {info.get('count')} aliases; examples={info.get('examples')[:8]}")
    lines.append("")
    lines.append("== Recall Source Mix ==")
    for probe_id, info in (report.get("recall_mix") or {}).items():
        lines.append(f"{probe_id}: {info.get('mix')}")
    lines.append("")
    lines.append("== Findings ==")
    for f in report.get("findings", []):
        lines.append(f"[{f['severity']}] {f['title']}: {f['detail']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only audit for Offline Hindsight v2 readiness")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--bank", default=DEFAULT_BANK)
    ap.add_argument("--offline-root", default=str(DEFAULT_OFFLINE_ROOT))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-recall-probes", action="store_true")
    args = ap.parse_args()

    report: dict[str, Any] = {"generated_at": datetime.now().isoformat(timespec="seconds")}
    client = HindsightNativeClient(api=args.api, bank=args.bank)
    try:
        report["health"] = client.health()
    except Exception as e:
        report["health"] = {"error": repr(e)}
    try:
        report["stats"] = client.stats()
    except Exception as e:
        report["stats"] = {"error": repr(e)}

    report["db"] = audit_db(args.bank, api=args.api, client=client)
    report["hindsight_store"] = report["db"]
    local = audit_local_outputs(Path(args.offline_root).expanduser())
    report["local_outputs"] = {k: v for k, v in local.items() if k != "doc_ids"}
    db_ids = db_offline_doc_ids(args.bank, api=args.api, client=client)
    local_ids = set(local.get("doc_ids", {}).keys())
    local_not_db = sorted(local_ids - db_ids)
    db_not_local = sorted(db_ids - local_ids)
    report["reconciliation"] = {
        "db_offline_docs": len(db_ids),
        "local_active_doc_ids": len(local_ids),
        "local_not_in_db_count": len(local_not_db),
        "local_not_in_db_examples": local_not_db[:10],
        "db_not_local_count": len(db_not_local),
        "db_not_local_examples": db_not_local[:10],
    }
    report["entities"] = audit_entities(args.bank, api=args.api, client=client)
    probes = build_recall_probes(local)
    report["recall_probes"] = probes
    report["recall_mix"] = {} if args.no_recall_probes else audit_recall_mix(args.api, args.bank, probes, client=client)
    report["findings"] = build_findings(report.get("stats") or {}, report["db"], report["local_outputs"], report["reconciliation"], report["entities"], report["recall_mix"])

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_text(report))


if __name__ == "__main__":
    main()
