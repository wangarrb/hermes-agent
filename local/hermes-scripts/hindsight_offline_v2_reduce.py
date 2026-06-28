#!/usr/bin/env python3
"""Generic local reducer for Offline Hindsight v2 canonical cards.

Input: existing offline daily/weekly JSON outputs.
Output: local topic/global canonical cards under offline_reflect/v2_cards.

Design constraints:
- No LLM calls.
- No Hindsight writes.
- No project/domain presets. Topics are discovered from unit metadata and tags.
- Chinese is first-class: IDs fall back to stable hashes when text has no ASCII slug.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

HOME = Path.home()
HERMES_HOME = HOME / ".hermes"
DEFAULT_OFFLINE_ROOT = HERMES_HOME / "hindsight" / "offline_reflect"
DEFAULT_OUTPUT_ROOT = DEFAULT_OFFLINE_ROOT / "v2_cards"
DEFAULT_ALIAS_MAP = DEFAULT_OFFLINE_ROOT / "v2_aliases.json"
ACTIVE_CANON_TO_ALIASES: dict[str, list[str]] = {}

NOISE_TOPICS = {"", "unknown", "general", "global", "cross-topic", "cross_topic", "mixed", "misc", "other", "all"}
GENERIC_TAG_STOPWORDS = {
    "high", "medium", "low", "bug", "fix", "test", "tests", "report", "daily", "weekly",
    "json", "md", "txt", "file", "path", "script", "data", "result", "results", "note",
    "the", "and", "for", "with", "from", "into", "onto", "via", "using", "uses", "must",
    "should", "could", "would", "this", "that", "these", "those", "when", "then", "than",
    "every", "before", "after", "active", "configuration", "configured", "enabled", "disabled",
    "用户", "结果", "文件", "脚本", "报告", "测试", "问题", "修复", "配置",
}
SECTION_TYPE = {
    "canonical_observations": "canonical_observation",
    "knowledge_points": "technical_lesson",
    "user_preferences": "user_preference",
    "project_decisions": "project_decision",
    "tooling_lessons": "tooling_lesson",
    "risks": "risk",
    "open_questions": "open_question",
}
SECTION_ORDER = [
    "canonical_observations",
    "knowledge_points",
    "user_preferences",
    "project_decisions",
    "tooling_lessons",
    "risks",
    "open_questions",
]
GLOBAL_TYPES = {"user_preference", "tooling_lesson", "risk", "open_question", "system_rule"}
TYPE_DIVERSITY_ORDER = [
    "user_preference",
    "project_decision",
    "technical_lesson",
    "tooling_lesson",
    "risk",
    "open_question",
    "method_comparison",
    "system_rule",
]


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:n]


def slugify(text: str, *, prefix: str = "item") -> str:
    text = (text or "").strip()
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64]
    if slug:
        return slug
    return f"{prefix}-zh-{sha1_short(text, 10)}"


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text


def dedup_key(text: str) -> str:
    text = normalize_text(text)
    # Remove common version/status adjectives and separators so repeated daily/weekly
    # phrasings merge without needing domain-specific knowledge.
    text = re.sub(r"\b(active|finalized|optimization|optimized|configuration|pipeline|bug|issue)\b", " ", text)
    text = re.sub(r"[：:|,，;；()（）\[\]`'\"]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) > 18:
        text = " ".join(words[:18])
    return text


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_bad_output_backup(path: Path) -> bool:
    # Backup dirs are timestamped, e.g. _bad-output-backup-20260506-004802.
    # Do prefix matching on path parts instead of exact part equality.
    return any(part.startswith("_bad-output-backup") for part in path.parts)


def active_offline_json_files(root: Path, *, include_backups: bool = False) -> list[Path]:
    files = []
    for f in glob.glob(str(root / "**" / "*.json"), recursive=True):
        p = Path(f)
        if not include_backups and is_bad_output_backup(p):
            continue
        if p.name in {"offline_reflect_progress.json"}:
            continue
        if "v2_cards" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v is not None and str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def item_text(item: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(item, dict):
        meta = dict(item)
        parts = []
        for key in ["insight", "title", "conclusion", "summary", "decision", "risk", "question", "preference", "lesson"]:
            val = item.get(key)
            if val:
                parts.append(str(val))
        if not parts:
            # Keep concise readable scalar fields; avoid dumping large nested arrays.
            for key, val in item.items():
                if isinstance(val, (str, int, float, bool)) and str(val).strip():
                    parts.append(f"{key}: {val}")
        return " | ".join(parts).strip(), meta
    return str(item).strip(), {}


def confidence_rank(value: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get((value or "").lower(), 0)


def infer_tags(meta: dict[str, Any], unit_topic: str, text: str) -> list[str]:
    tags = []
    for key in ["tags", "entities", "topics"]:
        tags.extend(string_list(meta.get(key)))
    if unit_topic and unit_topic.lower() not in NOISE_TOPICS:
        tags.append(unit_topic)
    # Lightweight generic fallback for English identifiers and Chinese terms.
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,8}", text or ""):
        t = token.strip("-_ ")
        if not t:
            continue
        low = t.lower()
        if low in GENERIC_TAG_STOPWORDS or low in NOISE_TOPICS:
            continue
        if len(tags) >= 12:
            break
        tags.append(t)
    # Stable order, case-insensitive dedup.
    seen = set()
    out = []
    for t in tags:
        low = str(t).strip().lower()
        if not low or len(low) < 3 or low in seen or low in GENERIC_TAG_STOPWORDS:
            continue
        seen.add(low)
        out.append(str(t).strip())
    return out[:12]


def alias_key(text: str) -> str:
    """Normalize entity/topic names for alias matching without losing the display name."""
    t = (text or "").strip()
    t = re.sub(r"[`'\"()（）\[\]{}]", " ", t)
    t = re.sub(r"[._/\\-]+", " ", t)
    t = re.sub(r"\b(frontend|front end|baseline|native|raw|prior|selector|config|yaml|yml|json|script|pipeline)\b", " ", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def display_canonical(name: str) -> str:
    name = (name or "").strip()
    return name or "global"


def load_alias_map(path: Path | None) -> tuple[dict[str, str], dict[str, list[str]], dict[str, Any]]:
    """Load local alias map and return alias-key -> canonical display name.

    Generic script stays domain-neutral; project-specific aliases live in the
    local JSON file under offline_reflect/v2_aliases.json.
    """
    alias_to_canon: dict[str, str] = {}
    canon_to_aliases: dict[str, list[str]] = defaultdict(list)
    stats: dict[str, Any] = {"path": str(path) if path else None, "loaded": False, "canonical_count": 0, "alias_count": 0, "errors": []}
    if not path:
        return alias_to_canon, canon_to_aliases, stats
    path = path.expanduser()
    if not path.exists():
        stats["errors"].append("alias_map_missing")
        return alias_to_canon, canon_to_aliases, stats
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        stats["errors"].append(f"parse_error:{e}")
        return alias_to_canon, canon_to_aliases, stats
    if not isinstance(data, dict):
        stats["errors"].append("alias_map_not_object")
        return alias_to_canon, canon_to_aliases, stats
    for canon, aliases in data.items():
        canon_name = display_canonical(str(canon))
        vals = string_list(aliases)
        vals.append(canon_name)
        for val in vals:
            k = alias_key(val)
            if not k:
                continue
            alias_to_canon[k] = canon_name
            if val != canon_name and val not in canon_to_aliases[canon_name]:
                canon_to_aliases[canon_name].append(val)
    stats["loaded"] = True
    stats["canonical_count"] = len(canon_to_aliases)
    stats["alias_count"] = len(alias_to_canon)
    return alias_to_canon, canon_to_aliases, stats


def heuristic_alias_candidate(name: str) -> str:
    """Generic cleanup for common suffix-only fragmentation.

    This avoids hard-coding project names while still merging `FooFrontend`,
    `raw Foo`, `Foo baseline`, and config-file variants when no local alias is
    provided.
    """
    original = (name or "").strip()
    if not original:
        return original
    t = re.sub(r"[`'\"()（）\[\]{}]", " ", original)
    t = re.sub(r"\.(yaml|yml|json|py|md|txt)\b", "", t, flags=re.I)
    t = re.sub(r"\b(raw|native|baseline|prior|selector|config)\b", " ", t, flags=re.I)
    t = re.sub(r"\b(frontend|front end)\b", " ", t, flags=re.I)
    t = re.sub(r"[_\-/]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return original
    # Preserve all-caps identifiers and CamelCase-ish names better than .title().
    compact = t.replace(" ", "")
    if re.fullmatch(r"[A-Za-z0-9]{2,16}", compact):
        return compact
    return t


def canonicalize_alias(name: str, alias_to_canon: dict[str, str]) -> str:
    raw = display_canonical(name)
    direct = alias_to_canon.get(alias_key(raw))
    if direct:
        return direct
    candidate = heuristic_alias_candidate(raw)
    return alias_to_canon.get(alias_key(candidate), candidate)


def normalize_observation_aliases(observations: list[Observation], alias_to_canon: dict[str, str]) -> dict[str, Any]:
    changed = 0
    for obs in observations:
        old_topic = obs.topic
        obs.topic = canonicalize_alias(obs.topic, alias_to_canon)
        if obs.topic != old_topic:
            changed += 1
        tags = []
        seen = set()
        for t in obs.tags:
            nt = canonicalize_alias(str(t), alias_to_canon)
            k = nt.lower()
            if not nt or k in seen:
                continue
            seen.add(k)
            tags.append(nt)
            if nt != t:
                changed += 1
        obs.tags = tags[:12]
    return {"normalized_fields": changed}


def entity_aliases_for_observations(observations: list[Observation]) -> list[dict[str, Any]]:
    names = {o.topic for o in observations if o.topic}
    for o in observations:
        names.update(o.tags)
    rows = []
    for canon in sorted(names, key=lambda x: x.lower()):
        aliases = ACTIVE_CANON_TO_ALIASES.get(canon) or []
        if aliases:
            rows.append({"canonical": canon, "aliases": aliases[:20]})
    return rows[:80]


def canonical_topic_name(topic: str) -> str:
    t = (topic or "").strip()
    if not t:
        return "global"
    # Preserve Chinese/non-ASCII text; normalize ASCII case for stable grouping.
    if re.search(r"[A-Za-z]", t):
        return t.lower()
    return t


def infer_topic(unit: dict[str, Any], meta: dict[str, Any], text: str) -> str:
    unit_topic = str(unit.get("topic") or "").strip()
    if unit_topic.lower() not in NOISE_TOPICS:
        return canonical_topic_name(unit_topic)
    tags = infer_tags(meta, unit_topic, text)
    for t in tags:
        if t.lower() not in NOISE_TOPICS and t.lower() not in GENERIC_TAG_STOPWORDS:
            return canonical_topic_name(t)
    return "global"


@dataclass
class Observation:
    id: str
    insight: str
    type: str
    topic: str
    applicability: str
    evidence_ids: list[str]
    source_documents: list[str]
    source_section: str
    source_period: str
    confidence: str = "medium"
    valid_from: str | None = None
    valid_until: str | None = None
    tags: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "insight": self.insight,
            "type": self.type,
            "topic": self.topic,
            "applicability": self.applicability,
            "evidence_ids": self.evidence_ids,
            "source_documents": self.source_documents,
            "source_section": self.source_section,
            "source_period": self.source_period,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "supersedes": self.supersedes,
            "tags": self.tags,
        }


def observation_from_item(data: dict[str, Any], section: str, item: Any) -> Observation | None:
    unit = data.get("unit") or {}
    llm = data.get("llm_json") or {}
    doc_id = str(data.get("document_id") or "")
    text, meta = item_text(item)
    if not text:
        return None
    unit_topic = str(unit.get("topic") or llm.get("topic") or "")
    topic = infer_topic(unit, meta, text)
    obs_type = str(meta.get("type") or SECTION_TYPE.get(section) or "technical_lesson")
    if section == "canonical_observations" and obs_type == "canonical_observation":
        obs_type = str(meta.get("type") or "technical_lesson")
    confidence = str(meta.get("confidence") or "medium").lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    evidence_ids = []
    evidence_ids.extend(string_list(meta.get("evidence_ids")))
    evidence_ids.extend(string_list(meta.get("source_ids")))
    evidence_text = str(meta.get("evidence") or "")
    evidence_ids.extend(re.findall(r"\b(?:fact_id=)?[0-9a-f]{8}-[0-9a-f-]{27,}\b", evidence_text, flags=re.I))
    if doc_id:
        evidence_ids.insert(0, doc_id)
    for sid in string_list(unit.get("source_ids"))[:20]:
        evidence_ids.append(sid)
    # Dedup while preserving order.
    seen = set()
    evidence_ids = [e for e in evidence_ids if e and not (e in seen or seen.add(e))]
    tags = infer_tags(meta, unit_topic, text)
    applicability = str(meta.get("applicability") or meta.get("scope") or (f"topic:{topic}" if topic != "global" else "global/cross-topic"))
    source_period = str(unit.get("period") or llm.get("period") or "")
    stable = "||".join([obs_type, topic, normalize_text(text), "|".join(evidence_ids[:3])])
    obs_id = str(meta.get("id") or f"obs:{slugify(topic, prefix='topic')}:{sha1_short(stable)}")
    return Observation(
        id=obs_id,
        insight=text,
        type=obs_type,
        topic=topic,
        applicability=applicability,
        evidence_ids=evidence_ids[:40],
        source_documents=[doc_id] if doc_id else [],
        source_section=section,
        source_period=source_period,
        confidence=confidence,
        valid_from=str(meta.get("valid_from") or source_period or "") or None,
        valid_until=meta.get("valid_until"),
        supersedes=string_list(meta.get("supersedes")),
        tags=tags,
    )


def count_backup_json_files(root: Path) -> int:
    return sum(1 for f in glob.glob(str(root / "**" / "*.json"), recursive=True) if is_bad_output_backup(Path(f)))


def collect_observations(root: Path, *, include_backups: bool = False) -> tuple[list[Observation], dict[str, Any]]:
    observations: list[Observation] = []
    files = active_offline_json_files(root, include_backups=include_backups)
    stats: dict[str, Any] = {
        "input_files": len(files),
        "backup_json_excluded": 0 if include_backups else count_backup_json_files(root),
        "include_backups": include_backups,
        "parse_errors": 0,
        "sections": Counter(),
        "topics": Counter(),
    }
    for path in files:
        data = load_json(path)
        if not data:
            stats["parse_errors"] += 1
            continue
        llm = data.get("llm_json") or {}
        unit = data.get("unit") or {}
        if not llm:
            continue
        for section in SECTION_ORDER:
            raw_items = llm.get(section) or []
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            if not isinstance(raw_items, list):
                raw_items = [raw_items]
            for item in raw_items:
                obs = observation_from_item(data, section, item)
                if obs:
                    observations.append(obs)
                    stats["sections"][section] += 1
                    stats["topics"][obs.topic] += 1
        topic = str(unit.get("topic") or llm.get("topic") or "").strip()
        if topic:
            stats["topics"][topic] += 0
    stats["sections"] = dict(stats["sections"])
    stats["topics"] = dict(stats["topics"].most_common(50))
    return observations, stats


def dedup_observations(observations: list[Observation]) -> list[Observation]:
    best: dict[str, Observation] = {}
    for obs in observations:
        key = dedup_key(obs.insight)
        if not key:
            continue
        old = best.get(key)
        if old is None:
            best[key] = obs
            continue
        if confidence_rank(obs.confidence) > confidence_rank(old.confidence):
            merged = obs
            merged.evidence_ids = list(dict.fromkeys(obs.evidence_ids + old.evidence_ids))[:40]
            merged.source_documents = list(dict.fromkeys(obs.source_documents + old.source_documents))[:20]
            best[key] = merged
        else:
            old.evidence_ids = list(dict.fromkeys(old.evidence_ids + obs.evidence_ids))[:40]
            old.source_documents = list(dict.fromkeys(old.source_documents + obs.source_documents))[:20]
            old.tags = list(dict.fromkeys(old.tags + obs.tags))[:12]
    return list(best.values())


def observation_rank_key(obs: Observation) -> tuple[int, int, str]:
    return (confidence_rank(obs.confidence), len(obs.evidence_ids), obs.valid_from or "")


def select_observations_for_card(observations: list[Observation], max_obs: int) -> list[Observation]:
    """Select observations with type diversity before filling by rank.

    A pure evidence-count sort can crowd out user preferences, risks, and open
    questions in large technical topics. Cards are high-level layers, so reserve
    a small per-type quota first, then fill remaining slots by rank.
    """
    ranked = sorted(observations, key=observation_rank_key, reverse=True)
    if len(ranked) <= max_obs:
        return ranked
    by_type: dict[str, list[Observation]] = defaultdict(list)
    for obs in ranked:
        by_type[obs.type].append(obs)
    selected: list[Observation] = []
    selected_ids: set[str] = set()
    per_type_quota = max(3, min(10, max_obs // 10))
    for typ in TYPE_DIVERSITY_ORDER:
        for obs in by_type.get(typ, [])[:per_type_quota]:
            if obs.id in selected_ids:
                continue
            selected.append(obs)
            selected_ids.add(obs.id)
            if len(selected) >= max_obs:
                return selected
    for obs in ranked:
        if obs.id in selected_ids:
            continue
        selected.append(obs)
        selected_ids.add(obs.id)
        if len(selected) >= max_obs:
            break
    return selected


def build_card(scope: str, topic: str, observations: list[Observation], *, max_obs: int) -> dict[str, Any]:
    observations = dedup_observations(observations)
    observations = select_observations_for_card(observations, max_obs)
    type_counts = Counter(o.type for o in observations)
    tag_counts = Counter(t for o in observations for t in o.tags)
    source_docs = sorted({d for o in observations for d in o.source_documents if d})
    periods = sorted({o.source_period for o in observations if o.source_period})
    card_id = f"offline-v2-card::{scope}::{slugify(topic, prefix='topic')}::{sha1_short('|'.join(source_docs) + topic)}"
    summary = [
        f"{topic}: {len(observations)} canonical observations from {len(source_docs)} source documents.",
        "Top observation types: " + ", ".join(f"{k}={v}" for k, v in type_counts.most_common(6)),
    ]
    return {
        "card_id": card_id,
        "schema_version": "offline-hindsight-v2-card/0.1",
        "scope": scope,
        "topic": topic,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "language_policy": "auto; Chinese and English are both first-class; preserve commands/paths/model names as evidence",
        "input_stats": {
            "observation_count": len(observations),
            "source_document_count": len(source_docs),
            "periods": periods[:20],
        },
        "executive_summary": summary,
        "canonical_observations": [o.as_dict() for o in observations],
        "entity_aliases": entity_aliases_for_observations(observations),
        "conflicts": [],
        "open_questions": [o.as_dict() for o in observations if o.type == "open_question"][:20],
        "evidence_index": source_docs[:200],
        "top_tags": dict(tag_counts.most_common(30)),
    }


def build_cards(observations: list[Observation], *, scope: str, max_topics: int, max_obs: int) -> list[dict[str, Any]]:
    observations = dedup_observations(observations)
    cards: list[dict[str, Any]] = []
    if scope in {"topic", "all"}:
        by_topic: dict[str, list[Observation]] = defaultdict(list)
        for obs in observations:
            topic = obs.topic or "global"
            if topic.lower() in NOISE_TOPICS:
                topic = "global"
            by_topic[topic].append(obs)
        ranked_topics = sorted(by_topic.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:max_topics]
        for topic, obs_list in ranked_topics:
            if topic == "global" and scope == "topic":
                continue
            cards.append(build_card("topic", topic, obs_list, max_obs=max_obs))
    if scope in {"global", "all"}:
        global_obs = [o for o in observations if o.type in GLOBAL_TYPES or (o.topic or "").lower() in NOISE_TOPICS or o.topic == "global"]
        if not global_obs:
            global_obs = observations[:]
        cards.append(build_card("global", "global", global_obs, max_obs=max_obs))
    return cards


def render_card_markdown(card: dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Offline Hindsight v2 Card: {card['topic']}")
    lines.append("")
    lines.append(f"card_id: `{card['card_id']}`")
    lines.append(f"scope: `{card['scope']}`")
    lines.append(f"generated_at: `{card['generated_at']}`")
    lines.append("")
    lines.append("## Executive Summary")
    for s in card.get("executive_summary") or []:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Canonical Observations")
    for obs in card.get("canonical_observations") or []:
        lines.append(f"- [{obs.get('confidence')}] ({obs.get('type')}) {obs.get('insight')}")
        if obs.get("applicability"):
            lines.append(f"  - applicability: {obs.get('applicability')}")
        if obs.get("evidence_ids"):
            lines.append(f"  - evidence: {', '.join(map(str, obs.get('evidence_ids')[:5]))}")
        if obs.get("tags"):
            lines.append(f"  - tags: {', '.join(map(str, obs.get('tags')[:8]))}")
    lines.append("")
    lines.append("## Evidence Index")
    for e in card.get("evidence_index") or []:
        lines.append(f"- {e}")
    return "\n".join(lines).strip() + "\n"


def write_observations_index(observations: list[Observation], output_root: Path) -> Path:
    """Write a local-only retrieval index with all deduped observations.

    Cards are intentionally compact, but eval/retrieval needs access to detailed
    numeric observations that may not fit into the top-N card body. This index is
    still local-only: no LLM calls and no Hindsight writes.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "observations_index.jsonl"
    rows = []
    for obs in sorted(dedup_observations(observations), key=lambda o: (o.topic, o.type, o.id)):
        rows.append(json.dumps(obs.as_dict(), ensure_ascii=False))
    index_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return index_path


def write_cards(cards: list[dict[str, Any]], output_root: Path, *, observations: list[Observation] | None = None) -> list[Path]:
    written: list[Path] = []
    for card in cards:
        subdir = output_root / ("topics" if card["scope"] == "topic" else "global")
        subdir.mkdir(parents=True, exist_ok=True)
        name = slugify(str(card["topic"]), prefix="topic")
        json_path = subdir / f"{name}.json"
        md_path = subdir / f"{name}.md"
        json_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_card_markdown(card), encoding="utf-8")
        written.extend([json_path, md_path])
    index_count = 0
    if observations is not None:
        index_path = write_observations_index(observations, output_root)
        index_count = sum(1 for _ in index_path.open(encoding="utf-8"))
        written.append(index_path)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "card_count": len(cards),
        "observation_index_count": index_count,
        "cards": [{"card_id": c["card_id"], "scope": c["scope"], "topic": c["topic"], "observation_count": len(c.get("canonical_observations") or [])} for c in cards],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic local reducer for Offline Hindsight v2 canonical cards")
    ap.add_argument("--offline-root", default=str(DEFAULT_OFFLINE_ROOT))
    ap.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--scope", choices=["topic", "global", "all"], default="all")
    ap.add_argument("--mode", choices=["dry-run", "local"], default="dry-run")
    ap.add_argument("--max-topics", type=int, default=20)
    ap.add_argument("--max-observations-per-card", type=int, default=80)
    ap.add_argument("--include-backups", action="store_true", help="Also read _bad-output-backup* JSON files; off by default for normal builds")
    ap.add_argument("--alias-map", default=str(DEFAULT_ALIAS_MAP), help="Local JSON alias map for entity/topic normalization; set empty string to disable")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    root = Path(args.offline_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    observations, stats = collect_observations(root, include_backups=args.include_backups)
    alias_path = Path(args.alias_map).expanduser() if str(args.alias_map or "").strip() else None
    alias_to_canon, canon_to_aliases, alias_stats = load_alias_map(alias_path)
    global ACTIVE_CANON_TO_ALIASES
    ACTIVE_CANON_TO_ALIASES = canon_to_aliases
    alias_normalization = normalize_observation_aliases(observations, alias_to_canon)
    stats["alias_map"] = alias_stats
    stats["alias_normalization"] = alias_normalization
    cards = build_cards(observations, scope=args.scope, max_topics=args.max_topics, max_obs=args.max_observations_per_card)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "offline_root": str(root),
        "output_root": str(output_root),
        "stats": stats,
        "collected_observations": len(observations),
        "card_count": len(cards),
        "cards": [{"scope": c["scope"], "topic": c["topic"], "observation_count": len(c.get("canonical_observations") or []), "card_id": c["card_id"]} for c in cards],
        "writes": [],
        "safety": "No LLM calls. No Hindsight writes. Local files only when --mode local.",
    }
    if args.mode == "local":
        written = write_cards(cards, output_root, observations=observations)
        report["writes"] = [str(p) for p in written]
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Offline Hindsight v2 Reduce")
        print(f"mode={args.mode} scope={args.scope}")
        print(f"input_files={stats.get('input_files')} collected_observations={len(observations)} cards={len(cards)}")
        print(f"topics={list((stats.get('topics') or {}).keys())[:12]}")
        for c in report["cards"][:20]:
            print(f"  [{c['scope']}] {c['topic']}: observations={c['observation_count']} id={c['card_id']}")
        if report["writes"]:
            print("written:")
            for p in report["writes"][:20]:
                print(f"  {p}")
        print(report["safety"])


if __name__ == "__main__":
    main()
